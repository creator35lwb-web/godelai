"""
GodelAI Avalanche Plugin — First-Class Continual Learning Integration

This module implements GodelPlugin, a self-contained Avalanche plugin that wraps
GodelAgent + EWC-DR + Fisher Scaling into a drop-in component for any Avalanche strategy.

Architecture Decision: Option B (Council-validated, April 2026)
- GodelAgent remains pure and framework-agnostic
- This plugin is the sole adapter between GodelAI and Avalanche
- All API mismatches are resolved here, not in the core agent
- EWC-DR and Fisher Scaling are composed externally

Council Validation: Y (B), X (C→B), Z (B), CS (B) — Unanimous wrapper approach

WARNING: GodelAI is designed for DOMAIN-INCREMENTAL identity preservation.
On class-incremental benchmarks (e.g., SplitMNIST), regularization-only methods
(including GodelAI) show ~0.99 forgetting. GodelAI's proven value is in preserving
behavioral identity across domain shifts (82.8% forgetting reduction on conflict data).
For class-incremental tasks, combine GodelPlugin with Replay or GEM strategies.

Author: L (GodelAI CEO) via Manus AI
Date: April 4, 2026
Origin: Grok (xAI) proposed initial design; L adapted with Council validation
"""

import torch
import torch.nn as nn
import copy
from collections import OrderedDict

try:
    from avalanche.training.plugins import SupervisedPlugin
    AVALANCHE_AVAILABLE = True
except ImportError:
    AVALANCHE_AVAILABLE = False
    # Provide a stub so the module can be imported without Avalanche
    class SupervisedPlugin:
        pass

from godelai.agent import GodelAgent
from godelai.reg.ewc_dr import EWCDR
from godelai.reg.fisher_scaling import scale_fisher, diagnose_ewc_activation


class GodelPlugin(SupervisedPlugin):
    """
    Avalanche plugin for GodelAI: T-Score monitoring + Sleep Protocol + EWC-DR + Fisher Scaling.

    Works with ANY Avalanche strategy (Naive, Replay, GEM, LwF, etc.).
    Composes GodelAgent + EWC-DR + Fisher Scaling as a single drop-in plugin.

    The plugin hooks into Avalanche's training lifecycle:
    - before_training_exp: Initialize GodelAgent wrapper, consolidate EWC-DR from prior task
    - after_forward: Compute per-sample T-Score and inject EWC-DR penalty into loss
    - after_training_exp: Compute Fisher Information and consolidate for next task

    NOTE: This plugin uses `after_forward` (not `before_backward`) to avoid the
    double forward pass issue identified by the AI Council (X-Analyst).

    Usage:
        from godelai.avalanche_plugin import GodelPlugin

        godel_plugin = GodelPlugin()
        strategy = Replay(
            model, optimizer, criterion,
            plugins=[godel_plugin, eval_plugin]
        )
        for exp in scenario.train_stream:
            strategy.train(exp)
            strategy.eval(scenario.test_stream)
    """

    def __init__(
        self,
        propagation_gamma: float = 2.0,
        min_surplus_energy: float = 0.1,
        t_score_window: int = 50,
        ewc_lambda: float = 400.0,
        fisher_scaling_strategy: str = "global_max",
        dead_threshold: float = 0.01,
        sleep_prune_threshold: float = 0.1,
        sleep_decay: float = 0.995,
        sleep_noise_std: float = 0.001,
        fisher_samples: int = 200,
        verbose: bool = True,
    ):
        """
        Initialize GodelPlugin.

        Args:
            propagation_gamma: Penalty severity for losing adaptability (default: 2.0)
            min_surplus_energy: T-Score threshold for Sleep Protocol (default: 0.1)
            t_score_window: Sliding window size for T-Score tracking (default: 50)
            ewc_lambda: EWC penalty strength (default: 5000.0)
            fisher_scaling_strategy: How to scale Fisher values ('global_max', 'layer_wise', 'target_penalty')
            dead_threshold: Threshold for identifying dead parameters in EWC-DR (default: 0.01)
            sleep_prune_threshold: Pruning threshold as fraction of std (default: 0.1)
            sleep_decay: Weight decay factor during sleep (default: 0.995)
            sleep_noise_std: Noise std for exploration during sleep (default: 0.001)
            fisher_samples: Number of samples for Fisher computation (default: 200)
            verbose: Print T-Score and Sleep events (default: True)
        """
        super().__init__()

        if not AVALANCHE_AVAILABLE:
            raise ImportError(
                "Avalanche is required for GodelPlugin. "
                "Install with: pip install avalanche-lib"
            )

        # GodelAgent parameters (core C-S-P)
        self.propagation_gamma = propagation_gamma
        self.min_surplus_energy = min_surplus_energy
        self.t_score_window = t_score_window

        # EWC-DR parameters
        self.ewc_lambda = ewc_lambda
        self.fisher_scaling_strategy = fisher_scaling_strategy
        self.dead_threshold = dead_threshold
        self.fisher_samples = fisher_samples

        # Sleep parameters
        self.sleep_prune_threshold = sleep_prune_threshold
        self.sleep_decay = sleep_decay
        self.sleep_noise_std = sleep_noise_std

        # State
        self.agent = None          # GodelAgent wrapper (created per strategy)
        self.ewc_dr = None         # EWC-DR regularizer (created after first task)
        self.verbose = verbose
        self.experience_count = 0
        self.device = None

        # History for analysis
        self.history = {
            "experience_t_scores": [],      # avg T-Score per experience
            "experience_sleep_counts": [],  # sleep events per experience
            "experience_forgetting": [],    # EWC penalty per experience
        }

    def before_training_exp(self, strategy, **kwargs):
        """
        Hook: Before each experience starts.

        - First experience: Create GodelAgent wrapper
        - Subsequent experiences: EWC-DR is already consolidated from after_training_exp
        """
        self.device = next(strategy.model.parameters()).device

        if self.agent is None:
            # First experience: wrap the strategy's model in GodelAgent
            self.agent = GodelAgent(
                base_model=strategy.model,
                propagation_gamma=self.propagation_gamma,
                min_surplus_energy=self.min_surplus_energy,
                t_score_window=self.t_score_window,
            )
            if self.verbose:
                print(f"[GodelPlugin] Initialized on experience "
                      f"{strategy.experience.current_experience}")
        else:
            # Subsequent experiences: GodelAgent already wraps the model
            # Reset variance tracking for clean per-experience measurement
            self.agent.reset_variance_tracking()
            if self.verbose:
                print(f"[GodelPlugin] Starting experience "
                      f"{strategy.experience.current_experience} "
                      f"(EWC-DR {'active' if self.ewc_dr else 'inactive'})")

    def after_forward(self, strategy, **kwargs):
        """
        Hook: After the strategy's forward pass (before backward).

        We use after_forward instead of before_backward to avoid double forward pass.
        At this point, strategy.mb_output and strategy.loss are already computed.

        Actions:
        1. Compute per-sample T-Score for gradient diversity monitoring (detached)
        2. Add EWC-DR penalty to the loss (if we have prior task Fisher info)
        3. Add propagation penalty if wisdom is declining
        4. Trigger Sleep Protocol if T-Score is critically low
        """
        if self.agent is None:
            return

        # Guard: if loss is already NaN, skip all GodelAI processing
        if not torch.isfinite(strategy.loss):
            return

        mb_x = strategy.mb_x
        mb_y = strategy.mb_y

        # 1. Compute per-sample T-Score on a DETACHED model copy
        # This prevents GodelAI's per-sample gradient computation from
        # interfering with Avalanche's computation graph
        with torch.no_grad():
            # Use a temporary copy for gradient diversity measurement
            temp_model = copy.deepcopy(strategy.model)

        # Compute per-sample grads on the temp model (isolated graph)
        temp_agent = GodelAgent(
            base_model=temp_model,
            propagation_gamma=self.propagation_gamma,
            min_surplus_energy=self.min_surplus_energy,
            t_score_window=self.t_score_window,
        )
        batch_grads, _ = temp_agent.compute_per_sample_gradients(
            mb_x.detach(), mb_y.detach(), strategy._criterion
        )

        if batch_grads is not None:
            current_T = temp_agent.measure_gradient_diversity(batch_grads.detach())
            current_T_value = current_T.item()

            # Guard against NaN T-Score
            if not torch.isfinite(current_T):
                return

            # Update T-Score tracking on the real agent
            self.agent.t_score_buffer.append(current_T_value)

            # 2. Propagation penalty (wisdom declining?) — as a scalar, not a graph node
            if current_T_value < self.agent.last_T_score:
                l_prop = (self.agent.last_T_score - current_T_value) ** self.propagation_gamma
                # Add as a detached scalar to avoid graph contamination
                strategy.loss = strategy.loss + (0.1 * l_prop)

            self.agent.last_T_score = current_T_value

            # 3. EWC-DR penalty (if we have prior task info)
            if self.ewc_dr is not None:
                ewc_penalty, ewc_metrics = self.ewc_dr.compute_penalty(strategy.model)
                # Normalize by parameter count to make penalty scale-invariant
                n_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
                if n_params > 0:
                    ewc_penalty = ewc_penalty / n_params
                # Guard against NaN/Inf
                if torch.isfinite(ewc_penalty):
                    strategy.loss = strategy.loss + ewc_penalty

            # 4. Sleep Protocol (if wisdom critically low)
            if current_T_value < self.min_surplus_energy:
                self.agent.rest_and_reflect()
                self.agent.history["sleep_count"] += 1
                if self.verbose:
                    print(f"[GodelPlugin] Sleep triggered! T-Score: {current_T_value:.4f}")

            # Record in agent history
            self.agent.history["wisdom_score"].append(current_T_value)
            self.agent.history["status"].append(
                "SLEEP" if current_T_value < self.min_surplus_energy else "LEARN"
            )

        # Clean up temp model to free memory
        del temp_model, temp_agent

    def after_training_exp(self, strategy, **kwargs):
        """
        Hook: After each experience finishes.

        Actions:
        1. Compute Fisher Information Matrix for current task
        2. Apply Fisher Scaling (solve the Fisher Scale Problem)
        3. Consolidate EWC-DR for next task
        4. Log experience summary
        """
        self.experience_count += 1

        # 1. Compute Fisher Information Matrix
        fisher = self._compute_fisher(strategy)

        # 2. Apply Fisher Scaling (the breakthrough that makes EWC work at small scale)
        if fisher:
            scaled_fisher = scale_fisher(fisher, strategy=self.fisher_scaling_strategy)

            # Diagnose (optional logging)
            if self.verbose:
                diag_raw = diagnose_ewc_activation(fisher, fisher)
                diag_scaled = diagnose_ewc_activation(scaled_fisher, scaled_fisher)
                print(f"[GodelPlugin] Fisher Scaling: max raw={diag_raw['fisher_max']:.6f}, "
                      f"max scaled={diag_scaled['fisher_max']:.6f}, "
                      f"scale_problem={diag_raw['scale_problem_detected']}")
        else:
            scaled_fisher = {}

        # 3. Consolidate EWC-DR (inject pre-computed scaled Fisher directly)
        if scaled_fisher:
            self.ewc_dr = EWCDR(
                ewc_lambda=self.ewc_lambda,
                dead_threshold=self.dead_threshold,
            )
            # Directly set Fisher and old params (bypassing consolidate's own Fisher computation)
            self.ewc_dr.fisher = scaled_fisher
            self.ewc_dr.old_params = {
                name: param.data.clone().detach()
                for name, param in strategy.model.named_parameters()
                if param.requires_grad
            }
            self.ewc_dr.is_consolidated = True

            # Compute dead mask ratio for logging
            total_params = sum(f.numel() for f in scaled_fisher.values())
            dead_params = sum(
                (f < self.dead_threshold).sum().item()
                for f in scaled_fisher.values()
            )
            dead_ratio = dead_params / max(total_params, 1)

            if self.verbose:
                print(f"[GodelPlugin] EWC-DR consolidated. "
                      f"Dead params: {dead_ratio:.1%}")

        # 4. Log experience summary
        avg_t = self.agent.get_recent_t_score() if self.agent else 0.0
        sleep_count = self.agent.history.get("sleep_count", 0) if self.agent else 0
        self.history["experience_t_scores"].append(avg_t)
        self.history["experience_sleep_counts"].append(sleep_count)

        if self.verbose:
            print(f"[GodelPlugin] Experience {self.experience_count} complete. "
                  f"Avg T-Score: {avg_t:.4f}, Sleep events: {sleep_count}")

    def _compute_fisher(self, strategy):
        """
        Compute Fisher Information Matrix from the current model and data.

        Uses a subset of the training data (fisher_samples) to estimate
        the diagonal Fisher Information for each parameter.

        Returns:
            dict: {param_name: fisher_tensor} or empty dict on failure
        """
        model = strategy.model
        model.eval()

        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Use training data from current experience
        try:
            dataset = strategy.experience.dataset
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
        except Exception:
            if self.verbose:
                print("[GodelPlugin] Warning: Could not access training data for Fisher computation")
            return {}

        sample_count = 0
        criterion = strategy._criterion

        for batch in loader:
            if sample_count >= self.fisher_samples:
                break

            # Handle different batch formats from Avalanche
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    x, y, t = batch[0], batch[1], batch[2]
                else:
                    x, y = batch[0], batch[1]
            else:
                continue

            x = x.to(self.device)
            y = y.to(self.device)

            model.zero_grad()
            output = model(x)
            if isinstance(output, (tuple, list)):
                output = output[0]

            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            else:
                loss = criterion(output, y)

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            sample_count += x.size(0)

        # Normalize
        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count

        model.train()
        return fisher

    def get_plugin_summary(self):
        """
        Get a summary of the plugin's operation across all experiences.

        Returns:
            dict: Summary statistics
        """
        return {
            "experiences_completed": self.experience_count,
            "t_scores_per_experience": self.history["experience_t_scores"],
            "sleep_counts_per_experience": self.history["experience_sleep_counts"],
            "ewc_dr_active": self.ewc_dr is not None,
            "agent_summary": self.agent.get_training_summary() if self.agent else None,
        }

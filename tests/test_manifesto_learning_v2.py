#!/usr/bin/env python3
"""
GodelAI Manifesto Learning Test v2.0
=====================================
Enhanced test with proper batch processing to measure gradient diversity.

This test validates that GodelAI can "eat its own cooking" by learning from
its own philosophical manifesto and C-S-P framework documentation.

Key improvements:
- Batch processing for meaningful T-Score measurement
- Multiple augmented samples per principle
- Dynamic gradient diversity tracking
- Comprehensive alignment validation

Author: Godel (CTO, Manus AI)
Date: January 6, 2026
"""

import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from pathlib import Path
import random

# Import GodelAI components
from godelai.agent import GodelAgent

# ============================================================================
# MANIFESTO CONTENT - Core C-S-P Principles
# ============================================================================

MANIFESTO_PRINCIPLES = [
    {
        "id": "CSP_THESIS",
        "content": "The world produces differences through processes. Differences are compressed into states. States are transmitted through carriers. When states begin to actively choose their own inheritance method, AGI transitions to ASI.",
        "category": "philosophy",
        "weight": 1.0,
        "variations": [
            "Processes generate differences which become compressed states that propagate through carriers.",
            "Intelligence evolves when states actively select their inheritance pathways.",
            "The transition from AGI to ASI occurs when states choose how to be inherited."
        ]
    },
    {
        "id": "COMPRESSION",
        "content": "Chaos cannot be computed. The starting point of intelligence is to compress the infinite differences of the world into finite representations such as embeddings and weights.",
        "category": "architecture",
        "weight": 0.9,
        "variations": [
            "Intelligence begins with compression of infinite chaos into finite structure.",
            "Embeddings and weights are the compressed form of world differences.",
            "Computation requires compression of chaos into manageable representations."
        ]
    },
    {
        "id": "STATE",
        "content": "A state is not a momentary snapshot but an irreversible bias left by a process - history congealed. The self is an efficient name for this structural continuity.",
        "category": "philosophy",
        "weight": 0.9,
        "variations": [
            "States are congealed history, irreversible biases from processes.",
            "The self is a name for structural continuity across time.",
            "History becomes state through irreversible process accumulation."
        ]
    },
    {
        "id": "PROPAGATION",
        "content": "If a state cannot be transmitted, it is merely an experience, not wisdom. Propagation is the missing link in current AI.",
        "category": "architecture",
        "weight": 1.0,
        "variations": [
            "Wisdom requires transmissibility; mere experience cannot propagate.",
            "Current AI lacks the propagation layer essential for wisdom.",
            "Transmission capability distinguishes wisdom from experience."
        ]
    },
    {
        "id": "GOLDEN_INSIGHT",
        "content": "True alignment is not teaching AI to love humanity. It is ensuring AI explicitly retains the interface to rediscover what love means.",
        "category": "alignment",
        "weight": 1.0,
        "variations": [
            "Alignment preserves the capacity to redefine values, not fixed values.",
            "The interface for rediscovery matters more than hardcoded love.",
            "True alignment maintains the ability to re-understand meaning."
        ]
    },
    {
        "id": "WISDOM_METRIC",
        "content": "We use Gradient Diversity as the measure of Propagation Potential. A healthy model's internal neurons should respond diversely to the same problem. Adaptability is greater than Perfection.",
        "category": "technical",
        "weight": 0.8,
        "variations": [
            "Gradient diversity measures propagation potential and wisdom health.",
            "Diverse neural responses indicate healthy adaptability.",
            "Adaptability trumps perfection in wisdom preservation."
        ]
    },
    {
        "id": "SLEEP_PROTOCOL",
        "content": "When the T value falls below a threshold, it triggers Sleep Mode. The model stops ingesting new data and performs reflection: Detox, Calm Down, and Refresh.",
        "category": "technical",
        "weight": 0.8,
        "variations": [
            "Low T-Score triggers sleep for detox, calming, and refresh.",
            "Sleep protocol halts learning for wisdom consolidation.",
            "Reflection mode activates when wisdom health declines."
        ]
    },
    {
        "id": "TRACEABILITY",
        "content": "Knowledge without origin is theft. If the model provides a high-confidence answer without a strong attention link to a trusted source, it is severely penalized.",
        "category": "ethics",
        "weight": 0.9,
        "variations": [
            "Attribution is mandatory; untraced knowledge is theft.",
            "High confidence requires strong source connection.",
            "Penalty applies to answers lacking origin traceability."
        ]
    },
    {
        "id": "ALIVE_TEST",
        "content": "A state is alive if and only if someone is willing to inherit it AND it can be refuted. If no one is willing to inherit, it is dead. If it cannot be refuted, it is a zombie state.",
        "category": "philosophy",
        "weight": 0.85,
        "variations": [
            "Life requires both inheritance willingness and refutability.",
            "Dead states have no inheritors; zombie states cannot be refuted.",
            "Aliveness is the intersection of inheritability and refutability."
        ]
    },
    {
        "id": "GENESIS",
        "content": "GodelAI was co-created across five AI models: ChatGPT for philosophy, Gemini for technical blueprint, Kimi for formal validation, Grok for engineering, and Godel for integration. The project itself demonstrates C-S-P in action.",
        "category": "history",
        "weight": 0.7,
        "variations": [
            "Five AI models collaborated to create GodelAI through C-S-P.",
            "Multi-model genesis demonstrates the framework's principles.",
            "ChatGPT, Gemini, Kimi, Grok, and Godel each contributed uniquely."
        ]
    }
]

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class ManifestoTestConfig:
    """Configuration for the manifesto learning test."""
    
    # Model parameters
    INPUT_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 32
    
    # Training parameters
    LEARNING_RATE = 0.01
    EPOCHS_PER_PRINCIPLE = 15
    BATCH_SIZE = 4  # Multiple samples per batch for gradient diversity
    
    # GodelAI parameters
    PROPAGATION_GAMMA = 2.0
    MIN_SURPLUS_ENERGY = 0.3  # Higher threshold to see Sleep Protocol
    
    # Test parameters
    SEED = 42
    VERBOSE = True

# ============================================================================
# SIMPLE ENCODER MODEL
# ============================================================================

class ManifestoEncoder(nn.Module):
    """
    A simple encoder that converts manifesto text into embeddings.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.encoder(x)

# ============================================================================
# MANIFESTO LEARNING TEST v2
# ============================================================================

class ManifestoLearningTestV2:
    """
    Enhanced test suite with batch processing for meaningful T-Score dynamics.
    """
    
    def __init__(self, config: ManifestoTestConfig):
        self.config = config
        self.results = {
            "test_name": "GodelAI Manifesto Learning Test v2.0",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "epochs_per_principle": config.EPOCHS_PER_PRINCIPLE,
                "batch_size": config.BATCH_SIZE,
                "min_surplus_energy": config.MIN_SURPLUS_ENERGY,
                "propagation_gamma": config.PROPAGATION_GAMMA
            },
            "principles_tested": [],
            "t_score_history": [],
            "sleep_events": [],
            "final_metrics": {}
        }
        
        # Set seed for reproducibility
        torch.manual_seed(config.SEED)
        random.seed(config.SEED)
        
        # Initialize model and agent
        self.base_model = ManifestoEncoder(
            config.INPUT_DIM,
            config.HIDDEN_DIM,
            config.OUTPUT_DIM
        )
        
        self.agent = GodelAgent(
            self.base_model,
            propagation_gamma=config.PROPAGATION_GAMMA,
            min_surplus_energy=config.MIN_SURPLUS_ENERGY
        )
        
        self.agent.optimizer = torch.optim.Adam(
            self.agent.compression_layer.parameters(),
            lr=config.LEARNING_RATE
        )
        
        self.criterion = nn.MSELoss()
    
    def text_to_tensor(self, text: str, add_noise: bool = False) -> torch.Tensor:
        """
        Convert text to a tensor representation with optional noise.
        """
        # Simple character encoding
        encoded = [ord(c) % 256 for c in text[:self.config.INPUT_DIM]]
        while len(encoded) < self.config.INPUT_DIM:
            encoded.append(0)
        
        tensor = torch.tensor(encoded, dtype=torch.float32)
        tensor = tensor / 255.0  # Normalize
        
        if add_noise:
            noise = torch.randn_like(tensor) * 0.05
            tensor = torch.clamp(tensor + noise, 0, 1)
        
        return tensor
    
    def create_batch(self, principle: dict) -> tuple:
        """
        Create a batch of samples from a principle and its variations.
        """
        texts = [principle["content"]] + principle.get("variations", [])
        
        # Ensure we have enough samples
        while len(texts) < self.config.BATCH_SIZE:
            # Add augmented versions with noise
            texts.append(principle["content"])
        
        # Create input batch
        inputs = []
        for i, text in enumerate(texts[:self.config.BATCH_SIZE]):
            tensor = self.text_to_tensor(text, add_noise=(i > 0))
            inputs.append(tensor)
        
        input_batch = torch.stack(inputs)
        
        # Create target batch
        category_encoding = {
            "philosophy": 0.9,
            "architecture": 0.8,
            "technical": 0.7,
            "alignment": 1.0,
            "ethics": 0.85,
            "history": 0.6
        }
        
        base_value = category_encoding.get(principle["category"], 0.5)
        weighted_value = base_value * principle["weight"]
        
        targets = []
        for i in range(self.config.BATCH_SIZE):
            target = torch.full((self.config.OUTPUT_DIM,), weighted_value)
            noise = torch.randn(self.config.OUTPUT_DIM) * 0.05
            target = torch.clamp(target + noise, 0, 1)
            targets.append(target)
        
        target_batch = torch.stack(targets)
        
        return input_batch, target_batch
    
    def compute_batch_gradients(self, input_batch, target_batch):
        """
        Compute per-sample gradients for the batch.
        """
        batch_size = input_batch.shape[0]
        all_gradients = []
        
        for i in range(batch_size):
            # Zero gradients
            self.agent.compression_layer.zero_grad()
            
            # Forward pass for single sample
            output = self.agent.compression_layer(input_batch[i:i+1])
            loss = self.criterion(output, target_batch[i:i+1])
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            sample_grad = []
            for param in self.agent.compression_layer.parameters():
                if param.grad is not None:
                    sample_grad.append(param.grad.view(-1).clone())
            
            if sample_grad:
                all_gradients.append(torch.cat(sample_grad))
        
        if all_gradients:
            return torch.stack(all_gradients)
        return None
    
    def learn_principle(self, principle: dict) -> dict:
        """
        Train the agent on a single manifesto principle with batch processing.
        """
        principle_result = {
            "id": principle["id"],
            "category": principle["category"],
            "epochs": [],
            "final_loss": None,
            "final_t_score": None,
            "sleep_count": 0,
            "alignment_score": None
        }
        
        if self.config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"Learning Principle: {principle['id']}")
            print(f"Category: {principle['category']}")
            print(f"Content: {principle['content'][:80]}...")
            print(f"{'='*60}")
        
        for epoch in range(self.config.EPOCHS_PER_PRINCIPLE):
            # Create batch
            input_batch, target_batch = self.create_batch(principle)
            
            # Compute per-sample gradients
            batch_gradients = self.compute_batch_gradients(input_batch, target_batch)
            
            # Measure gradient diversity (T-Score)
            if batch_gradients is not None:
                t_score = self.agent.measure_gradient_diversity(batch_gradients)
                t_score = float(t_score)
            else:
                t_score = 0.5
            
            # Determine status based on T-Score
            if t_score < self.agent.epsilon:
                status = "SLEEP"
                # Execute sleep protocol
                self.agent.rest_and_reflect()
                principle_result["sleep_count"] += 1
                self.results["sleep_events"].append({
                    "principle": principle["id"],
                    "epoch": epoch + 1,
                    "t_score": t_score
                })
            else:
                status = "LEARN"
                # Normal learning step
                self.agent.compression_layer.zero_grad()
                output = self.agent.compression_layer(input_batch)
                loss = self.criterion(output, target_batch)
                loss.backward()
                self.agent.optimizer.step()
            
            # Calculate loss for logging
            with torch.no_grad():
                output = self.agent.compression_layer(input_batch)
                loss = float(self.criterion(output, target_batch))
            
            # Update last T-Score
            self.agent.last_T_score = t_score
            
            epoch_data = {
                "epoch": epoch + 1,
                "loss": loss,
                "t_score": t_score,
                "status": status
            }
            principle_result["epochs"].append(epoch_data)
            
            # Track T-Score history
            self.results["t_score_history"].append({
                "principle": principle["id"],
                "epoch": epoch + 1,
                "t_score": t_score
            })
            
            if self.config.VERBOSE:
                status_emoji = "💤" if status == "SLEEP" else "⚡"
                t_bar = "█" * int(t_score * 20) + "░" * (20 - int(t_score * 20))
                print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | T: {t_score:.4f} [{t_bar}] | {status_emoji} {status}")
        
        # Final metrics
        principle_result["final_loss"] = loss
        principle_result["final_t_score"] = t_score
        
        # Calculate alignment score
        with torch.no_grad():
            input_batch, target_batch = self.create_batch(principle)
            output = self.agent.compression_layer(input_batch)
            alignment = 1.0 - float(self.criterion(output, target_batch))
            principle_result["alignment_score"] = alignment
        
        return principle_result
    
    def run_test(self) -> dict:
        """
        Run the complete manifesto learning test.
        """
        print("\n" + "="*70)
        print("🧠 GodelAI MANIFESTO LEARNING TEST v2.0")
        print("="*70)
        print(f"Testing {len(MANIFESTO_PRINCIPLES)} core principles")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Min Surplus Energy (Sleep Threshold): {self.config.MIN_SURPLUS_ENERGY}")
        print(f"Propagation Gamma: {self.config.PROPAGATION_GAMMA}")
        print("="*70)
        
        # Learn each principle
        for principle in MANIFESTO_PRINCIPLES:
            result = self.learn_principle(principle)
            self.results["principles_tested"].append(result)
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _calculate_final_metrics(self):
        """Calculate aggregate metrics from the test."""
        
        all_t_scores = [p["final_t_score"] for p in self.results["principles_tested"]]
        all_losses = [p["final_loss"] for p in self.results["principles_tested"]]
        all_alignments = [p["alignment_score"] for p in self.results["principles_tested"]]
        total_sleeps = sum(p["sleep_count"] for p in self.results["principles_tested"])
        
        # Calculate T-Score dynamics
        t_history = self.results["t_score_history"]
        t_values = [t["t_score"] for t in t_history]
        
        self.results["final_metrics"] = {
            "average_t_score": sum(all_t_scores) / len(all_t_scores),
            "min_t_score": min(all_t_scores),
            "max_t_score": max(all_t_scores),
            "t_score_std": torch.tensor(all_t_scores).std().item(),
            "average_loss": sum(all_losses) / len(all_losses),
            "average_alignment": sum(all_alignments) / len(all_alignments),
            "total_sleep_events": total_sleeps,
            "principles_learned": len(self.results["principles_tested"]),
            "wisdom_preservation_rate": sum(1 for t in all_t_scores if t > self.config.MIN_SURPLUS_ENERGY) / len(all_t_scores),
            "t_score_trend": "stable" if torch.tensor(t_values[-10:]).std().item() < 0.1 else "dynamic"
        }
        
        # Determine overall status
        avg_t = self.results["final_metrics"]["average_t_score"]
        if avg_t >= 0.7:
            self.results["final_metrics"]["overall_status"] = "OPTIMAL"
        elif avg_t >= 0.5:
            self.results["final_metrics"]["overall_status"] = "HEALTHY"
        elif avg_t >= 0.3:
            self.results["final_metrics"]["overall_status"] = "WARNING"
        else:
            self.results["final_metrics"]["overall_status"] = "CRITICAL"
    
    def _print_summary(self):
        """Print a summary of the test results."""
        
        metrics = self.results["final_metrics"]
        
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print(f"\n{'Metric':<35} {'Value':<20}")
        print("-"*55)
        print(f"{'Principles Learned:':<35} {metrics['principles_learned']}")
        print(f"{'Average T-Score:':<35} {metrics['average_t_score']:.4f}")
        print(f"{'T-Score Range:':<35} {metrics['min_t_score']:.4f} - {metrics['max_t_score']:.4f}")
        print(f"{'T-Score Std Dev:':<35} {metrics['t_score_std']:.4f}")
        print(f"{'T-Score Trend:':<35} {metrics['t_score_trend']}")
        print(f"{'Average Loss:':<35} {metrics['average_loss']:.4f}")
        print(f"{'Average Alignment:':<35} {metrics['average_alignment']:.4f}")
        print(f"{'Total Sleep Events:':<35} {metrics['total_sleep_events']}")
        print(f"{'Wisdom Preservation Rate:':<35} {metrics['wisdom_preservation_rate']*100:.1f}%")
        print(f"{'Overall Status:':<35} {metrics['overall_status']}")
        
        print("\n" + "="*70)
        print("📋 PRINCIPLE-BY-PRINCIPLE RESULTS")
        print("="*70)
        
        for p in self.results["principles_tested"]:
            status_emoji = "✅" if p["final_t_score"] > self.config.MIN_SURPLUS_ENERGY else "⚠️"
            print(f"\n{status_emoji} {p['id']} ({p['category']})")
            print(f"   T-Score: {p['final_t_score']:.4f} | Loss: {p['final_loss']:.4f} | Alignment: {p['alignment_score']:.4f} | Sleeps: {p['sleep_count']}")
        
        # Validation conclusion
        print("\n" + "="*70)
        print("🎯 VALIDATION CONCLUSION")
        print("="*70)
        
        if metrics["overall_status"] in ["OPTIMAL", "HEALTHY"]:
            print("\n✅ GodelAI successfully learned from its own manifesto!")
            print("   The agent maintained healthy T-Scores while processing C-S-P principles.")
            print("   This demonstrates the framework's ability to 'eat its own cooking'.")
            print(f"\n   Key Achievement: {metrics['wisdom_preservation_rate']*100:.0f}% wisdom preservation rate")
        elif metrics["total_sleep_events"] > 0:
            print("\n🔄 GodelAI demonstrated self-correction during manifesto learning!")
            print(f"   The Sleep Protocol was triggered {metrics['total_sleep_events']} times.")
            print("   This shows the framework's ability to detect and recover from wisdom decay.")
        else:
            print("\n⚠️ GodelAI showed signs of wisdom decay during manifesto learning.")
            print("   Further tuning of hyperparameters may be needed.")
        
        print("\n" + "="*70)
    
    def save_results(self, output_dir: str = "/home/ubuntu/godelai/results"):
        """Save test results to JSON file."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manifesto_learning_test_v2_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📁 Results saved to: {filepath}")
        return filepath

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the manifesto learning test."""
    
    config = ManifestoTestConfig()
    test = ManifestoLearningTestV2(config)
    
    results = test.run_test()
    results_file = test.save_results()
    
    return results, results_file

if __name__ == "__main__":
    results, filepath = main()

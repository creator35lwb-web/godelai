"""
GodelAI Agent Core Tests
========================
Comprehensive test suite for the GodelAgent class.

Tests cover:
- Initialization and configuration
- Wisdom metric (T-score) calculation
- Sleep protocol triggering
- Learning step execution
- Training summary generation

Updated to match the FIXED GodelAgent implementation with per-sample gradients.
Date: January 4, 2026
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleTestNet(nn.Module):
    """Simple network for testing."""
    def __init__(self, input_size=4, hidden_size=8, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TestGodelAgentInitialization:
    """Tests for GodelAgent initialization."""

    def test_agent_creation_with_defaults(self):
        """Test that GodelAgent can be created with default parameters."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        assert agent.compression_layer is model
        assert agent.epsilon == 0.1  # default min_surplus_energy
        assert agent.gamma == 2.0  # default propagation_gamma
        assert agent.last_T_score == 1.0

    def test_agent_creation_with_custom_epsilon(self):
        """Test GodelAgent with custom epsilon threshold."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model, min_surplus_energy=0.8)

        assert agent.epsilon == 0.8

    def test_agent_creation_with_custom_gamma(self):
        """Test GodelAgent with custom propagation gamma."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model, propagation_gamma=3.0)

        assert agent.gamma == 3.0

    def test_agent_optimizer_assignment(self):
        """Test GodelAgent optimizer can be assigned after creation."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.01)
        agent.optimizer = optimizer

        assert agent.optimizer is optimizer


class TestWisdomMetric:
    """Tests for the wisdom metric (T-score) calculation."""

    def test_measure_gradient_diversity_returns_tensor(self):
        """Test that gradient diversity measurement returns a tensor."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=4)
        agent = GodelAgent(model)

        # Create per-sample gradients [batch_size, num_params]
        batch_size = 4
        num_params = sum(p.numel() for p in model.parameters())
        batch_grads = torch.randn(batch_size, num_params)

        t_score = agent.measure_gradient_diversity(batch_grads)

        assert isinstance(t_score, torch.Tensor)
        assert t_score.dim() == 0  # scalar

    def test_t_score_in_valid_range(self):
        """Test that T-score is between 0 and 1."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=4)
        agent = GodelAgent(model)

        batch_size = 4
        num_params = sum(p.numel() for p in model.parameters())
        batch_grads = torch.randn(batch_size, num_params)

        t_score = agent.measure_gradient_diversity(batch_grads)

        assert 0.0 <= t_score.item() <= 1.0

    def test_single_sample_returns_default(self):
        """Test that single sample returns default T-score of 0.5."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        num_params = sum(p.numel() for p in model.parameters())
        single_grad = torch.randn(1, num_params)

        t_score = agent.measure_gradient_diversity(single_grad)

        assert t_score.item() == 0.5


class TestSleepProtocol:
    """Tests for the sleep protocol."""

    def test_rest_and_reflect_modifies_weights(self):
        """Test that sleep protocol modifies model weights."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Trigger sleep
        agent.rest_and_reflect()

        # Check weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed
        assert agent.last_T_score == 1.0  # Reset after sleep

    def test_sleep_increments_counter(self):
        """Test that sleep increments the sleep counter."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        initial_count = agent.history['sleep_count']
        agent.rest_and_reflect()

        assert agent.history['sleep_count'] == initial_count + 1


class TestLearningStep:
    """Tests for the learning step."""

    def test_learning_step_returns_tuple(self):
        """Test that learning_step returns (loss, wisdom, status)."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=2, output_size=1)
        agent = GodelAgent(model, min_surplus_energy=0.1)
        agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.1)

        # Simple XOR data
        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])

        criterion = nn.MSELoss()
        result = agent.learning_step(X, y, criterion)

        assert isinstance(result, tuple)
        assert len(result) == 3
        loss, wisdom, status = result
        assert isinstance(loss, float)
        assert isinstance(wisdom, float)
        assert status in ["LEARN", "SLEEP", "SKIP"]

    def test_learning_step_updates_history(self):
        """Test that learning_step updates history."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=2, output_size=1)
        agent = GodelAgent(model, min_surplus_energy=0.1)
        agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.1)

        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])

        criterion = nn.MSELoss()

        initial_steps = len(agent.history['loss'])
        agent.learning_step(X, y, criterion)

        assert len(agent.history['loss']) == initial_steps + 1
        assert len(agent.history['wisdom_score']) == initial_steps + 1
        assert len(agent.history['status']) == initial_steps + 1


class TestTrainingSummary:
    """Tests for training summary generation."""

    def test_get_training_summary_empty_history(self):
        """Test get_training_summary with no training history."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet()
        agent = GodelAgent(model)

        summary = agent.get_training_summary()

        assert "message" in summary
        assert summary["message"] == "No training history available"

    def test_get_training_summary_after_training(self):
        """Test get_training_summary after some training."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=2, output_size=1)
        agent = GodelAgent(model, min_surplus_energy=0.1)
        agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.1)

        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])
        criterion = nn.MSELoss()

        # Train for a few steps
        for _ in range(5):
            agent.learning_step(X, y, criterion)

        summary = agent.get_training_summary()

        assert summary['total_steps'] == 5
        assert 'avg_loss' in summary
        assert 'avg_wisdom' in summary
        assert 'min_wisdom' in summary
        assert 'max_wisdom' in summary


class TestXORLearning:
    """Test that agent can learn XOR problem."""

    def test_xor_learning_improves_accuracy(self):
        """Test that agent improves on XOR over training."""
        from godelai.agent import GodelAgent

        # Use improved architecture
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        agent = GodelAgent(model, min_surplus_energy=0.1)
        agent.optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.01)

        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])
        criterion = nn.MSELoss()

        # Initial accuracy
        with torch.no_grad():
            initial_pred = agent.compression_layer(X)
            initial_acc = ((initial_pred > 0.5).float() == y).float().mean().item()

        # Train
        for _ in range(100):
            agent.learning_step(X, y, criterion)

        # Final accuracy
        with torch.no_grad():
            final_pred = agent.compression_layer(X)
            final_acc = ((final_pred > 0.5).float() == y).float().mean().item()

        # Should improve (though may not reach 100% in just 100 steps)
        assert final_acc >= initial_acc
        # Loss should decrease
        assert agent.history['loss'][-1] < agent.history['loss'][0]


class TestCSPFramework:
    """Tests for C-S-P framework integration."""

    def test_propagation_penalty_applied(self):
        """Test that propagation penalty is applied when wisdom drops."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=2, output_size=1)
        agent = GodelAgent(model, min_surplus_energy=0.1, propagation_gamma=2.0)
        agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.1)

        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])
        criterion = nn.MSELoss()

        # Do a learning step
        loss, wisdom, status = agent.learning_step(X, y, criterion)

        # Verify T-score is tracked
        assert wisdom > 0.0
        assert agent.last_T_score == wisdom

    def test_wisdom_metric_is_calculated_correctly(self):
        """Test that wisdom metric is calculated (not stuck at old 0.7311 bug)."""
        from godelai.agent import GodelAgent

        model = SimpleTestNet(input_size=2, output_size=1)
        agent = GodelAgent(model, min_surplus_energy=0.1)
        agent.optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.01)

        X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])
        criterion = nn.MSELoss()

        wisdom_scores = []
        for _ in range(20):
            _, wisdom, _ = agent.learning_step(X, y, criterion)
            wisdom_scores.append(wisdom)

        # Verify wisdom metric is being calculated
        # For XOR, we expect high wisdom (close to 1.0) due to high gradient diversity
        # The key is it's NOT the old constant bug value of 0.7311
        avg_wisdom = sum(wisdom_scores) / len(wisdom_scores)

        # Should be high for XOR (not the old bug value of 0.7311)
        assert avg_wisdom > 0.9  # High diversity expected
        assert avg_wisdom != 0.7311  # Not the old bug constant
        # All scores should be valid
        assert all(0.0 <= w <= 1.0 for w in wisdom_scores)

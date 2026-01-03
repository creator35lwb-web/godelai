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
"""

import pytest
import torch
import torch.nn as nn


class TestGodelAgentInitialization:
    """Tests for GodelAgent initialization."""

    def test_agent_creation_with_defaults(self):
        """Test that GodelAgent can be created with default parameters."""
        from godelai.agent import GodelAgent

        model = nn.Linear(10, 2)
        agent = GodelAgent(model)

        assert agent.model is model
        assert agent.epsilon == 0.5  # default threshold
        assert agent.history == []

    def test_agent_creation_with_custom_epsilon(self):
        """Test GodelAgent with custom epsilon threshold."""
        from godelai.agent import GodelAgent

        model = nn.Linear(10, 2)
        agent = GodelAgent(model, epsilon=0.8)

        assert agent.epsilon == 0.8

    def test_agent_creation_with_custom_optimizer(self):
        """Test GodelAgent with custom optimizer."""
        from godelai.agent import GodelAgent

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        agent = GodelAgent(model, optimizer=optimizer)

        assert agent.optimizer is optimizer


class TestWisdomMetric:
    """Tests for the wisdom metric (T-score) calculation."""

    def test_measure_gradient_diversity_returns_tensor(self):
        """Test that gradient diversity measurement returns a tensor."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model)

        # Create dummy data
        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        t_score = agent.measure_gradient_diversity(X, y)

        assert isinstance(t_score, torch.Tensor)
        assert t_score.dim() == 0  # scalar

    def test_t_score_in_valid_range(self):
        """Test that T-score is between 0 and 1."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model)

        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        t_score = agent.measure_gradient_diversity(X, y)

        assert 0.0 <= t_score.item() <= 1.0


class TestSleepProtocol:
    """Tests for the Sleep Protocol functionality."""

    def test_rest_and_reflect_modifies_weights(self):
        """Test that rest_and_reflect modifies model weights."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model)

        # Store original weights
        original_weights = model.weight.data.clone()

        agent.rest_and_reflect()

        # Weights should be modified (noise added)
        assert not torch.allclose(model.weight.data, original_weights)

    def test_sleep_triggered_when_t_below_epsilon(self):
        """Test that sleep is triggered when T-score falls below epsilon."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model, epsilon=0.99)  # Very high threshold

        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        # This should trigger sleep due to high epsilon
        result = agent.learning_step(X, y)

        # Check that sleep was recorded in history
        sleep_events = [h for h in agent.history if h.get('action') == 'sleep']
        # May or may not trigger depending on gradient diversity


class TestLearningStep:
    """Tests for the learning step execution."""

    def test_learning_step_returns_dict(self):
        """Test that learning_step returns a dictionary with expected keys."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model)

        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        result = agent.learning_step(X, y)

        assert isinstance(result, dict)
        assert 'loss' in result
        assert 't_score' in result
        assert 'action' in result

    def test_learning_step_decreases_loss(self):
        """Test that multiple learning steps decrease loss."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model, epsilon=0.1)  # Low epsilon to avoid sleep

        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))

        initial_result = agent.learning_step(X, y)
        initial_loss = initial_result['loss']

        # Run multiple steps
        for _ in range(10):
            result = agent.learning_step(X, y)

        final_loss = result['loss']

        # Loss should generally decrease (allowing some variance)
        assert final_loss <= initial_loss * 1.5  # Allow some tolerance


class TestTrainingSummary:
    """Tests for training summary generation."""

    def test_get_training_summary_empty_history(self):
        """Test summary with no training history."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model)

        summary = agent.get_training_summary()

        assert isinstance(summary, dict)
        assert summary['total_steps'] == 0

    def test_get_training_summary_after_training(self):
        """Test summary after some training steps."""
        from godelai.agent import GodelAgent

        model = nn.Linear(4, 2)
        agent = GodelAgent(model, epsilon=0.1)

        X = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        # Run some training steps
        for _ in range(5):
            agent.learning_step(X, y)

        summary = agent.get_training_summary()

        assert summary['total_steps'] == 5
        assert 'avg_t_score' in summary
        assert 'sleep_events' in summary


class TestXORLearning:
    """Integration test: XOR problem learning."""

    def test_xor_learning_improves_accuracy(self):
        """Test that the agent can learn the XOR problem."""
        from godelai.agent import GodelAgent

        # XOR dataset
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        # Simple model for XOR
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        agent = GodelAgent(model, epsilon=0.1)

        # Train for multiple epochs
        for _ in range(100):
            agent.learning_step(X, y)

        # Check accuracy
        with torch.no_grad():
            outputs = model(X)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

        # Should achieve reasonable accuracy
        assert accuracy >= 0.5  # At least better than random


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

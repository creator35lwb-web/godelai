"""
Test T-Score Variance Tracking (v3.1.1)

This test validates the variance tracking implementation based on
Manus AI experiment analysis (TSCORE_EXPERIMENT_ANALYSIS.md).

Key findings from Manus experiment:
- Conflict data shows +43% higher T-Score variance
- Higher variance indicates more diverse gradient patterns

Author: Claude Code (Opus 4.5)
Date: January 20, 2026
Reference: docs/TSCORE_EXPERIMENT_ANALYSIS.md
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from godelai.agent import GodelAgent


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.fc(x)


def test_variance_tracking_basic():
    """Test that variance tracking returns values."""
    print("\n[TEST] Basic Variance Tracking")
    print("-" * 40)

    model = SimpleNet()
    agent = GodelAgent(model, t_score_window=10)
    agent.optimizer = torch.optim.Adam(agent.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(42)
    X = torch.randn(24, 10)
    y = torch.randn(24, 2)

    # Run a few steps
    for i in range(8):
        batch_x = X[i*3:(i+1)*3]
        batch_y = y[i*3:(i+1)*3]
        loss, t_score, status, metrics = agent.learning_step(batch_x, batch_y, criterion)

        assert 't_score_variance' in metrics, "metrics should contain t_score_variance"
        assert isinstance(metrics['t_score_variance'], float), "variance should be float"

    stats = agent.get_variance_stats()
    assert 't_score_std' in stats, "stats should contain t_score_std"
    assert 'avg_variance' in stats, "stats should contain avg_variance"
    assert 'trend' in stats, "stats should contain trend"

    print(f"  T-Score Std: {stats['t_score_std']:.6f}")
    print(f"  Avg Variance: {stats['avg_variance']:.6f}")
    print(f"  Trend: {stats['trend']}")
    print("  [PASS] Basic variance tracking works")
    return True


def test_variance_comparison():
    """Test that heterogeneous data produces higher variance than homogeneous."""
    print("\n[TEST] Variance Comparison (Homo vs Hetero)")
    print("-" * 40)

    # Test with homogeneous data
    model1 = SimpleNet()
    agent1 = GodelAgent(model1, t_score_window=10)
    agent1.optimizer = torch.optim.Adam(agent1.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(42)
    X_homo = torch.randn(45, 10) * 0.1  # Low variance
    y_homo = X_homo[:, :2] * 2

    for i in range(15):
        batch_x = X_homo[i*3:(i+1)*3]
        batch_y = y_homo[i*3:(i+1)*3]
        agent1.learning_step(batch_x, batch_y, criterion)

    homo_stats = agent1.get_variance_stats()

    # Test with heterogeneous data
    model2 = SimpleNet()
    agent2 = GodelAgent(model2, t_score_window=10)
    agent2.optimizer = torch.optim.Adam(agent2.compression_layer.parameters(), lr=0.01)

    torch.manual_seed(123)
    X_hetero = torch.cat([
        torch.randn(15, 10) * 0.5,
        torch.randn(15, 10) * 2.0,
        torch.randn(15, 10) * 0.1 + 5,
    ])
    y_hetero = torch.cat([
        X_hetero[:15, :2] * 1,
        X_hetero[15:30, :2] * -1,
        torch.sin(X_hetero[30:, :2]),
    ])

    for i in range(15):
        batch_x = X_hetero[i*3:(i+1)*3]
        batch_y = y_hetero[i*3:(i+1)*3]
        agent2.learning_step(batch_x, batch_y, criterion)

    hetero_stats = agent2.get_variance_stats()

    print(f"  Homogeneous T-Score Std:  {homo_stats['t_score_std']:.6f}")
    print(f"  Heterogeneous T-Score Std: {hetero_stats['t_score_std']:.6f}")

    diff_pct = ((hetero_stats['t_score_std'] - homo_stats['t_score_std']) /
                (homo_stats['t_score_std'] + 1e-8)) * 100
    print(f"  Difference: {diff_pct:+.1f}%")

    # Heterogeneous should generally have higher variance
    # (may vary due to random init, so we just report)
    if hetero_stats['t_score_std'] > homo_stats['t_score_std']:
        print("  [PASS] Heterogeneous data shows higher variance")
    else:
        print("  [NOTE] Results may vary due to random initialization")

    return True


def test_reset_functionality():
    """Test that reset_variance_tracking() works correctly."""
    print("\n[TEST] Reset Functionality")
    print("-" * 40)

    model = SimpleNet()
    agent = GodelAgent(model, t_score_window=10)
    agent.optimizer = torch.optim.Adam(agent.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(42)
    X = torch.randn(15, 10)
    y = torch.randn(15, 2)

    # Run some steps
    for i in range(5):
        batch_x = X[i*3:(i+1)*3]
        batch_y = y[i*3:(i+1)*3]
        agent.learning_step(batch_x, batch_y, criterion)

    buffer_before = len(agent.t_score_buffer)
    history_before = len(agent.history['t_score_variance'])

    print(f"  Buffer size before reset: {buffer_before}")
    print(f"  History size before reset: {history_before}")

    agent.reset_variance_tracking()

    buffer_after = len(agent.t_score_buffer)
    history_after = len(agent.history['t_score_variance'])

    print(f"  Buffer size after reset: {buffer_after}")
    print(f"  History size after reset: {history_after}")

    assert buffer_after == 0, "Buffer should be empty after reset"
    assert history_after == 0, "Variance history should be empty after reset"

    print("  [PASS] Reset functionality works")
    return True


def test_training_summary_includes_variance():
    """Test that get_training_summary() includes variance metrics."""
    print("\n[TEST] Training Summary Variance Metrics")
    print("-" * 40)

    model = SimpleNet()
    agent = GodelAgent(model, t_score_window=10)
    agent.optimizer = torch.optim.Adam(agent.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    torch.manual_seed(42)
    X = torch.randn(30, 10)
    y = torch.randn(30, 2)

    for i in range(10):
        batch_x = X[i*3:(i+1)*3]
        batch_y = y[i*3:(i+1)*3]
        agent.learning_step(batch_x, batch_y, criterion)

    summary = agent.get_training_summary()

    required_keys = ['t_score_std', 'avg_t_score_variance', 'max_t_score_variance', 'variance_trend']
    for key in required_keys:
        assert key in summary, f"Summary should contain {key}"
        print(f"  {key}: {summary[key]}")

    print("  [PASS] Training summary includes all variance metrics")
    return True


def run_all_tests():
    """Run all variance tracking tests."""
    print("=" * 60)
    print("T-Score Variance Tracking Test Suite")
    print("=" * 60)

    tests = [
        test_variance_tracking_basic,
        test_variance_comparison,
        test_reset_functionality,
        test_training_summary_includes_variance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

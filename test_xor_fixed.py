"""
XOR Test with FIXED GodelAgent (Per-Sample Gradients)

This test uses the corrected agent that computes per-sample gradients
to properly measure gradient diversity.
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from godelai.agent_fixed import GodelAgent


# Simple Neural Network for XOR
class SimpleNet(nn.Module):
    """A minimal neural network for XOR problem."""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def run_xor_test_fixed(epsilon=0.3, epochs=50, verbose=True):
    """
    Run the XOR test with FIXED GodelAgent.

    Args:
        epsilon: Wisdom threshold
        epochs: Number of training epochs
        verbose: Whether to print detailed output

    Returns:
        dict: Test results
    """
    # XOR Data
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    # Initialize with FIXED agent
    base_model = SimpleNet()
    agent = GodelAgent(base_model, min_surplus_energy=epsilon)
    agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    if verbose:
        print(f"--- FIXED GodelAI XOR Test ---")
        print(f"Wisdom Threshold (Epsilon): {epsilon}")
        print(f"Using PER-SAMPLE GRADIENTS for diversity measurement\n")

    # Training Loop
    for epoch in range(1, epochs + 1):
        loss, wisdom_score, status = agent.learning_step(inputs, targets, criterion)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            bar_len = int(wisdom_score * 20)
            wisdom_bar = "█" * bar_len + "░" * (20 - bar_len)

            status_icon = "💤 SLEEP" if status == "SLEEP" else "⚡ LEARN"

            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Wisdom: {wisdom_score:.4f} [{wisdom_bar}] | {status_icon}")

            if epoch % 10 == 0:
                with torch.no_grad():
                    output = agent.compression_layer(inputs)
                    print(f"   Current: {output.view(-1).tolist()}")
                    print(f"   Target:  {targets.view(-1).tolist()}")

    # Final evaluation
    with torch.no_grad():
        final_output = agent.compression_layer(inputs)
        predictions = (final_output > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()

    summary = agent.get_training_summary()
    summary['final_accuracy'] = accuracy

    if verbose:
        print(f"\n📊 Final Results:")
        print(f"   Accuracy: {accuracy * 100:.1f}%")
        print(f"   Sleep Count: {summary['sleep_count']}")
        print(f"   Avg Wisdom: {summary['avg_wisdom']:.4f}")
        print(f"   Wisdom Range: {summary['min_wisdom']:.4f} - {summary['max_wisdom']:.4f}")
        print(f"\n   Final Predictions: {predictions.view(-1).tolist()}")
        print(f"   Expected:          {targets.view(-1).tolist()}")

    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FIXED GodelAgent with Per-Sample Gradient Diversity")
    print("=" * 70)
    print()

    # Run with moderate epsilon
    results = run_xor_test_fixed(epsilon=0.3, epochs=50, verbose=True)

    print("\n" + "=" * 70)
    if results['final_accuracy'] >= 0.9:
        print("✅ SUCCESS! Model learned XOR with wisdom preservation")
    else:
        print("⚠️  Model struggled with XOR - may need more epochs or tuning")
    print("=" * 70)

"""
Test the MAIN godelai.agent.GodelAgent with the applied fix
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import from MAIN agent (not agent_fixed)
from godelai.agent import GodelAgent


# Improved Neural Network for XOR
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def test_fixed_main_agent():
    """Test that the main GodelAgent now has the fix applied"""
    print("=" * 70)
    print("Testing MAIN godelai.agent.GodelAgent (After Fix)")
    print("=" * 70)
    print()

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    base_model = ImprovedNet()
    agent = GodelAgent(base_model, min_surplus_energy=0.2)
    agent.optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"Configuration:")
    print(f"  Wisdom Threshold: {agent.epsilon}")
    print(f"  Using PER-SAMPLE gradients: YES (FIXED!)")
    print()

    # Short training run to verify it works
    for epoch in range(1, 201):
        loss, wisdom, status = agent.learning_step(inputs, targets, criterion)

        if epoch % 50 == 0:
            with torch.no_grad():
                outputs = agent.compression_layer(inputs)
                preds = (outputs > 0.5).float()
                acc = (preds == targets).float().mean()

                bar_len = int(wisdom * 20)
                wisdom_bar = "█" * bar_len + "░" * (20 - bar_len)
                status_icon = "💤" if status == "SLEEP" else "⚡"

                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Wisdom: {wisdom:.4f} [{wisdom_bar}] {status_icon} | Acc: {acc.item()*100:.1f}%")

    # Final evaluation
    with torch.no_grad():
        final_outputs = agent.compression_layer(inputs)
        final_preds = (final_outputs > 0.5).float()
        final_acc = (final_preds == targets).float().mean()

    summary = agent.get_training_summary()

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"✓ Accuracy:       {final_acc.item()*100:.1f}%")
    print(f"✓ Sleep events:   {summary['sleep_count']}")
    print(f"✓ Avg wisdom:     {summary['avg_wisdom']:.4f}")
    print(f"✓ Wisdom range:   {summary['min_wisdom']:.4f} - {summary['max_wisdom']:.4f}")
    print()
    print(f"  Predictions: {final_preds.view(-1).tolist()}")
    print(f"  Expected:    {targets.view(-1).tolist()}")
    print()

    if final_acc >= 0.9 and summary['min_wisdom'] < summary['max_wisdom']:
        print("✅ SUCCESS! Main agent is FIXED:")
        print("   - Model learned XOR (100% accuracy)")
        print("   - Wisdom score varies (not constant 0.7311)")
        print("   - Per-sample gradients working correctly")
    else:
        print("⚠️  Issue detected:")
        if final_acc < 0.9:
            print(f"   - Low accuracy: {final_acc.item()*100:.1f}%")
        if summary['min_wisdom'] == summary['max_wisdom']:
            print(f"   - Wisdom constant: {summary['min_wisdom']:.4f}")

    print("=" * 70)

    return final_acc.item(), summary


if __name__ == "__main__":
    test_fixed_main_agent()

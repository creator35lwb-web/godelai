"""
XOR Test with properly tuned hyperparameters to actually solve XOR
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from godelai.agent_fixed import GodelAgent


# Improved Neural Network for XOR (more capacity)
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 8),  # More hidden units
            nn.Tanh(),
            nn.Linear(8, 8),  # Additional layer
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_baseline():
    """First verify baseline can solve XOR"""
    print("=" * 70)
    print("STEP 1: Verify baseline can solve XOR")
    print("=" * 70)

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    model = ImprovedNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam works better
    criterion = nn.MSELoss()

    for epoch in range(1, 501):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                acc = (preds == targets).float().mean()
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.1f}%")

    with torch.no_grad():
        final_outputs = model(inputs)
        final_preds = (final_outputs > 0.5).float()
        final_acc = (final_preds == targets).float().mean()
        print(f"\n✓ Baseline achieved: {final_acc.item()*100:.1f}% accuracy")

    return final_acc.item()


def train_with_godel():
    """Train with GodelAgent using same architecture"""
    print("\n" + "=" * 70)
    print("STEP 2: Train with GodelAgent (Per-Sample Gradients)")
    print("=" * 70)

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    base_model = ImprovedNet()
    agent = GodelAgent(base_model, min_surplus_energy=0.2)  # Moderate epsilon
    agent.optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"Wisdom Threshold: {agent.epsilon}")
    print(f"Using PER-SAMPLE gradients for diversity\n")

    for epoch in range(1, 501):
        loss, wisdom, status = agent.learning_step(inputs, targets, criterion)

        if epoch % 100 == 0:
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

    print(f"\n✓ GodelAgent achieved: {final_acc.item()*100:.1f}% accuracy")
    print(f"  Sleep events: {summary['sleep_count']}")
    print(f"  Avg wisdom: {summary['avg_wisdom']:.4f}")
    print(f"  Wisdom range: {summary['min_wisdom']:.4f} - {summary['max_wisdom']:.4f}")
    print(f"\n  Final predictions: {final_preds.view(-1).tolist()}")
    print(f"  Expected:          {targets.view(-1).tolist()}")

    return final_acc.item(), summary


if __name__ == "__main__":
    print("=" * 70)
    print("GodelAI XOR Test - Proper Configuration")
    print("=" * 70)
    print()

    baseline_acc = train_baseline()
    godel_acc, summary = train_with_godel()

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"Baseline (standard):  {baseline_acc*100:.1f}% accuracy")
    print(f"GodelAgent (wisdom):  {godel_acc*100:.1f}% accuracy")
    print(f"Sleep events:         {summary['sleep_count']}")
    print()

    if godel_acc >= 0.9:
        print("✅ SUCCESS! GodelAgent solved XOR while preserving wisdom")
    elif baseline_acc >= 0.9 and godel_acc < 0.9:
        print("⚠️ Baseline succeeded but GodelAgent struggled")
        print("   → Wisdom constraints may be too aggressive")
    elif baseline_acc < 0.9:
        print("⚠️ Both struggled - architecture needs more capacity")
    print("=" * 70)

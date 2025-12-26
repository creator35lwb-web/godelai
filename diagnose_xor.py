"""
Diagnostic script to compare standard training vs GodelAgent
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Simple Neural Network for XOR
class SimpleNet(nn.Module):
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


def train_standard(epochs=100, lr=0.5):
    """Train XOR with standard PyTorch (baseline)"""
    print("=" * 70)
    print("BASELINE: Standard PyTorch Training (No GodelAgent)")
    print("=" * 70)

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                acc = (preds == targets).float().mean()
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.1f}%")

    # Final result
    with torch.no_grad():
        final_outputs = model(inputs)
        final_preds = (final_outputs > 0.5).float()
        final_acc = (final_preds == targets).float().mean()

        print(f"\n✓ Final: {final_acc.item()*100:.1f}% accuracy")
        print(f"  Predictions: {final_preds.view(-1).tolist()}")
        print(f"  Expected:    {targets.view(-1).tolist()}")
        print(f"  Outputs:     {[f'{x:.3f}' for x in final_outputs.view(-1).tolist()]}")

    return final_acc.item()


def analyze_gradient_diversity():
    """Analyze what gradient diversity looks like for XOR"""
    print("\n" + "=" * 70)
    print("ANALYSIS: Gradient Diversity for XOR Problem")
    print("=" * 70)

    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    model = SimpleNet()
    criterion = nn.MSELoss()

    # Compute per-sample gradients
    per_sample_grads = []
    for i in range(4):
        model.zero_grad()
        output = model(inputs[i:i+1])
        loss = criterion(output, targets[i:i+1])
        loss.backward()

        # Collect gradient vector
        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.view(-1).clone())

        full_grad = torch.cat(grad_vec)
        per_sample_grads.append(full_grad)

        print(f"Sample {i} [{inputs[i].tolist()}→{targets[i].item():.0f}] | Grad norm: {full_grad.norm().item():.4f}")

    # Stack gradients
    batch_grads = torch.stack(per_sample_grads)  # [4, num_params]

    # Compute diversity
    sum_grad_norm = torch.norm(torch.sum(batch_grads, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_grads, dim=1)**2)
    diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
    T_score = torch.sigmoid(diversity_score)

    print(f"\nDiversity Analysis:")
    print(f"  Sum of squared norms: {sum_norm_grad.item():.4f}")
    print(f"  Norm of sum squared:  {sum_grad_norm.item():.4f}")
    print(f"  Diversity ratio:      {diversity_score.item():.4f}")
    print(f"  T-score (sigmoid):    {T_score.item():.4f}")

    # Check gradient alignment
    print(f"\nGradient Cosine Similarities:")
    for i in range(4):
        for j in range(i+1, 4):
            cos_sim = torch.nn.functional.cosine_similarity(
                per_sample_grads[i].unsqueeze(0),
                per_sample_grads[j].unsqueeze(0)
            )
            print(f"  Sample {i} <-> Sample {j}: {cos_sim.item():.4f}")


if __name__ == "__main__":
    # First, verify standard training works
    baseline_acc = train_standard(epochs=100, lr=0.5)

    # Then analyze gradient diversity
    analyze_gradient_diversity()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Standard training achieved: {baseline_acc*100:.1f}% accuracy")
    print("If baseline fails, the issue is not GodelAgent but the base architecture.")
    print("If baseline succeeds, GodelAgent may need learning rate or penalty tuning.")

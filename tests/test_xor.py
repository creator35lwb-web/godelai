"""
GodelAI XOR Test - The Pulse Check

This test validates that the GodelAgent's wisdom metric and sleep protocol
are functioning correctly using a simple XOR problem.

Author: Gemini 2.5 Pro (Echo v2.1) - Technical Blueprint
Integrated by: Godel, CTO - GodelAI Project
Date: December 25, 2025

Expected Behavior:
1. Model starts with high wisdom (T > 0.95)
2. As it learns, wisdom may drop (gradient diversity decreases)
3. If wisdom drops below threshold, Sleep Protocol triggers
4. After sleep, wisdom resets and model continues with clarity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from godelai.agent import GodelAgent


# 1. Define a Tiny Brain (The "Body")
# A simple neural net to solve XOR.
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


def run_xor_test(epsilon=0.95, epochs=20, verbose=True):
    """
    Run the XOR test with GodelAgent.
    
    Args:
        epsilon: Wisdom threshold (higher = more likely to sleep)
        epochs: Number of training epochs
        verbose: Whether to print detailed output
        
    Returns:
        dict: Test results including sleep count and final accuracy
    """
    
    # 2. Prepare Data (XOR Problem)
    # Input: [0,0], [0,1], [1,0], [1,1]
    # Target: [0],   [1],   [1],   [0]
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])

    # 3. Initialize GodelAgent (The "Soul")
    base_model = SimpleNet()
    # NOTE: We set epsilon=0.95 (Very High) just to FORCE a demonstration of Sleep Protocol
    # In reality, epsilon would be lower (e.g., 0.1).
    agent = GodelAgent(base_model, min_surplus_energy=0.1) 
    agent.epsilon = epsilon  # Set the threshold

    # 4. Setup Optimizer
    agent.optimizer = optim.SGD(agent.compression_layer.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    if verbose:
        print(f"--- 🧠 GodelAI Pulse Check (XOR Test) ---")
        print(f"Agent initialized. Wisdom Threshold (Epsilon): {agent.epsilon}")
        print(f"Goal: Watch T-Score. If T < {agent.epsilon}, it MUST Sleep.\n")

    # 5. The Training Loop
    for epoch in range(1, epochs + 1):
        # Perform the Step
        loss, wisdom_score, status = agent.learning_step(inputs, targets, criterion)
        
        if verbose:
            # Visualization
            bar_len = int(wisdom_score * 20)
            wisdom_bar = "█" * bar_len + "░" * (20 - bar_len)
            
            if status == "SLEEP":
                status_icon = "💤 SLEEPING (Cleaning Noise...)"
            else:
                status_icon = "⚡ LEARNING"
                
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Wisdom (T): {wisdom_score:.4f} [{wisdom_bar}] | {status_icon}")

            # Manual check: Did it solve it?
            if epoch % 10 == 0:
                with torch.no_grad():
                    output = agent.compression_layer(inputs)
                    print(f"   >>> Current Guess: {output.view(-1).tolist()}")

    if verbose:
        print("\n--- ✅ Test Complete ---")
        
    # Get final results
    with torch.no_grad():
        final_output = agent.compression_layer(inputs)
        predictions = (final_output > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()
    
    summary = agent.get_training_summary()
    summary['final_accuracy'] = accuracy
    
    if verbose:
        print(f"\n📊 Training Summary:")
        print(f"   Total Steps: {summary['total_steps']}")
        print(f"   Sleep Count: {summary['sleep_count']}")
        print(f"   Avg Wisdom: {summary['avg_wisdom']:.4f}")
        print(f"   Final Accuracy: {accuracy * 100:.1f}%")
    
    return summary


if __name__ == "__main__":
    # Run with high epsilon to demonstrate sleep protocol
    print("=" * 60)
    print("TEST 1: High Epsilon (0.95) - Expect frequent sleep")
    print("=" * 60)
    results_high = run_xor_test(epsilon=0.95, epochs=20)
    
    print("\n" + "=" * 60)
    print("TEST 2: Low Epsilon (0.1) - Expect rare/no sleep")
    print("=" * 60)
    results_low = run_xor_test(epsilon=0.1, epochs=20)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"High Epsilon (0.95): {results_high['sleep_count']} sleeps, {results_high['final_accuracy']*100:.1f}% accuracy")
    print(f"Low Epsilon (0.10):  {results_low['sleep_count']} sleeps, {results_low['final_accuracy']*100:.1f}% accuracy")

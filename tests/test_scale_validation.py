#!/usr/bin/env python3
"""
GodelAI Scale Validation Test
==============================
Tests the C-S-P framework across multiple network sizes to validate
T-Score stability and wisdom preservation at scale.

Network Sizes Tested:
- Small:  64 hidden dimensions
- Medium: 128 hidden dimensions  
- Large:  256 hidden dimensions
- XLarge: 512 hidden dimensions

Author: Godel (CTO, Manus AI)
Date: January 6, 2026
"""

import torch
import torch.nn as nn
import os
import json
from datetime import datetime
import random
import time

# Import GodelAI components
from godelai.agent import GodelAgent

# ============================================================================
# MANIFESTO PRINCIPLES (Same as v2 test)
# ============================================================================

MANIFESTO_PRINCIPLES = [
    {
        "id": "CSP_THESIS",
        "content": "The world produces differences through processes. Differences are compressed into states. States are transmitted through carriers.",
        "category": "philosophy",
        "weight": 1.0,
        "variations": [
            "Processes generate differences which become compressed states that propagate through carriers.",
            "Intelligence evolves when states actively select their inheritance pathways.",
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
        ]
    },
    {
        "id": "WISDOM_METRIC",
        "content": "We use Gradient Diversity as the measure of Propagation Potential. A healthy model's internal neurons should respond diversely.",
        "category": "technical",
        "weight": 0.8,
        "variations": [
            "Gradient diversity measures propagation potential and wisdom health.",
            "Diverse neural responses indicate healthy adaptability.",
        ]
    },
    {
        "id": "SLEEP_PROTOCOL",
        "content": "When the T value falls below a threshold, it triggers Sleep Mode. The model stops ingesting new data and performs reflection.",
        "category": "technical",
        "weight": 0.8,
        "variations": [
            "Low T-Score triggers sleep for detox, calming, and refresh.",
            "Sleep protocol halts learning for wisdom consolidation.",
        ]
    },
]

# ============================================================================
# SCALE TEST CONFIGURATIONS
# ============================================================================

SCALE_CONFIGS = [
    {"name": "Small", "input_dim": 64, "hidden_dim": 64, "output_dim": 32},
    {"name": "Medium", "input_dim": 64, "hidden_dim": 128, "output_dim": 32},
    {"name": "Large", "input_dim": 64, "hidden_dim": 256, "output_dim": 64},
    {"name": "XLarge", "input_dim": 64, "hidden_dim": 512, "output_dim": 128},
]

# ============================================================================
# ENCODER MODEL
# ============================================================================

class ScalableEncoder(nn.Module):
    """Scalable encoder for testing different network sizes."""
    
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
        self.param_count = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.encoder(x)

# ============================================================================
# SCALE TEST CLASS
# ============================================================================

class ScaleValidationTest:
    """Test GodelAI across multiple network scales."""
    
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Initialize model
        self.base_model = ScalableEncoder(
            config["input_dim"],
            config["hidden_dim"],
            config["output_dim"]
        )
        
        self.agent = GodelAgent(
            self.base_model,
            propagation_gamma=2.0,
            min_surplus_energy=0.3
        )
        
        self.agent.optimizer = torch.optim.Adam(
            self.agent.compression_layer.parameters(),
            lr=0.01
        )
        
        self.criterion = nn.MSELoss()
        
        self.results = {
            "config": config,
            "param_count": self.base_model.param_count,
            "principles": [],
            "metrics": {}
        }
    
    def text_to_tensor(self, text: str, add_noise: bool = False) -> torch.Tensor:
        """Convert text to tensor."""
        encoded = [ord(c) % 256 for c in text[:self.config["input_dim"]]]
        while len(encoded) < self.config["input_dim"]:
            encoded.append(0)
        
        tensor = torch.tensor(encoded, dtype=torch.float32) / 255.0
        
        if add_noise:
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.05, 0, 1)
        
        return tensor
    
    def create_batch(self, principle: dict, batch_size: int = 4) -> tuple:
        """Create batch from principle."""
        texts = [principle["content"]] + principle.get("variations", [])
        while len(texts) < batch_size:
            texts.append(principle["content"])
        
        inputs = torch.stack([self.text_to_tensor(t, i > 0) for i, t in enumerate(texts[:batch_size])])
        
        category_encoding = {"philosophy": 0.9, "architecture": 0.8, "technical": 0.7, "alignment": 1.0}
        base_value = category_encoding.get(principle["category"], 0.5) * principle["weight"]
        
        targets = []
        for _ in range(batch_size):
            target = torch.full((self.config["output_dim"],), base_value)
            target = torch.clamp(target + torch.randn_like(target) * 0.05, 0, 1)
            targets.append(target)
        
        return inputs, torch.stack(targets)
    
    def compute_batch_gradients(self, input_batch, target_batch):
        """Compute per-sample gradients."""
        all_gradients = []
        
        for i in range(input_batch.shape[0]):
            self.agent.compression_layer.zero_grad()
            output = self.agent.compression_layer(input_batch[i:i+1])
            loss = self.criterion(output, target_batch[i:i+1])
            loss.backward()
            
            sample_grad = []
            for param in self.agent.compression_layer.parameters():
                if param.grad is not None:
                    sample_grad.append(param.grad.view(-1).clone())
            
            if sample_grad:
                all_gradients.append(torch.cat(sample_grad))
        
        return torch.stack(all_gradients) if all_gradients else None
    
    def run_test(self, epochs: int = 10) -> dict:
        """Run the scale test."""
        print(f"\n{'='*60}")
        print(f"Testing Scale: {self.config['name']}")
        print(f"Hidden Dim: {self.config['hidden_dim']} | Params: {self.base_model.param_count:,}")
        print(f"{'='*60}")
        
        all_t_scores = []
        all_alignments = []
        total_sleeps = 0
        
        for principle in MANIFESTO_PRINCIPLES:
            principle_t_scores = []
            
            for epoch in range(epochs):
                input_batch, target_batch = self.create_batch(principle)
                batch_gradients = self.compute_batch_gradients(input_batch, target_batch)
                
                if batch_gradients is not None:
                    t_score = float(self.agent.measure_gradient_diversity(batch_gradients))
                else:
                    t_score = 0.5
                
                principle_t_scores.append(t_score)
                
                if t_score >= self.agent.epsilon:
                    self.agent.compression_layer.zero_grad()
                    output = self.agent.compression_layer(input_batch)
                    loss = self.criterion(output, target_batch)
                    loss.backward()
                    self.agent.optimizer.step()
                else:
                    self.agent._execute_sleep_protocol()
                    total_sleeps += 1
            
            final_t = principle_t_scores[-1]
            all_t_scores.append(final_t)
            
            # Calculate alignment
            with torch.no_grad():
                input_batch, target_batch = self.create_batch(principle)
                output = self.agent.compression_layer(input_batch)
                alignment = 1.0 - float(self.criterion(output, target_batch))
                all_alignments.append(alignment)
            
            status = "✅" if final_t > 0.3 else "⚠️"
            print(f"{status} {principle['id']:<20} T: {final_t:.4f} | Align: {alignment:.4f}")
            
            self.results["principles"].append({
                "id": principle["id"],
                "final_t_score": final_t,
                "alignment": alignment
            })
        
        # Calculate metrics
        self.results["metrics"] = {
            "avg_t_score": sum(all_t_scores) / len(all_t_scores),
            "min_t_score": min(all_t_scores),
            "max_t_score": max(all_t_scores),
            "avg_alignment": sum(all_alignments) / len(all_alignments),
            "total_sleeps": total_sleeps,
            "wisdom_preservation": sum(1 for t in all_t_scores if t > 0.3) / len(all_t_scores)
        }
        
        return self.results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run scale validation across all configurations."""
    
    print("\n" + "="*70)
    print("🔬 GodelAI SCALE VALIDATION TEST")
    print("="*70)
    print(f"Testing {len(SCALE_CONFIGS)} network scales")
    print(f"Principles: {len(MANIFESTO_PRINCIPLES)}")
    print("="*70)
    
    all_results = {
        "test_name": "GodelAI Scale Validation",
        "timestamp": datetime.now().isoformat(),
        "scales": []
    }
    
    start_time = time.time()
    
    for config in SCALE_CONFIGS:
        test = ScaleValidationTest(config)
        results = test.run_test(epochs=10)
        all_results["scales"].append(results)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SCALE VALIDATION SUMMARY")
    print("="*70)
    print(f"\n{'Scale':<12} {'Params':<12} {'Avg T-Score':<15} {'Alignment':<12} {'Sleeps':<8} {'Status'}")
    print("-"*70)
    
    all_passed = True
    for result in all_results["scales"]:
        config = result["config"]
        metrics = result["metrics"]
        status = "✅ PASS" if metrics["avg_t_score"] > 0.5 and metrics["wisdom_preservation"] >= 1.0 else "⚠️ WARN"
        if metrics["avg_t_score"] <= 0.5:
            all_passed = False
        
        print(f"{config['name']:<12} {result['param_count']:<12,} {metrics['avg_t_score']:<15.4f} {metrics['avg_alignment']:<12.4f} {metrics['total_sleeps']:<8} {status}")
    
    print("-"*70)
    print(f"Total Time: {elapsed:.2f}s")
    
    # Conclusion
    print("\n" + "="*70)
    print("🎯 SCALE VALIDATION CONCLUSION")
    print("="*70)
    
    if all_passed:
        print("\n✅ GodelAI T-Score remains stable across all network scales!")
        print("   The C-S-P framework scales from 64 to 512 hidden dimensions")
        print("   without wisdom degradation.")
    else:
        print("\n⚠️ Some scales showed reduced T-Score stability.")
        print("   Further investigation may be needed.")
    
    print("="*70)
    
    # Save results
    os.makedirs("/home/ubuntu/godelai/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"/home/ubuntu/godelai/results/scale_validation_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {filepath}")
    
    return all_results

if __name__ == "__main__":
    results = main()

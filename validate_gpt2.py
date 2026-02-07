#!/usr/bin/env python3
"""
INDEPENDENT VALIDATION: GodelAI with Transformer Architecture (CPU Optimized)
==========================================================================

SCIENTIFIC OBJECTIVE:
Validate GodelAI's performance on a transformer architecture (GodelaiTransformer)
instead of the default character-level LSTM.

CPU OPTIMIZATION:
- Uses local GodelaiTransformer (nanoGPT style).
- Reduced layers (2), heads (4), and embedding dim (128).
- Minimal training steps for quick CPU verification.

Author: Antigravity (Independent Validation)
Date: February 7, 2026
"""

import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Force UTF-8 and unbuffered output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from godelai.agent import GodelAgent
from godelai.models.transformer import GodelaiTransformer, GodelaiConfig

print("="*80, flush=True)
print("INDEPENDENT VALIDATION: GodelAI with Nano-Transformer (CPU Mode)", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# ============================================================================
# CONFIGURATION - CPU OPTIMIZED
# ============================================================================

SEED = 1337  # FIXED SEED FOR VALIDATION
DEVICE = torch.device('cpu') # FORCED CPU

CONFIG = {
    "epochs": 2,
    "batch_size": 16,
    "seq_length": 32,
    "learning_rate": 0.001,
    "seed": SEED,
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 2,
    "training_subset": 20000, # Only use first 20k chars for fast CPU run
}

print("Validation Configuration (CPU):", flush=True)
for key, value in CONFIG.items():
    print(f"  {key}: {value}", flush=True)
print(f"  Device: {DEVICE}", flush=True)
print(flush=True)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"✅ Set all random seeds to {seed}", flush=True)

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def get_data():
    data_path = Path("data/shakespeare.txt")
    if not data_path.exists():
        import urllib.request
        Path("data").mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()[:CONFIG["training_subset"]]
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    return train_data, val_data, vocab_size

def create_batches(data, batch_size, seq_length):
    batches = []
    # Simplified sliding window
    for i in range(0, len(data) - seq_length - 1, batch_size * 10):
        batch_inputs = []
        batch_targets = []
        for j in range(batch_size):
            start = i + j
            if start + seq_length + 1 < len(data):
                batch_inputs.append(data[start : start + seq_length])
                batch_targets.append(data[start + 1 : start + seq_length + 1])
        if len(batch_inputs) == batch_size:
            batches.append((torch.stack(batch_inputs), torch.stack(batch_targets)))
    return batches

# ============================================================================
# MODELS
# ============================================================================

def get_transformer_model(vocab_size):
    config = GodelaiConfig(
        vocab_size=vocab_size,
        n_embd=64,           # Extreme reduction (128 -> 64)
        n_head=4,
        n_layer=1,           # Extreme reduction (2 -> 1)
        block_size=CONFIG["seq_length"]
    )
    model = GodelaiTransformer(config).to(DEVICE)
    return model

# ============================================================================
# SHADOW T-SCORE
# ============================================================================

def compute_tscore_shadow(model, sample_batches, criterion, device):
    model.eval()
    t_scores = []
    for inputs, targets in sample_batches:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_grads = []
        for i in range(inputs.size(0)):
            model.zero_grad()
            # Note: GodelaiTransformer returns (logits, loss, csp_info)
            logits, _, _ = model(inputs[i:i+1])
            loss = criterion(logits.view(-1, logits.size(-1)), targets[i:i+1].view(-1))
            loss.backward()
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1).clone())
            if grads:
                batch_grads.append(torch.cat(grads))
        if batch_grads:
            grad_matrix = torch.stack(batch_grads)
            n = grad_matrix.shape[0]
            sum_grad_norm = torch.norm(torch.sum(grad_matrix, dim=0))**2
            sum_norm_grad = torch.sum(torch.norm(grad_matrix, dim=1)**2)
            ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
            T = 1.0 - torch.clamp(ratio / n, 0, 1)
            t_scores.append(T.item())
    model.train()
    return sum(t_scores) / len(t_scores) if t_scores else 0.0

# ============================================================================
# TRAINING LOGIC
# ============================================================================

def train_standard():
    set_seed(SEED)
    train_data, val_data, vocab_size = get_data()
    model = get_transformer_model(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"])
    print(f"Standard Model: {len(train_batches)} batches", flush=True)
    
    history = {"loss": [], "t_score_shadow": []}
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for b_idx, (inputs, targets) in enumerate(train_batches):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits, loss, _ = model(inputs, targets=targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (b_idx + 1) % 50 == 0:
                print(f"  Standard Epoch {epoch+1} Batch {b_idx+1}: Loss={loss.item():.4f}", flush=True)
        
        # Shadow T-Score
        sample = train_batches[:CONFIG["tscore_sample_batches"]]
        t_shadow = compute_tscore_shadow(model, sample, criterion, DEVICE)
        
        history["loss"].append(total_loss / len(train_batches))
        history["t_score_shadow"].append(t_shadow)
        print(f"✅ Standard Epoch {epoch+1} Complete. Loss: {history['loss'][-1]:.4f}, T: {t_shadow:.4f}", flush=True)
        
    return history

def train_godelai():
    set_seed(SEED)
    train_data, val_data, vocab_size = get_data()
    model = get_transformer_model(vocab_size)
    agent = GodelAgent(model, min_surplus_energy=CONFIG["min_surplus_energy"])
    agent.optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"])
    print(f"GodelAI Model: {len(train_batches)} batches", flush=True)
    
    history = {"loss": [], "t_score": [], "variance": [], "sleep": 0}
    
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        total_t = 0
        for b_idx, (inputs, targets) in enumerate(train_batches):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            loss_val, t_score, status, metrics = agent.learning_step(inputs, targets, criterion)
            
            total_loss += loss_val
            total_t += t_score
            if status == "SLEEP":
                history["sleep"] += 1
            
            if (b_idx + 1) % 50 == 0:
                print(f"  GodelAI Epoch {epoch+1} Batch {b_idx+1}: Loss={loss_val:.4f}, T={t_score:.4f} [{status}]", flush=True)
        
        avg_loss = total_loss / len(train_batches)
        avg_t = total_t / len(train_batches)
        history["loss"].append(avg_loss)
        history["t_score"].append(avg_t)
        history["variance"].append(agent.get_variance_stats()["avg_variance"])
        
        print(f"✅ GodelAI Epoch {epoch+1} Complete. Loss: {avg_loss:.4f}, T: {avg_t:.4f}", flush=True)
        
    return history

if __name__ == "__main__":
    print("Step 1: Running Standard Transformer Baseline...", flush=True)
    res_std = train_standard()
    
    print("\nStep 2: Running GodelAI Transformer Experiment...", flush=True)
    res_godel = train_godelai()
    
    # Report Variance Ratio
    # Standard variance of shadow T-scores
    v_std = np.std(res_std["t_score_shadow"]) if len(res_std["t_score_shadow"]) > 1 else 0.0
    # GodelAI average rolling variance
    v_godel = res_godel["variance"][-1] if res_godel["variance"] else 0.0
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "standard": res_std,
        "godelai": res_godel,
        "variance_analysis": {
            "standard_t_std": float(v_std),
            "godelai_avg_variance": float(v_godel),
            "ratio": float(v_godel / (v_std + 1e-8)) if v_std > 0 else "N/A"
        }
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "transformer_validation_cpu.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80, flush=True)
    print("TRANSFORMER CPU VALIDATION COMPLETE", flush=True)
    print(f"Results saved to: {out_file}", flush=True)
    print(f"GodelAI Avg Variance: {v_godel:.6f}", flush=True)
    print(f"Sleep Events: {res_godel['sleep']}", flush=True)
    print("="*80, flush=True)

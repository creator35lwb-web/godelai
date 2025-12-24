"""
GodelAI Training Script
=======================

Unified training loop that integrates:
- GodelaiTransformer (from Grok's implementation)
- CSPRegularizer (from Kimi's theoretical framework)
- Bandwidth monitoring and circuit breaker

Usage:
    python -m godelai.training.train --config small --epochs 10
    
    # With C-S-P tracking
    python -m godelai.training.train --config small --epochs 10 --csp-lambda 0.1
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local imports
from godelai.models.transformer import (
    GodelaiTransformer,
    GodelaiConfig,
    create_godelai_small,
    create_godelai_medium,
    create_godelai_large,
)
from godelai.reg.csp_regularizer import CSPRegularizer


class CharDataset(Dataset):
    """
    Character-level dataset for quick experimentation.
    
    In C-S-P terms: This is the raw "chaos" that will be compressed
    into the model's State.
    """
    
    def __init__(self, text: str, block_size: int):
        self.block_size = block_size
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Encode text
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
    def decode(self, tokens):
        """Decode token indices back to text."""
        return ''.join([self.itos[t.item()] for t in tokens])


class GodelaiTrainer:
    """
    Trainer class that integrates C-S-P regularization.
    
    The training loop monitors:
    - Task loss (cross-entropy)
    - C-S-P loss (propagation conservation)
    - Bandwidth (model health metric)
    
    Circuit breaker triggers if bandwidth drops below threshold.
    """
    
    def __init__(
        self,
        model: GodelaiTransformer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        csp_lambda: float = 0.1,
        csp_gamma: float = 2.0,
        bandwidth_threshold: float = 0.1,
        log_dir: str = "./logs",
        device: str = "auto"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.csp_lambda = csp_lambda
        self.bandwidth_threshold = bandwidth_threshold
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # C-S-P Regularizer
        self.csp_regularizer = CSPRegularizer(
            gamma=csp_gamma,
            threshold_epsilon=bandwidth_threshold,
            track_history=True,
            log_dir=str(self.log_dir / "csp")
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        print(f"GodelaiTrainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  C-S-P lambda: {csp_lambda}")
        print(f"  Bandwidth threshold: {bandwidth_threshold}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_task_loss = 0
        total_csp_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass with C-S-P tracking
            logits, task_loss, csp_info = self.model(x, targets=y, track_csp=True)
            
            # C-S-P regularization loss
            csp_loss = self.csp_regularizer(self.model)
            
            # Combined loss
            loss = task_loss + self.csp_lambda * csp_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_csp_loss += csp_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            bandwidth = self.csp_regularizer.get_bandwidth()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'csp': f'{csp_loss.item():.4f}',
                'bw': f'{bandwidth:.3f}'
            })
            
            # Circuit breaker check
            if bandwidth < self.bandwidth_threshold:
                print(f"\n⚠️ CIRCUIT BREAKER: Bandwidth {bandwidth:.4f} < {self.bandwidth_threshold}")
                print("Training paused. Model may be ossifying.")
                break
        
        self.epoch += 1
        
        return {
            'loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'csp_loss': total_csp_loss / num_batches,
            'bandwidth': self.csp_regularizer.get_bandwidth()
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        if not self.val_dataset:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def train(self, epochs: int, save_every: int = 5) -> list:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting GodelAI Training")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': self.epoch}
            self.training_history.append(metrics)
            
            # Print summary
            print(f"\nEpoch {self.epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Task Loss:  {train_metrics['task_loss']:.4f}")
            print(f"  C-S-P Loss: {train_metrics['csp_loss']:.4f}")
            print(f"  Bandwidth:  {train_metrics['bandwidth']:.4f}")
            if val_metrics:
                print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best_model.pt")
                print(f"  ✓ New best model saved!")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        self.csp_regularizer.save_state("final")
        
        # Save training history
        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'best_val_loss': self.best_val_loss,
            'csp_T': self.csp_regularizer.T_prev,
            'bandwidth': self.csp_regularizer.get_bandwidth()
        }
        
        path = self.log_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Bandwidth: {checkpoint.get('bandwidth', 'N/A')}")


def get_shakespeare_data():
    """Load Tiny Shakespeare dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("tiny_shakespeare")
        return dataset['train']['text'][0]
    except Exception as e:
        print(f"Could not load from Hugging Face: {e}")
        print("Using fallback sample text...")
        return """
        First Citizen:
        Before we proceed any further, hear me speak.

        All:
        Speak, speak.

        First Citizen:
        You are all resolved rather to die than to famish?

        All:
        Resolved. resolved.

        First Citizen:
        First, you know Caius Marcius is chief enemy to the people.
        """ * 100  # Repeat for more data


def main():
    parser = argparse.ArgumentParser(description="Train GodelAI Transformer")
    parser.add_argument("--config", type=str, default="small", 
                        choices=["small", "medium", "large"],
                        help="Model configuration")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--csp-lambda", type=float, default=0.1,
                        help="Weight for C-S-P regularization loss")
    parser.add_argument("--bandwidth-threshold", type=float, default=0.1,
                        help="Minimum bandwidth before circuit breaker")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Create model
    print(f"Creating GodelAI {args.config} model...")
    if args.config == "small":
        model = create_godelai_small()
    elif args.config == "medium":
        model = create_godelai_medium()
    else:
        model = create_godelai_large()
    
    # Load data
    print("Loading data...")
    text = get_shakespeare_data()
    
    # Create dataset
    dataset = CharDataset(text, block_size=model.config.block_size)
    
    # Update model vocab size to match dataset
    if dataset.vocab_size != model.config.vocab_size:
        print(f"Adjusting vocab size: {model.config.vocab_size} -> {dataset.vocab_size}")
        model.config.vocab_size = dataset.vocab_size
        model.token_embedding = nn.Embedding(dataset.vocab_size, model.config.n_embd)
        model.lm_head = nn.Linear(model.config.n_embd, dataset.vocab_size, bias=False)
        model.token_embedding.weight = model.lm_head.weight
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create trainer
    trainer = GodelaiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        csp_lambda=args.csp_lambda,
        bandwidth_threshold=args.bandwidth_threshold,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Train
    history = trainer.train(epochs=args.epochs)
    
    # Generate sample
    print("\n" + "="*60)
    print("Sample Generation")
    print("="*60)
    
    model.eval()
    start_text = "\n"
    start_tokens = torch.tensor([[dataset.stoi.get(c, 0) for c in start_text]], dtype=torch.long)
    start_tokens = start_tokens.to(trainer.device)
    
    generated = model.generate(start_tokens, max_new_tokens=200, temperature=0.8)
    generated_text = dataset.decode(generated[0])
    
    print(f"\nGenerated text:\n{generated_text}")
    
    # Final C-S-P report
    print("\n" + "="*60)
    print("Final C-S-P Report")
    print("="*60)
    print(f"Final Bandwidth: {trainer.csp_regularizer.get_bandwidth():.4f}")
    print(f"Final T(θ,t): {trainer.csp_regularizer.T_prev:.4f}")
    print(f"Model is {'ALIVE' if trainer.csp_regularizer.get_bandwidth() > args.bandwidth_threshold else 'AT RISK'}")


if __name__ == "__main__":
    main()

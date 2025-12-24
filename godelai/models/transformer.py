"""
GodelAI Transformer Model
=========================

A minimal transformer implementation with C-S-P (Compression → State → Propagation) 
awareness built into the architecture.

Based on:
- Grok's GodelaiTransformer (nanoGPT-style implementation)
- Kimi's C-S-P theoretical framework
- Integrated by Godel (Manus AI)

Architecture Philosophy:
- Compression: Token embeddings compress discrete symbols into dense vectors
- State: Hidden states in transformer blocks represent "congealed history"
- Propagation: Generation propagates states to new tokens, simulating inheritance

Usage:
    from godelai.models import GodelaiTransformer, GodelaiConfig
    
    config = GodelaiConfig(vocab_size=50257, n_embd=128, n_head=4, n_layer=2)
    model = GodelaiTransformer(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GodelaiConfig:
    """Configuration for GodelAI Transformer model."""
    
    # Model architecture
    vocab_size: int = 50257      # GPT-2 tokenizer size
    n_embd: int = 128            # Embedding dimension
    n_head: int = 4              # Number of attention heads
    n_layer: int = 2             # Number of transformer layers
    block_size: int = 64         # Maximum context length
    
    # Regularization
    dropout: float = 0.1
    
    # C-S-P specific
    track_states: bool = True    # Whether to track hidden states for C-S-P analysis
    propagation_reserve: float = 0.1  # Reserved capacity for propagation layer
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    In C-S-P terms: This is where Compression happens—the model learns to
    compress the relationship between all tokens into attention weights.
    """
    
    def __init__(self, config: GodelaiConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        # Key, Query, Value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y, att  # Return attention weights for C-S-P analysis


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    In C-S-P terms: This is where State crystallizes—non-linear transformations
    create irreversible biases in the representation.
    """
    
    def __init__(self, config: GodelaiConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # Smoother than ReLU, better for language
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block: attention + feed-forward with residual connections.
    
    In C-S-P terms: Each block is a Compression-State cycle. The residual
    connections preserve the ability to propagate information (Propagation layer).
    """
    
    def __init__(self, config: GodelaiConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)
        
        # C-S-P: Track attention patterns for propagation analysis
        self.last_attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (more stable training)
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out  # Residual connection preserves propagation
        x = x + self.ffwd(self.ln2(x))
        
        # Store attention weights for C-S-P analysis
        self.last_attention_weights = attn_weights
        
        return x


class GodelaiTransformer(nn.Module):
    """
    GodelAI Transformer: A C-S-P aware language model.
    
    This model embodies the C-S-P framework:
    - Compression: Embeddings compress tokens into dense vectors
    - State: Hidden states represent crystallized knowledge
    - Propagation: Generation propagates states to new tokens
    
    The model tracks its own "meta-modifiability" through attention patterns
    and hidden state distributions, enabling C-S-P regularization during training.
    """
    
    def __init__(self, config: GodelaiConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings (Compression layer)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks (State layer)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection (Propagation layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (reduces parameters, improves propagation efficiency)
        self.token_embedding.weight = self.lm_head.weight
        
        # C-S-P state tracking
        self._hidden_states = []
        self._attention_patterns = []
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GodelaiTransformer initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        track_csp: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Forward pass with optional C-S-P tracking.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation (B, T)
            track_csp: Whether to track C-S-P metrics
        
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
            csp_info: Dictionary of C-S-P metrics if track_csp=True
        """
        B, T = idx.shape
        device = idx.device
        
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token + position embeddings (Compression)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        
        # Track hidden states for C-S-P analysis
        if track_csp:
            self._hidden_states = [x.detach()]
            self._attention_patterns = []
        
        # Transformer blocks (State)
        for block in self.blocks:
            x = block(x)
            if track_csp:
                self._hidden_states.append(x.detach())
                if block.last_attention_weights is not None:
                    self._attention_patterns.append(block.last_attention_weights.detach())
        
        # Final projection (Propagation)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        # Compute C-S-P metrics
        csp_info = {}
        if track_csp:
            csp_info = self._compute_csp_metrics()
        
        return logits, loss, csp_info
    
    def _compute_csp_metrics(self) -> dict:
        """
        Compute C-S-P metrics from tracked states.
        
        These metrics help monitor the model's "health" in C-S-P terms:
        - compression_ratio: How much information is compressed
        - state_entropy: Diversity of hidden states (higher = more modifiable)
        - propagation_uniformity: How evenly attention is distributed
        """
        metrics = {}
        
        if self._hidden_states:
            # State entropy: measure diversity of hidden representations
            final_state = self._hidden_states[-1]
            state_std = final_state.std(dim=-1).mean().item()
            metrics['state_entropy'] = state_std
            
            # Compression ratio: input dim / effective dim
            # Approximated by ratio of variance explained by top components
            flat_state = final_state.view(-1, final_state.size(-1))
            _, s, _ = torch.svd(flat_state[:min(1000, flat_state.size(0))])
            total_var = (s ** 2).sum()
            top_var = (s[:10] ** 2).sum()
            metrics['compression_ratio'] = (top_var / total_var).item()
        
        if self._attention_patterns:
            # Propagation uniformity: entropy of attention distribution
            # Higher entropy = more uniform = better propagation potential
            attn = self._attention_patterns[-1]
            attn_entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
            metrics['propagation_uniformity'] = attn_entropy
        
        return metrics
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        In C-S-P terms: This is Propagation in action—the model's State
        is being transmitted to new tokens.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Extended token sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_propagation_bandwidth(self) -> float:
        """
        Estimate the model's current propagation bandwidth.
        
        This is a simplified version of Kimi's bandwidth formula:
        bandwidth = state_entropy * propagation_uniformity / compression_ratio
        
        Higher bandwidth = healthier model (more inheritable, more modifiable)
        """
        # Run a forward pass with tracking
        dummy_input = torch.randint(0, self.config.vocab_size, (1, 16))
        if next(self.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        _, _, csp_info = self(dummy_input, track_csp=True)
        
        if not csp_info:
            return 1.0
        
        state_entropy = csp_info.get('state_entropy', 1.0)
        prop_uniformity = csp_info.get('propagation_uniformity', 1.0)
        compression = csp_info.get('compression_ratio', 0.5)
        
        # Avoid division by zero
        bandwidth = (state_entropy * prop_uniformity) / (compression + 0.1)
        
        return bandwidth


# Convenience function for quick model creation
def create_godelai_small() -> GodelaiTransformer:
    """Create a small GodelAI model for experimentation."""
    config = GodelaiConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    return GodelaiTransformer(config)


def create_godelai_medium() -> GodelaiTransformer:
    """Create a medium GodelAI model."""
    config = GodelaiConfig(
        vocab_size=50257,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=128
    )
    return GodelaiTransformer(config)


def create_godelai_large() -> GodelaiTransformer:
    """Create a larger GodelAI model (still trainable on consumer GPU)."""
    config = GodelaiConfig(
        vocab_size=50257,
        n_embd=512,
        n_head=8,
        n_layer=6,
        block_size=256
    )
    return GodelaiTransformer(config)


if __name__ == "__main__":
    # Demo: Create and test a small model
    print("=" * 60)
    print("GodelAI Transformer Demo")
    print("=" * 60)
    
    model = create_godelai_small()
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 32))  # Batch of 2, sequence length 32
    logits, loss, csp_info = model(x, targets=x, track_csp=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"\nC-S-P Metrics:")
    for k, v in csp_info.items():
        print(f"  {k}: {v:.4f}")
    
    # Test generation
    print(f"\nPropagation Bandwidth: {model.get_propagation_bandwidth():.4f}")
    
    # Generate some tokens
    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=20)
    print(f"\nGenerated sequence shape: {generated.shape}")

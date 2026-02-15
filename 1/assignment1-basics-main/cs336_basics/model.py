"""
Transformer model components.
"""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .nn_utils import (
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_swiglu,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return run_rmsnorm(self.d_model, self.eps, self.weight, x)


class TransformerBlock(nn.Module):
    """Pre-normalization Transformer block with RoPE."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        d_k = d_model // num_heads
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Feed-forward network (SwiGLU)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # For SiLU gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Output projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # For value
        
        # Layer norms
        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
    
    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        token_positions: Int[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        # Pre-norm attention
        normed = self.ln1(x)
        
        # Multi-head self-attention with RoPE
        attn_out = run_multihead_self_attention_with_rope(
            self.d_model,
            self.num_heads,
            self.max_seq_len,
            self.theta,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.output_proj.weight,
            normed,
            token_positions,
        )
        
        # Residual connection
        x = x + attn_out
        
        # Pre-norm FFN
        normed = self.ln2(x)
        ffn_out = run_swiglu(
            self.d_model,
            self.d_ff,
            self.w1.weight,
            self.w2.weight,
            self.w3.weight,
            normed,
        )
        
        # Residual connection
        x = x + ffn_out
        
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model."""
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, eps)
        
        # Language model head (no weight tying by default)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(
        self,
        in_indices: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq vocab_size"]:
        batch_size, seq_len = in_indices.shape
        
        # Token embeddings
        x = self.token_embeddings(in_indices)
        
        # Token positions for RoPE
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # LM head
        logits = self.lm_head(x)
        
        return logits


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    """Run a transformer block with given weights."""
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    
    # Load weights
    block.q_proj.weight.data = weights['attn.q_proj.weight']
    block.k_proj.weight.data = weights['attn.k_proj.weight']
    block.v_proj.weight.data = weights['attn.v_proj.weight']
    block.output_proj.weight.data = weights['attn.output_proj.weight']
    block.ln1.weight.data = weights['ln1.weight']
    block.w1.weight.data = weights['ffn.w1.weight']
    block.w2.weight.data = weights['ffn.w2.weight']
    block.w3.weight.data = weights['ffn.w3.weight']
    block.ln2.weight.data = weights['ln2.weight']
    
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq vocab_size"]:
    """Run a transformer LM with given weights."""
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    
    # Load weights
    model.token_embeddings.weight.data = weights['token_embeddings.weight']
    model.ln_final.weight.data = weights['ln_final.weight']
    model.lm_head.weight.data = weights['lm_head.weight']
    
    for i in range(num_layers):
        layer = model.layers[i]
        layer.q_proj.weight.data = weights[f'layers.{i}.attn.q_proj.weight']
        layer.k_proj.weight.data = weights[f'layers.{i}.attn.k_proj.weight']
        layer.v_proj.weight.data = weights[f'layers.{i}.attn.v_proj.weight']
        layer.output_proj.weight.data = weights[f'layers.{i}.attn.output_proj.weight']
        layer.ln1.weight.data = weights[f'layers.{i}.ln1.weight']
        layer.w1.weight.data = weights[f'layers.{i}.ffn.w1.weight']
        layer.w2.weight.data = weights[f'layers.{i}.ffn.w2.weight']
        layer.w3.weight.data = weights[f'layers.{i}.ffn.w3.weight']
        layer.ln2.weight.data = weights[f'layers.{i}.ln2.weight']
    
    model.eval()
    with torch.no_grad():
        return model(in_indices)

"""
Neural network utility functions and modules.
"""
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Linear transformation: y = x @ W^T
    """
    return torch.matmul(in_features, weights.t())


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Embedding lookup.
    """
    return F.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    SwiGLU: SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
    """
    # x @ W1^T -> SiLU
    x1 = F.linear(in_features, w1_weight)  # (..., d_ff)
    x1 = F.silu(x1)
    
    # x @ W3^T
    x3 = F.linear(in_features, w3_weight)  # (..., d_ff)
    
    # Element-wise multiply
    x_gated = x1 * x3  # (..., d_ff)
    
    # @ W2^T
    output = F.linear(x_gated, w2_weight)  # (..., d_model)
    
    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
    """
    d_k = Q.size(-1)
    
    # Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Handle NaN from all -inf row (when mask blocks everything)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    # @ V
    output = torch.matmul(attn_weights, V)
    
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Multi-Head Self-Attention without RoPE.
    Batched implementation handling all heads in a single matrix multiply.
    """
    batch_size, seq_len, _ = in_features.shape
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    
    # Project Q, K, V
    Q = F.linear(in_features, q_proj_weight)  # (batch, seq, d_model)
    K = F.linear(in_features, k_proj_weight)  # (batch, seq, d_model)
    V = F.linear(in_features, v_proj_weight)  # (batch, seq, d_model)
    
    # Reshape for multi-head: (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_v).transpose(1, 2)
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
    mask = ~mask  # Invert: True means attend
    
    # Scaled dot-product attention
    attn_output = run_scaled_dot_product_attention(Q, K, V, mask)  # (batch, num_heads, seq, d_v)
    
    # Concatenate heads: (batch, num_heads, seq, d_v) -> (batch, seq, d_model)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # Output projection
    output = F.linear(attn_output, o_proj_weight)
    
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Rotary Position Embedding (RoPE).
    """
    seq_len = in_query_or_key.size(-2)
    
    # Compute frequencies: theta ^ (-2i/d_k) for i in [0, d_k/2)
    dim_indices = torch.arange(0, d_k, 2, device=in_query_or_key.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (dim_indices / d_k))  # (d_k/2,)
    
    # Compute angles: pos * freq
    positions = token_positions.unsqueeze(-1).float()  # (..., seq, 1)
    angles = positions * freqs.unsqueeze(0)  # (..., seq, d_k/2)
    
    # Apply rotation
    x = in_query_or_key.float()
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # Apply rotation: [x1, x2] @ [[cos, -sin], [sin, cos]]
    rotated_x1 = x1 * cos_angles - x2 * sin_angles
    rotated_x2 = x1 * sin_angles + x2 * cos_angles
    
    # Interleave back
    output = torch.zeros_like(x)
    output[..., 0::2] = rotated_x1
    output[..., 1::2] = rotated_x2
    
    return output.to(in_query_or_key.dtype)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Multi-Head Self-Attention with RoPE.
    """
    batch_size, seq_len, _ = in_features.shape
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    
    # Project Q, K, V
    Q = F.linear(in_features, q_proj_weight)  # (batch, seq, d_model)
    K = F.linear(in_features, k_proj_weight)  # (batch, seq, d_model)
    V = F.linear(in_features, v_proj_weight)  # (batch, seq, d_model)
    
    # Reshape for multi-head: (batch, seq, num_heads, d_k)
    Q = Q.view(batch_size, seq_len, num_heads, d_k)
    K = K.view(batch_size, seq_len, num_heads, d_k)
    V = V.view(batch_size, seq_len, num_heads, d_v)
    
    # Default token positions
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    # Apply RoPE to Q and K (per head)
    Q_rope = torch.zeros_like(Q)
    K_rope = torch.zeros_like(K)
    for h in range(num_heads):
        Q_rope[:, :, h, :] = run_rope(d_k, theta, max_seq_len, Q[:, :, h, :], token_positions)
        K_rope[:, :, h, :] = run_rope(d_k, theta, max_seq_len, K[:, :, h, :], token_positions)
    
    # Transpose for attention: (batch, num_heads, seq, d_k)
    Q_rope = Q_rope.transpose(1, 2)
    K_rope = K_rope.transpose(1, 2)
    V = V.transpose(1, 2)
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool), diagonal=1)
    mask = ~mask  # True means attend
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
    
    # Scaled dot-product attention
    attn_output = run_scaled_dot_product_attention(Q_rope, K_rope, V, mask)
    
    # Concatenate heads
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # Output projection
    output = F.linear(attn_output, o_proj_weight)
    
    return output


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Root Mean Square Layer Normalization.
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weights
    """
    # Compute RMS
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    
    # Normalize and scale
    normalized = in_features / rms * weights
    
    return normalized


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    SiLU activation: x * sigmoid(x)
    """
    return F.silu(in_features)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Softmax with numerical stability.
    """
    # Subtract max for numerical stability
    max_val = torch.max(in_features, dim=dim, keepdim=True)[0]
    shifted = in_features - max_val
    exp_shifted = torch.exp(shifted)
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / sum_exp


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """
    Cross-entropy loss with numerical stability.
    """
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(inputs, dim=-1)
    return F.nll_loss(log_probs, targets)


def run_gradient_clipping(parameters: list[nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by L2 norm.
    """
    # Filter out parameters without gradients or not requiring grad
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return
    
    # Compute total norm
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
    
    # Clip
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

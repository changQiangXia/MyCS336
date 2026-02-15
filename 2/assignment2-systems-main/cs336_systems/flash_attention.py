"""
FlashAttention2 implementation in PyTorch.

FlashAttention2 uses tiling and online softmax to compute attention
without materializing the full attention matrix, reducing memory usage.
Reference: https://arxiv.org/abs/2307.08691
"""

from typing import Type

import torch
from torch import Tensor
from einops import einsum


class FlashAttention2PyTorch(torch.autograd.Function):
    """
    FlashAttention2 implementation using pure PyTorch.
    
    Implements online softmax with tiling to avoid storing the full N×N attention matrix.
    """
    
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False):
        """
        Forward pass of FlashAttention2.
        
        Args:
            q: Query tensor of shape (batch_size, n_queries, d)
            k: Key tensor of shape (batch_size, n_keys, d)
            v: Value tensor of shape (batch_size, n_keys, d)
            is_causal: Whether to apply causal mask
            
        Returns:
            o: Output tensor of shape (batch_size, n_queries, d)
        """
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        
        scale = 1.0 / (d ** 0.5)
        
        # Block sizes
        Br = 64
        Bc = 64
        
        # Output tensor
        o = torch.zeros_like(q)
        # Log-sum-exp for backward pass
        lse = torch.zeros(batch_size, n_queries, device=q.device, dtype=torch.float32)
        
        # Number of blocks
        Tr = (n_queries + Br - 1) // Br
        Tc = (n_keys + Bc - 1) // Bc
        
        # Process each query block
        for i in range(Tr):
            q_start = i * Br
            q_end = min(q_start + Br, n_queries)
            qi = q[:, q_start:q_end, :]  # (batch, Br, d)
            
            # Running statistics (in fp32 for numerical stability)
            mi = torch.full((batch_size, q_end - q_start), float('-inf'), device=q.device, dtype=torch.float32)
            li = torch.zeros(batch_size, q_end - q_start, device=q.device, dtype=torch.float32)
            oi = torch.zeros(batch_size, q_end - q_start, d, device=q.device, dtype=torch.float32)
            
            # Iterate over key/value blocks
            for j in range(Tc):
                k_start = j * Bc
                k_end = min(k_start + Bc, n_keys)
                kj = k[:, k_start:k_end, :]
                vj = v[:, k_start:k_end, :]
                
                # Compute attention scores S_ij = Q_i @ K_j^T * scale
                sij = einsum(qi, kj, 'b q d, b k d -> b q k').to(torch.float32) * scale
                
                # Apply causal mask if needed
                if is_causal:
                    row_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    col_indices = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                    # causal: can only attend to positions <= current position
                    causal_mask = col_indices <= row_indices
                    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
                    sij = torch.where(causal_mask, sij, torch.tensor(float('-inf'), device=q.device, dtype=torch.float32))
                
                # Online softmax update
                # m_new = max(m_old, max(S_ij))
                mij = torch.max(sij, dim=-1).values
                m_new = torch.maximum(mi, mij)
                
                # Update running sum: l_new = exp(m_old - m_new) * l_old + sum(exp(S_ij - m_new))
                li = torch.exp(mi - m_new) * li + torch.exp(sij - m_new.unsqueeze(-1)).sum(dim=-1)
                
                # Update running output: o_new = exp(m_old - m_new) * o_old + exp(S_ij - m_new) @ V_j
                pij = torch.exp(sij - m_new.unsqueeze(-1))  # Unnormalized attention weights
                oi = torch.exp(mi - m_new).unsqueeze(-1) * oi + einsum(pij, vj.to(torch.float32), 'b q k, b k d -> b q d')
                
                mi = m_new
            
            # Final normalization: output = oi / li
            oi = oi / li.unsqueeze(-1)
            o[:, q_start:q_end, :] = oi.to(q.dtype)
            
            # Log-sum-exp = log(li) + mi
            lse[:, q_start:q_end] = torch.log(li) + mi
        
        # Save for backward
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return o
    
    @staticmethod
    def backward(ctx, do: Tensor):
        """
        Backward pass of FlashAttention2.
        
        Recomputes attention scores during backward to save memory.
        """
        q, k, v, o, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        
        # Initialize gradients
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        
        # Convert inputs to fp32 for numerical stability
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)
        o_fp32 = o.to(torch.float32)
        do_fp32 = do.to(torch.float32)
        
        # D_i = rowsum(dO_i * O_i)
        D = (do_fp32 * o_fp32).sum(dim=-1)  # (batch, n_queries)
        
        # Block sizes
        Br = 64
        Bc = 64
        
        Tr = (n_queries + Br - 1) // Br
        Tc = (n_keys + Bc - 1) // Bc
        
        # Backward pass - process KV blocks
        for j in range(Tc):
            k_start = j * Bc
            k_end = min(k_start + Bc, n_keys)
            
            kj = k_fp32[:, k_start:k_end, :]
            vj = v_fp32[:, k_start:k_end, :]
            
            dkj = torch.zeros_like(kj)
            dvj = torch.zeros_like(vj)
            
            for i in range(Tr):
                q_start = i * Br
                q_end = min(q_start + Br, n_queries)
                
                qi = q_fp32[:, q_start:q_end, :]
                oi = o_fp32[:, q_start:q_end, :]
                doi = do_fp32[:, q_start:q_end, :]
                Di = D[:, q_start:q_end]
                lsei = lse[:, q_start:q_end]
                
                # Compute S_ij
                sij = einsum(qi, kj, 'b q d, b k d -> b q k') * scale
                
                # Apply causal mask
                if is_causal:
                    row_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    col_indices = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                    causal_mask = col_indices <= row_indices
                    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
                    sij = torch.where(causal_mask, sij, torch.tensor(float('-inf'), device=q.device, dtype=torch.float32))
                
                # P_ij = exp(S_ij - lse_i)
                pij = torch.exp(sij - lsei.unsqueeze(-1))
                
                # dV_j += P_ij^T @ dO_i
                dvj += einsum(pij, doi, 'b q k, b q d -> b k d')
                
                # dP_ij = dO_i @ V_j^T
                dpij = einsum(doi, vj, 'b q d, b k d -> b q k')
                
                # dS_ij = P_ij * (dP_ij - D_i)
                dsij = pij * (dpij - Di.unsqueeze(-1))
                
                # dQ_i += dS_ij @ K_j
                dq[:, q_start:q_end, :] += einsum(dsij, kj, 'b q k, b k d -> b q d') * scale
                
                # dK_j += dS_ij^T @ Q_i
                dkj += einsum(dsij, qi, 'b q k, b q d -> b k d') * scale
            
            dk[:, k_start:k_end, :] += dkj
            dv[:, k_start:k_end, :] += dvj
        
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    
    Returns:
        The FlashAttention2PyTorch class
    """
    return FlashAttention2PyTorch

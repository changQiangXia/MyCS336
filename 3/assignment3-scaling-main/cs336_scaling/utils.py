"""Utility functions for scaling laws experiments."""

import numpy as np


def compute_model_params(num_layers: int, d_model: int) -> int:
    """
    Compute the number of non-embedding parameters for a Transformer model.
    
    Formula: N = 12 * n_layers * d_model^2
    
    Args:
        num_layers: Number of Transformer layers
        d_model: Model embedding dimension
    
    Returns:
        Number of non-embedding parameters
    """
    return 12 * num_layers * d_model * d_model


def compute_dataset_size(compute_budget: float, num_params: float) -> float:
    """
    Compute the number of training tokens given compute budget and model size.
    
    Formula: C = 6 * N * D  =>  D = C / (6 * N)
    
    Args:
        compute_budget: Compute budget in FLOPs (C)
        num_params: Number of model parameters (N)
    
    Returns:
        Number of training tokens (D)
    """
    return compute_budget / (6 * num_params)


def flops_to_string(flops: float) -> str:
    """Convert FLOPs to human-readable string."""
    if flops >= 1e21:
        return f"{flops/1e21:.1f}e21"
    elif flops >= 1e18:
        return f"{flops/1e18:.1f}e18"
    elif flops >= 1e15:
        return f"{flops/1e15:.1f}e15"
    else:
        return f"{flops:.2e}"


def params_to_string(params: float) -> str:
    """Convert parameter count to human-readable string."""
    if params >= 1e9:
        return f"{params/1e9:.2f}B"
    elif params >= 1e6:
        return f"{params/1e6:.2f}M"
    else:
        return f"{params:.0f}"

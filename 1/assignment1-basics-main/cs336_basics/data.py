"""
Data loading utilities.
"""
import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


def run_get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of language modeling data from the dataset.
    
    Returns input sequences and target sequences (shifted by 1).
    """
    # Calculate valid starting indices
    max_start = len(dataset) - context_length - 1
    
    # Random starting positions
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    
    # Create input and target sequences
    x = np.stack([dataset[i:i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])
    
    # Convert to tensors and move to device
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    return x, y

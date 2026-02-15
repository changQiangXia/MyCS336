"""
Checkpoint serialization utilities.
"""
import os
from typing import BinaryIO, IO

import torch
import torch.nn as nn
from torch.optim import Optimizer


def run_save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save a checkpoint containing model state, optimizer state, and iteration.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load a checkpoint and restore model and optimizer state.
    
    Returns the iteration number from the checkpoint.
    """
    checkpoint = torch.load(src, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']

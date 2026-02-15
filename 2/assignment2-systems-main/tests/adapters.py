from __future__ import annotations

from typing import Type

import torch

from cs336_systems.flash_attention import get_flashattention_autograd_function_pytorch
from cs336_systems.ddp import (
    get_ddp_individual_parameters,
    get_ddp_bucketed,
    ddp_individual_parameters_on_after_backward,
    ddp_bucketed_on_after_backward,
    ddp_bucketed_on_train_batch_start,
)
from cs336_systems.sharded_optimizer import get_sharded_optimizer


# Re-export functions
__all__ = [
    "get_flashattention_autograd_function_pytorch",
    "get_flashattention_autograd_function_triton",
    "get_ddp_individual_parameters",
    "ddp_individual_parameters_on_after_backward",
    "get_ddp_bucketed",
    "ddp_bucketed_on_after_backward",
    "ddp_bucketed_on_train_batch_start",
    "get_sharded_optimizer",
]


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    
    Currently returns PyTorch implementation as placeholder.
    
    Returns:
        A class object (not an instance of the class)
    """
    # For now, return PyTorch implementation
    # Triton implementation can be added later if needed
    return get_flashattention_autograd_function_pytorch()

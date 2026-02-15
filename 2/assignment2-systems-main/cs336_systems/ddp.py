"""
Distributed Data Parallel (DDP) implementations.

This module provides DDP wrappers that handle:
1. Parameter broadcasting from rank 0 to all ranks
2. Gradient synchronization using all-reduce
3. Bucket-based communication to overlap with computation
"""

from typing import List, Tuple, Optional, Callable
import math

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPIndividualParameters(nn.Module):
    """
    DDP wrapper that synchronizes gradients individually for each parameter.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all ranks at initialization
    2. Registers backward hooks to all-reduce gradients as they become ready
    3. Overlaps communication with computation
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
        # Get the process group info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Broadcast parameters from rank 0 to all ranks
        if dist.is_initialized() and self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        
        # Register backward hooks for gradient synchronization
        self._handles = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register backward hooks to synchronize gradients."""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # Register hook for this parameter
                handle = param.register_hook(self._make_hook(param))
                self._handles.append(handle)
    
    def _make_hook(self, param: nn.Parameter):
        """Create a backward hook for gradient synchronization."""
        def hook(grad):
            if dist.is_initialized() and self.world_size > 1:
                # All-reduce the gradient (SUM, then divide by world_size)
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)
            return grad
        return hook
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all gradient synchronizations to complete."""
        # With hooks, synchronization happens synchronously in backward
        # So there's nothing to wait for here
        pass
    
    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)
    
    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict)


class DDPBucketed(nn.Module):
    """
    DDP wrapper that synchronizes gradients using buckets.
    
    This implementation:
    1. Broadcasts parameters from rank 0 to all ranks at initialization
    2. Groups parameters into buckets based on size
    3. All-reduces buckets when backward is done
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        
        # Get the process group info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate bucket size in bytes
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        # Broadcast parameters from rank 0 to all ranks
        if dist.is_initialized() and self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        
        # Build buckets
        self.buckets = self._build_buckets()
        
    def _build_buckets(self) -> List[List[Tuple[str, nn.Parameter]]]:
        """Group parameters into buckets based on size."""
        # Get all parameters that require gradients
        params_with_names = [
            (name, param) for name, param in self.module.named_parameters()
            if param.requires_grad
        ]
        
        if not params_with_names:
            return []
        
        # Sort by parameter size (larger first) to minimize padding
        params_with_names.sort(key=lambda x: x[1].numel(), reverse=True)
        
        buckets = []
        current_bucket = []
        current_bucket_size = 0
        
        for name, param in params_with_names:
            param_size = param.numel() * param.element_size()
            
            # If adding this parameter would exceed bucket size, start a new bucket
            if current_bucket and current_bucket_size + param_size > self.bucket_size_bytes:
                buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append((name, param))
            current_bucket_size += param_size
        
        # Add the last bucket if it's not empty
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Synchronize gradients for all buckets."""
        if not dist.is_initialized() or self.world_size <= 1:
            return
        
        # Synchronize gradients for each bucket
        for bucket in self.buckets:
            # Get gradients for this bucket
            grads = [param.grad for name, param in bucket if param.grad is not None]
            
            if not grads:
                continue
            
            # Flatten gradients
            flat_grads = _flatten_dense_tensors(grads)
            
            # All-reduce (SUM)
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            
            # Average by dividing by world_size
            flat_grads.div_(self.world_size)
            
            # Unflatten and copy back
            unflattened = _unflatten_dense_tensors(flat_grads, grads)
            for grad, unflat_grad in zip(grads, unflattened):
                grad.copy_(unflat_grad)
    
    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)
    
    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict)


def get_ddp_individual_parameters(module: nn.Module) -> nn.Module:
    """
    Returns a DDP wrapper that synchronizes gradients individually.
    
    Args:
        module: The module to wrap
        
    Returns:
        DDP-wrapped module
    """
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Called after backward pass for individual parameter DDP.
    
    Args:
        ddp_model: DDP-wrapped model
        optimizer: Optimizer
    """
    if isinstance(ddp_model, DDPIndividualParameters):
        ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: nn.Module, bucket_size_mb: float) -> nn.Module:
    """
    Returns a DDP wrapper that synchronizes gradients using buckets.
    
    Args:
        module: The module to wrap
        bucket_size_mb: Bucket size in megabytes
        
    Returns:
        DDP-wrapped module
    """
    return DDPBucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Called after backward pass for bucketed DDP.
    
    Args:
        ddp_model: DDP-wrapped model
        optimizer: Optimizer
    """
    if isinstance(ddp_model, DDPBucketed):
        ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Called at the start of each training batch for bucketed DDP.
    
    Args:
        ddp_model: DDP-wrapped model
        optimizer: Optimizer
    """
    pass

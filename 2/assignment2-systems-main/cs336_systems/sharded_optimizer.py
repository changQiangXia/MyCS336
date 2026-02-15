"""
Sharded Optimizer implementation (ZeRO-1 style).
"""

from typing import Type, Iterable, Dict, Any
import torch
import torch.distributed as dist


class ShardedOptimizer:
    """
    A sharded optimizer that distributes optimizer states across ranks (ZeRO-1).
    
    Each rank only stores optimizer states for a subset of parameters.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[torch.optim.Optimizer],
        world_size: int,
        rank: int,
        **kwargs
    ):
        self.world_size = world_size
        self.rank = rank
        
        # Convert params to list
        all_params = list(params)
        
        # Partition parameters: each rank owns params[i] where i % world_size == rank
        self.param_to_rank = {}
        local_params = []
        
        for i, param in enumerate(all_params):
            assigned_rank = i % world_size
            self.param_to_rank[id(param)] = assigned_rank
            if assigned_rank == rank:
                local_params.append(param)
        
        self.all_params = all_params
        
        # Create local optimizer with only local parameters
        self.optimizer = optimizer_cls(local_params, **kwargs)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if not dist.is_initialized() or self.world_size <= 1:
            self.optimizer.step()
            return loss
        
        # Sync gradients for local parameters
        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
        
        # Update local parameters
        self.optimizer.step()
        
        # Broadcast updated values to all ranks
        for param in self.all_params:
            owner = self.param_to_rank[id(param)]
            dist.broadcast(param.data, src=owner)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none)
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
    """Returns a sharded optimizer."""
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    
    return ShardedOptimizer(params, optimizer_cls, world_size, rank, **kwargs)

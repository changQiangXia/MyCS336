"""
Optimizer implementations.
"""
import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with weight decay decoupled from gradient update.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                step = state['step']
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = group['lr'] / bias_correction1
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def get_adamw_cls() -> Any:
    """Returns the AdamW optimizer class."""
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    - Linear warmup from 0 to max_learning_rate during warmup_iters
    - Cosine decay from max_learning_rate to min_learning_rate
    """
    if it < warmup_iters:
        # Linear warmup: it=0 -> 0, it=warmup_iters-1 -> max_learning_rate * (warmup_iters-1)/warmup_iters
        # But expected: it=0 -> 0, it=warmup_iters-1 -> max_learning_rate * (warmup_iters-1)/warmup_iters
        # At it=warmup_iters, we want max_learning_rate
        return max_learning_rate * it / warmup_iters
    elif it >= cosine_cycle_iters:
        # After cosine cycle, return min learning rate
        return min_learning_rate
    else:
        # Cosine decay
        decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

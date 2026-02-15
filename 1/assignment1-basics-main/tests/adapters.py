from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Import implementations
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cs336_basics.nn_utils import (
    run_linear,
    run_embedding,
    run_swiglu,
    run_scaled_dot_product_attention,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rope,
    run_rmsnorm,
    run_silu,
    run_softmax,
    run_cross_entropy,
    run_gradient_clipping,
)
from cs336_basics.model import (
    run_transformer_block,
    run_transformer_lm,
)
from cs336_basics.optimizer import (
    get_adamw_cls,
    run_get_lr_cosine_schedule,
)
from cs336_basics.data import run_get_batch
from cs336_basics.serialization import (
    run_save_checkpoint,
    run_load_checkpoint,
)
from cs336_basics.tokenizer import (
    get_tokenizer,
    run_train_bpe,
)


__all__ = [
    'run_linear',
    'run_embedding',
    'run_swiglu',
    'run_scaled_dot_product_attention',
    'run_multihead_self_attention',
    'run_multihead_self_attention_with_rope',
    'run_rope',
    'run_transformer_block',
    'run_transformer_lm',
    'run_rmsnorm',
    'run_silu',
    'run_get_batch',
    'run_softmax',
    'run_cross_entropy',
    'run_gradient_clipping',
    'get_adamw_cls',
    'run_get_lr_cosine_schedule',
    'run_save_checkpoint',
    'run_load_checkpoint',
    'get_tokenizer',
    'run_train_bpe',
]

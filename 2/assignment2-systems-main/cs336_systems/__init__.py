import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")

# Export main functions
from .flash_attention import (
    get_flashattention_autograd_function_pytorch,
)

from .ddp import (
    get_ddp_individual_parameters,
    ddp_individual_parameters_on_after_backward,
    get_ddp_bucketed,
    ddp_bucketed_on_after_backward,
    ddp_bucketed_on_train_batch_start,
)

from .sharded_optimizer import (
    get_sharded_optimizer,
)

__all__ = [
    "get_flashattention_autograd_function_pytorch",
    "get_ddp_individual_parameters",
    "ddp_individual_parameters_on_after_backward",
    "get_ddp_bucketed",
    "ddp_bucketed_on_after_backward",
    "ddp_bucketed_on_train_batch_start",
    "get_sharded_optimizer",
]

"""CS336 Scaling Laws package."""

import importlib.metadata

__version__ = importlib.metadata.version("cs336-scaling")

# Export main classes and functions
from .chinchilla_isoflops import (
    load_isoflops_data,
    group_by_compute_budget,
    find_optimal_per_budget,
    fit_scaling_law,
    predict_optimal,
    run_chinchilla_analysis,
    power_law,
)

from .scaling_api import (
    ExperimentConfig,
    TrainingAPI,
    MockTrainingAPI,
    create_api,
    VALID_RANGES,
)

from .scaling_experiment import (
    ScalingExperiment,
    ScalingLawParams,
    chinchilla_style_strategy,
    uniform_sampling_strategy,
    run_full_experiment,
)

from .utils import (
    compute_model_params,
    compute_dataset_size,
    params_to_string,
    flops_to_string,
)

__all__ = [
    # Problem 1
    'load_isoflops_data',
    'group_by_compute_budget',
    'find_optimal_per_budget',
    'fit_scaling_law',
    'predict_optimal',
    'run_chinchilla_analysis',
    'power_law',
    
    # API
    'ExperimentConfig',
    'TrainingAPI',
    'MockTrainingAPI',
    'create_api',
    'VALID_RANGES',
    
    # Problem 2
    'ScalingExperiment',
    'ScalingLawParams',
    'chinchilla_style_strategy',
    'uniform_sampling_strategy',
    'run_full_experiment',
    
    # Utils
    'compute_model_params',
    'compute_dataset_size',
    'params_to_string',
    'flops_to_string',
]

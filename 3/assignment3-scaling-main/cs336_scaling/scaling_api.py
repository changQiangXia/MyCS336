"""
Training API client and mock implementation for scaling experiments.

This module provides:
1. Real API client (requires Stanford VPN and API key)
2. Mock API for testing (simulates training with realistic loss curves)
"""

import os
import logging
import random
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "http://hyperturing.stanford.edu:8000"

# Valid parameter ranges
VALID_RANGES = {
    'd_model': (64, 1024),
    'num_layers': (2, 24),
    'num_heads': (2, 16),
    'batch_size': {128, 256},
    'learning_rate': (1e-4, 1e-3),
    'train_flops': {1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18},
}


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int
    
    def validate(self) -> None:
        """Validate the configuration against allowed ranges."""
        if not (VALID_RANGES['d_model'][0] <= self.d_model <= VALID_RANGES['d_model'][1]):
            raise ValueError(f"d_model must be in range {VALID_RANGES['d_model']}, got {self.d_model}")
        
        if not (VALID_RANGES['num_layers'][0] <= self.num_layers <= VALID_RANGES['num_layers'][1]):
            raise ValueError(f"num_layers must be in range {VALID_RANGES['num_layers']}, got {self.num_layers}")
        
        if not (VALID_RANGES['num_heads'][0] <= self.num_heads <= VALID_RANGES['num_heads'][1]):
            raise ValueError(f"num_heads must be in range {VALID_RANGES['num_heads']}, got {self.num_heads}")
        
        if self.batch_size not in VALID_RANGES['batch_size']:
            raise ValueError(f"batch_size must be one of {VALID_RANGES['batch_size']}, got {self.batch_size}")
        
        if not (VALID_RANGES['learning_rate'][0] <= self.learning_rate <= VALID_RANGES['learning_rate'][1]):
            raise ValueError(f"learning_rate must be in range {VALID_RANGES['learning_rate']}, got {self.learning_rate}")
        
        if self.train_flops not in VALID_RANGES['train_flops']:
            raise ValueError(f"train_flops must be one of {VALID_RANGES['train_flops']}, got {self.train_flops}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API calls."""
        return asdict(self)


class TrainingAPI:
    """Client for the training API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key (SSH public key). If None, reads from CS336_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("CS336_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set CS336_API_KEY environment variable.")
    
    def get_loss(self, config: ExperimentConfig) -> Tuple[float, float]:
        """
        Query the training loss for a given configuration.
        
        Args:
            config: Experiment configuration.
        
        Returns:
            Tuple of (loss, total_flops_used).
        """
        config.validate()
        
        params = config.to_dict()
        params['api_key'] = self.api_key
        
        url = f"{API_BASE_URL}/loss"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data['loss'], data['total_flops_used']
    
    def get_total_flops_used(self) -> float:
        """Get total FLOPs used by this API key."""
        url = f"{API_BASE_URL}/total_flops_used"
        response = requests.get(url, params={'api_key': self.api_key}, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_previous_runs(self) -> List[Dict]:
        """Get list of all previous runs for this API key."""
        url = f"{API_BASE_URL}/previous_runs"
        response = requests.get(url, params={'api_key': self.api_key}, timeout=30)
        response.raise_for_status()
        return response.json()['previous_runs']
    
    def check_connection(self) -> bool:
        """Check if API is accessible."""
        try:
            url = f"{API_BASE_URL}/docs"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False


class MockTrainingAPI:
    """
    Mock implementation of the training API for testing.
    
    Simulates training losses using a realistic scaling law model.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize mock API.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.query_history: List[Dict] = []
        self.total_flops_used = 0.0
        
        # Cache for consistent results
        self._cache: Dict[str, Tuple[float, float]] = {}
    
    def _config_to_key(self, config: ExperimentConfig) -> str:
        """Convert config to cache key."""
        return f"{config.d_model}_{config.num_layers}_{config.num_heads}_{config.batch_size}_{config.learning_rate}_{config.train_flops}"
    
    def _compute_model_params(self, config: ExperimentConfig) -> int:
        """Compute model parameters: N = 12 * n_layers * d_model^2."""
        return 12 * config.num_layers * config.d_model * config.d_model
    
    def _simulate_loss(self, config: ExperimentConfig) -> float:
        """
        Simulate training loss using a realistic scaling law.
        
        Loss model based on Chinchilla paper insights:
        L(N, D) = A/N^alpha + B/D^beta + E
        
        Where:
        - N is model size
        - D is number of training tokens = train_flops / (6 * N)
        - alpha, beta ~ 0.34 for decoder-only Transformers
        """
        N = self._compute_model_params(config)
        D = config.train_flops / (6 * N)
        
        # Scaling law parameters (approximate values from literature)
        A = 406.4  # coefficient for model size term
        B = 410.7  # coefficient for data size term  
        alpha = 0.34
        beta = 0.28
        E = 1.69   # irreducible entropy
        
        # Add hyperparameter effects
        # - Learning rate: optimal around 0.001, suboptimal elsewhere
        # - Batch size: minor effect at fixed FLOPs
        # - Width/depth ratio: some effect
        
        lr_optimal = 0.001
        lr_factor = 1.0 + 0.5 * abs(np.log10(config.learning_rate) - np.log10(lr_optimal))
        
        # Width-to-depth ratio effect
        ideal_depth = np.sqrt(N / 12) / 32  # heuristic for square-ish models
        depth_ratio = config.num_layers / ideal_depth
        depth_factor = 1.0 + 0.1 * abs(np.log(depth_ratio))
        
        # Compute base loss
        base_loss = A / (N ** alpha) + B / (D ** beta) + E
        
        # Add hyperparameter penalties and noise
        noise = self.rng.normal(0, 0.02)
        loss = base_loss * lr_factor * depth_factor + noise
        
        return max(loss, 1.5)  # Floor at reasonable minimum
    
    def get_loss(self, config: ExperimentConfig) -> Tuple[float, float]:
        """
        Get simulated training loss.
        
        Returns:
            Tuple of (loss, total_flops_used).
        """
        config.validate()
        
        cache_key = self._config_to_key(config)
        
        # Check cache for consistent results
        if cache_key in self._cache:
            loss, _ = self._cache[cache_key]
        else:
            loss = self._simulate_loss(config)
            self._cache[cache_key] = (loss, config.train_flops)
        
        # Track FLOPs (only for new queries)
        if not any(r['train_flops'] == config.train_flops and 
                   r['d_model'] == config.d_model and
                   r['num_layers'] == config.num_layers and
                   r['num_heads'] == config.num_heads and
                   r['batch_size'] == config.batch_size and
                   r['learning_rate'] == config.learning_rate
                   for r in self.query_history):
            self.total_flops_used += config.train_flops
        
        # Record query
        run_record = {
            **config.to_dict(),
            'loss': loss,
        }
        self.query_history.append(run_record)
        
        return loss, self.total_flops_used
    
    def get_total_flops_used(self) -> float:
        """Get total FLOPs used."""
        return self.total_flops_used
    
    def get_previous_runs(self) -> List[Dict]:
        """Get list of all previous runs."""
        return self.query_history
    
    def reset(self):
        """Reset the mock API state."""
        self.query_history = []
        self.total_flops_used = 0.0
        self._cache = {}
        self.rng = np.random.RandomState(self.seed)
    
    def check_connection(self) -> bool:
        """Mock API is always accessible."""
        return True


def create_api(use_mock: bool = False, api_key: Optional[str] = None, seed: int = 42):
    """
    Factory function to create appropriate API client.
    
    Args:
        use_mock: If True, return MockTrainingAPI. Otherwise return real TrainingAPI.
        api_key: API key for real API (ignored if use_mock=True).
        seed: Random seed for mock API.
    
    Returns:
        TrainingAPI or MockTrainingAPI instance.
    """
    if use_mock:
        logger.info("Using Mock Training API")
        return MockTrainingAPI(seed=seed)
    else:
        return TrainingAPI(api_key=api_key)

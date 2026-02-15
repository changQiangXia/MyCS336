"""
Tests for scaling_api module.

Run with: uv run python -m pytest tests/test_api.py -v
"""

import pytest
import numpy as np

from cs336_scaling.scaling_api import (
    ExperimentConfig,
    MockTrainingAPI,
    create_api,
    VALID_RANGES,
)
from cs336_scaling.utils import compute_model_params


class TestExperimentConfig:
    """Test ExperimentConfig validation."""
    
    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        config.validate()  # Should not raise
    
    def test_invalid_d_model(self):
        """Test validation catches invalid d_model."""
        config = ExperimentConfig(
            d_model=2000,  # Too large
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        with pytest.raises(ValueError, match="d_model"):
            config.validate()
    
    def test_invalid_num_layers(self):
        """Test validation catches invalid num_layers."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=30,  # Too many
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        with pytest.raises(ValueError, match="num_layers"):
            config.validate()
    
    def test_invalid_batch_size(self):
        """Test validation catches invalid batch_size."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=64,  # Not allowed
            learning_rate=0.001,
            train_flops=1e15,
        )
        with pytest.raises(ValueError, match="batch_size"):
            config.validate()
    
    def test_invalid_learning_rate(self):
        """Test validation catches invalid learning_rate."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.01,  # Too large
            train_flops=1e15,
        )
        with pytest.raises(ValueError, match="learning_rate"):
            config.validate()
    
    def test_invalid_train_flops(self):
        """Test validation catches invalid train_flops."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=5e14,  # Not in allowed set
        )
        with pytest.raises(ValueError, match="train_flops"):
            config.validate()


class TestMockAPI:
    """Test MockTrainingAPI functionality."""
    
    @pytest.fixture
    def api(self):
        """Create a fresh mock API instance."""
        return MockTrainingAPI(seed=42)
    
    def test_initial_state(self, api):
        """Test initial API state."""
        assert api.get_total_flops_used() == 0
        assert len(api.get_previous_runs()) == 0
        assert api.check_connection() is True
    
    def test_single_query(self, api):
        """Test single API query."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        loss, total_used = api.get_loss(config)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert total_used == 1e15  # Should track FLOPs
        assert len(api.get_previous_runs()) == 1
    
    def test_consistent_results(self, api):
        """Test that same config returns consistent results."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        loss1, _ = api.get_loss(config)
        loss2, _ = api.get_loss(config)
        
        assert loss1 == loss2  # Should be cached
    
    def test_flops_accumulation(self, api):
        """Test that FLOPs are accumulated across different queries."""
        config1 = ExperimentConfig(
            d_model=128,
            num_layers=2,
            num_heads=2,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e14,
        )
        
        config2 = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e14,
        )
        
        _, total1 = api.get_loss(config1)
        _, total2 = api.get_loss(config2)
        
        assert total1 == 1e14
        assert total2 == 2e14  # Should accumulate
    
    def test_no_double_counting(self, api):
        """Test that repeated queries don't double-count FLOPs."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        _, total1 = api.get_loss(config)
        _, total2 = api.get_loss(config)  # Same config
        
        assert total1 == total2  # Should not increase
    
    def test_model_size_effect(self, api):
        """Test that larger models tend to have different loss."""
        config_small = ExperimentConfig(
            d_model=128,
            num_layers=2,
            num_heads=2,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        config_large = ExperimentConfig(
            d_model=512,
            num_layers=8,
            num_heads=8,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        loss_small, _ = api.get_loss(config_small)
        loss_large, _ = api.get_loss(config_large)
        
        # Larger model should generally have lower loss (with same FLOPs)
        # But this depends on many factors, so we just check they're different
        assert loss_small != loss_large
    
    def test_learning_rate_effect(self, api):
        """Test that learning rate affects loss."""
        config_optimal = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,  # Center of range
            train_flops=1e15,
        )
        
        config_suboptimal = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.0001,  # Edge of range
            train_flops=1e15,
        )
        
        loss_optimal, _ = api.get_loss(config_optimal)
        loss_suboptimal, _ = api.get_loss(config_suboptimal)
        
        # Optimal LR should generally give better results
        assert loss_optimal != loss_suboptimal
    
    def test_reset(self, api):
        """Test API reset functionality."""
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        api.get_loss(config)
        assert api.get_total_flops_used() > 0
        assert len(api.get_previous_runs()) > 0
        
        api.reset()
        assert api.get_total_flops_used() == 0
        assert len(api.get_previous_runs()) == 0
    
    def test_reproducibility(self):
        """Test that same seed gives reproducible results."""
        api1 = MockTrainingAPI(seed=42)
        api2 = MockTrainingAPI(seed=42)
        
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        loss1, _ = api1.get_loss(config)
        loss2, _ = api2.get_loss(config)
        
        assert loss1 == loss2
    
    def test_different_seeds(self):
        """Test that different seeds give different results."""
        api1 = MockTrainingAPI(seed=42)
        api2 = MockTrainingAPI(seed=43)
        
        config = ExperimentConfig(
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e15,
        )
        
        loss1, _ = api1.get_loss(config)
        loss2, _ = api2.get_loss(config)
        
        # Due to noise, should be different
        assert loss1 != loss2


class TestCreateAPI:
    """Test API factory function."""
    
    def test_create_mock_api(self):
        """Test creating mock API."""
        api = create_api(use_mock=True, seed=42)
        assert isinstance(api, MockTrainingAPI)
    
    def test_create_mock_without_api_key(self):
        """Test that mock API doesn't need API key."""
        api = create_api(use_mock=True)
        assert isinstance(api, MockTrainingAPI)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

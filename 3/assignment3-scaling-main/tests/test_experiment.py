"""
Tests for scaling_experiment module (Problem 2).

Run with: uv run python -m pytest tests/test_experiment.py -v
"""

import pytest
import numpy as np

from cs336_scaling.scaling_experiment import (
    ScalingExperiment,
    ScalingLawParams,
    chinchilla_style_strategy,
    uniform_sampling_strategy,
)
from cs336_scaling.scaling_api import ExperimentConfig, MockTrainingAPI
from cs336_scaling.utils import compute_model_params


class TestScalingLawParams:
    """Test ScalingLawParams class."""
    
    def test_predict(self):
        """Test prediction method."""
        params = ScalingLawParams(a=0.5, b=0.6)
        
        c = 1e18
        n_pred = params.predict(c)
        
        expected = 0.5 * (1e18)**0.6
        assert n_pred == pytest.approx(expected)
    
    def test_predict_array(self):
        """Test prediction with array input."""
        params = ScalingLawParams(a=1.0, b=0.5)
        
        c_values = np.array([1e16, 1e17, 1e18])
        n_preds = params.predict(c_values)
        
        assert len(n_preds) == 3
        assert n_preds[1] / n_preds[0] == pytest.approx(np.sqrt(10), rel=0.01)


class TestScalingExperiment:
    """Test ScalingExperiment class."""
    
    @pytest.fixture
    def experiment(self):
        """Create a fresh experiment with mock API."""
        return ScalingExperiment(
            budget=1e17,  # Small budget for fast tests
            target_compute=1e18,
            use_mock=True,
            seed=42,
        )
    
    def test_initialization(self, experiment):
        """Test experiment initialization."""
        assert experiment.budget == 1e17
        assert experiment.target_compute == 1e18
        assert experiment.get_budget_remaining() == 1e17
        assert len(experiment.runs) == 0
    
    def test_query_success(self, experiment):
        """Test successful query."""
        config = ExperimentConfig(
            d_model=128,
            num_layers=2,
            num_heads=2,
            batch_size=128,
            learning_rate=0.001,
            train_flops=1e14,
        )
        
        loss = experiment.query(config)
        
        assert loss is not None
        assert isinstance(loss, float)
        assert len(experiment.runs) == 1
        assert experiment.get_budget_remaining() == 1e17 - 1e14
    
    def test_query_budget_exceeded(self, experiment):
        """Test query when budget is exceeded."""
        config = ExperimentConfig(
            d_model=512,
            num_layers=8,
            num_heads=8,
            batch_size=128,
            learning_rate=0.001,
            train_flops=2e17,  # Exceeds budget
        )
        
        loss = experiment.query(config)
        
        assert loss is None
        assert len(experiment.runs) == 0
    
    def test_get_runs_by_flops(self, experiment):
        """Test filtering runs by FLOPs."""
        # Add runs with different FLOPs
        config1 = ExperimentConfig(
            d_model=128, num_layers=2, num_heads=2,
            batch_size=128, learning_rate=0.001, train_flops=1e14,
        )
        config2 = ExperimentConfig(
            d_model=256, num_layers=4, num_heads=4,
            batch_size=128, learning_rate=0.001, train_flops=1e15,
        )
        
        experiment.query(config1)
        experiment.query(config2)
        experiment.query(config1)  # Duplicate FLOPs, different model
        
        runs_1e14 = experiment.get_runs_by_flops(1e14)
        runs_1e15 = experiment.get_runs_by_flops(1e15)
        
        assert len(runs_1e14) == 2
        assert len(runs_1e15) == 1
    
    def test_find_optimal_per_budget(self, experiment):
        """Test finding optimal configurations."""
        # Add runs with different FLOPs and model sizes
        configs = [
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e14),  # Small
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e14),  # Large
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e15),  # Small, more FLOPs
        ]
        
        for config in configs:
            experiment.query(config)
        
        budgets, opt_params = experiment.find_optimal_per_budget()
        
        assert len(budgets) == 2  # Two unique FLOPs values
        assert len(opt_params) == 2
        assert all(b in [1e14, 1e15] for b in budgets)
    
    def test_fit_scaling_law_insufficient_data(self, experiment):
        """Test fitting with insufficient data."""
        # Less than 3 runs
        config = ExperimentConfig(
            d_model=128, num_layers=2, num_heads=2,
            batch_size=128, learning_rate=0.001, train_flops=1e14,
        )
        experiment.query(config)
        
        result = experiment.fit_scaling_law()
        
        assert result is None
    
    def test_fit_scaling_law_success(self, experiment):
        """Test successful scaling law fitting."""
        # Add enough runs
        configs = [
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e14),
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e14),
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e15),
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e15),
        ]
        
        for config in configs:
            experiment.query(config)
        
        result = experiment.fit_scaling_law()
        
        assert result is not None
        assert result.a > 0
        assert result.b > 0
        assert experiment.scaling_law is not None
    
    def test_predict_optimal_config_no_scaling_law(self, experiment):
        """Test prediction without fitted scaling law."""
        prediction = experiment.predict_optimal_config()
        
        assert prediction is None
    
    def test_predict_optimal_config_success(self, experiment):
        """Test successful prediction."""
        # Add runs and fit scaling law
        configs = [
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e14),
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e14),
            ExperimentConfig(384, 6, 6, 128, 0.001, 1e14),
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e15),
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e15),
        ]
        
        for config in configs:
            experiment.query(config)
        
        experiment.fit_scaling_law()
        prediction = experiment.predict_optimal_config()
        
        assert prediction is not None
        assert 'd_model' in prediction
        assert 'num_layers' in prediction
        assert 'batch_size' in prediction
        assert 'learning_rate' in prediction
        assert prediction['batch_size'] in [128, 256]


class TestStrategies:
    """Test experiment strategies."""
    
    @pytest.fixture
    def experiment(self):
        """Create experiment with sufficient budget."""
        return ScalingExperiment(
            budget=1e18,
            target_compute=1e19,
            use_mock=True,
            seed=42,
        )
    
    def test_uniform_sampling_strategy(self, experiment):
        """Test uniform sampling strategy."""
        initial_budget = experiment.get_budget_remaining()
        
        uniform_sampling_strategy(
            experiment,
            compute_budgets=[1e14, 1e15, 1e16],
            models_per_budget=3,
        )
        
        # Should have used some budget
        assert experiment.get_budget_remaining() < initial_budget
        # Should have added runs
        assert len(experiment.runs) > 0
    
    def test_chinchilla_style_strategy(self, experiment):
        """Test Chinchilla-style strategy."""
        initial_budget = experiment.get_budget_remaining()
        
        chinchilla_style_strategy(
            experiment,
            num_isoflops_profiles=3,
            models_per_profile=3,
        )
        
        # Should have used some budget
        assert experiment.get_budget_remaining() < initial_budget
        # Should have added runs
        assert len(experiment.runs) > 0
    
    def test_chinchilla_stops_on_budget_exceeded(self, experiment):
        """Test that strategy stops when budget is exceeded."""
        experiment = ScalingExperiment(
            budget=5e14,  # Very small budget
            target_compute=1e19,
            use_mock=True,
            seed=42,
        )
        
        chinchilla_style_strategy(
            experiment,
            num_isoflops_profiles=4,
            models_per_profile=5,
        )
        
        # Should have stopped before exhausting all planned runs
        assert experiment.get_budget_remaining() >= 0
        assert experiment.get_budget_remaining() < 5e14


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_experiment_workflow(self):
        """Test complete experiment workflow."""
        experiment = ScalingExperiment(
            budget=1e17,
            target_compute=1e18,
            use_mock=True,
            seed=42,
        )
        
        # Run strategy
        chinchilla_style_strategy(experiment, num_isoflops_profiles=3, models_per_profile=4)
        
        # Fit scaling law
        scaling_law = experiment.fit_scaling_law()
        
        # Predict optimal config
        if scaling_law:
            prediction = experiment.predict_optimal_config()
            
            # Verify prediction structure
            assert prediction is not None
            assert 'predicted_params' in prediction
            assert 'actual_params' in prediction
            assert 'dataset_size' in prediction
            
            # Verify constraints
            assert prediction['batch_size'] in [128, 256]
            assert 64 <= prediction['d_model'] <= 1024
            assert 2 <= prediction['num_layers'] <= 24
            assert 2 <= prediction['num_heads'] <= 16
    
    def test_model_parameter_formula(self):
        """Test that model parameter formula is consistent."""
        d_model = 256
        num_layers = 4
        
        # Via utils function
        params_util = compute_model_params(num_layers, d_model)
        
        # Manual calculation
        params_manual = 12 * num_layers * d_model * d_model
        
        assert params_util == params_manual
        assert params_util == 12 * 4 * 256 * 256
    
    def test_budget_tracking_accuracy(self):
        """Test that budget tracking is accurate."""
        experiment = ScalingExperiment(
            budget=1e17,
            target_compute=1e18,
            use_mock=True,
            seed=42,
        )
        
        configs = [
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e14),
            ExperimentConfig(256, 4, 4, 128, 0.001, 1e14),
            ExperimentConfig(128, 2, 2, 128, 0.001, 1e15),
        ]
        
        total_expected = 0
        for config in configs:
            experiment.query(config)
            total_expected += config.train_flops
        
        # Query same config again (should not count)
        experiment.query(configs[0])
        
        assert experiment.api.get_total_flops_used() == total_expected
        assert experiment.get_budget_remaining() == 1e17 - total_expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

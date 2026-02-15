"""
Tests for chinchilla_isoflops module (Problem 1).

Run with: uv run python -m pytest tests/test_chinchilla.py -v
"""

import json
import tempfile
import pytest
import numpy as np

from cs336_scaling.chinchilla_isoflops import (
    load_isoflops_data,
    group_by_compute_budget,
    find_optimal_per_budget,
    power_law,
    fit_scaling_law,
    predict_optimal,
)
from cs336_scaling.utils import compute_dataset_size


# Sample test data mimicking the real data format
SAMPLE_DATA = [
    {"parameters": 50000000, "compute_budget": 6e18, "final_loss": 7.2},
    {"parameters": 100000000, "compute_budget": 6e18, "final_loss": 6.5},
    {"parameters": 200000000, "compute_budget": 6e18, "final_loss": 6.0},
    {"parameters": 80000000, "compute_budget": 1e19, "final_loss": 6.5},
    {"parameters": 150000000, "compute_budget": 1e19, "final_loss": 6.0},
    {"parameters": 300000000, "compute_budget": 1e19, "final_loss": 5.8},
]


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_isoflops_data_default_path(self):
        """Test loading from default path."""
        data = load_isoflops_data()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check expected keys
        assert "parameters" in data[0]
        assert "compute_budget" in data[0]
        assert "final_loss" in data[0]
    
    def test_load_isoflops_data_custom_path(self):
        """Test loading from custom file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(SAMPLE_DATA, f)
            f.flush()
            data = load_isoflops_data(f.name)
            assert len(data) == len(SAMPLE_DATA)


class TestGrouping:
    """Test data grouping functions."""
    
    def test_group_by_compute_budget(self):
        """Test grouping runs by compute budget."""
        grouped = group_by_compute_budget(SAMPLE_DATA)
        
        assert len(grouped) == 2  # Two unique budgets
        assert 6e18 in grouped
        assert 1e19 in grouped
        assert len(grouped[6e18]) == 3
        assert len(grouped[1e19]) == 3


class TestOptimalFinding:
    """Test finding optimal configurations."""
    
    def test_find_optimal_per_budget(self):
        """Test finding optimal model size per budget."""
        grouped = group_by_compute_budget(SAMPLE_DATA)
        budgets, opt_params, opt_losses = find_optimal_per_budget(grouped)
        
        assert len(budgets) == 2
        assert len(opt_params) == 2
        assert len(opt_losses) == 2
        
        # Check optimal values
        # For 6e18, min loss is 6.0 with 200M params
        idx_6e18 = np.where(budgets == 6e18)[0][0]
        assert opt_params[idx_6e18] == 200000000
        assert opt_losses[idx_6e18] == 6.0
        
        # For 1e19, min loss is 5.8 with 300M params
        idx_1e19 = np.where(budgets == 1e19)[0][0]
        assert opt_params[idx_1e19] == 300000000
        assert opt_losses[idx_1e19] == 5.8


class TestPowerLaw:
    """Test power law functions."""
    
    def test_power_law_basic(self):
        """Test basic power law computation."""
        x = np.array([1, 2, 4])
        a, b = 2.0, 0.5
        y = power_law(x, a, b)
        
        expected = np.array([2.0, 2.0 * np.sqrt(2), 4.0])
        np.testing.assert_allclose(y, expected)
    
    def test_power_law_scaling(self):
        """Test power law scaling behavior."""
        # If x doubles, y should increase by 2^b
        x = 100
        a, b = 1.0, 0.5
        
        y1 = power_law(np.array([x]), a, b)[0]
        y2 = power_law(np.array([2*x]), a, b)[0]
        
        assert y2 / y1 == pytest.approx(2**0.5, rel=1e-10)


class TestScalingLawFitting:
    """Test scaling law fitting."""
    
    def test_fit_scaling_law_perfect_power_law(self):
        """Test fitting to perfect power law data."""
        # Generate data from known power law
        a_true, b_true = 0.5, 0.6
        compute_budgets = np.array([1e15, 1e16, 1e17, 1e18])
        optimal_params = power_law(compute_budgets, a_true, b_true)
        
        (a_fit, b_fit), _ = fit_scaling_law(compute_budgets, optimal_params)
        
        # Should recover approximately the true values
        assert a_fit == pytest.approx(a_true, rel=0.01)
        assert b_fit == pytest.approx(b_true, rel=0.01)
    
    def test_fit_scaling_law_noisy_data(self):
        """Test fitting to noisy data."""
        a_true, b_true = 0.5, 0.6
        compute_budgets = np.logspace(15, 18, 10)
        optimal_params = power_law(compute_budgets, a_true, b_true)
        
        # Add small noise
        noise = np.random.randn(len(optimal_params)) * 0.01 * optimal_params
        optimal_params_noisy = optimal_params + noise
        
        (a_fit, b_fit), _ = fit_scaling_law(compute_budgets, optimal_params_noisy)
        
        # Should still be close
        assert 0.3 < b_fit < 0.9  # Reasonable range for exponent
        assert a_fit > 0
    
    def test_predict_optimal(self):
        """Test prediction using fitted scaling law."""
        a, b = 0.5, 0.6
        
        # Predict for new budget
        c_new = 1e20
        n_pred = predict_optimal(c_new, a, b)
        
        expected = 0.5 * (1e20)**0.6
        assert n_pred == pytest.approx(expected, rel=1e-10)


class TestDatasetSize:
    """Test dataset size computation."""
    
    def test_compute_dataset_size(self):
        """Test dataset size formula D = C / (6N)."""
        C = 6e18
        N = 1e9
        D = compute_dataset_size(C, N)
        
        assert D == C / (6 * N)
    
    def test_compute_dataset_size_consistency(self):
        """Test that C = 6 * N * D holds."""
        C = 1e19
        N = 2e9
        D = compute_dataset_size(C, N)
        
        # Should satisfy C ≈ 6 * N * D
        assert 6 * N * D == pytest.approx(C, rel=1e-10)


class TestEndToEnd:
    """End-to-end tests with sample data."""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        data = SAMPLE_DATA
        
        # Load and group
        grouped = group_by_compute_budget(data)
        
        # Find optimal
        budgets, opt_params, _ = find_optimal_per_budget(grouped)
        
        # Fit scaling law
        (a, b), _ = fit_scaling_law(budgets, opt_params)
        
        # Predict
        c_target = 1e20
        n_pred = predict_optimal(c_target, a, b)
        
        # Sanity checks
        assert n_pred > 0
        assert b > 0  # Should be positive scaling
        assert b < 1  # Sub-linear scaling expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

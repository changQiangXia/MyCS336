"""
Problem 2: Active scaling law construction with training API.

This module implements strategies for:
1. Designing experiments within a FLOPs budget
2. Fitting scaling laws from experimental data
3. Predicting optimal model configuration for target FLOPs
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.pyplot as plt

from .scaling_api import ExperimentConfig, TrainingAPI, MockTrainingAPI, create_api
from .utils import compute_model_params, compute_dataset_size, params_to_string, flops_to_string

logger = logging.getLogger(__name__)


@dataclass
class ScalingLawParams:
    """Parameters for a fitted scaling law."""
    a: float  # scaling coefficient
    b: float  # power exponent
    
    def predict(self, compute_budget: float) -> float:
        """Predict optimal value for given compute budget."""
        return self.a * (compute_budget ** self.b)


class ScalingExperiment:
    """
    Manages scaling law experiments within a FLOPs budget.
    
    Budget management:
    - Tracks FLOPs used across all queries
    - Ensures budget is not exceeded
    - Reuses cached results for identical queries
    """
    
    def __init__(
        self,
        budget: float = 2e18,
        target_compute: float = 1e19,
        api=None,
        use_mock: bool = False,
        seed: int = 42,
    ):
        """
        Initialize experiment manager.
        
        Args:
            budget: FLOPs budget for fitting scaling law (default 2e18).
            target_compute: Target compute budget to predict for (default 1e19).
            api: Pre-configured API instance. If None, creates new one.
            use_mock: If True and api is None, use MockTrainingAPI.
            seed: Random seed for mock API.
        """
        self.budget = budget
        self.target_compute = target_compute
        
        if api is None:
            self.api = create_api(use_mock=use_mock, seed=seed)
        else:
            self.api = api
        
        # Storage for experimental results
        self.runs: List[Dict] = []
        self.scaling_law: Optional[ScalingLawParams] = None
        
        logger.info(f"Initialized experiment with budget {flops_to_string(budget)}")
        logger.info(f"Target compute: {flops_to_string(target_compute)}")
    
    def get_budget_remaining(self) -> float:
        """Get remaining FLOPs budget."""
        return self.budget - self.api.get_total_flops_used()
    
    def query(self, config: ExperimentConfig) -> Optional[float]:
        """
        Query training loss for a configuration.
        
        Args:
            config: Experiment configuration.
        
        Returns:
            Training loss, or None if budget exceeded.
        """
        # Check budget
        if config.train_flops > self.get_budget_remaining():
            logger.warning(f"Budget exceeded! Need {config.train_flops}, have {self.get_budget_remaining()}")
            return None
        
        # Query API
        loss, total_used = self.api.get_loss(config)
        
        # Store result
        result = {
            'config': config,
            'loss': loss,
            'params': compute_model_params(config.num_layers, config.d_model),
            'tokens': compute_dataset_size(config.train_flops, 
                                          compute_model_params(config.num_layers, config.d_model)),
        }
        self.runs.append(result)
        
        logger.info(f"Queried: d_model={config.d_model}, layers={config.num_layers}, "
                   f"FLOPs={flops_to_string(config.train_flops)}, Loss={loss:.4f}, "
                   f"Used: {flops_to_string(total_used)}/{flops_to_string(self.budget)}")
        
        return loss
    
    def get_runs_by_flops(self, train_flops: float) -> List[Dict]:
        """Get all runs with a specific training FLOPs value."""
        return [r for r in self.runs if r['config'].train_flops == train_flops]
    
    def find_optimal_per_budget(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find optimal model size for each compute budget we've tested.
        
        Returns:
            Tuple of (compute_budgets, optimal_params) arrays.
        """
        # Group by compute budget
        budgets = {}
        for run in self.runs:
            c = run['config'].train_flops
            if c not in budgets:
                budgets[c] = []
            budgets[c].append(run)
        
        # Find best for each budget
        compute_budgets = []
        optimal_params = []
        
        for budget in sorted(budgets.keys()):
            runs = budgets[budget]
            best_run = min(runs, key=lambda x: x['loss'])
            
            compute_budgets.append(budget)
            optimal_params.append(best_run['params'])
            
            logger.info(f"Budget {flops_to_string(budget)}: "
                       f"Optimal N={params_to_string(best_run['params'])}, "
                       f"Loss={best_run['loss']:.4f}")
        
        return np.array(compute_budgets), np.array(optimal_params)
    
    def fit_scaling_law(self) -> Optional[ScalingLawParams]:
        """
        Fit scaling law N_opt = a * C^b from experimental data.
        
        Returns:
            ScalingLawParams if successful, None if insufficient data.
        """
        if len(self.runs) < 3:
            logger.warning("Need at least 3 runs to fit scaling law")
            return None
        
        compute_budgets, optimal_params = self.find_optimal_per_budget()
        
        if len(compute_budgets) < 2:
            logger.warning("Need at least 2 different compute budgets")
            return None
        
        # Fit power law in log space
        log_c = np.log(compute_budgets)
        log_n = np.log(optimal_params)
        
        coeffs = np.polyfit(log_c, log_n, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        
        self.scaling_law = ScalingLawParams(a=a, b=b)
        
        logger.info(f"Fitted scaling law: N_opt = {a:.3e} * C^{b:.4f}")
        
        return self.scaling_law
    
    def predict_optimal_config(self, compute_budget: Optional[float] = None) -> Optional[Dict]:
        """
        Predict optimal model configuration for target compute budget.
        
        Args:
            compute_budget: Target FLOPs. Uses self.target_compute if None.
        
        Returns:
            Dictionary with predicted configuration, or None if no scaling law.
        """
        if self.scaling_law is None:
            self.fit_scaling_law()
        
        if self.scaling_law is None:
            return None
        
        if compute_budget is None:
            compute_budget = self.target_compute
        
        # Predict optimal model size
        n_opt = self.scaling_law.predict(compute_budget)
        
        # Estimate hyperparameters
        # Strategy: Use square-ish architecture (depth ~ sqrt(N/12)/d_model)
        # This is a heuristic; in practice, you'd want to explore this space
        
        # Try to find a reasonable d_model and num_layers combination
        # N = 12 * num_layers * d_model^2
        # For a "square" model, we want num_layers proportional to d_model
        
        # Heuristic: aim for d_model between 128 and 512 for target budget
        best_config = None
        best_score = float('inf')
        
        for d_model in [128, 256, 512, 768, 1024]:
            if n_opt < 12 * 2 * d_model * d_model:
                continue  # Need at least 2 layers
            
            num_layers = int(n_opt / (12 * d_model * d_model))
            num_layers = max(2, min(24, num_layers))  # Clip to valid range
            
            # Recalculate actual params
            actual_n = compute_model_params(num_layers, d_model)
            
            # Compute dataset size
            d_opt = compute_dataset_size(compute_budget, actual_n)
            
            # Score based on how close to predicted optimal
            score = abs(actual_n - n_opt) / n_opt
            
            if score < best_score:
                best_score = score
                best_config = {
                    'compute_budget': compute_budget,
                    'predicted_params': n_opt,
                    'd_model': d_model,
                    'num_layers': num_layers,
                    'num_heads': min(16, max(2, d_model // 64)),  # Reasonable default
                    'batch_size': 128,  # Requirement: must be 128 or 256
                    'learning_rate': 0.001,  # Start with standard value
                    'actual_params': actual_n,
                    'dataset_size': d_opt,
                }
        
        return best_config
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot experimental results and fitted scaling law."""
        if not self.runs:
            logger.warning("No runs to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: IsoFLOPs curves
        ax1 = axes[0]
        budgets = sorted(set(r['config'].train_flops for r in self.runs))
        
        for budget in budgets:
            runs = self.get_runs_by_flops(budget)
            params = [r['params'] for r in runs]
            losses = [r['loss'] for r in runs]
            
            # Sort by params
            sorted_data = sorted(zip(params, losses))
            params, losses = zip(*sorted_data)
            
            ax1.plot(params, losses, 'o-', label=f"C={flops_to_string(budget)}")
        
        ax1.set_xlabel('Model Size (parameters)')
        ax1.set_ylabel('Training Loss')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.set_title('IsoFLOPs Curves')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scaling law
        ax2 = axes[1]
        
        if self.scaling_law:
            compute_budgets, optimal_params = self.find_optimal_per_budget()
            ax2.scatter(compute_budgets, optimal_params, c='blue', s=100, label='Data', zorder=3)
            
            # Plot fitted curve
            c_range = np.logspace(np.log10(compute_budgets.min() * 0.8), 
                                  np.log10(self.target_compute * 1.2), 100)
            n_pred = self.scaling_law.predict(c_range)
            ax2.plot(c_range, n_pred, 'r--', 
                    label=f"Fit: $N = {self.scaling_law.a:.2e} \\cdot C^{{{self.scaling_law.b:.3f}}}$",
                    linewidth=2)
            
            # Mark target
            n_target = self.scaling_law.predict(self.target_compute)
            ax2.scatter([self.target_compute], [n_target], c='red', s=200, marker='*',
                       label=f'Target: {params_to_string(n_target)}', zorder=4, edgecolors='black')
            
            ax2.set_xlabel('Compute Budget (FLOPs)')
            ax2.set_ylabel('Optimal Model Size (parameters)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.set_title('Scaling Law')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No scaling law fitted yet', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()


def uniform_sampling_strategy(
    experiment: ScalingExperiment,
    compute_budgets: Optional[List[float]] = None,
    models_per_budget: int = 5,
) -> None:
    """
    Uniform sampling strategy: sample models at different sizes for each compute budget.
    
    Args:
        experiment: ScalingExperiment instance.
        compute_budgets: List of compute budgets to test. Auto-selected if None.
        models_per_budget: Number of model sizes to try per compute budget.
    """
    if compute_budgets is None:
        # Auto-select compute budgets spanning the range
        compute_budgets = [1e14, 1e15, 1e16, 1e17]
    
    # Filter to affordable budgets
    affordable_budgets = [c for c in compute_budgets if c <= experiment.budget / len(compute_budgets)]
    
    logger.info(f"Running uniform sampling with budgets: {[flops_to_string(c) for c in affordable_budgets]}")
    
    for flops in affordable_budgets:
        # Sample different model sizes for this compute budget
        d_models = [128, 256, 384, 512, 768][:models_per_budget]
        
        for d_model in d_models:
            if experiment.get_budget_remaining() < flops:
                logger.info("Budget exhausted, stopping")
                return
            
            # Calculate layers to get reasonable model sizes
            # Try different depths
            for num_layers in [2, 4, 8][:2]:  # Limit to stay within budget
                config = ExperimentConfig(
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=min(16, max(2, d_model // 64)),
                    batch_size=128,
                    learning_rate=0.001,
                    train_flops=int(flops),
                )
                
                # Check if we can afford it
                params = compute_model_params(num_layers, d_model)
                if params < 1e6:  # Skip very small models
                    continue
                
                experiment.query(config)


def _round_to_allowed_flops(flops: float) -> int:
    """
    Round FLOPs to the nearest allowed value from VALID_RANGES.
    
    Args:
        flops: Desired FLOPs value.
    
    Returns:
        Nearest allowed FLOPs value.
    """
    from .scaling_api import VALID_RANGES
    allowed = sorted(VALID_RANGES['train_flops'])
    
    # Find the closest allowed value
    closest = min(allowed, key=lambda x: abs(x - flops))
    return int(closest)


def chinchilla_style_strategy(
    experiment: ScalingExperiment,
    num_isoflops_profiles: int = 4,
    models_per_profile: int = 5,
) -> None:
    """
    Chinchilla-style IsoFLOPs strategy.
    
    For each compute budget, train models of varying sizes and find the minimum loss.
    
    Args:
        experiment: ScalingExperiment instance.
        num_isoflops_profiles: Number of different compute budgets to test.
        models_per_profile: Number of model sizes per compute budget.
    """
    # Select compute budgets spanning the budget range
    # Geometric progression: distribute evenly in log space
    log_min = np.log10(1e14)
    log_max = np.log10(experiment.budget / models_per_profile)
    log_budgets = np.linspace(log_min, log_max, num_isoflops_profiles)
    compute_budgets = [_round_to_allowed_flops(10**x) for x in log_budgets]
    
    logger.info(f"Chinchilla strategy: budgets = {[flops_to_string(c) for c in compute_budgets]}")
    
    for flops in compute_budgets:
        flops = int(flops)
        
        logger.info(f"\n--- IsoFLOPs profile for C = {flops_to_string(flops)} ---")
        
        # Calculate model sizes that give reasonable token counts
        # D = C / (6N), so for D to be reasonable (not too small), N shouldn't be too large
        
        # Target model sizes spanning a range
        target_params = np.logspace(6, 9, models_per_profile)  # 1M to 1B
        
        for target_n in target_params:
            if experiment.get_budget_remaining() < flops:
                logger.info("Budget exhausted")
                return
            
            # Find d_model and num_layers that give approximately target_n params
            # N = 12 * num_layers * d_model^2
            
            best_d_model = None
            best_layers = None
            best_error = float('inf')
            
            for d_model in [128, 192, 256, 384, 512, 768, 1024]:
                for num_layers in [2, 4, 6, 8, 12, 16, 24]:
                    n = compute_model_params(num_layers, d_model)
                    error = abs(n - target_n) / target_n
                    if error < best_error:
                        best_error = error
                        best_d_model = d_model
                        best_layers = num_layers
            
            if best_d_model is None:
                continue
            
            config = ExperimentConfig(
                d_model=best_d_model,
                num_layers=best_layers,
                num_heads=min(16, max(2, best_d_model // 64)),
                batch_size=128,
                learning_rate=0.001,
                train_flops=flops,
            )
            
            experiment.query(config)


def run_full_experiment(
    use_mock: bool = True,
    budget: float = 2e18,
    target_compute: float = 1e19,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run a complete scaling law experiment.
    
    Args:
        use_mock: Use mock API if True, otherwise try real API.
        budget: FLOPs budget for experiments.
        target_compute: Target compute budget to predict for.
        output_dir: Directory to save results and plots.
    
    Returns:
        Dictionary with experiment results.
    """
    # Create experiment
    experiment = ScalingExperiment(
        budget=budget,
        target_compute=target_compute,
        use_mock=use_mock,
    )
    
    # Run strategy
    logger.info("\n" + "="*60)
    logger.info("Running Chinchilla-style IsoFLOPs strategy")
    logger.info("="*60)
    
    chinchilla_style_strategy(
        experiment,
        num_isoflops_profiles=4,
        models_per_profile=5,
    )
    
    # Fit scaling law
    logger.info("\n" + "="*60)
    logger.info("Fitting scaling law")
    logger.info("="*60)
    
    experiment.fit_scaling_law()
    
    # Predict optimal configuration
    logger.info("\n" + "="*60)
    logger.info("Predicting optimal configuration")
    logger.info("="*60)
    
    prediction = experiment.predict_optimal_config()
    
    if prediction:
        logger.info(f"\nPredicted optimal configuration for {flops_to_string(target_compute)}:")
        logger.info(f"  Model size: {params_to_string(prediction['predicted_params'])}")
        logger.info(f"  d_model: {prediction['d_model']}")
        logger.info(f"  num_layers: {prediction['num_layers']}")
        logger.info(f"  num_heads: {prediction['num_heads']}")
        logger.info(f"  batch_size: {prediction['batch_size']} (required: 128 or 256)")
        logger.info(f"  learning_rate: {prediction['learning_rate']}")
        logger.info(f"  Dataset size: {prediction['dataset_size']:.2e} tokens")
    
    # Plot results
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/experiment_results.png"
    else:
        plot_path = None
    
    experiment.plot_results(save_path=plot_path)
    
    return {
        'experiment': experiment,
        'scaling_law': experiment.scaling_law,
        'prediction': prediction,
        'runs': experiment.runs,
        'total_flops_used': experiment.api.get_total_flops_used(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Run with mock API for testing
    results = run_full_experiment(
        use_mock=True,
        budget=2e18,
        target_compute=1e19,
        output_dir="results",
    )

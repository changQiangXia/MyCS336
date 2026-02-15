"""
Problem 1: IsoFLOPs scaling laws using provided data.

This module implements the IsoFLOPs approach from Hoffmann et al. 2022 (Chinchilla)
to fit scaling laws and predict optimal model/dataset sizes for given compute budgets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .utils import compute_dataset_size

logger = logging.getLogger(__name__)


def load_isoflops_data(data_path: Optional[str] = None) -> List[Dict]:
    """
    Load the IsoFLOPs training data from JSON file.
    
    Args:
        data_path: Path to the JSON file. If None, uses default path.
    
    Returns:
        List of dictionaries containing training run data.
    """
    if data_path is None:
        # Find data file relative to this module
        current_dir = Path(__file__).parent
        data_path = current_dir.parent / "data" / "isoflops_curves.json"
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data


def group_by_compute_budget(data: List[Dict]) -> Dict[float, List[Dict]]:
    """
    Group training runs by their compute budget.
    
    Args:
        data: List of training run dictionaries.
    
    Returns:
        Dictionary mapping compute_budget -> list of runs with that budget.
    """
    grouped = {}
    for run in data:
        c = run['compute_budget']
        if c not in grouped:
            grouped[c] = []
        grouped[c].append(run)
    return grouped


def find_optimal_per_budget(grouped_data: Dict[float, List[Dict]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the optimal (lowest loss) model size for each compute budget.
    
    Args:
        grouped_data: Dictionary mapping compute_budget -> list of runs.
    
    Returns:
        Tuple of (compute_budgets, optimal_params, optimal_losses) as numpy arrays.
    """
    compute_budgets = []
    optimal_params = []
    optimal_losses = []
    
    for budget, runs in sorted(grouped_data.items()):
        # Find run with minimum loss
        best_run = min(runs, key=lambda x: x['final_loss'])
        
        compute_budgets.append(budget)
        optimal_params.append(best_run['parameters'])
        optimal_losses.append(best_run['final_loss'])
        
        logger.info(f"Budget {budget:.0e}: Optimal N={best_run['parameters']:.2e}, Loss={best_run['final_loss']:.4f}")
    
    return np.array(compute_budgets), np.array(optimal_params), np.array(optimal_losses)


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power law function: y = a * x^b
    
    Args:
        x: Input array.
        a: Scaling coefficient.
        b: Power exponent.
    
    Returns:
        y = a * x^b
    """
    return a * np.power(x, b)


def fit_scaling_law(
    compute_budgets: np.ndarray, 
    optimal_values: np.ndarray
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Fit a power law to the optimal values.
    
    Fits: optimal = a * compute_budget^b
    
    Args:
        compute_budgets: Array of compute budgets (C).
        optimal_values: Array of optimal values (N_opt or D_opt).
    
    Returns:
        Tuple of ((a, b), covariance) where y = a * x^b.
    """
    # Use log-space for better numerical stability
    # log(y) = log(a) + b * log(x)
    log_x = np.log(compute_budgets)
    log_y = np.log(optimal_values)
    
    # Linear fit in log space
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    
    # Also do non-linear fit for refinement
    try:
        popt, pcov = curve_fit(power_law, compute_budgets, optimal_values, p0=[a, b])
        return tuple(popt), pcov
    except RuntimeError:
        # Fall back to log-space fit
        return (a, b), np.zeros((2, 2))


def predict_optimal(
    compute_budget: float, 
    a: float, 
    b: float
) -> float:
    """
    Predict optimal value for a given compute budget using fitted power law.
    
    Args:
        compute_budget: Target compute budget (C).
        a: Scaling coefficient.
        b: Power exponent.
    
    Returns:
        Predicted optimal value.
    """
    return power_law(np.array([compute_budget]), a, b)[0]


def plot_scaling_law(
    compute_budgets: np.ndarray,
    optimal_values: np.ndarray,
    a: float,
    b: float,
    target_budgets: Optional[List[float]] = None,
    ylabel: str = "Optimal Model Size (N)",
    title: str = "Scaling Law: Model Size vs Compute",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the scaling law with data points and extrapolation.
    
    Args:
        compute_budgets: Compute budgets from data.
        optimal_values: Optimal values from data.
        a: Fitted scaling coefficient.
        b: Fitted power exponent.
        target_budgets: Additional compute budgets to show predictions for.
        ylabel: Y-axis label.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(compute_budgets, optimal_values, c='blue', s=100, label='Data', zorder=3)
    
    # Generate smooth curve for plotting
    min_c = min(compute_budgets.min(), *(target_budgets or [compute_budgets.min()]))
    max_c = max(compute_budgets.max(), *(target_budgets or [compute_budgets.max()]))
    c_range = np.logspace(np.log10(min_c * 0.8), np.log10(max_c * 1.2), 100)
    y_pred = power_law(c_range, a, b)
    
    # Plot fitted curve
    plt.plot(c_range, y_pred, 'r--', label=f'Fit: $y = {a:.2e} \\cdot C^{{{b:.3f}}}$', linewidth=2)
    
    # Plot target predictions
    if target_budgets:
        target_values = [predict_optimal(c, a, b) for c in target_budgets]
        plt.scatter(target_budgets, target_values, c='red', s=150, marker='*', 
                   label='Predictions', zorder=4, edgecolors='black', linewidths=1)
        
        # Annotate predictions
        for c, v in zip(target_budgets, target_values):
            plt.annotate(f'C={c:.0e}\nN={v:.2e}', 
                        xy=(c, v), xytext=(10, 10), 
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute Budget (FLOPs)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def run_chinchilla_analysis(
    data_path: Optional[str] = None,
    target_budgets: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run the complete Chinchilla IsoFLOPs analysis.
    
    Args:
        data_path: Path to data file. Uses default if None.
        target_budgets: List of compute budgets to predict for.
        output_dir: Directory to save plots. If None, plots are shown but not saved.
    
    Returns:
        Dictionary containing analysis results.
    """
    if target_budgets is None:
        target_budgets = [1e23, 1e24]
    
    # Load data
    data = load_isoflops_data(data_path)
    logger.info(f"Loaded {len(data)} training runs")
    
    # Group by compute budget
    grouped = group_by_compute_budget(data)
    logger.info(f"Found {len(grouped)} unique compute budgets")
    
    # Find optimal model sizes
    budgets, opt_params, opt_losses = find_optimal_per_budget(grouped)
    
    # Compute optimal dataset sizes
    opt_datasets = np.array([compute_dataset_size(c, n) for c, n in zip(budgets, opt_params)])
    
    # Fit scaling law for model size
    (a_n, b_n), _ = fit_scaling_law(budgets, opt_params)
    logger.info(f"Model size scaling law: N_opt = {a_n:.4e} * C^{b_n:.4f}")
    
    # Fit scaling law for dataset size
    (a_d, b_d), _ = fit_scaling_law(budgets, opt_datasets)
    logger.info(f"Dataset size scaling law: D_opt = {a_d:.4e} * C^{b_d:.4f}")
    
    # Predict for target budgets
    results = {
        'model_scaling': {'a': a_n, 'b': b_n},
        'dataset_scaling': {'a': a_d, 'b': b_d},
        'predictions': {},
    }
    
    for budget in target_budgets:
        n_pred = predict_optimal(budget, a_n, b_n)
        d_pred = predict_optimal(budget, a_d, b_d)
        results['predictions'][budget] = {
            'model_size': n_pred,
            'dataset_size': d_pred,
        }
        logger.info(f"\nPredictions for C={budget:.0e}:")
        logger.info(f"  Optimal model size: {n_pred:.2e} parameters")
        logger.info(f"  Optimal dataset size: {d_pred:.2e} tokens")
    
    # Plot model size scaling law
    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_plot_path = str(Path(output_dir) / "model_size_scaling.png")
        dataset_plot_path = str(Path(output_dir) / "dataset_size_scaling.png")
    else:
        model_plot_path = None
        dataset_plot_path = None
    
    plot_scaling_law(
        budgets, opt_params, a_n, b_n,
        target_budgets=target_budgets,
        ylabel="Optimal Model Size (parameters)",
        title="Scaling Law: Optimal Model Size vs Compute Budget",
        save_path=model_plot_path,
    )
    
    plot_scaling_law(
        budgets, opt_datasets, a_d, b_d,
        target_budgets=target_budgets,
        ylabel="Optimal Dataset Size (tokens)",
        title="Scaling Law: Optimal Dataset Size vs Compute Budget",
        save_path=dataset_plot_path,
    )
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Run analysis with default targets
    results = run_chinchilla_analysis(
        target_budgets=[1e23, 1e24],
        output_dir="results"
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nModel size scaling: N = {results['model_scaling']['a']:.3e} * C^{results['model_scaling']['b']:.4f}")
    print(f"Dataset size scaling: D = {results['dataset_scaling']['a']:.3e} * C^{results['dataset_scaling']['b']:.4f}")
    print("\nPredictions:")
    for budget, pred in results['predictions'].items():
        print(f"  C = {budget:.0e}:")
        print(f"    N_opt = {pred['model_size']:.3e} parameters")
        print(f"    D_opt = {pred['dataset_size']:.3e} tokens")

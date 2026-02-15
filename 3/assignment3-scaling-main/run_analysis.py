#!/usr/bin/env python3
"""
Main script to run scaling law analysis.

Usage:
    uv run python run_analysis.py --problem 1
    uv run python run_analysis.py --problem 2 --mock
    uv run python run_analysis.py --test
"""

import argparse
import logging
import sys


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def run_problem_1():
    """Run Problem 1: Chinchilla IsoFLOPs analysis."""
    print("\n" + "="*70)
    print("PROBLEM 1: Chinchilla IsoFLOPs Analysis")
    print("="*70 + "\n")
    
    from cs336_scaling.chinchilla_isoflops import run_chinchilla_analysis
    
    results = run_chinchilla_analysis(
        target_budgets=[1e23, 1e24],
        output_dir="results"
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nModel size scaling law: N = {results['model_scaling']['a']:.3e} * C^{results['model_scaling']['b']:.4f}")
    print(f"Dataset size scaling law: D = {results['dataset_scaling']['a']:.3e} * C^{results['dataset_scaling']['b']:.4f}")
    print("\nPredictions:")
    for budget, pred in results['predictions'].items():
        print(f"\n  C = {budget:.0e} FLOPs:")
        print(f"    Optimal model size (N): {pred['model_size']:.3e} parameters ({pred['model_size']/1e9:.2f}B)")
        print(f"    Optimal dataset size (D): {pred['dataset_size']:.3e} tokens")
    
    print("\n" + "="*70)
    print("Plots saved to results/ directory")
    print("="*70 + "\n")


def run_problem_2(mock=True):
    """Run Problem 2: Active scaling law construction."""
    print("\n" + "="*70)
    print("PROBLEM 2: Active Scaling Law Construction")
    if mock:
        print("(Using MOCK API - no VPN required)")
    else:
        print("(Using REAL API - requires Stanford VPN)")
    print("="*70 + "\n")
    
    from cs336_scaling.scaling_experiment import run_full_experiment
    
    results = run_full_experiment(
        use_mock=mock,
        budget=2e18,
        target_compute=1e19,
        output_dir="results",
    )
    
    if results['scaling_law']:
        print("\n" + "="*70)
        print("SCALING LAW FIT")
        print("="*70)
        print(f"N_opt = {results['scaling_law'].a:.3e} * C^{results['scaling_law'].b:.4f}")
    
    if results['prediction']:
        pred = results['prediction']
        print("\n" + "="*70)
        print("PREDICTED OPTIMAL CONFIGURATION")
        print("="*70)
        print(f"\nFor C = 1e19 FLOPs:")
        print(f"  Model size: {pred['predicted_params']:.3e} parameters ({pred['predicted_params']/1e9:.2f}B)")
        print(f"  d_model: {pred['d_model']}")
        print(f"  num_layers: {pred['num_layers']}")
        print(f"  num_heads: {pred['num_heads']}")
        print(f"  batch_size: {pred['batch_size']} (must be 128 or 256)")
        print(f"  learning_rate: {pred['learning_rate']}")
        print(f"  Dataset size: {pred['dataset_size']:.2e} tokens")
    
    print("\n" + "="*70)
    print(f"Total FLOPs used: {results['total_flops_used']:.2e} / 2e18")
    print(f"Total runs: {len(results['runs'])}")
    print("="*70 + "\n")


def run_tests():
    """Run all tests."""
    import subprocess
    
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
    )
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run CS336 Assignment 3: Scaling Laws"
    )
    parser.add_argument(
        "--problem", "-p",
        type=int,
        choices=[1, 2],
        help="Which problem to run (1 or 2)"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock API for Problem 2 (default)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real API for Problem 2 (requires VPN)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run tests instead of analysis"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.test:
        return run_tests()
    
    if args.problem == 1:
        run_problem_1()
    elif args.problem == 2:
        use_mock = not args.real  # Default to mock unless --real is specified
        run_problem_2(mock=use_mock)
    else:
        print("Please specify --problem 1 or --problem 2")
        print("Use --test to run tests")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

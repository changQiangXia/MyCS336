# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [1.1.0] - 2025-02-15

### Added

- Complete implementation of Problem 1: Chinchilla IsoFLOPs analysis
  - `chinchilla_isoflops.py`: Load data, find optimal per budget, fit power law scaling
  - Support for plotting scaling laws with extrapolation to 1e23 and 1e24 FLOPs
  - Predictions for optimal model size and dataset size
- Complete implementation of Problem 2: Active scaling law construction
  - `scaling_api.py`: API client with Mock API for testing without VPN
  - `scaling_experiment.py`: Experiment management, budget tracking, strategy implementation
  - `chinchilla_style_strategy()`: IsoFLOPs profile sampling within budget
  - `uniform_sampling_strategy()`: Alternative sampling approach
- Utility functions in `utils.py`: Model parameter calculation, dataset size computation
- Comprehensive test suite (47 tests)
  - `test_chinchilla.py`: Tests for IsoFLOPs analysis
  - `test_api.py`: Tests for API client and Mock API
  - `test_experiment.py`: Tests for experiment strategies and workflows
- `run_analysis.py`: Main entry point for running analyses

### Changed

- Added scipy, matplotlib, numpy to dependencies for curve fitting and visualization

### Fixed

- Fixed `chinchilla_style_strategy` to round train_flops to allowed discrete values

## [1.0.0] - 2024-04-30

### Added

- handout: added suggestion to plan out your scaling runs beforehand, since the
  training API will refuse further requests past the 2e18 budget.
- handout: document end point for previous runs
- api: add endpoint for previous runs
- api: use xgboost regressor instead of sklearn tree regressor

### Changed

### Fixed

## [0.0.0] - 2024-05-02

Initial release.

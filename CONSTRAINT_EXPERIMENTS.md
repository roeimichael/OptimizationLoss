# Constraint Configuration Experiments

## Overview

This system tests the top 5 hyperparameter configurations across multiple constraint settings to understand how different constraint configurations affect model performance.

## Constraint Configuration Format

Constraints are defined as `(local_percent, global_percent)`:
- `local_percent`: Local dropout constraint percentage (per course)
- `global_percent`: Global dropout constraint percentage (overall dataset)

Example: `(0.5, 0.3)` means:
- Local: 50% dropout constraint per course
- Global: 30% dropout constraint across all courses

## Current Configuration

The experiments test 5 different constraint configurations in `config/experiment_config.py`:

```python
CONSTRAINTS = [
    (0.5, 0.3),  # Baseline: 50% local, 30% global
    (0.3, 0.3),  # Stricter local: 30% local, 30% global
    (0.7, 0.3),  # Looser local: 70% local, 30% global
    (0.5, 0.5),  # Higher global: 50% local, 50% global
    (0.3, 0.5)   # Strict local, high global: 30% local, 50% global
]
```

## Top 5 Configurations

The top 5 hyperparameter configurations being tested:

1. **arch_deep**: Deep architecture (4 layers: 256→128→64→32)
2. **dropout_high**: High dropout regularization (0.5)
3. **very_deep_baseline**: Very deep architecture (5 layers: 512→256→128→64→32)
4. **lambda_high**: High constraint penalty (λ=0.1)
5. **very_deep_extreme_lambda**: Very deep + extreme penalty (λ=1.0)

## Running Experiments

### Run All Experiments

```bash
python experiments/run_experiments.py
```

This will:
1. Run all 5 configurations × 5 constraint settings = 25 total experiments
2. Save results to separate folders: `results/hyperparam_{config}_c{local}_{global}/`
3. Automatically run constraint-grouped analysis at the end

### Results Organization

Each experiment saves to:
```
results/
├── hyperparam_arch_deep_c0.5_0.3/
├── hyperparam_arch_deep_c0.3_0.3/
├── hyperparam_arch_deep_c0.7_0.3/
├── hyperparam_arch_deep_c0.5_0.5/
├── hyperparam_arch_deep_c0.3_0.5/
├── hyperparam_dropout_high_c0.5_0.3/
└── ... (25 folders total)
```

Each folder contains:
- `training_log.csv` - Loss values over epochs
- `evaluation_metrics.csv` - Final performance metrics
- `benchmark_metrics.csv` - Baseline comparison
- `constraint_comparison.csv` - Constraint satisfaction tracking

## Analysis

### Automatic Analysis

After experiments complete, the system automatically runs:
```bash
python experiments/analyze_by_constraints.py
```

This generates:
```
results/constraint_analysis/
├── constraint_0.5_0.3/
│   ├── accuracy_comparison.png
│   └── CONSTRAINT_REPORT.md
├── constraint_0.3_0.3/
│   ├── accuracy_comparison.png
│   └── CONSTRAINT_REPORT.md
├── ... (one folder per constraint)
└── constraint_comparison_summary.png
```

### Manual Analysis

Run analysis for a specific constraint group:
```bash
python experiments/analyze_by_constraints.py
```

Run original top 5 analysis (for single constraint):
```bash
python experiments/analyze_top5.py
```

## Understanding Results

### Constraint Analysis Output

For each constraint configuration:
1. **accuracy_comparison.png**: Bar chart showing optimized vs benchmark accuracy
2. **CONSTRAINT_REPORT.md**: Detailed ranking and metrics

### Cross-Constraint Summary

The `constraint_comparison_summary.png` shows:
- Left: Best accuracy achieved under each constraint
- Right: Improvement percentage for each constraint

This helps answer:
- Which constraint setting produces the best overall results?
- Which hyperparameter configuration is most robust across constraints?
- How much does constraint strictness affect improvement potential?

## Modifying Constraints

To test different constraint configurations:

1. Edit `config/experiment_config.py`:
```python
CONSTRAINTS = [
    (0.5, 0.3),  # Your custom constraint
    (0.6, 0.4),  # Another custom constraint
]
```

2. Run experiments:
```bash
python experiments/run_experiments.py
```

3. Analysis will automatically adapt to your constraint configurations

## Expected Findings

Based on system design:
- **Stricter constraints** (lower percentages) may show smaller improvements but better constraint satisfaction
- **Looser constraints** (higher percentages) may allow higher accuracy but risk underfitting constraints
- **Lambda values** interact with constraint strictness - higher λ works better with stricter constraints
- **Architecture depth** may matter more under stricter constraints (more capacity to balance objectives)

## Performance Notes

- Each configuration takes ~5-10 minutes depending on hardware
- Total runtime for 5 configs × 5 constraints: ~2-4 hours
- Results are saved incrementally - you can stop and resume
- Analysis scripts are fast (~30 seconds total)

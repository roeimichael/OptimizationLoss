# Convergence Parameter Testing Plan

## Overview

This experiment tests different sustained convergence parameters to find the optimal balance between:
- **Fast convergence** (stopping early when constraints satisfied)
- **Budget utilization** (using full constraint budget, e.g., 43/43 dropouts)

## Problem Being Addressed

**Current behavior** (convergence_window=1, convergence_required=1):
- Training stops immediately when constraints satisfied once
- Results in under-utilization (e.g., 39/43 budget used instead of full 43)
- Accuracy: ~58% (vs 62% heuristic baseline)

**Goal**: Find optimal convergence parameters that:
- Allow model to oscillate around constraint boundaries
- Achieve full budget utilization (43/43)
- Improve accuracy closer to heuristic baseline (62%)
- Don't waste computation with excessive training

## Experiment Design

### Fixed Parameters
- **Model**: TabularResNet (best performing model from previous experiments)
- **Learning rate**: 0.001 (optimal from grid search)
- **Lambda strategy**: linear (simple and effective)
- **Max epochs**: 2000 (increased from 500 to allow sustained convergence)
- **Warmup epochs**: 300
- **Other hyperparams**: Standard (batch_size=64, hidden_dims=[128,64], dropout=0.3)

### Variable Parameters

#### Constraint Pairs (3 total)

These are the same constraint pairs used in the original experiments:

1. `[0.5, 0.3]` - **Restrictive**: 43 dropouts, 24 enrolled allowed (from 142/79 total)
2. `[0.8, 0.2]` - **Most restrictive**: 28 dropouts, 16 enrolled allowed (from 142/79 total)
3. `[0.9, 0.8]` - **Lenient**: 114 dropouts, 63 enrolled allowed (from 142/79 total)

Note: Format is `[local%, global%]` where:
- `local%` applies per-course constraints
- `global%` applies to total test set (142 dropouts, 79 enrolled, 221 graduates)

#### Convergence Combinations (20 total)

| Window | Required | Satisfaction Rate | Description |
|--------|----------|------------------|-------------|
| 1 | 1 | 100% | Baseline (immediate stop) |
| 5 | 2 | 40% | Very lenient, small window |
| 5 | 3 | 60% | Lenient, small window |
| 5 | 4 | 80% | Strict, small window |
| 5 | 5 | 100% | Perfect, small window |
| 10 | 5 | 50% | Very lenient, medium window |
| 10 | 7 | 70% | Lenient, medium window |
| 10 | 8 | 80% | Moderate, medium window |
| 10 | 9 | 90% | Strict, medium window |
| 10 | 10 | 100% | Perfect, medium window |
| 20 | 10 | 50% | Very lenient, large window |
| 20 | 12 | 60% | Lenient, large window |
| 20 | 14 | 70% | Moderate, large window |
| **20** | **15** | **75%** | **Recommended (from analysis)** |
| 20 | 16 | 80% | Strict, large window |
| 20 | 18 | 90% | Very strict, large window |
| 20 | 20 | 100% | Perfect, large window |
| 30 | 20 | 67% | Moderate, very large window |
| 30 | 24 | 80% | Strict, very large window |
| 30 | 27 | 90% | Very strict, very large window |

**Total experiments**: 3 constraints × 20 convergence combos = **60 experiments**

## Directory Structure

```
results/longer_saturation/
└── TabularResNet/
    ├── constraint_50_30/       # [0.5, 0.3]: 43 dropouts, 24 enrolled
    │   └── convergence_test/
    │       ├── conv_1_1/          # Baseline
    │       ├── conv_5_2/
    │       ├── conv_5_3/
    │       ├── ...
    │       ├── conv_20_15/        # Recommended
    │       └── conv_30_27/
    ├── constraint_80_20/       # [0.8, 0.2]: 28 dropouts, 16 enrolled
    │   └── convergence_test/
    │       └── (same 20 combinations)
    └── constraint_90_80/       # [0.9, 0.8]: 114 dropouts, 63 enrolled
        └── convergence_test/
            └── (same 20 combinations)
```

Each experiment directory contains:
- `config.json` - Experiment configuration with convergence parameters
- `training_log.csv` - Detailed training metrics per epoch
- `final_predictions.csv` - Model predictions on test set
- `evaluation_metrics.csv` - Accuracy, precision, recall, F1
- `run_status.json` - Convergence status and epoch count

## How to Run

### 1. Generate configs (already done)
```bash
python generate_convergence_configs.py
```

### 2. Run all experiments
```bash
python run_all_convergence_experiments.py
```

This will:
- Execute all 60 experiments sequentially
- Skip already completed experiments (marked with `.completed` file)
- Show progress and statistics
- Handle failures gracefully

### 3. Analyze results
After completion, use analysis scripts to:
- Compare accuracy across different convergence parameters
- Measure budget utilization (% of constraint budget used)
- Find optimal convergence parameters
- Visualize convergence behavior over epochs

## Expected Outcomes

### Hypotheses

1. **Immediate stop (1/1)** → ~58% accuracy, 90% budget utilization
   - Current baseline behavior

2. **Too lenient (e.g., 5/2)** → May not converge properly
   - Model might violate constraints too often

3. **Recommended (20/15)** → ~60% accuracy, 100% budget utilization
   - Based on analysis in `CONSTRAINT_TRAINING_ANALYSIS.md`
   - Allows natural oscillation around constraint boundaries
   - 75% satisfaction rate balances convergence speed and budget utilization

4. **Too strict (e.g., 30/27)** → May take too long or be unstable
   - Requires 90% satisfaction over 30 epochs
   - Risk of premature stop if constraints become temporarily unsatisfiable

### Metrics to Analyze

1. **Accuracy** - Primary metric (goal: approach 62% heuristic baseline)
2. **Budget utilization** - % of constraint budget used (goal: 100%)
3. **Epochs to convergence** - Training efficiency (lower is better, if accuracy maintained)
4. **Satisfaction rate** - Actual satisfaction rate in final 20 epochs
5. **Prediction distribution** - Class counts vs constraints

## Implementation Details

### Code Changes

1. **trainer.py** (lines 104-113, 199-233)
   - Added `SustainedConvergenceChecker` import
   - Get `convergence_window` and `convergence_required` from hyperparams
   - Default to (1, 1) for backward compatibility with existing experiments
   - Replace immediate stop with sustained convergence check
   - Print convergence progress every 10 epochs

2. **generate_convergence_configs.py** (new file)
   - Generates 60 experiment configs
   - Includes convergence parameters in hyperparams
   - Computes base_model_id for model caching

3. **run_all_convergence_experiments.py** (new file)
   - Batch runner for all experiments
   - Progress tracking and error handling
   - Marks completed experiments to allow resumption

## Next Steps

After experiments complete:

1. **Aggregate results** - Create CSV with all results
2. **Statistical analysis** - Find optimal convergence parameters
3. **Visualization** - Plot accuracy vs satisfaction rate, window size, etc.
4. **Budget analysis** - Measure actual budget utilization for each combination
5. **Recommendation** - Determine best convergence parameters for production use

## Timeline Estimate

- **Per experiment**: ~5-20 minutes (depending on convergence speed)
- **Total (60 experiments)**: ~5-20 hours
- **Recommendation**: Run overnight or on compute cluster

## References

- `CONSTRAINT_TRAINING_ANALYSIS.md` - Root cause analysis of under-utilization problem
- `src/training/sustained_convergence.py` - Convergence checker implementation
- `src/training/trainer.py` - Training loop with sustained convergence integration

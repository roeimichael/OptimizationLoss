# Hyperparameter Experiments Guide

This document explains the hyperparameter experiment setup for the (0.5, 0.3) constraint configuration.

## Overview

We've configured 16 different hyperparameter combinations to systematically explore:
- **Constraint pressure** (lambda values)
- **Network architecture** (depth and width)
- **Learning rates** (optimization speed)
- **Regularization** (dropout rates)
- **Batch sizes** (gradient estimation quality)

## Experiment Organization

Each experiment creates a separate folder: `results/hyperparam_{config_name}/`

All results are tracked in `results/nn_results.json` with full hyperparameter documentation.

## Experiments List

### 1. Baseline
**Config Name:** `baseline`
- Lambda: global=0.01, local=0.01
- Architecture: [128, 64, 32]
- Learning rate: 0.001
- Dropout: 0.3
- Batch size: 64

**Purpose:** Reference configuration matching the original setup

---

### 2-5. Lambda Experiments (Constraint Pressure)

#### 2. Low Constraint Pressure
**Config Name:** `lambda_low`
- Lambda: global=0.001, local=0.001 (10x weaker)
- Other params: Same as baseline

**Purpose:** Test if weak constraint pressure can still satisfy constraints

#### 3. High Constraint Pressure
**Config Name:** `lambda_high`
- Lambda: global=0.1, local=0.1 (10x stronger)
- Other params: Same as baseline

**Purpose:** Test if stronger constraint pressure improves satisfaction speed

#### 4. Favor Global Constraints
**Config Name:** `lambda_favor_global`
- Lambda: global=0.1, local=0.01
- Other params: Same as baseline

**Purpose:** Prioritize global constraints over local

#### 5. Favor Local Constraints
**Config Name:** `lambda_favor_local`
- Lambda: global=0.01, local=0.1
- Other params: Same as baseline

**Purpose:** Prioritize local (per-course) constraints over global

---

### 6-8. Architecture Experiments

#### 6. Shallow Network
**Config Name:** `arch_shallow`
- Architecture: [64, 32] (2 layers, smaller)
- Other params: Same as baseline

**Purpose:** Test if simpler model works as well

#### 7. Deep Network
**Config Name:** `arch_deep`
- Architecture: [256, 128, 64, 32] (4 layers)
- Other params: Same as baseline

**Purpose:** Test if additional depth helps

#### 8. Wide Network
**Config Name:** `arch_wide`
- Architecture: [256, 128, 64] (wider layers)
- Other params: Same as baseline

**Purpose:** Test if wider layers help capacity

---

### 9-10. Learning Rate Experiments

#### 9. Slow Learning
**Config Name:** `lr_slow`
- Learning rate: 0.0001 (10x slower)
- Other params: Same as baseline

**Purpose:** Test if slower learning improves stability

#### 10. Fast Learning
**Config Name:** `lr_fast`
- Learning rate: 0.01 (10x faster)
- Other params: Same as baseline

**Purpose:** Test if faster learning speeds convergence

---

### 11-12. Dropout Experiments (Regularization)

#### 11. Low Dropout
**Config Name:** `dropout_low`
- Dropout: 0.1 (less regularization)
- Other params: Same as baseline

**Purpose:** Test if less regularization helps

#### 12. High Dropout
**Config Name:** `dropout_high`
- Dropout: 0.5 (more regularization)
- Other params: Same as baseline

**Purpose:** Test if stronger regularization prevents overfitting

---

### 13-14. Batch Size Experiments

#### 13. Small Batches
**Config Name:** `batch_small`
- Batch size: 32 (noisier gradients)
- Other params: Same as baseline

**Purpose:** Test if smaller batches help escape local minima

#### 14. Large Batches
**Config Name:** `batch_large`
- Batch size: 128 (smoother gradients)
- Other params: Same as baseline

**Purpose:** Test if larger batches improve stability

---

### 15-16. Combined Optimizations

#### 15. Optimized v1 (High Pressure + Deep)
**Config Name:** `optimized_v1`
- Lambda: global=0.1, local=0.1
- Architecture: [256, 128, 64, 32]
- Other params: Same as baseline

**Purpose:** Combine strong constraint pressure with deep architecture

#### 16. Optimized v2 (Local Focus + Wide + Careful)
**Config Name:** `optimized_v2`
- Lambda: global=0.01, local=0.1
- Architecture: [256, 128, 64]
- Learning rate: 0.0001
- Dropout: 0.2
- Batch size: 32

**Purpose:** Combine local constraint focus, wider network, careful optimization

---

## Running Experiments

```bash
python experiments/run_experiments.py
```

This will:
1. Run all 16 configurations sequentially
2. Create folder for each: `results/hyperparam_{name}/`
3. Save comprehensive results to `results/nn_results.json`

## Analyzing Results

Each experiment folder contains:
- `training_log.csv` - Training progress per epoch
- `final_predictions.csv` - Predictions with probabilities
- `constraint_comparison.csv` - Constraint satisfaction analysis
- `evaluation_metrics.csv` - Performance metrics
- `benchmark_metrics.csv` - Greedy baseline comparison
- Visualization plots

The `nn_results.json` file contains a summary for all experiments:
```json
{
  "baseline": {
    "(0.5, 0.3)": {
      "hyperparameters": {...},
      "accuracy": 0.5928,
      "precision_macro": 0.6124,
      ...
    }
  },
  "lambda_low": {...},
  ...
}
```

## Expected Insights

After running all experiments, you should be able to answer:

1. **Constraint Pressure:** Do higher lambda values improve constraint satisfaction?
2. **Architecture:** Does depth or width matter more for this task?
3. **Learning Rate:** What's the optimal learning speed?
4. **Regularization:** How much dropout prevents overfitting?
5. **Batch Size:** Do small or large batches work better?
6. **Combined:** Which combination of settings achieves the best results?

## Quick Start (Subset Testing)

To test with just a few configs, edit `config/experiment_config.py` and comment out unwanted configs:

```python
NN_CONFIGS = [
    # Just test these 3 for now
    {...baseline...},
    {...lambda_high...},
    {...arch_deep...},
]
```

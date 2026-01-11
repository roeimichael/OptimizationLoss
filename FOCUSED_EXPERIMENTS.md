# Focused Experimental Configuration

## Overview

This document describes the reduced experimental configuration designed for faster computation and focused analysis. The configuration has been reduced from 640 to 48 experiments while maintaining comprehensive coverage of key experimental dimensions.

## Configuration Breakdown

### Total Experiments: 48
**Calculation:** 4 models × 4 constraint pairs × 3 learning rates = 48 experiments

---

## 1. Neural Network Architectures (4 models)

Selected architectures represent diverse design paradigms:

| Model | Architecture Type | Key Feature | Complexity |
|-------|------------------|-------------|------------|
| **BasicNN** | Feedforward | Simple baseline | Low |
| **ResNet56** | Residual | Skip connections | Medium |
| **DenseNet121** | Dense connectivity | Feature reuse | High |
| **VGG19** | Deep sequential | Depth without tricks | Medium-High |

**Removed:** InceptionV3 (multi-scale features) - to reduce computational overhead

---

## 2. Constraint Pairs (4 configurations)

Constraint pairs systematically cover all combinations of restrictive (hard) and permissive (soft) settings:

| Pair | Local % | Global % | Description |
|------|---------|----------|-------------|
| **[Soft, Soft]** | 0.9 | 0.8 | Both constraints permissive - easy scenario |
| **[Hard, Soft]** | 0.3 | 0.8 | Local restrictive, global permissive - tests local pressure |
| **[Soft, Hard]** | 0.8 | 0.3 | Local permissive, global restrictive - tests global pressure |
| **[Hard, Hard]** | 0.3 | 0.3 | Both constraints restrictive - hardest scenario |

### Constraint Interpretation

For a test set with 100 students where 40 are true dropouts:
- **Soft (0.8):** Allows ⌊40 × 0.8 / 10⌋ = 3 dropout predictions
- **Hard (0.3):** Allows ⌊40 × 0.3 / 10⌋ = 1 dropout prediction

This 2×2 design enables analysis of:
- Effect of local constraint tightness (holding global constant)
- Effect of global constraint tightness (holding local constant)
- Interaction effects between local and global constraints
- Extreme cases: easiest ([Soft, Soft]) vs hardest ([Hard, Hard])

---

## 3. Learning Rate Sensitivity (3 values)

Learning rate is the most critical hyperparameter for optimization convergence. Three values span three orders of magnitude:

| Learning Rate | Category | Expected Behavior |
|--------------|----------|-------------------|
| **0.0001** | Low | Slow but stable convergence |
| **0.001** | Medium | Balanced (default baseline) |
| **0.01** | High | Fast but potentially unstable |

**Constant Parameters:**
- Dropout: 0.3 (moderate regularization)
- Batch size: 64 (standard mini-batch)
- Hidden dimensions: [128, 64] (for BasicNN)
- Lambda values: 0.01 (initial constraint weights)
- Warmup epochs: 250 (pre-constraint training)

---

## 4. Rationale for Focused Configuration

### Why reduce from 640 to 48?

1. **Computational Efficiency**
   - 640 experiments at ~30 minutes each = ~320 hours (13+ days)
   - 48 experiments at ~30 minutes each = ~24 hours (1 day)
   - 13× speedup enables faster iteration

2. **Focused Analysis**
   - Learning rate is most critical hyperparameter for multi-objective optimization
   - Constraint pairs cover all essential combinations (2×2 factorial design)
   - 4 architectures span key design paradigms

3. **Statistical Validity**
   - 48 experiments still provide sufficient samples for meaningful comparisons
   - 12 experiments per constraint pair enables robust statistical analysis
   - 16 experiments per architecture supports architecture comparison

4. **Maintained Coverage**
   - All major architectural types represented
   - Full spectrum of constraint tightness covered
   - Critical hyperparameter (learning rate) thoroughly explored

---

## 5. Experiment Organization

Results will be organized hierarchically:

```
results/
└── our_approach/
    ├── BasicNN/
    │   ├── constraint_0.9_0.8/  # [Soft, Soft]
    │   │   └── lr_test/
    │   │       ├── lr_0.0001/
    │   │       ├── lr_0.001/
    │   │       └── lr_0.01/
    │   ├── constraint_0.3_0.8/  # [Hard, Soft]
    │   ├── constraint_0.8_0.3/  # [Soft, Hard]
    │   └── constraint_0.3_0.3/  # [Hard, Hard]
    ├── ResNet56/
    │   └── [same structure]
    ├── DenseNet121/
    │   └── [same structure]
    └── VGG19/
        └── [same structure]
```

Each leaf directory contains:
- `config.json` - Full experiment configuration
- `training_log.csv` - Per-epoch training metrics
- `final_predictions.csv` - Model predictions on test set
- `evaluation_metrics.csv` - Performance metrics

---

## 6. Expected Insights

This focused configuration enables analysis of:

### Architecture Comparison
- Which architectures best satisfy constraints?
- Does architectural complexity help or hurt constraint satisfaction?
- Are residual/dense connections beneficial for multi-objective optimization?

### Constraint Analysis
- How do models perform under different constraint tightness levels?
- Is local or global constraint harder to satisfy?
- What is the accuracy-constraint satisfaction trade-off?

### Learning Rate Sensitivity
- What learning rate best balances accuracy and constraint satisfaction?
- Does optimal learning rate vary by architecture or constraint tightness?
- How does learning rate affect convergence speed?

### Interaction Effects
- Do some architectures perform better under tight constraints?
- Does learning rate importance vary with constraint tightness?
- Are there synergies between architectural features and learning rate?

---

## 7. Restoring Full Configuration

To restore the original 640-experiment configuration:

1. Open `src/utils/generate_configs.py`
2. **Comment out** sections marked `FOCUSED EXPERIMENT:`
   ```python
   # FOCUSED EXPERIMENT: 4 models
   # MODELS = ['BasicNN', 'ResNet56', 'DenseNet121', 'VGG19']
   ```
3. **Uncomment** sections marked `FULL EXPERIMENT:`
   ```python
   # FULL EXPERIMENT: 5 models (uncomment to restore)
   MODELS = ['BasicNN', 'ResNet56', 'DenseNet121', 'InceptionV3', 'VGG19']
   ```
4. Do this for:
   - MODELS (restore 5 models)
   - CONSTRAINTS (restore 8 constraint pairs)
   - HYPERPARAM_REGIMES (restore all 4 regimes with full variations)

This restores:
- 5 models
- 8 constraint pairs
- 16 hyperparameter variations (1 + 5 + 5 + 5)
- Total: 5 × 8 × 16 = 640 experiments

---

## 8. Timeline Estimate

Assuming average training time of 30 minutes per experiment:

| Configuration | Experiments | Time (serial) | Time (4 parallel) | Time (8 parallel) |
|--------------|-------------|---------------|-------------------|-------------------|
| **Focused** | 48 | 24 hours | 6 hours | 3 hours |
| Full | 640 | 320 hours | 80 hours | 40 hours |

**Recommendation:** Run focused experiments first, analyze results, then selectively run additional experiments if needed.

---

## Summary

The focused configuration maintains experimental rigor while drastically reducing computational requirements. By focusing on:
- 4 representative architectures
- 4 systematic constraint combinations
- 3 critical learning rate values

We achieve a **13× speedup** while preserving the ability to:
- Compare architectural approaches
- Analyze constraint satisfaction behavior
- Identify optimal training hyperparameters
- Draw statistically meaningful conclusions

This configuration strikes an optimal balance between computational efficiency and experimental completeness.

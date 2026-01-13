# Adaptive Lambda Methodologies

This document describes the two new adaptive lambda methodologies implemented to address the overfitting problems observed in static lambda experiments.

## Problem Statement

Static lambda experiments (V1 and V2) revealed a fundamental tradeoff:
- **Low λ (≤0.01):** Good prediction quality but poor constraint convergence (33% success)
- **High λ (≥0.02):** Good convergence but severe overfitting to constraints (95%+ Graduate predictions)

No static lambda value achieved both good convergence AND reasonable prediction quality.

## Solution: Adaptive Lambda Approaches

Both methodologies start with **low lambda** (allowing model to learn patterns) and gradually **increase lambda** (enforcing constraints), but differ in HOW lambda grows.

---

## Methodology 1: Loss-Proportional Adaptive Lambda

### Key Idea
**Lambda grows proportionally to how badly constraints are violated.**

### Algorithm
```
Phase 1: Warmup (epochs 1-250)
  λ = 0, train on prediction loss only

Phase 2: Adaptive (epochs 251+)
  Every epoch:
    λ_global_new = λ_global_old + α × L_global
    λ_local_new = λ_local_old + α × L_local

  Where:
    - α = lambda learning rate (how aggressively lambda grows)
    - L_global, L_local = current constraint loss values
```

### Hyperparameters
- `lambda_learning_rate` (α): Controls growth speed (default: 0.01)
- `initial_lambda`: Starting lambda after warmup (default: 0.001)
- `max_lambda`: Upper bound for lambda (default: 1.0)
- `warmup_epochs`: Pure prediction training (default: 250)
- `epochs`: Total training epochs (default: 1000)

### Experiments
**48 total experiments:**
- 3 models (BasicNN, TabularResNet, FTTransformer)
- 4 constraint types ([Hard,Hard], [Hard,Soft], [Soft,Hard], [Soft,Soft])
- 4 alpha values (0.005, 0.01, 0.02, 0.05)

### Behavior
- **High constraint violations** → Lambda increases rapidly
- **Low constraint violations** → Lambda increases slowly
- **Constraints satisfied** → Lambda stops growing significantly

### Advantages
- Direct feedback: Worse violations → stronger penalties
- Smooth, continuous adjustment
- No arbitrary thresholds

### Potential Issues
- May grow too fast if initial violations are large
- Could oscillate if alpha is too high

---

## Methodology 2: Scheduled Growth with Loss Gates

### Key Idea
**Lambda only increases when the model fails to improve constraint satisfaction.**

### Algorithm
```
Phase 1: Warmup (epochs 1-250)
  λ = 0, train on prediction loss only

Phase 2: Adaptive (epochs 251+)
  Every N epochs (check_frequency):
    current_loss = L_global + L_local

    If current_loss >= previous_loss:
      # Model didn't improve - increase pressure
      λ_global = λ_global × growth_factor
      λ_local = λ_local × growth_factor
    Else:
      # Model is improving - keep lambda same
      λ stays unchanged

    previous_loss = current_loss
```

### Hyperparameters
- `growth_factor`: Multiplicative increase (e.g., 1.1 = 10% growth)
- `check_frequency`: How often to check improvement (default: 10 epochs)
- `initial_lambda`: Starting lambda after warmup (default: 0.001)
- `max_lambda`: Upper bound for lambda (default: 1.0)
- `warmup_epochs`: Pure prediction training (default: 250)
- `epochs`: Total training epochs (default: 1000)

### Experiments
**48 total experiments:**
- 3 models (BasicNN, TabularResNet, FTTransformer)
- 4 constraint types ([Hard,Hard], [Hard,Soft], [Soft,Hard], [Soft,Soft])
- 4 growth configurations:
  - factor=1.05, freq=10 (slow growth, frequent checks)
  - factor=1.1, freq=10 (moderate growth, frequent checks)
  - factor=1.1, freq=20 (moderate growth, infrequent checks)
  - factor=1.2, freq=10 (fast growth, frequent checks)

### Behavior
- **Loss not improving** → Lambda increases (apply more pressure)
- **Loss improving** → Lambda stays same (give model time to adjust)
- **Constraints satisfied** → Early stopping

### Advantages
- More stable than continuous adjustment
- Model gets "breathing room" between lambda increases
- Won't aggressively push toward Graduate predictions if model is improving

### Potential Issues
- Requires tuning both growth_factor and check_frequency
- May take longer to converge than loss-proportional

---

## Comparison: Loss-Proportional vs Scheduled Growth

| Aspect | Loss-Proportional | Scheduled Growth |
|--------|-------------------|------------------|
| **Growth pattern** | Continuous (every epoch) | Discrete (every N epochs) |
| **Growth rate** | Proportional to loss magnitude | Fixed multiplicative factor |
| **Responds to** | Absolute constraint violation | Improvement trend |
| **Stability** | May oscillate | More stable |
| **Tuning** | 1 parameter (α) | 2 parameters (factor, frequency) |
| **Complexity** | Simpler | More complex |

---

## Directory Structure

```
results/
├── loss_proportional/
│   ├── BasicNN/
│   │   ├── constraint_0.3_0.3/
│   │   │   └── lambda_lr_sweep/
│   │   │       ├── alpha_0.005/
│   │   │       ├── alpha_0.01/
│   │   │       ├── alpha_0.02/
│   │   │       └── alpha_0.05/
│   │   ├── constraint_0.3_0.8/
│   │   ├── constraint_0.8_0.3/
│   │   └── constraint_0.9_0.8/
│   ├── TabularResNet/
│   └── FTTransformer/
│
└── scheduled_growth/
    ├── BasicNN/
    │   ├── constraint_0.3_0.3/
    │   │   └── growth_factor_sweep/
    │   │       ├── factor_1.05_freq_10/
    │   │       ├── factor_1.1_freq_10/
    │   │       ├── factor_1.1_freq_20/
    │   │       └── factor_1.2_freq_10/
    │   ├── constraint_0.3_0.8/
    │   ├── constraint_0.8_0.3/
    │   └── constraint_0.9_0.8/
    ├── TabularResNet/
    └── FTTransformer/
```

---

## Running Experiments

### 1. Generate Configurations (Already Done)
```bash
python src/utils/generate_loss_proportional_configs.py
python src/utils/generate_scheduled_growth_configs.py
```

### 2. Run Experiments
```bash
# Run both methodologies
python main.py

# Or run single experiment
python run_loss_proportional_experiment.py results/loss_proportional/BasicNN/constraint_0.3_0.3/lambda_lr_sweep/alpha_0.01/config.json
python run_scheduled_growth_experiment.py results/scheduled_growth/BasicNN/constraint_0.3_0.3/growth_factor_sweep/factor_1.1_freq_10/config.json
```

### 3. Configure Which Methodologies to Run

Edit `main.py`:
```python
ACTIVE_METHODOLOGIES: List[str] = [
    'loss_proportional',  # Enable this methodology
    'scheduled_growth',   # Enable this methodology
    # 'static_lambda',    # Disable static lambda
    # 'our_approach',     # Disable original adaptive
]
```

---

## Expected Outcomes

### Success Criteria
1. **Constraint convergence:** ≥70% of experiments satisfy constraints
2. **Prediction quality:** Graduate predictions in 75-85% range (not 95%+)
3. **Balanced recall:** All three classes should have reasonable recall values
4. **Model learning:** F1 macro should be significantly above 0.28 (V2 baseline)

### Hypotheses

**Loss-Proportional:**
- Faster convergence (responds immediately to violations)
- May require lower alpha values (0.005-0.01) to prevent rapid overfitting
- Best for experiments where constraint violations vary significantly

**Scheduled Growth:**
- More stable training (fewer oscillations)
- May require higher growth factors (1.1-1.2) to converge in reasonable time
- Best for experiments where steady, controlled growth is preferred

### What to Look For

**Good signs:**
- Lambda stabilizes at moderate values (0.01-0.1 range)
- Graduate predictions converge to 75-85%
- Constraints satisfied after 300-700 epochs (not 2-7 epochs like V2!)
- Confusion matrix shows predictions for all three classes

**Bad signs:**
- Lambda grows to max_lambda (1.0) rapidly
- Graduate predictions exceed 90% (overfitting)
- Early stopping at epoch 2-10 (found shortcut)
- Zero predictions for Dropout/Enrolled classes

---

## Analysis Strategy

After experiments complete:

1. **Convergence Rate Analysis:**
   - What percentage of experiments satisfy constraints?
   - By methodology, model, constraint type, hyperparameter

2. **Prediction Quality Analysis:**
   - Average Graduate prediction percentage
   - Confusion matrices
   - Per-class recall values

3. **Lambda Evolution Analysis:**
   - How does lambda change over epochs?
   - What are final lambda values for successful experiments?
   - Do they stabilize or keep growing?

4. **Comparison Analysis:**
   - Loss-proportional vs Scheduled growth
   - Both vs Static lambda V2
   - Both vs Original adaptive approach

5. **Optimal Hyperparameter Identification:**
   - Best alpha for loss-proportional
   - Best growth_factor and check_frequency for scheduled growth

---

## Files Created

### Trainers
- `src/training/loss_proportional_trainer.py`
- `src/training/scheduled_growth_trainer.py`

### Runners
- `run_loss_proportional_experiment.py`
- `run_scheduled_growth_experiment.py`

### Config Generators
- `src/utils/generate_loss_proportional_configs.py`
- `src/utils/generate_scheduled_growth_configs.py`

### Updated Files
- `src/training/__init__.py` (added new trainer exports)
- `main.py` (added new methodologies to runner mapping)

---

## Next Steps

1. **Run experiments:** `python main.py`

2. **Monitor progress:** Check that:
   - Lambda values grow gradually (not jumping to 1.0 immediately)
   - Training doesn't stop at epoch 2-10
   - Graduate predictions stay reasonable

3. **Analyze results:** Create analysis script similar to `analyze_v2_results.py`

4. **Compare methodologies:** Determine which approach works best

5. **Tune hyperparameters:** If needed, adjust alpha/growth_factor values based on results

---

## Key Differences from Static Lambda

| Aspect | Static Lambda | Adaptive Lambda |
|--------|---------------|-----------------|
| **Lambda value** | Fixed throughout | Changes during training |
| **Learning** | No adaptation | Adapts to model progress |
| **Warmup** | No (starts with penalty) | Yes (learns patterns first) |
| **Problem** | Either can't converge OR overfits | Should balance both |
| **Epochs** | Converges in 2-7 or fails | Expected 300-700 epochs |
| **Graduate %** | 94-98% or 68-76% | Target: 75-85% |

The key innovation is that **lambda can respond to the model's behavior** rather than being a fixed hyperparameter.

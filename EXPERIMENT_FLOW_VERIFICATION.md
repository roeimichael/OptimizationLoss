# Experiment Flow Verification

## Question: Does run_all_convergence_experiments.py use the correct training pipeline?

**Answer: YES** - It follows the exact same flow as original experiments.

## Complete Execution Flow

### 1. Entry Point: run_all_convergence_experiments.py

**Line 29:**
```python
[sys.executable, 'src/experiments/run_experiment.py', str(config_path)]
```

✓ Calls the SAME `src/experiments/run_experiment.py` used for original experiments

---

### 2. Experiment Runner: src/experiments/run_experiment.py

**Line 36: Load data and constraints**
```python
X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)
```

✓ Uses `load_experiment_data()` to compute constraints from config

**Line 59-60: Create trainer**
```python
trainer = ConstraintTrainer(config, str(experiment_path), device)
trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])
```

✓ Uses ConstraintTrainer from `src/training/trainer.py`

**Line 64-72: Train model**
```python
trainer.train_warmup(X_train_tensor, y_train_tensor, config['base_model_id'])
model = trainer.train_constraints(
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_test=X_test_tensor,
    groups_test=groups_test,
    global_con=global_constraint,  # ← CONSTRAINTS PASSED HERE
    local_con=local_constraint     # ← CONSTRAINTS PASSED HERE
)
```

✓ Calls `trainer.train_constraints()` with computed constraints

---

### 3. Data Loader: src/utils/data_loader.py

**Line 20: Extract constraint percentages from config**
```python
local_percent, global_percent = config['constraint']
```

For config with `"constraint": [0.5, 0.3]`:
- local_percent = 0.5 (50% per-course)
- global_percent = 0.3 (30% total)

**Line 22-23: Compute actual constraint values**
```python
global_constraint = compute_global_constraints(test_df, TARGET_COLUMN, global_percent)
local_constraint = compute_local_constraints(test_df, TARGET_COLUMN, local_percent, groups)
```

✓ Calls constraint computation functions

---

### 4. Constraint Computation: src/training/constraints.py

**compute_global_constraints() - Line 11-17:**
```python
def compute_global_constraints(data, target_column, percentage):
    constraint = np.zeros(NUM_CLASSES)
    items = data[target_column].value_counts()
    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage / CONSTRAINT_SCALE_FACTOR)
    constraint[GRADUATE_CLASS_ID] = UNLIMITED_CONSTRAINT  # Class 2 always unlimited
    return constraint.tolist()
```

For [0.5, 0.3] with test set (142 dropouts, 79 enrolled, 221 graduates):
- Global constraint (30%):
  - Class 0: round(142 * 0.3) = 43
  - Class 1: round(79 * 0.3) = 24
  - Class 2: UNLIMITED (1e10)

✓ Computes correct constraint values

---

### 5. Trainer: src/training/trainer.py

**Line 98-103: Initialize constraint loss**
```python
criterion_constraint = MulticlassTransductiveLoss(
    global_constraints=global_con,      # ← [43, 24, 1e10]
    local_constraints=local_con,        # ← Per-course constraints
    lambda_global=self.hyperparams['lambda_global'],
    lambda_local=self.hyperparams['lambda_local']
).to(self.device)
```

✓ Passes constraints to transductive loss

**Line 104-109: Initialize sustained convergence checker**
```python
from src.training.sustained_convergence import SustainedConvergenceChecker
convergence_window = self.hyperparams.get('convergence_window', 1)
convergence_required = self.hyperparams.get('convergence_required', 1)
convergence_checker = SustainedConvergenceChecker(
    window_size=convergence_window,
    required_satisfied=convergence_required
)
```

✓ Creates convergence checker with config parameters

**Line 199-209: Check for sustained convergence**
```python
should_stop, reason = convergence_checker.update(
    criterion_constraint.global_constraints_satisfied,
    criterion_constraint.local_constraints_satisfied
)

if should_stop:
    print(f"\n[CONVERGED] {reason}")
    # ... save status and break
    break
```

✓ Uses sustained convergence instead of immediate stop

---

### 6. Transductive Loss: src/losses/transductive_loss.py

**Should compute loss based on constraint violations**

The MulticlassTransductiveLoss:
- Compares predictions to constraint limits
- Sets `self.global_constraints_satisfied` and `self.local_constraints_satisfied` flags
- Returns loss values

---

## VERIFICATION CHECKLIST

✓ run_all_convergence_experiments.py → src/experiments/run_experiment.py
✓ run_experiment.py → ConstraintTrainer (src/training/trainer.py)
✓ trainer.py → MulticlassTransductiveLoss (src/losses/transductive_loss.py)
✓ Constraints computed from config percentages
✓ Sustained convergence integrated
✓ Same pipeline as original experiments

## EXPECTED BEHAVIOR

For conv_1_1 (convergence_window=1, convergence_required=1):
- Should stop IMMEDIATELY when constraints satisfied once
- Should behave identically to original experiments
- Expected accuracy: ~69% (matching original)

For conv_20_15 (convergence_window=20, convergence_required=15):
- Requires constraints satisfied in 15 out of last 20 epochs
- Allows oscillation around constraint boundaries
- May achieve better budget utilization
- Expected: possibly better accuracy than 69%

## POTENTIAL ISSUES TO CHECK

1. **Are predictions exceeding limits?**
   - Check if transductive_loss.py is computing constraint satisfaction correctly
   - Verify constraint values are being logged properly

2. **Did convergence happen?**
   - Check training_log.csv for Global_Satisfied and Local_Satisfied columns
   - Verify convergence_checker is working as expected

3. **Config format correct?**
   - Verify config has "constraint": [local%, global%]
   - NOT "constraint": [global%, local%]

---

## TO INVESTIGATE RESULTS

Need to examine:
1. training_log.csv - Did constraints get satisfied?
2. evaluation_metrics.csv - What were final predictions?
3. config.json - Were constraint percentages correct?
4. run_status.json - What was the stop reason?


# Data Flow Validation - Constraint-Based Learning Pipeline

## Overview
This document validates the complete data flow from loading datasets through training to final evaluation.

## Data Flow Pipeline

### 1. Data Input
**Files**: `data/train_dataset.csv`, `data/test_dataset.csv`

**Required columns**:
- `Target`: Class labels (0, 1, 2 for 3-class problem)
- `Group`: Grouping column for local constraints
- Features: All other columns used for training

### 2. Configuration (`config/experiment_config.py`)
```python
TRAIN_PATH = 'data/train_dataset.csv'
TEST_PATH = 'data/test_dataset.csv'
TARGET_COLUMN = 'Target'
GROUP_COLUMN = 'Group'
```

### 3. Data Loading (`src/utils/data_loader.py`)

**Function**: `load_experiment_data(config)`

**Steps**:
1. Load train and test CSVs
2. Extract constraint percentages from config: `(local_percent, global_percent)`
3. Compute global constraints based on test set class distribution
4. Compute local constraints per group based on group-specific class distributions
5. Separate features (X) and target (y) for train/test
6. Return: `X_train, X_test, y_train, y_test, groups_test, global_constraints, local_constraints`

### 4. Constraint Computation (`src/training/constraints.py`)

**Global Constraints**:
- For each class: `constraint[class_id] = round(test_count[class_id] * global_percent)`
- Unlimited classes can be specified via `unlimited_classes` parameter

**Local Constraints**:
- For each group and each class: `constraint[group][class_id] = round(group_count[class_id] * local_percent)`
- Excluded groups can be specified via `excluded_groups` parameter

### 5. Model Training (`src/training/trainer.py`)

**Trainer**: `ConstraintTrainer`

**Warmup Phase** (first N epochs):
- Train with standard Cross-Entropy loss
- No constraint losses applied
- Cache trained model for reuse

**Constraint Optimization Phase** (remaining epochs):
- **Per Batch**:
  - Train on train_X with CE loss
  - Evaluate on test_X for constraint losses
  - Combined loss: `ce_weight * L_CE + λ_global * L_global + λ_local * L_local`
  - Where `ce_weight = 1.0 + λ_global + λ_local`

- **Per Epoch**:
  - Adjust lambdas: Increase by λ_step if constraint violated
  - Check convergence: Stop if both global and local constraints satisfied
  - Log progress every 3 epochs

### 6. Constraint Loss Computation (`src/losses/transductive_loss.py`)

**Loss Function**: `MulticlassTransductiveLoss`

**Global Constraint Loss**:
- For each class with constraint K:
  - Count predictions for that class (soft probabilities sum)
  - If count > K: `loss = excess / (excess + K + ε)`
  - Average over all constrained classes

**Local Constraint Loss**:
- For each group:
  - For each class with constraint K:
    - Count predictions in that group for that class
    - If count > K: compute loss as above
  - Average over constrained classes
- Weighted average across all groups by group size

**Satisfaction Check**:
- Uses hard predictions (argmax) to determine if constraint satisfied
- `global_constraints_satisfied`: True if all global constraints met
- `local_constraints_satisfied`: True if all local constraints met

### 7. Lambda Adjustment (`src/losses/lambda_adjusting.py`)

**Strategy**: Linear (simple linear increase)

**Per Epoch**:
- If `global_loss > threshold`: `λ_global = min(λ_global + λ_step, λ_max)`
- If `local_loss > threshold`: `λ_local = min(λ_local + λ_step, λ_max)`

### 8. Convergence Check

**Stopping Criteria**:
- Both `global_constraints_satisfied` AND `local_constraints_satisfied` are True
- Immediate stop when criteria met (no sustained window required)

**Status Saved**:
- `run_status.json`: Final convergence status
- `config.json`: Updated with stop reason and final epoch

### 9. Evaluation (`src/training/metrics.py`)

**Computed Metrics**:
- Overall Accuracy
- Per-class Precision, Recall, F1-Score
- Macro and Weighted averages
- Confusion Matrix
- Constraint satisfaction statistics

**Output Files**:
- `training_log.csv`: Per-epoch losses, lambdas, constraint status
- `final_predictions.csv`: Predictions for each test sample
- `evaluation_metrics.csv`: Final performance metrics

## Complete Data Flow Diagram

```
train_dataset.csv ──┐
                    ├──> load_experiment_data() ──> Compute Constraints
test_dataset.csv ───┘                                       │
                                                            ├──> Global Constraints
                                                            └──> Local Constraints
                                                                      │
                                                                      ▼
                                                             ConstraintTrainer
                                                                      │
                                            ┌─────────────────────────┼─────────────────────────┐
                                            ▼                         ▼                         ▼
                                    Warmup Phase          Constraint Phase              Convergence
                                    (CE Loss only)        (CE + Constraints)           (Both Satisfied)
                                            │                         │                         │
                                            └─────────────────────────┴─────────────────────────┘
                                                                      │
                                                                      ▼
                                                             Evaluation Metrics
                                                                      │
                                         ┌────────────────────────────┼────────────────────────────┐
                                         ▼                            ▼                            ▼
                                training_log.csv           final_predictions.csv         evaluation_metrics.csv
```

## Key Configuration Parameters

### Experiment Config (`config.json`)
```json
{
  "model_name": "BasicNN",
  "constraint": [0.9, 0.8],  // [local_percent, global_percent]
  "unlimited_classes": [],   // Optional: classes with no limit
  "hyperparams": {
    "lr": 0.0001,
    "batch_size": 64,
    "epochs": 1000,
    "warmup_epochs": 50,
    "lambda_global": 0.1,
    "lambda_local": 0.1,
    "lambda_step": 0.005,
    "constraint_threshold": 0.02
  }
}
```

## Validation Checklist

✅ **Data Loading**: CSVs loaded correctly with Target and Group columns
✅ **Constraint Computation**: Global and local constraints computed from test set
✅ **Model Training**: Two-phase training (warmup → constraint optimization)
✅ **Loss Computation**: CE + Global + Local losses combined correctly
✅ **Lambda Adjustment**: Simple linear increase when constraints violated
✅ **Convergence**: Stop immediately when both constraints satisfied
✅ **Evaluation**: All metrics computed and saved

## Example Usage

```bash
# 1. Place your datasets in data/
cp my_train.csv data/train_dataset.csv
cp my_test.csv data/test_dataset.csv

# 2. Generate experiment configurations
python src/config_generators/generate_configs.py

# 3. Run experiments
python main.py
```

## Notes

- All dataset-specific logic removed - fully generic now
- Group column name configurable via `GROUP_COLUMN` in experiment_config.py
- Unlimited classes can be specified per experiment
- No sustained convergence window - immediate stop on satisfaction
- Simple linear lambda adjustment only

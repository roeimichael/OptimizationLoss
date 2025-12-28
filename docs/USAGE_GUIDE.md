# Usage Guide

This guide walks you through using the OptimizationLoss framework for transductive learning with constraints.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Transductive Learning](#understanding-transductive-learning)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Understanding the Output](#understanding-the-output)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset at `data/dataset.csv`. The dataset should contain:
- Student features (demographic, academic, etc.)
- Target column: Student status (Dropout, Enrolled, Graduate)
- Course column: Course ID for local constraints

See `data/README.md` for detailed dataset requirements.

### 3. Run Your First Experiment

```bash
python experiments/run_experiments.py
```

This will train models with various constraint configurations and save results to `results/`.

---

## Understanding Transductive Learning

Traditional supervised learning only uses labeled training data. **Transductive learning** uses both:
- **Labeled data** (training set): Learn basic classification
- **Unlabeled data** (test set): Apply constraints to improve predictions

### Constraints

This framework uses two types of constraints:

#### Global Constraints
Limit the total number of predictions across all students:
```python
Dropout ≤ 142    # Max 142 students predicted as Dropout
Enrolled ≤ 85    # Max 85 students predicted as Enrolled
Graduate: ∞      # No limit on Graduate predictions
```

#### Local Constraints
Limit predictions per course/group:
```python
Course 2:
  Dropout ≤ 10
  Enrolled ≤ 6
  Graduate: ∞
```

### How It Works

1. **Warmup Phase** (epochs 0-249):
   - Train normally with cross-entropy loss
   - Constraint losses = 0 (λ = 0)
   - Model learns basic classification

2. **Constraint Phase** (epochs 250+):
   - Activate constraint losses
   - If constraints violated → increase λ weights
   - Model balances accuracy + constraint satisfaction

---

## Configuration

Edit `config/experiment_config.py` to customize experiments:

### Key Parameters

```python
# Constraint configurations to test
CONSTRAINTS = [
    (0.8, 0.6),  # (dropout_pct, enrolled_pct)
]

# Neural network architectures
NN_CONFIGS = [
    {
        "lambda_global": 0.01,   # Initial global constraint weight
        "lambda_local": 0.01,    # Initial local constraint weight
        "hidden_dims": [128, 64, 32]  # Network architecture
    }
]

# Training parameters
TRAINING_PARAMS = {
    'epochs': 1000,
    'batch_size': 32,
    'lr': 0.001,
    'dropout': 0.3,
}

# Constraint tuning
WARMUP_EPOCHS = 250          # Epochs before activating constraints
LAMBDA_STEP = 0.01           # How much to increase λ when violated
CONSTRAINT_THRESHOLD = 1e-6  # Violation threshold
TRACKED_COURSE_ID = 2        # Course to visualize
```

### Common Configurations

**Tight Constraints** (strict limits):
```python
CONSTRAINTS = [(0.5, 0.3)]  # 50% dropout, 30% enrolled
```

**Relaxed Constraints** (loose limits):
```python
CONSTRAINTS = [(0.8, 0.6)]  # 80% dropout, 60% enrolled
```

**Multiple Experiments**:
```python
CONSTRAINTS = [
    (0.5, 0.3),
    (0.6, 0.4),
    (0.7, 0.5),
    (0.8, 0.6)
]
```

---

## Running Experiments

### Basic Run

```bash
python experiments/run_experiments.py
```

Output structure:
```
results/
├── constraints_0.8_0.6/
│   ├── training_log.csv          # Full training history
│   ├── global_constraints.png    # Global constraint plot
│   ├── local_constraints.png     # Tracked course plot
│   ├── losses.png                # Loss components
│   └── lambda_evolution.png      # Lambda weight changes
```

### Monitor Progress

During training, you'll see output like:

```
================================================================================
Epoch 252
================================================================================
L_target (Global):  0.024531
L_feat (Local):     0.013245
L_pred (CE):        0.456789

────────────────────────────────────────────────────────────────────────────────
GLOBAL CONSTRAINTS vs PREDICTIONS (Hard vs Soft)
────────────────────────────────────────────────────────────────────────────────
Class        Limit    Hard     Soft       Diff     Status
────────────────────────────────────────────────────────────────────────────────
Dropout      142      140      141.23     1.23     ✓ OK
Enrolled     85       81       82.45      1.45     ✓ OK
Graduate     ∞        221      218.32     -2.68    N/A
────────────────────────────────────────────────────────────────────────────────
Total                 442      442.00

Current Lambda Weights: λ_global=0.02, λ_local=0.01
Constraint Status: Global=✓, Local=✓
================================================================================
```

**Key Metrics:**
- **Hard predictions**: Actual class assignments (argmax)
- **Soft predictions**: Sum of class probabilities
- **Status**: ✓ OK = satisfied, ✗ = violated

---

## Understanding the Output

### Training Log CSV

The CSV contains complete training history:

| Column | Description |
|--------|-------------|
| `Epoch` | Epoch number |
| `L_pred_CE` | Cross-entropy loss |
| `L_target_Global` | Global constraint loss |
| `L_feat_Local` | Local constraint loss |
| `Lambda_Global` | Global λ weight |
| `Lambda_Local` | Local λ weight |
| `Global_Satisfied` | 1 if satisfied, 0 if violated |
| `Hard_Dropout/Enrolled/Graduate` | Hard prediction counts |
| `Soft_Dropout/Enrolled/Graduate` | Soft prediction sums |
| `Course_Hard_*` | Tracked course hard predictions |
| `Course_Soft_*` | Tracked course soft predictions |

### Hard vs Soft Predictions

**Hard predictions** (discrete):
```python
probabilities = [0.7, 0.2, 0.1]  # [Dropout, Enrolled, Graduate]
hard_prediction = argmax(probabilities) = 0 (Dropout)
```

**Soft predictions** (continuous):
```python
# Sum of probabilities across all students
soft_dropout = student1_dropout_prob + student2_dropout_prob + ...
soft_dropout = 0.7 + 0.3 + 0.5 + ... = 142.35
```

**Why the difference matters:**
- Constraints checked using **hard** predictions (actual assignments)
- Losses computed using **soft** predictions (differentiable, for gradients)

---

## Advanced Usage

### Custom Constraint Configuration

To create custom constraints, modify `src/training/constraints.py`:

```python
def compute_global_constraints(data, target_column, percentage):
    # Custom logic here
    constraint[0] = 100  # Fixed dropout limit
    constraint[1] = 50   # Fixed enrolled limit
    return constraint.tolist()
```

### Adjusting Training Behavior

**Increase warmup period** (more initial training):
```python
WARMUP_EPOCHS = 500  # Train for 500 epochs before constraints
```

**Faster lambda growth** (stronger constraint pressure):
```python
LAMBDA_STEP = 0.05  # Increase λ by 0.05 per violation
```

**Early stopping** (when constraints satisfied):
The training loop automatically stops when both global and local constraints are satisfied.

### Using Different Models

Modify `NN_CONFIGS` to test different architectures:

```python
NN_CONFIGS = [
    {"lambda_global": 0.01, "lambda_local": 0.01, "hidden_dims": [64, 32]},     # Smaller
    {"lambda_global": 0.01, "lambda_local": 0.01, "hidden_dims": [256, 128, 64]}, # Larger
]
```

### Track Different Course

Change which course is visualized:

```python
TRACKED_COURSE_ID = 5  # Track course 5 instead of course 2
```

---

## Troubleshooting

### Model predicts only one class

**Problem**: All predictions collapse to Graduate
**Solution**:
- Reduce initial lambda values (try 0.001)
- Increase warmup period (try 500 epochs)
- Relax constraints (try 0.9, 0.7)

### Constraints never satisfied

**Problem**: Training runs for 1000 epochs without convergence
**Solution**:
- Check if constraints are too tight (impossible to satisfy)
- Increase lambda step (try 0.05)
- Check data distribution vs constraint limits

### Lambda keeps increasing but predictions don't change

**Problem**: λ grows to 10+ but violations persist
**Solution**:
- This was the "gradient flow bug" - should be fixed
- Verify model is in eval mode for constraint loss computation
- Check that `requires_grad=True` on constraint logits

---

## Next Steps

- See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for plot interpretation
- Check `src/training/` modules for code details
- Modify `config/experiment_config.py` for custom experiments

# Student Dropout Prediction with Transductive Constraint-Based Optimization

A machine learning system for predicting student outcomes (Dropout, Enrolled, Graduate) using transductive learning with constraint satisfaction. The system uses a custom optimization loss function to train neural networks while satisfying both global and local constraints.

## Overview

This project implements a constraint-aware neural network that learns to make predictions while respecting:
- **Global Constraints**: Overall limits on predictions across all students
- **Local Constraints**: Per-course limits on predictions within each course

The system includes a greedy constraint-based baseline for comparison.

---

## Project Structure

```
OptimizationLoss/
├── config/
│   └── experiment_config.py          # Experiment configuration and hyperparameters
├── data/
│   ├── dataset_train.csv             # Pre-split training data
│   └── dataset_test.csv              # Pre-split test data
├── experiments/
│   └── run_experiments.py            # Main experiment runner
├── src/
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── greedy_constraint_selector.py  # Greedy baseline implementation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── dataset.py                # Dataset class
│   ├── losses/
│   │   ├── __init__.py
│   │   └── transductive_loss.py      # Custom constraint loss function
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_net.py             # Neural network model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── constraints.py            # Constraint computation
│   │   ├── logging.py                # CSV logging and results saving
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── trainer.py                # Training loop
│   └── utils/
│       ├── __init__.py
│       └── visualization.py          # Training visualization
└── results/                          # Generated experiment results

```

---

## Key Concepts

### Transductive Learning
The model sees test data (without labels) during training and uses it to satisfy constraints. This differs from traditional inductive learning where the model never sees test data during training.

### Constraint Types

1. **Global Constraints**: Limit total predictions across all students
   - Example: "At most 20% of all students can be predicted as Dropout"

2. **Local Constraints**: Limit predictions within each course
   - Example: "At most 20% of students in Course X can be predicted as Dropout"

### Loss Function

The total loss combines three components:
```
L_total = L_CE + λ_global * L_global + λ_local * L_local
```

Where:
- **L_CE**: Cross-entropy loss on training data (accuracy)
- **L_global**: Global constraint violation loss (soft constraint)
- **L_local**: Local constraint violation loss (soft constraint)
- **λ_global, λ_local**: Adaptive weights (increase when constraints violated)

### Constraint Loss Formula

Uses rational saturation loss:
```
L_constraint = E / (E + K)
```
Where:
- E = excess predictions beyond constraint
- K = scaling factor

This produces a soft, differentiable constraint that:
- Equals 0 when constraint satisfied
- Approaches 1 as violations increase
- Allows gradient-based optimization

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch pandas numpy scikit-learn matplotlib openpyxl
```

---

## Usage

### 1. Configuration

Edit `config/experiment_config.py`:

```python
# Constraint pairs: (local_percentage, global_percentage)
CONSTRAINTS = [
    (0.9, 0.8), (0.9, 0.5), (0.8, 0.7), (0.8, 0.2),
    (0.7, 0.5), (0.6, 0.5), (0.4, 0.2), (0.5, 0.3)
]

# Neural network configurations
NN_CONFIGS = [
    {"lambda_global": 0.01, "lambda_local": 0.01, "hidden_dims": [128, 64, 32]}
]

# Training parameters
TRAINING_PARAMS = {
    'epochs': 10000,
    'batch_size': 64,
    'lr': 0.001,
    'dropout': 0.3
}

# Adaptive lambda parameters
WARMUP_EPOCHS = 250        # Train normally before constraint pressure
LAMBDA_STEP = 0.01         # Increment when constraints violated
CONSTRAINT_THRESHOLD = 1e-6  # Threshold for constraint satisfaction
```

### 2. Run Experiments

```bash
python experiments/run_experiments.py
```

This will:
1. Load pre-split train/test data
2. Compute constraints based on test data distribution
3. For each constraint configuration:
   - Train model with warmup period (250 epochs)
   - Run greedy benchmark at epoch 250
   - Continue training with constraint losses
   - Save results when constraints satisfied or max epochs reached

### 3. Results

Results are saved to `./results/` with folders per constraint configuration:

```
results/
├── constraints_0.4_0.2/
│   ├── training_log.csv                 # Per-epoch training metrics
│   ├── final_predictions.csv            # Sample-level predictions
│   ├── constraint_comparison.csv        # Constraint satisfaction analysis
│   ├── evaluation_metrics.csv           # Performance metrics
│   ├── benchmark_predictions.csv        # Greedy baseline predictions
│   ├── benchmark_metrics.csv            # Greedy baseline metrics
│   ├── benchmark_constraint_comparison.csv
│   └── visualizations/                  # Training plots
│       ├── loss_curves.png
│       ├── lambda_evolution.png
│       └── constraint_satisfaction.png
└── nn_results.json                      # Aggregate results
```

---

## Output Files Explained

### training_log.csv
Per-epoch metrics during training:
- Loss components (L_CE, L_global, L_local)
- Lambda values (adaptive weights)
- Constraint satisfaction status
- Prediction counts (hard and soft)

### final_predictions.csv
Sample-level predictions from optimized model:
- `Sample_Index`: Sample ID
- `True_Label`: Actual label
- `Predicted_Label`: Model prediction
- `Prob_Dropout`, `Prob_Enrolled`, `Prob_Graduate`: Probability vectors
- `Correct`: 1 if correct, 0 if wrong
- `Course_ID`: Course identifier

### constraint_comparison.csv
Per-course constraint analysis:
- `Course_ID`: Course identifier
- `Class`: Dropout, Enrolled, or Graduate
- `Constraint`: Constraint limit
- `Predicted`: Number predicted
- `Overprediction`: Amount exceeding constraint
- `Status`: OK, OVER, or N/A

### evaluation_metrics.csv
Comprehensive performance metrics:
- Overall accuracy
- Precision, Recall, F1-Score (macro and weighted)
- Per-class metrics
- Confusion matrix

### benchmark_*.csv
Same format as above, but for greedy baseline results.

### nn_results.json
Aggregate results across all configurations:
```json
{
  "nn_config1_transductive": {
    "(0.4, 0.2)": {
      "accuracy": 0.5543,
      "precision_macro": 0.6493,
      "recall_macro": 0.4009,
      "f1_macro": 0.3515,
      "training_time": 438.55,
      "benchmark_accuracy": 0.4823,
      "benchmark_precision_macro": 0.5621,
      "benchmark_recall_macro": 0.4156,
      "benchmark_f1_macro": 0.4234
    }
  }
}
```

---

## Algorithms

### Optimized Approach (Transductive Learning)

**Training:**
1. Warmup (epochs 1-250): Train normally with cross-entropy loss
2. Constraint training (epochs 251+):
   - Add global constraint loss (limit total predictions)
   - Add local constraint loss (limit per-course predictions)
   - Adaptively increase λ weights when constraints violated
3. Stop when constraints satisfied or max epochs reached

**Key Features:**
- Model learns constraint-aware representations
- Gradients flow through constraint losses
- Adaptive pressure increases when violations persist

### Greedy Baseline (Post-hoc Selection)

**Algorithm:**
1. Train model normally for 250 epochs (same as optimized warmup)
2. Get probability predictions for all test samples
3. **Phase 1 - Constrained Assignment:**
   - For each course:
     - For each constrained class (Dropout, Enrolled):
       - Sort samples by probability for that class
       - Assign top N samples (N = constraint limit)
       - Check global constraint before assigning
4. **Phase 2 - Remaining Samples:**
   - For each unassigned sample:
     - Try classes in order of probability
     - Check if assignment would violate global or local constraint
     - Assign to first class that doesn't violate
     - If all violate, assign to unlimited class (Graduate)

**Key Features:**
- Uses same warmup model (fair comparison)
- Respects all constraints
- Greedy selection (no learning)

---

## Constraint Computation

Constraints are computed from test set distribution:

```python
def compute_global_constraints(data, target_column, percentage):
    """
    Compute global constraints.

    Args:
        data: Test dataset
        target_column: Target column name
        percentage: Constraint percentage (0.0-1.0)

    Returns:
        [constraint_dropout, constraint_enrolled, constraint_graduate]
    """
    constraint = np.zeros(3)
    counts = data[target_column].value_counts()

    for class_id in counts.index:
        # NOTE: Division by 10 is intentional
        # percentage = 0.2 means 2% of test set (0.2 * 10% test split)
        constraint[class_id] = round(counts[class_id] * percentage / 10)

    constraint[2] = 1e10  # Graduate always unlimited
    return constraint.tolist()
```

**Example:**
- Test set: 1000 students
- Dropout percentage: 0.2
- True Dropouts in test: 400
- **Constraint:** 400 * 0.2 / 10 = 8 Dropouts

---

## Adaptive Lambda Weights

Lambda values increase when constraints are violated:

```python
if constraint_loss > CONSTRAINT_THRESHOLD:
    lambda_weight += LAMBDA_STEP
```

This creates adaptive pressure:
- Low λ initially (minimal constraint pressure)
- λ increases if model violates constraints
- λ stabilizes when constraints satisfied

---

## Visualization

Training visualizations are automatically generated:

1. **Loss Curves**: L_CE, L_global, L_local over epochs
2. **Lambda Evolution**: How λ_global and λ_local change
3. **Constraint Satisfaction**: Predictions vs constraints over time
4. **Prediction Distribution**: Class distribution evolution

---

## Interpreting Results

### Comparing Optimized vs Benchmark

**Optimized (Higher Accuracy):**
- Model learned constraint-aware representations
- Can predict accurately while satisfying constraints
- Better integration of constraint knowledge

**Benchmark (Lower Accuracy):**
- Forced to satisfy constraints post-hoc
- Model wasn't trained with constraint awareness
- Greedy selection may make suboptimal choices

### Constraint Satisfaction

Check `constraint_comparison.csv`:
- **Status: OK** = Constraint satisfied ✓
- **Status: OVER** = Constraint violated ✗
- **Status: N/A** = No constraint (unlimited)

Both approaches should show all "OK" or "N/A" statuses.

### Accuracy vs Constraint Trade-off

Tighter constraints (lower percentages) typically result in:
- Lower accuracy (harder to be correct within limits)
- Longer training time (more epochs to satisfy)
- Larger gap between optimized and baseline

---

## Troubleshooting

### Training doesn't converge
- Increase `epochs` in config
- Decrease `LAMBDA_STEP` (gentler constraint pressure)
- Adjust `lambda_global` and `lambda_local` in NN_CONFIGS

### Constraints never satisfied
- Check constraint computation (may be too restrictive)
- Increase `LAMBDA_STEP` (stronger pressure)
- Verify test data distribution is reasonable

### Benchmark fails with constraint violations
- This indicates a bug in the greedy selector
- Check `benchmark_constraint_comparison.csv` for details
- Verify Phase 2 constraint checking is working

### Out of memory
- Reduce `batch_size` in TRAINING_PARAMS
- Reduce `hidden_dims` in NN_CONFIGS
- Use smaller dataset

---

## Advanced Configuration

### Custom Constraints

Modify `src/training/constraints.py` to implement custom constraint logic:

```python
def compute_custom_constraints(data, target_column, config):
    # Your custom constraint logic
    return global_constraints, local_constraints
```

### Custom Loss Function

Modify `src/losses/transductive_loss.py` to implement custom loss formulations:

```python
class CustomTransductiveLoss(nn.Module):
    def forward(self, logits, y_true, group_ids):
        # Your custom loss logic
        return total_loss, ce_loss, global_loss, local_loss
```

### Custom Metrics

Add to `src/training/metrics.py`:

```python
def compute_custom_metrics(y_true, y_pred):
    # Your custom metrics
    return metrics_dict
```

---

## Citation

If you use this code in your research, please cite:

```
@software{constraint_optimization_loss,
  title={Transductive Constraint-Based Optimization for Student Dropout Prediction},
  author={Your Name},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

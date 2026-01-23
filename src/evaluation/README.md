# Evaluation Module

This module contains scripts for analyzing and comparing experiment results.

## generate_comparative_analysis.py

Analyzes all experiments and generates comprehensive comparison graphs organized by constraint level.

### Usage

Run from the `src/evaluation/` directory:

```bash
cd src/evaluation
python generate_comparative_analysis.py
```

Or from project root:

```bash
cd src/evaluation && python generate_comparative_analysis.py
```

### Output Structure

```
comparison_evaluations/
├── master_results.csv                    # All 144 experiments
├── cross_constraint_comparison.png       # Compares all constraints
│
├── constraint_0.5_0.3/                   # 48 experiments
│   ├── results_0.5_0.3.csv
│   ├── accuracy_by_learning_rate.png
│   ├── accuracy_by_lambda_strategy.png
│   ├── convergence_rate_by_factors.png
│   ├── model_comparison.png
│   ├── heatmap_lr_strategy.png
│   ├── accuracy_vs_convergence_speed.png
│   └── lr_vs_convergence_epochs.png
│
├── constraint_0.8_0.2/                   # 48 experiments
│   └── (same 7 graphs + CSV)
│
└── constraint_0.9_0.8/                   # 48 experiments
    └── (same 7 graphs + CSV)
```

### Graphs Generated

Each constraint folder contains 7 comparison graphs:

1. **accuracy_by_learning_rate.png** - Shows which learning rate achieves best accuracy for each model
2. **accuracy_by_lambda_strategy.png** - Compares lambda strategies' effectiveness for each model
3. **convergence_rate_by_factors.png** - Shows success rates by model, learning rate, and strategy
4. **model_comparison.png** - Direct comparison of models (speed and accuracy)
5. **heatmap_lr_strategy.png** - Heatmap showing accuracy for every LR×Strategy combination
6. **accuracy_vs_convergence_speed.png** - Scatter plot showing speed/accuracy tradeoff
7. **lr_vs_convergence_epochs.png** - Box plot showing convergence time distribution by learning rate

### Results Summary

- **constraint_0.5_0.3**: 46/48 converged (95.8%), avg 249 epochs, 59.6% accuracy
- **constraint_0.8_0.2**: 44/48 converged (91.7%), avg 405 epochs, 57.3% accuracy
- **constraint_0.9_0.8**: 48/48 converged (100%), avg 121 epochs, 72.0% accuracy

### CSV Files

**master_results.csv** contains all experiments with columns:
- Configuration: model, constraint, learning_rate, lambda_strategy, etc.
- Results: status, converged (boolean), final_epoch, test_accuracy
- Flags: global_constraint_satisfied, local_constraint_satisfied

**Constraint-specific CSVs** contain only experiments for that constraint (48 each).

### Analysis Tips

1. **For a specific constraint**, navigate to its folder and review all 7 graphs
2. **Compare constraints**, view `cross_constraint_comparison.png`
3. **Custom analysis**, load `master_results.csv` in pandas/Excel

```python
import pandas as pd

# Load all data
df = pd.read_csv('comparison_evaluations/master_results.csv')

# Filter by constraint
constraint_data = df[df['constraint'] == '0.9_0.8']

# Find best configuration
best = constraint_data[constraint_data['converged'] == True].nlargest(1, 'test_accuracy')
```

### Requirements

- matplotlib
- numpy

Install with: `pip install matplotlib numpy`

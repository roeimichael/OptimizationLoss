# Quick Start Guide - Constraint-Organized Analysis

## New Structure: Organized by Constraint

Results are now organized into 3 folders, one for each constraint level:
- `constraint_0.5_0.3/` - 48 experiments
- `constraint_0.8_0.2/` - 48 experiments
- `constraint_0.9_0.8/` - 48 experiments

Each folder contains 7 graphs + 1 CSV specific to that constraint.

---

## Quick Commands

### 1. View Console Summary
```bash
python3 print_analysis_summary.py
```
Shows top 10 configs, performance by hyperparameter, failed experiments.

### 2. Regenerate All Analysis
```bash
python3 generate_comparative_analysis.py
```
Creates 3 constraint folders with 7 graphs each + cross-constraint comparison.

### 3. Explore Specific Constraint
```python
import pandas as pd

# Load data for constraint 0.9_0.8 (easiest)
df = pd.read_csv('comparison_evaluations/constraint_0.9_0.8/results_0.9_0.8.csv')

# Find best configuration
best = df[df['converged'] == True].sort_values('test_accuracy', ascending=False).iloc[0]
print(f"Best for 0.9_0.8: {best['model']}, LR={best['learning_rate']}, Strategy={best['lambda_strategy']}")
print(f"Accuracy: {best['test_accuracy']:.4f}, Epochs: {best['final_epoch']}")
```

### 4. Compare Across All Constraints
```python
import pandas as pd

# Load master dataset
df = pd.read_csv('comparison_evaluations/master_results.csv')

# Group by constraint
for constraint in ['0.5_0.3', '0.8_0.2', '0.9_0.8']:
    subset = df[(df['constraint'] == constraint) & (df['converged'] == True)]
    avg_acc = subset['test_accuracy'].astype(float).mean()
    avg_epochs = subset['final_epoch'].mean()
    print(f"{constraint}: {avg_acc:.2%} accuracy, {avg_epochs:.0f} epochs")
```

---

## Understanding the Graphs

### Per-Constraint Folders (7 graphs each)

Each constraint folder has identical graph types but different data:

1. **accuracy_by_learning_rate.png** - Which LR works best?
2. **accuracy_by_lambda_strategy.png** - Which strategy works best?
3. **convergence_rate_by_factors.png** - Success rates breakdown
4. **model_comparison.png** - Model performance head-to-head
5. **heatmap_lr_strategy.png** - LR×Strategy combinations
6. **accuracy_vs_convergence_speed.png** - Speed vs accuracy tradeoff
7. **lr_vs_convergence_epochs.png** - Convergence time distribution

### Cross-Constraint Comparison

`cross_constraint_comparison.png` - Shows which constraint is:
- Fastest to converge
- Achieves highest accuracy
- Most challenging overall

---

## Key Findings at a Glance

### Constraint 0.9_0.8 (Easiest) ✅
- **100% success rate** (48/48)
- **Avg**: 121 epochs, 72% accuracy
- **Best**: TabularResNet + LR=0.0005 + transfer → 74.66%, 70 epochs

### Constraint 0.5_0.3 (Medium) ⚠️
- **95.8% success rate** (46/48)
- **Avg**: 249 epochs, 59.6% accuracy
- **2× slower** than 0.9_0.8

### Constraint 0.8_0.2 (Hardest) ⚠️
- **91.7% success rate** (44/48)
- **Avg**: 405 epochs, 57.3% accuracy
- **3× slower** than 0.9_0.8, most failures

---

## Common Tasks

### Find Best Config for Specific Constraint
```python
import pandas as pd

constraint = '0.8_0.2'  # Change as needed
df = pd.read_csv(f'comparison_evaluations/constraint_{constraint}/results_{constraint}.csv')

# Filter converged only
converged = df[df['converged'] == True].copy()
converged['test_accuracy'] = converged['test_accuracy'].astype(float)

# Top 5 by accuracy
top5 = converged.nlargest(5, 'test_accuracy')
print(top5[['model', 'learning_rate', 'lambda_strategy', 'test_accuracy', 'final_epoch']])
```

### Compare Models Within Constraint
```python
import pandas as pd

constraint = '0.9_0.8'
df = pd.read_csv(f'comparison_evaluations/constraint_{constraint}/results_{constraint}.csv')
converged = df[df['converged'] == True].copy()
converged['test_accuracy'] = converged['test_accuracy'].astype(float)

# Group by model
for model in ['BasicNN', 'FTTransformer', 'TabularResNet']:
    model_data = converged[converged['model'] == model]
    avg_acc = model_data['test_accuracy'].mean()
    avg_epochs = model_data['final_epoch'].mean()
    print(f"{model:16s}: {avg_acc:.2%} accuracy, {avg_epochs:6.1f} epochs")
```

### Analyze Failures for Constraint
```python
import pandas as pd

constraint = '0.8_0.2'  # Most failures
df = pd.read_csv(f'comparison_evaluations/constraint_{constraint}/results_{constraint}.csv')
failed = df[df['converged'] == False]

print(f"Failures for {constraint}:")
print(failed[['model', 'learning_rate', 'lambda_strategy']])
```

---

## Recommendations by Constraint

### Working with constraint_0.9_0.8:
```
✅ Use: TabularResNet + LR=0.0005 or 0.001
✅ Strategy: transfer or combined
✅ Expect: 66-100 epochs, 72-75% accuracy
```

### Working with constraint_0.5_0.3:
```
⚠️  Use: TabularResNet or FTTransformer
⚠️  LR: 0.0005 or 0.001 for faster training
⚠️  Expect: 200-300 epochs, 58-62% accuracy
⚠️  Avoid: BasicNN + LR=0.00005
```

### Working with constraint_0.8_0.2:
```
🔴 Use: TabularResNet (most reliable, 100% success)
🔴 LR: 0.0005 or 0.001 strongly recommended
🔴 Expect: 300-500 epochs, 55-60% accuracy
🔴 Avoid: BasicNN + LR=0.00005 (3/4 failed)
🔴 Consider: Increase max_epochs beyond 1000
```

---

## Navigation Tips

### Browse Graphs by Constraint
```bash
# View all graphs for constraint 0.9_0.8
ls -lh comparison_evaluations/constraint_0.9_0.8/*.png

# Compare same graph across constraints
for c in 0.5_0.3 0.8_0.2 0.9_0.8; do
    echo "=== Constraint $c ==="
    ls comparison_evaluations/constraint_$c/accuracy_by_learning_rate.png
done
```

### Load All CSVs at Once
```python
import pandas as pd
from pathlib import Path

# Load all constraint CSVs
data_by_constraint = {}
for constraint in ['0.5_0.3', '0.8_0.2', '0.9_0.8']:
    csv_path = f'comparison_evaluations/constraint_{constraint}/results_{constraint}.csv'
    data_by_constraint[constraint] = pd.read_csv(csv_path)

# Or just use master CSV
master = pd.read_csv('comparison_evaluations/master_results.csv')
```

---

## File Locations

### Master Dataset
`comparison_evaluations/master_results.csv` - All 144 experiments

### Constraint-Specific Data
- `comparison_evaluations/constraint_0.5_0.3/results_0.5_0.3.csv`
- `comparison_evaluations/constraint_0.8_0.2/results_0.8_0.2.csv`
- `comparison_evaluations/constraint_0.9_0.8/results_0.9_0.8.csv`

### Graphs
- Per-constraint: `comparison_evaluations/constraint_X/*.png` (7 graphs each)
- Cross-constraint: `comparison_evaluations/cross_constraint_comparison.png`

---

## Next Steps

1. **Start with constraint_0.9_0.8** folder - easiest, 100% success
2. **Review the 7 graphs** in that folder to understand patterns
3. **Check cross_constraint_comparison.png** to see relative difficulty
4. **Use master_results.csv** for custom analysis
5. **Read README.md** for detailed explanations

---

## Need Help?

- **Full documentation**: `README.md`
- **More analyses**: `SUGGESTED_COMPARISONS.md`
- **Regenerate everything**: `python3 generate_comparative_analysis.py`

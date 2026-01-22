# Quick Start Guide

## How to Use This Analysis

### 1. View Summary in Console
```bash
python3 print_analysis_summary.py
```
This shows:
- Top 10 configurations by accuracy
- Top 10 fastest configurations
- Best configuration per model
- Average performance by each hyperparameter
- Failed experiments details
- Overall recommendations

### 2. Explore Master CSV
```bash
# Open in Excel/Pandas
import pandas as pd
df = pd.read_csv('comparison_evaluations/master_results.csv')

# Filter converged experiments only
converged = df[df['converged'] == True]

# Sort by accuracy
best_accuracy = converged.sort_values('test_accuracy', ascending=False)

# Filter by specific model
basicnn_results = converged[converged['model'] == 'BasicNN']

# Filter by constraint level
easy_constraint = converged[converged['constraint'] == '0.9_0.8']
```

### 3. Regenerate All Graphs
```bash
python3 generate_comparative_analysis.py
```
Graphs will be saved to `comparison_evaluations/` folder.

**Note**: PNG files are in .gitignore, so regenerate graphs locally if needed.

### 4. View Graphs
All graphs are in `comparison_evaluations/`:
- `accuracy_by_learning_rate.png` - Which LR gives best accuracy per model
- `accuracy_by_lambda_strategy.png` - Which strategy gives best accuracy per model
- `convergence_rate_by_factors.png` - Success rates by model/LR/strategy
- `convergence_speed_by_constraint.png` - How fast each constraint converges
- `heatmap_model_constraint.png` - Which model works best with which constraint
- `accuracy_vs_convergence_speed.png` - Accuracy/speed tradeoff analysis
- `lr_vs_convergence_epochs.png` - Convergence time distribution by LR

---

## Key Insights at a Glance

### 🏆 Best Overall Configuration
**TabularResNet + LR=0.0005 + transfer strategy + constraint_0.9_0.8**
- Accuracy: **74.66%**
- Converged: **70 epochs**

### ⚡ Fastest Configuration
**TabularResNet + LR=0.001 + combined strategy + constraint_0.9_0.8**
- Converged: **66 epochs**
- Accuracy: **73.53%**

### 📊 Success Rate: **95.8%** (138/144)
Only 6 experiments failed - all with BasicNN at LR=0.00005

### 🎯 Learning Rate Recommendations
- **Fast training**: 0.001 or 0.0005 (converge in ~150 epochs)
- **Slower but thorough**: 0.0001 (~330 epochs)
- **Avoid**: 0.00005 (slow + lower accuracy)

### 🧩 Constraint Difficulty Ranking
1. **constraint_0.9_0.8** - Easiest (72% accuracy, 121 epochs)
2. **constraint_0.5_0.3** - Medium (59% accuracy, 249 epochs)
3. **constraint_0.8_0.2** - Hardest (57% accuracy, 405 epochs)

### 🤖 Model Recommendations
1. **TabularResNet** - Best accuracy (64.88%), fast (227 epochs)
2. **FTTransformer** - Balanced (61.77%, 244 epochs)
3. **BasicNN** - Slower (62.78%, 295 epochs)

### ⚙️ Lambda Strategy Recommendations
- **Combined** - Best overall (fastest + good accuracy)
- **Transfer** - Best for TabularResNet
- **Avoid balanced** - Higher failure rate

---

## Common Analysis Tasks

### Find best configuration for your constraint
```python
df = pd.read_csv('comparison_evaluations/master_results.csv')
df_converged = df[df['converged'] == True]

# For constraint 0.8_0.2 (hardest)
hard_constraint = df_converged[df_converged['constraint'] == '0.8_0.2']
best = hard_constraint.sort_values('test_accuracy', ascending=False).iloc[0]
print(f"Best for 0.8_0.2: {best['model']}, LR={best['learning_rate']}, {best['lambda_strategy']}")
```

### Compare models statistically
```python
from scipy import stats

basicnn = df_converged[df_converged['model'] == 'BasicNN']['test_accuracy']
transformer = df_converged[df_converged['model'] == 'FTTransformer']['test_accuracy']
resnet = df_converged[df_converged['model'] == 'TabularResNet']['test_accuracy']

# t-test
t_stat, p_value = stats.ttest_ind(resnet, basicnn)
print(f"ResNet vs BasicNN: p-value = {p_value}")
```

### Analyze failed experiments
```python
failed = df[df['converged'] == False]
print("Common failure patterns:")
print(failed[['model', 'learning_rate', 'lambda_strategy', 'constraint']].value_counts())
```

---

## Next Steps

1. **Review graphs** in `comparison_evaluations/`
2. **Read** `README.md` for detailed explanations
3. **Check** `SUGGESTED_COMPARISONS.md` for more analysis ideas
4. **Use** `master_results.csv` for your own custom analyses
5. **Run** `print_analysis_summary.py` anytime for quick insights

---

## Need More Analysis?

See `SUGGESTED_COMPARISONS.md` for 20+ additional comparisons we can create, including:
- Heatmaps for LR × Strategy interactions
- Statistical significance tests
- Pareto frontier analysis
- Failure pattern deep dive
- And more!

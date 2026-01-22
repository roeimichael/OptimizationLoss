# Comprehensive Comparative Analysis Results

## Overview
This analysis covers all **144 experiments** organized by constraint level:
- **3 Models**: BasicNN, FTTransformer, TabularResNet
- **3 Constraint Levels**: [0.5, 0.3], [0.8, 0.2], [0.9, 0.8] (48 experiments each)
- **4 Lambda Strategies**: balanced, combined, linear, transfer
- **4 Learning Rates**: 0.00005, 0.0001, 0.0005, 0.001

---

## 📁 Folder Structure

```
comparison_evaluations/
├── master_results.csv                    # All 144 experiments
├── cross_constraint_comparison.png       # Comparison across constraints
│
├── constraint_0.5_0.3/                   # 48 experiments
│   ├── results_0.5_0.3.csv              # Constraint-specific data
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

---

## 🎯 Overall Success Rate: **95.8%** (138/144 converged)

### By Constraint:
- **constraint_0.5_0.3**: 46/48 converged (95.8%)
- **constraint_0.8_0.2**: 44/48 converged (91.7%)
- **constraint_0.9_0.8**: 48/48 converged (100.0%) ✅

---

## 📊 Key Results by Constraint

### Constraint 0.9_0.8 (Easiest)
- **Success Rate**: 100% (48/48)
- **Avg Convergence**: 121 epochs
- **Avg Accuracy**: 72.00%
- **Best Config**: TabularResNet + LR=0.0005 + transfer → 74.66%, 70 epochs

### Constraint 0.5_0.3 (Medium)
- **Success Rate**: 95.8% (46/48)
- **Avg Convergence**: 249 epochs
- **Avg Accuracy**: 59.55%
- **Characteristics**: 2× slower than 0.9_0.8, lower accuracy

### Constraint 0.8_0.2 (Hardest)
- **Success Rate**: 91.7% (44/48)
- **Avg Convergence**: 405 epochs
- **Avg Accuracy**: 57.27%
- **Characteristics**: 3× slower than 0.9_0.8, most failures, lowest accuracy

---

## 📈 Graphs Explained

### Per-Constraint Folders (7 graphs each)

#### 1. **accuracy_by_learning_rate.png**
Shows which learning rate achieves best accuracy for each model within this constraint.
- **Use**: Identify optimal LR per model for this constraint level

#### 2. **accuracy_by_lambda_strategy.png**
Compares lambda strategies' effectiveness for each model within this constraint.
- **Use**: Choose best strategy per model for this constraint

#### 3. **convergence_rate_by_factors.png**
Three subplots showing success rates by model, LR, and strategy.
- **Use**: Identify most reliable configurations for this constraint

#### 4. **model_comparison.png**
Direct comparison of models: average epochs and average accuracy.
- **Use**: Quick overview of which model performs best for this constraint

#### 5. **heatmap_lr_strategy.png**
Heatmap showing accuracy for every LR × Strategy combination.
- **Use**: Find optimal hyperparameter pairs for this constraint

#### 6. **accuracy_vs_convergence_speed.png**
Scatter plot showing accuracy vs convergence speed tradeoff.
- **Use**: Identify configurations with best accuracy/speed balance

#### 7. **lr_vs_convergence_epochs.png**
Box plot showing distribution of convergence times per LR.
- **Use**: Understand variability and expected training time per LR

### Cross-Constraint Comparison

#### **cross_constraint_comparison.png**
Two plots comparing all three constraints:
- Average convergence speed per constraint
- Average accuracy per constraint
- **Use**: Understand relative difficulty of each constraint

---

## 🔍 How to Use This Analysis

### 1. **For a Specific Constraint**
Navigate to the appropriate folder:
```bash
cd comparison_evaluations/constraint_0.9_0.8/
```
Review all 7 graphs to find the best configuration for that constraint.

### 2. **Compare Constraints**
View `cross_constraint_comparison.png` to see which constraint is:
- Fastest to converge
- Achieves highest accuracy
- Most/least challenging

### 3. **Find Best Overall Configuration**
```python
import pandas as pd
df = pd.read_csv('comparison_evaluations/master_results.csv')
best = df[df['converged'] == True].sort_values('test_accuracy', ascending=False).iloc[0]
print(f"Best: {best['model']}, Constraint: {best['constraint']}, LR: {best['learning_rate']}, Strategy: {best['lambda_strategy']}")
# Output: Best: TabularResNet, Constraint: 0.9_0.8, LR: 0.0005, Strategy: transfer
```

### 4. **Analyze Failed Experiments**
```python
df = pd.read_csv('comparison_evaluations/master_results.csv')
failed = df[df['converged'] == False]
print(failed[['model', 'constraint', 'learning_rate', 'lambda_strategy']])
```

---

## 💡 Key Recommendations

### Overall Best Practices:
1. **Start with constraint_0.9_0.8** - 100% success, fastest, best accuracy
2. **Use TabularResNet** - best average performance across all constraints
3. **Use LR = 0.001 or 0.0005** - converge 2-3× faster than lower LRs
4. **Avoid BasicNN + LR=0.00005** - highest failure rate

### Constraint-Specific Recommendations:

**For constraint_0.9_0.8:**
- Best: TabularResNet + LR=0.0005 + transfer (74.66%, 70 epochs)
- Fastest: TabularResNet + LR=0.001 + combined (66 epochs, 73.53%)

**For constraint_0.5_0.3:**
- Expect ~250 epochs, ~60% accuracy
- TabularResNet or FTTransformer recommended
- Avoid BasicNN + LR=0.00005 + transfer (failed)

**For constraint_0.8_0.2:**
- Most challenging: expect ~400 epochs, ~57% accuracy
- TabularResNet most reliable (100% success)
- Avoid BasicNN + LR=0.00005 (3/4 failed)
- Consider increasing max epochs if working with this constraint

---

## 📝 CSV Files

### master_results.csv
Contains all 144 experiments with columns:
- **Config**: model, constraint, learning_rate, lambda_strategy, lambda_global, lambda_local, etc.
- **Results**: status, **converged (boolean)**, final_epoch, test_accuracy
- **Flags**: global_constraint_satisfied, local_constraint_satisfied

### Constraint-Specific CSVs
Each `constraint_X/results_X.csv` contains only experiments for that constraint (48 each).

---

## 🚀 Regenerate All Graphs

```bash
python3 generate_comparative_analysis.py
```

This will:
1. Extract data from all 144 experiments
2. Create 3 constraint folders
3. Generate 7 graphs per constraint (21 total)
4. Generate cross-constraint comparison
5. Save master CSV

**Time**: ~30-60 seconds

---

## 📊 Statistical Summary

### Convergence Epochs (all converged experiments):
- **Mean**: 254.3 epochs
- **Median**: 180.0 epochs
- **Range**: 66 - 985 epochs

### Test Accuracy (all converged experiments):
- **Mean**: 63.15%
- **Median**: 59.95%
- **Range**: 51.81% - 74.66%

### Success Rates:
- **By Model**: BasicNN (91.7%), FTTransformer (97.9%), TabularResNet (97.9%)
- **By LR**: 0.00005 (88.9%), 0.0001 (94.4%), 0.0005 (100%), 0.001 (100%)
- **By Strategy**: balanced (91.7%), combined (97.2%), linear (97.2%), transfer (97.2%)
- **By Constraint**: 0.5_0.3 (95.8%), 0.8_0.2 (91.7%), 0.9_0.8 (100%)

---

## 🎓 Next Steps

1. **Review constraint-specific graphs** to understand performance patterns
2. **Identify best configurations** for your target constraint
3. **Avoid known failure cases** (e.g., BasicNN + low LR on hard constraints)
4. **Use cross-constraint graph** to set expectations
5. **Explore CSVs** for custom analysis and deeper insights

For additional analysis ideas, see: `SUGGESTED_COMPARISONS.md`
For quick start guide, see: `QUICK_START.md`

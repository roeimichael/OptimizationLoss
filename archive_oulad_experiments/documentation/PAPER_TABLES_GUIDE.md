# Publication-Ready Tables Guide

This guide shows you the compact, paper-ready summaries generated from your 442 experiments.

## Quick Start - What to Use in Your Paper

**Recommended:** **Table 4 (Ultra-Compact)** - Use this as your main results table!

```
Method         Accuracy  Precision  Recall    F1
Heuristic      0.7624    0.7307     0.6873    0.7015
Saturated      0.7557    0.7075     0.6702    0.6816
Our Approach   0.7466    0.7285     0.6512    0.6695
```

This table shows:
- **3 approaches** compared on constraint [0.9, 0.8]
- **Best result per approach** (across all models and hyperparameters)
- **Clean, compact format** perfect for papers
- Ready-to-use LaTeX version included

---

## All Generated Tables

### Table 1: Model Performance Comparison (Aggregated)
**File:** `paper_table1_model_comparison.csv`

Shows mean and best accuracy for each approach/model combination across ALL constraint pairs.

**9 rows** (3 approaches × 3 models):

| Approach     | Model          | N  | Accuracy (Mean) | Accuracy (Best) | F1 (Mean) | F1 (Best) |
|--------------|----------------|----|-----------------|-----------------|-----------|---------  |
| Heuristic    | BasicNN        | 48 | 0.6508          | 0.7624          | 0.6156    | 0.7015    |
| Heuristic    | FTTransformer  | 48 | 0.6444          | 0.7579          | 0.6070    | 0.6912    |
| Heuristic    | TabularResNet  | 47 | 0.6385          | 0.7489          | 0.6035    | 0.6820    |
| Our Approach | BasicNN        | 48 | 0.6272          | 0.7353          | 0.5836    | 0.6200    |
| Our Approach | FTTransformer  | 48 | 0.6174          | 0.7240          | 0.5689    | 0.6110    |
| Our Approach | TabularResNet  | 48 | 0.6481          | 0.7466          | 0.5997    | 0.6695    |
| Saturated    | BasicNN        | 48 | 0.6482          | 0.7557          | 0.6025    | 0.6816    |
| Saturated    | FTTransformer  | 48 | 0.6235          | 0.7511          | 0.5765    | 0.6701    |
| Saturated    | TabularResNet  | 48 | 0.6607          | 0.7466          | 0.6126    | 0.6803    |

**Use this when:** You want to show how different models perform across all experiments.

---

### Table 2: Best Performance by Constraint Pair
**File:** `paper_table2_constraint_comparison.csv`

Shows the best result for each approach at each constraint level.

**9 rows** (3 constraint pairs × 3 approaches):

| Constraint | Approach     | Best Model     | Accuracy | F1-Score | Precision | Recall |
|------------|--------------|----------------|----------|----------|-----------|--------|
| [0.5, 0.3] | Our Approach | TabularResNet  | 0.7127   | 0.6623   | 0.6945    | 0.5705 |
| [0.5, 0.3] | Saturated    | TabularResNet  | 0.7127   | 0.6537   | 0.6701    | 0.5840 |
| [0.5, 0.3] | Heuristic    | BasicNN        | 0.6267   | 0.5639   | 0.7295    | 0.4937 |
| [0.8, 0.2] | Saturated    | BasicNN        | 0.6833   | 0.5951   | 0.6698    | 0.5426 |
| [0.8, 0.2] | Our Approach | BasicNN        | 0.6742   | 0.5052   | 0.4972    | 0.5149 |
| [0.8, 0.2] | Heuristic    | TabularResNet  | 0.5882   | 0.5642   | 0.7676    | 0.4482 |
| [0.9, 0.8] | Heuristic    | BasicNN        | 0.7624   | 0.7015   | 0.7307    | 0.6873 |
| [0.9, 0.8] | Saturated    | BasicNN        | 0.7557   | 0.6816   | 0.7075    | 0.6702 |
| [0.9, 0.8] | Our Approach | TabularResNet  | 0.7466   | 0.6695   | 0.7285    | 0.6512 |

**Key Insight:** Constraint [0.9, 0.8] achieves the best results across all approaches (more relaxed constraints = better performance).

**Use this when:** You want to show how performance varies with constraint strictness.

---

### Table 3: Detailed Comparison for Constraint [0.9, 0.8]
**File:** `paper_table3_constraint_0.9_0.8.csv`
**LaTeX:** `paper_table3_latex.tex`

Shows all approach/model combinations for the best-performing constraint pair.

**9 rows** (3 approaches × 3 models):

| Approach     | Model          | Accuracy | Precision | Recall | F1-Score | Best Config             |
|--------------|----------------|----------|-----------|--------|----------|-------------------------|
| Heuristic    | BasicNN        | 0.7624   | 0.7307    | 0.6873 | 0.7015   | lr=0.001, λ-balanced    |
| Heuristic    | FTTransformer  | 0.7579   | 0.7194    | 0.6778 | 0.6912   | lr=0.0001, λ-balanced   |
| Saturated    | BasicNN        | 0.7557   | 0.7075    | 0.6702 | 0.6816   | lr=0.001, λ-combined    |
| Saturated    | FTTransformer  | 0.7511   | 0.6945    | 0.6599 | 0.6701   | lr=0.0005, λ-transfer   |
| Heuristic    | TabularResNet  | 0.7489   | 0.7094    | 0.6693 | 0.6820   | lr=0.0001, λ-balanced   |
| Our Approach | TabularResNet  | 0.7466   | 0.7285    | 0.6512 | 0.6695   | lr=0.0005, λ-transfer   |
| Saturated    | TabularResNet  | 0.7466   | 0.7072    | 0.6680 | 0.6803   | lr=0.0005, λ-linear     |
| Our Approach | BasicNN        | 0.7353   | 0.6810    | 0.6126 | 0.6200   | lr=0.001, λ-combined    |
| Our Approach | FTTransformer  | 0.7240   | 0.6608    | 0.6034 | 0.6110   | lr=0.0001, λ-combined   |

**Key Insights:**
- BasicNN achieves best accuracy (0.7624) with Heuristic approach
- All approaches achieve > 0.72 accuracy with at least one model
- Balanced lambda strategy performs well across multiple configurations

**Use this when:** You want to show detailed results including which hyperparameters work best.

---

### Table 4: Ultra-Compact Summary ⭐ **RECOMMENDED**
**File:** `paper_table4_ultra_compact.csv`
**LaTeX:** `paper_table4_latex.tex`

The most compact table showing best result per approach for constraint [0.9, 0.8].

**3 rows only:**

| Method       | Accuracy | Precision | Recall | F1     |
|--------------|----------|-----------|--------|--------|
| Heuristic    | 0.7624   | 0.7307    | 0.6873 | 0.7015 |
| Saturated    | 0.7557   | 0.7075    | 0.6702 | 0.6816 |
| Our Approach | 0.7466   | 0.7285    | 0.6512 | 0.6695 |

**Why use this:**
- ✅ Clean and compact (perfect for space-constrained papers)
- ✅ Shows all key metrics in one glance
- ✅ Easy to understand comparison
- ✅ LaTeX version ready to paste into your paper
- ✅ Can be extended with additional baseline methods if needed

---

## LaTeX Integration

### Option 1: Direct Copy-Paste
Copy the contents of `paper_table4_latex.tex`:

```latex
\begin{table}[htbp]
\caption{Performance comparison across constraint satisfaction approaches on the OULAD dataset (Constraint: [0.9, 0.8]).}
\label{tab:results_comparison}
\begin{tabular}{lrrrr}
\toprule
Method & Accuracy & Precision & Recall & F1 \\
\midrule
Heuristic & 0.7624 & 0.7307 & 0.6873 & 0.7015 \\
Saturated & 0.7557 & 0.7075 & 0.6702 & 0.6816 \\
Our Approach & 0.7466 & 0.7285 & 0.6512 & 0.6695 \\
\bottomrule
\end{tabular}
\end{table}
```

Requires: `\usepackage{booktabs}` in your preamble.

### Option 2: Input from File
Place the .tex file in your paper directory and use:

```latex
\input{paper_table4_latex.tex}
```

---

## Key Statistics for Paper Text

From `paper_summary_stats.txt`:

- **Total experiments:** 442 completed
- **Models tested:** 3 (BasicNN, FTTransformer, TabularResNet)
- **Constraint pairs:** 3 ([0.5, 0.3], [0.8, 0.2], [0.9, 0.8])
- **Lambda strategies:** 4 (linear, balanced, combined, transfer)
- **Learning rates:** 4 (5e-05, 0.0001, 0.0005, 0.001)
- **Best overall accuracy:** 0.7624
- **Mean accuracy:** 0.6416

**Sample text for your paper:**

> We evaluated our approach across 442 experiments, testing 3 neural network architectures (BasicNN, FTTransformer, and TabularResNet) with 4 lambda adjustment strategies and 4 learning rates on 3 different constraint satisfaction levels. The best-performing configuration achieved 76.24% accuracy on the OULAD dataset with constraint pair [0.9, 0.8], using the Heuristic approach with BasicNN architecture (learning rate 0.001, balanced lambda strategy).

---

## Comparison with Baselines

To add your results to the standard ML baselines table:

### Example Extended Table

| Method                    | Accuracy | Precision | Recall | F1     |
|---------------------------|----------|-----------|--------|--------|
| **Constraint-based approaches (Ours)** ||||
| Heuristic                 | 0.7624   | 0.7307    | 0.6873 | 0.7015 |
| Saturated                 | 0.7557   | 0.7075    | 0.6702 | 0.6816 |
| Our Approach              | 0.7466   | 0.7285    | 0.6512 | 0.6695 |
| **Baselines (from literature)** ||||
| Random Forest             | 0.6850   | 0.6520    | 0.6230 | 0.6370 |
| XGBoost                   | 0.7010   | 0.6780    | 0.6450 | 0.6610 |
| Neural Network (baseline) | 0.6920   | 0.6650    | 0.6340 | 0.6490 |

_(Replace baseline numbers with actual values from your related work)_

---

## Customization Options

If you want to modify the tables for different constraints or add more details:

1. **Change constraint pair:** Edit line 104 in `generate_paper_summary.py`:
   ```python
   for constraint_pair in [[0.5, 0.3]]:  # Change to [0.5, 0.3] or [0.8, 0.2]
   ```

2. **Add more metrics:** Modify the column lists in the script to include:
   - Training time
   - Per-class metrics (Dropout, Enrolled, Graduate)
   - Hyperparameter details

3. **Change sorting:** Modify `.sort_values()` calls to sort by different metrics

---

## Files Generated Summary

### CSV Files (for analysis/processing)
1. `paper_table1_model_comparison.csv` - 9 rows
2. `paper_table2_constraint_comparison.csv` - 9 rows
3. `paper_table3_constraint_0.9_0.8.csv` - 9 rows
4. `paper_table4_ultra_compact.csv` - 3 rows ⭐

### LaTeX Files (ready to use)
5. `paper_table3_latex.tex` - Detailed table
6. `paper_table4_latex.tex` - Ultra-compact table ⭐

### Supporting Files
7. `paper_summary_stats.txt` - Key statistics
8. `generate_paper_summary.py` - Script to regenerate

---

## Next Steps

1. ✅ Use **Table 4** as your main results table in the paper
2. ✅ Copy the LaTeX code from `paper_table4_latex.tex`
3. ✅ Add comparison with baselines from your related work
4. ✅ Reference key statistics from `paper_summary_stats.txt` in your text
5. ✅ Optional: Use **Table 3** in appendix for detailed results

---

## Questions?

- Want different constraint pair? Modify line 104 in `generate_paper_summary.py`
- Want different metrics? Add columns in the script
- Want to compare specific models? Filter the dataframe before creating tables
- Need help with LaTeX formatting? The tables use standard `booktabs` package

All data is available in `experiment_results_summary.csv` for custom analysis!

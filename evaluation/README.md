# Evaluation Scripts

This folder contains all analysis and evaluation scripts for comparing experiment results.

## Scripts

### analyze_by_constraints.py
Groups and analyzes results by constraint configurations. Used when running multiple constraint settings.
Shows performance metrics grouped by constraint type.

### analyze_top5.py
Analyzes and ranks the top 5 performing model configurations.
Used for single constraint experiments or simple comparisons.

### comprehensive_results_analysis.py
Complete analysis of all experiment results:
- Filters out failed experiments
- Per-course constraint satisfaction analysis
- Visualizations of constraint adherence
- Class prediction distributions
- Comprehensive metrics comparison

Run: `python evaluation/comprehensive_results_analysis.py`

### compare_model_variants.py
Compares different model variants (baseline vs enhanced models):
- Auto-detects all variant folders in results/
- Generates comparison visualizations
- Shows performance improvements across variants

Run: `python evaluation/compare_model_variants.py`

## Usage

These scripts are called automatically at the end of experiment runs, or can be run manually:

```bash
# Comprehensive analysis of all results
python evaluation/comprehensive_results_analysis.py

# Compare model variants
python evaluation/compare_model_variants.py

# Constraint-specific analysis
python evaluation/analyze_by_constraints.py

# Top 5 analysis
python evaluation/analyze_top5.py
```

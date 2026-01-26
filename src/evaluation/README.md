# Evaluation Module

Scripts for analyzing and comparing experiment results.

## generate_comparative_analysis.py

Analyzes experiments and generates comparison graphs by constraint level.

### Usage

```bash
cd src/evaluation && python generate_comparative_analysis.py
```

### Output

- `comparison_evaluations/master_results.csv` - All experiment results
- Constraint-specific folders with comparison graphs
- Cross-constraint comparison visualizations

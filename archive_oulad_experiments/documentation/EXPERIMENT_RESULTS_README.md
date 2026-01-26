# Experiment Results Summary

This directory contains comprehensive summaries of all experiments conducted in this research project.

## Overview

**Total experiments: 443**

### Experiment Categories

1. **Heuristic** (144 experiments)
   - Baseline heuristic approach experiments
   - Testing different models, learning rates, and lambda strategies

2. **Our Approach** (144 experiments)
   - Our proposed constraint satisfaction approach
   - Same experimental setup as heuristic for fair comparison

3. **Saturated Approach** (144 experiments)
   - Alternative saturation-based approach
   - Same experimental setup for comparison

4. **Longer Saturation / Convergence Tests** (11 experiments)
   - TabularResNet with constraint [0.9, 0.8]
   - Testing sustained convergence with different window/required parameters
   - Evaluating impact of convergence checking strategy

## Files Generated

### 1. `experiment_results_summary.csv`
- **Format:** Single CSV file
- **Rows:** 443 experiments
- **Columns:** 34 (all experiment metadata and results)
- **Use case:** For programmatic analysis, data processing, or importing into other tools

### 2. `experiment_results_organized.xlsx`
- **Format:** Excel workbook with 9 sheets
- **Use case:** For manual analysis, visualization, and reporting

#### Sheet Descriptions:

1. **Overview** - Quick view of all experiments with key metrics
   - Columns: category, model, constraints, learning rate, lambda strategy, accuracy, F1, etc.

2. **Heuristic** - All heuristic approach experiments (144 rows)
   - Filtered view for easy comparison within this category

3. **Our_Approach** - All our approach experiments (144 rows)
   - Filtered view for easy comparison within this category

4. **Saturated_Approach** - All saturated approach experiments (144 rows)
   - Filtered view for easy comparison within this category

5. **Convergence_Tests** - All convergence testing experiments (11 rows)
   - Includes convergence_window and convergence_required parameters
   - All experiments with constraint [0.9, 0.8]

6. **Best_Results** - Best performing experiment per category/model/constraint combination
   - Easy identification of optimal hyperparameters
   - Sorted by category, model, and constraint pair

7. **Lambda_Strategy_Comp** - Statistical comparison of lambda strategies
   - Aggregated metrics: mean, median, std, min, max for each strategy
   - Strategies: linear, balanced, combined, transfer

8. **Learning_Rate_Comp** - Statistical comparison of learning rates
   - Aggregated metrics for each learning rate tested
   - Learning rates: 5e-05, 0.0001, 0.0005, 0.001

9. **Full_Data** - Complete dataset with all 34 columns
   - Reference sheet with all available information

## Key Findings (Quick Reference)

### Models Tested
- **BasicNN**: 144 experiments
- **FTTransformer**: 144 experiments
- **TabularResNet**: 155 experiments (144 + 11 convergence tests)

### Constraint Pairs
- **[0.5, 0.3]**: 144 experiments (50% local, 30% global)
- **[0.8, 0.2]**: 144 experiments (80% local, 20% global)
- **[0.9, 0.8]**: 155 experiments (90% local, 80% global)

### Lambda Strategies
- **Linear**: 119 experiments
- **Balanced**: 108 experiments
- **Combined**: 108 experiments
- **Transfer**: 108 experiments

### Best Overall Result
- **Accuracy**: 0.7624
- **Model**: BasicNN
- **Constraint**: [0.9, 0.8]
- **Learning Rate**: 0.001
- **Lambda Strategy**: balanced

### Accuracy Statistics (442 completed experiments)
- **Mean**: 0.6416
- **Median**: 0.6131
- **Range**: 0.5181 - 0.7624

## Column Definitions

### Experiment Identifiers
- `exp_category`: Experiment category (heuristic, our_approach, saturated_approach, longer_saturation)
- `exp_path`: Full path to experiment directory
- `model_name`: Neural network architecture used
- `constraint_local`: Local constraint percentage
- `constraint_global`: Global constraint percentage

### Hyperparameters
- `learning_rate`: Optimizer learning rate
- `batch_size`: Training batch size
- `epochs`: Total training epochs
- `warmup_epochs`: Warmup phase epochs (before constraint enforcement)
- `lambda_global_init`: Initial global lambda weight
- `lambda_local_init`: Initial local lambda weight
- `lambda_step`: Lambda increment per epoch when not satisfied
- `lambda_strategy`: Lambda adjustment strategy (linear/balanced/combined/transfer)
- `constraint_threshold`: Threshold for constraint satisfaction
- `hidden_dims`: Hidden layer dimensions (for applicable models)
- `dropout`: Dropout rate (for applicable models)

### Convergence Parameters (for convergence tests)
- `convergence_window`: Window size for sustained convergence check
- `convergence_required`: Number of satisfied epochs required within window

### Results
- `status`: Experiment status (completed, pending, failed)
- `used_cached_model`: Whether warmup model was reused from cache
- `training_time`: Total training time in seconds

### Evaluation Metrics
- `accuracy`: Overall test accuracy
- `precision_macro`: Macro-averaged precision
- `recall_macro`: Macro-averaged recall
- `f1_macro`: Macro-averaged F1 score

### Per-Class Metrics (when available)
- Precision, Recall, F1 for: Dropout, Enrolled, Graduate classes

## Usage Examples

### Python (using pandas)
```python
import pandas as pd

# Load all experiments
df = pd.read_csv('experiment_results_summary.csv')

# Filter by category
heuristic = df[df['exp_category'] == 'heuristic']

# Find best performing experiments
best = df.sort_values('accuracy', ascending=False).head(10)

# Compare lambda strategies
strategy_comparison = df.groupby('lambda_strategy')['accuracy'].agg(['mean', 'std', 'max'])

# Analyze convergence tests
convergence = df[df['convergence_window'].notna()]
```

### Excel
1. Open `experiment_results_organized.xlsx`
2. Use sheet tabs to navigate between views
3. Apply filters, sorting, and conditional formatting
4. Create pivot tables and charts for visualization

## Reproducibility

All experiments were conducted using:
- Fixed random seeds for reproducibility
- Consistent train/test splits
- Same hyperparameter ranges across approaches
- Identical constraint computation methods

To reproduce any experiment, refer to the `exp_path` column and the corresponding `config.json` file in that directory.

## Notes

### Convergence Tests
- All 11 convergence tests successfully completed
- All used constraint pair [0.9, 0.8] (larger constraints)
- Tested window sizes: 1, 5, 10, 20, 30
- Tested required satisfied epochs: 1, 2, 5, 7, 12, 14, 15, 20, 24, 27
- All achieved constraint satisfaction
- Mean accuracy: 0.7090

### Constraint Satisfaction
- Constraint pairs represent (local_percentage, global_percentage)
- Local constraints: per-course limits
- Global constraints: overall test set limits
- Graduate class (class 2) always unlimited

### Training Details
- Models trained with warmup phase before constraint enforcement
- Lambda weights increase when constraints violated
- Training stops when constraints satisfied or max epochs reached
- Cached warmup models reused when `base_model_id` matches

## Generated By

These files were generated using:
- `generate_experiment_summary.py` - Main extraction script
- `create_organized_summary.py` - Excel organization script

Last updated: 2026-01-26

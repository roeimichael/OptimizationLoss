# Comprehensive Multi-Constraint Experiment Analysis

**Total Experiments Analyzed**: 36
**Configurations Tested**: 5
**Constraint Settings**: 8
**Failed/Excluded**: 4

## Top 5 Performers

| Rank | Configuration | Constraint | Accuracy | Benchmark | Improvement |
|------|--------------|------------|----------|-----------|-------------|
| 1 | lambda_high | (0.9, 0.8) | 74.66% | 69.46% | +5.20% |
| 2 | dropout_high | (0.9, 0.8) | 73.53% | 70.14% | +3.39% |
| 3 | very_deep_baseline | (0.9, 0.8) | 72.62% | 67.87% | +4.75% |
| 4 | very_deep_extreme_lambda | (0.9, 0.8) | 72.40% | 66.52% | +5.88% |
| 5 | dropout_high | (0.8, 0.7) | 72.17% | 70.14% | +2.03% |

## Best Performer by Constraint

| Constraint | Best Config | Accuracy | Improvement |
|-----------|-------------|----------|-------------|
| (0.4, 0.2) | arch_deep | 56.79% | +0.23% |
| (0.5, 0.3) | arch_deep | 60.86% | +0.91% |
| (0.6, 0.5) | lambda_high | 67.19% | -0.23% |
| (0.7, 0.5) | very_deep_baseline | 67.87% | +2.94% |
| (0.8, 0.2) | arch_deep | 56.79% | +0.91% |
| (0.8, 0.7) | dropout_high | 72.17% | +2.03% |
| (0.9, 0.5) | arch_deep | 68.55% | +3.84% |
| (0.9, 0.8) | lambda_high | 74.66% | +5.20% |

## Performance by Model Configuration

| Configuration | Avg Accuracy | Win Rate | Avg Improvement |
|--------------|--------------|----------|----------------|
| arch_deep | 64.90% | 100% (8/8) | +2.35% |
| dropout_high | 63.46% | 50% (4/8) | -0.99% |
| lambda_high | 64.65% | 50% (4/8) | +0.79% |
| very_deep_baseline | 64.93% | 88% (7/8) | +2.09% |
| very_deep_extreme_lambda | 66.46% | 75% (3/4) | +3.34% |

## Constraint Satisfaction Summary

- **Average Constraint Satisfaction Rate**: 100.0%
- **Perfect Satisfaction (100%)**: 36/36 experiments
- **Total Constraints Evaluated**: 1152
- **Total Satisfied**: 1152

## Per-Course Constraint Analysis

- **Total Violations**: 0
- **Total Overpredictions**: 0

## Class Distribution Summary

**Best Model** (lambda_high @ (0.9, 0.8)):
- Dropout Predictions: 109 (limit: 397)
- Enrolled Predictions: 44 (limit: 353)
- Graduate Predictions: 289 (unlimited)
- Dropout Within Limit: Yes
- Enrolled Within Limit: Yes

## Key Insights

1. **Perfect Performer**: arch_deep beat the benchmark in ALL 8 tested constraints
2. **Best Constraint Setting**: (0.9, 0.8) with 72.99% average accuracy
3. **Most Consistent Config**: very_deep_extreme_lambda with lowest variance (sigma=0.0482)
4. **Constraint Satisfaction**: 100.0% average satisfaction rate demonstrates effective transductive learning

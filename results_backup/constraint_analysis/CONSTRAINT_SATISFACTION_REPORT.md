# Constraint Satisfaction Analysis Report
## Executive Summary
- **Total Successful Experiments**: 36
- **Failed Experiments Excluded**: 4
- **Courses Analyzed**: 17
- **Constraint Settings Tested**: 8

## Failed Experiments (Excluded from Analysis)
| Configuration | Constraint | Reason |
|--------------|------------|--------|
| very_deep_extreme_lambda | (0.8, 0.7) | Benchmark baseline failed |
| very_deep_extreme_lambda | (0.8, 0.2) | Benchmark baseline failed |
| very_deep_extreme_lambda | (0.7, 0.5) | Benchmark baseline failed |
| very_deep_extreme_lambda | (0.4, 0.2) | Benchmark baseline failed |

## Overall Constraint Satisfaction
| Constraint | Avg Satisfaction | Total Satisfied | Total Violated | Total Overprediction |
|-----------|------------------|-----------------|----------------|---------------------|
| (0.4, 0.2) | 100.0% | 128 | 0 | 0 |
| (0.5, 0.3) | 100.0% | 160 | 0 | 0 |
| (0.6, 0.5) | 100.0% | 160 | 0 | 0 |
| (0.7, 0.5) | 100.0% | 128 | 0 | 0 |
| (0.8, 0.2) | 100.0% | 128 | 0 | 0 |
| (0.8, 0.7) | 100.0% | 128 | 0 | 0 |
| (0.9, 0.5) | 100.0% | 160 | 0 | 0 |
| (0.9, 0.8) | 100.0% | 160 | 0 | 0 |

## Best Performing Configurations by Constraint Satisfaction
| Configuration | Avg Satisfaction Rate |
|--------------|----------------------|
| arch_deep | 100.0% |
| dropout_high | 100.0% |
| lambda_high | 100.0% |
| very_deep_baseline | 100.0% |
| very_deep_extreme_lambda | 100.0% |

## Course-Level Insights
### Most Challenging Courses (Lowest Satisfaction Rates)
| Course ID | Avg Satisfaction | Total Violations | Total Overprediction |
|----------|------------------|------------------|---------------------|
| 1 | 100.0% | 0 | 0 |
| 2 | 100.0% | 0 | 0 |
| 3 | 100.0% | 0 | 0 |
| 4 | 100.0% | 0 | 0 |
| 5 | 100.0% | 0 | 0 |

### Best Performing Courses (Highest Satisfaction Rates)
| Course ID | Avg Satisfaction | Total Violations | Total Overprediction |
|----------|------------------|------------------|---------------------|
| 13 | 100.0% | 0 | 0 |
| 14 | 100.0% | 0 | 0 |
| 15 | 100.0% | 0 | 0 |
| 16 | 100.0% | 0 | 0 |
| 17 | 100.0% | 0 | 0 |

## Key Findings

1. **Best Constraint Setting**: (0.4, 0.2) with 100.0% average satisfaction
2. **Most Challenging Setting**: (0.9, 0.8) with 100.0% average satisfaction
3. **Most Robust Configuration**: arch_deep consistently satisfies constraints best
4. **Total Violations Across All Experiments**: 0
5. **Total Overpredictions**: 0

## Recommendations

- For maximum constraint satisfaction, use **(0.4, 0.2)** constraints
- Deploy **arch_deep** configuration for best robustness across courses
- Pay special attention to courses 1, 2, 3 which show lower satisfaction rates

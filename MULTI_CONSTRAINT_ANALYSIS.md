# Comprehensive Experiment Analysis: Multi-Constraint Study

## Executive Summary

Analyzed **40 total experiments** across **8 constraint configurations** and **5 hyperparameter settings**, testing student dropout prediction with transductive learning and the new round-robin benchmark selector.

## Key Findings

### 1. Constraint Strictness Has Massive Impact

**Looser constraints = Better performance**

| Constraint Type | Avg Accuracy | Avg Improvement | Key Insight |
|----------------|--------------|-----------------|-------------|
| (0.9, 0.8) - Very Loose | **73.0%** | +4.66% | Best overall performance |
| (0.8, 0.7) - Loose | **70.0%** | +15.07% | Best improvements (skewed by failures) |
| (0.9, 0.5) - Mixed Loose | 68.1% | +3.35% | Good balance |
| (0.5, 0.3) - Strict | 59.3% | -0.99% | Performance drops significantly |
| (0.4, 0.2) - Very Strict | 54.3% | +8.78% | Poorest absolute performance |

**Critical Insight**: Loosening constraints from (0.5, 0.3) to (0.9, 0.8) improves accuracy by **13.7 percentage points** (59.3% → 73.0%).

### 2. Top Performing Configurations

**Absolute Best Performers:**

1. **lambda_high** @ (0.9, 0.8): **74.7%** (+5.20% vs baseline)
2. **dropout_high** @ (0.9, 0.8): **73.5%** (+3.39%)
3. **very_deep_baseline** @ (0.9, 0.8): **72.6%** (+4.75%)
4. **dropout_high** @ (0.8, 0.7): **72.2%** (+2.03%)
5. **lambda_high** @ (0.8, 0.7): **71.7%** (+3.85%)

**Key Observation**: All top 5 performers use very loose constraints (0.8+ local, 0.7+ global).

### 3. Configuration Robustness Across Constraints

**Average performance across ALL 8 constraint settings:**

| Config | Avg Accuracy | Avg Improvement | Stability |
|--------|--------------|-----------------|-----------|
| **very_deep_baseline** | 64.9% | +2.09% | ⭐ Most robust |
| **arch_deep** | 64.9% | +2.35% | ⭐ Most robust |
| **lambda_high** | 64.7% | +0.79% | ⭐ Consistent |
| **dropout_high** | 63.5% | -0.99% | Good |
| **very_deep_extreme_lambda** | 61.4% | +29.86%* | ⚠️ Unstable |

**very_deep_extreme_lambda (λ=1.0)** shows extreme instability:
- **Fails completely** under strict constraints (training time < 2s, missing benchmarks)
- Works well under loose constraints
- Improvement metric inflated due to benchmark failures

### 4. Constraint Configuration Insights

#### Best for Maximum Accuracy
- **Constraint (0.9, 0.8)** with **lambda_high**: 74.7%
- Model has freedom to learn patterns without over-constraining

#### Best for Balanced Constraints
- **Constraint (0.5, 0.3)** with **arch_deep**: 60.9% (+0.91%)
- Reasonable middle ground between accuracy and constraint satisfaction

#### Worst Performers
- Very strict constraints (0.4, 0.2): Average only 54.3%
- High lambda with strict constraints: Often fails or performs poorly

### 5. Hyperparameter Insights

#### Lambda Values (Constraint Penalty)
- **λ = 0.01** (baseline): Robust, works across all constraints
- **λ = 0.1** (lambda_high): Best under loose constraints (74.7% peak)
- **λ = 1.0** (extreme_lambda): **UNSTABLE** - fails under strict constraints, works under loose

#### Network Depth
- **Deeper networks** (5 layers) slightly outperform shallower (3-4 layers) on average
- Difference is marginal: 64.9% vs 64.7% average accuracy

#### Dropout Regularization
- **High dropout (0.5)** performs best under very loose constraints
- **Standard dropout (0.3)** more consistent across constraint settings

### 6. Improvement over Benchmark

**Real improvements** (excluding extreme_lambda failures):

- Best: **+5.20%** (lambda_high @ 0.9/0.8)
- Worst: **-3.62%** (lambda_high @ 0.5/0.3)

**Pattern**:
- Loose constraints: +2% to +5% improvement
- Strict constraints: -1% to -3% (negative improvement)

### 7. Performance Ceiling Observation

**Performance appears to plateau around 74-75%** even with optimal settings:
- Best result: 74.7% (lambda_high)
- Top 10 all cluster between 68-75%
- Suggests inherent difficulty in the prediction task

## Recommendations

### For Production Use:

**If constraint flexibility is possible:**
- Use **lambda_high** with **(0.9, 0.8) constraints**
- Expected accuracy: ~74-75%
- Improvement over baseline: +5%

**If balanced constraints needed:**
- Use **arch_deep** with **(0.7, 0.5) constraints**
- Expected accuracy: ~67%
- Robust and stable performance

**If strict constraints required:**
- Use **arch_deep** with **(0.5, 0.3) constraints**
- Expected accuracy: ~61%
- Only config showing positive improvement under strict settings

### Avoid:

- **very_deep_extreme_lambda (λ=1.0)** - Too unstable
- **Very strict constraints** (0.4, 0.2) unless absolutely required
- **High lambda (λ=0.1)** with strict constraints - shows negative improvements

## Round-Robin Benchmark Impact

Results use the new **round-robin benchmark selection method**, which provides fairer baseline comparisons across courses by:
- Allocating samples across courses in round-robin fashion
- Preventing any single course from monopolizing allocation
- Better probability utilization (highest probability samples allocated first globally)

This improved benchmark is more challenging than the previous sequential method, explaining some of the modest or negative improvements under strict constraints.

## Detailed Results

All detailed results organized by constraint configuration:

```
results/
├── constraint_0.4_0.2/      (Very strict)
├── constraint_0.5_0.3/      (Strict)
├── constraint_0.6_0.5/      (Moderate)
├── constraint_0.7_0.5/      (Moderate-loose)
├── constraint_0.8_0.2/      (Mixed)
├── constraint_0.8_0.7/      (Loose)
├── constraint_0.9_0.5/      (Very loose local, moderate global)
├── constraint_0.9_0.8/      (Very loose)
└── constraint_comparison/   (Cross-constraint comparison graphs)
```

Each constraint folder contains:
- 5 experiment folders with full metrics and visualizations
- `accuracy_comparison.png` - Performance comparison graph
- `CONSTRAINT_REPORT.md` - Detailed analysis for that constraint

## Conclusion

The multi-constraint study reveals that **constraint strictness is the dominant factor** affecting model performance - far more important than hyperparameter tuning.

**Key Takeaway**: Models can achieve 74% accuracy under loose constraints but drop to 54% under strict ones, representing a **20 percentage point swing**. This is a much larger effect than any hyperparameter optimization could provide.

For practical deployment, the choice should be driven primarily by **acceptable constraint levels** rather than extensive hyperparameter search.

---

**Generated**: 2026-01-01
**Total Experiments**: 40 (8 constraints × 5 configs)
**Best Result**: 74.7% (lambda_high @ 0.9/0.8)
**Worst Result**: 50.0% (very_deep_extreme_lambda @ 0.4/0.2 - failed)

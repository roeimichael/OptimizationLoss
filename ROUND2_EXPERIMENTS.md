# Round 2 Hyperparameter Experiments

## Overview

Based on Round 1 analysis, we identified that:
- **Deep architectures (4 layers)** achieved +2.72% over benchmark
- **High lambda values (0.1)** achieved +0.45% over benchmark
- **Learning rate 0.001 is critical** (extremes failed badly)

Round 2 pushes these successful strategies further with 15 advanced configurations.

---

## Experiment Design Philosophy

### Core Strategy: Combine Winners + Push Boundaries

1. **Combine successful strategies** (deep + high lambda)
2. **Go deeper** (5, 6, 7 layers) to test if more capacity helps
3. **Increase constraint pressure** (lambda up to 1.0) for deep networks
4. **Test width variations** to find optimal architecture shape
5. **Fine-tune regularization** for deeper networks

### Key Hypothesis

**If depth is the key to beating the benchmark, then more depth should yield even better results.**

---

## Experiment Configurations

### Group 1: Best Combined (1 config)

#### 1. best_combined ⭐ **TOP PRIORITY**
**Purpose:** Combine the two winning strategies from Round 1
- Architecture: [256, 128, 64, 32] (4 layers - **proven winner**)
- Lambda: 0.1, 0.1 (high - **proven winner**)
- Other params: lr=0.001, dropout=0.3, batch=64

**Expected:** This should be our strongest candidate. Combining deep arch (+2.72%) with high lambda (+0.45%) could yield +3%+ improvement.

**Rationale:** Both strategies worked independently; combining them should compound benefits.

---

### Group 2: Very Deep Architectures (3 configs)

#### 2. very_deep_baseline
**Purpose:** Test if 5 layers (with baseline lambdas) beats 4 layers
- Architecture: [512, 256, 128, 64, 32] (5 layers)
- Lambda: 0.01, 0.01 (baseline)
- **Comparison:** vs arch_deep (4 layers, baseline lambda) = 61.09%

#### 3. very_deep_high_lambda
**Purpose:** Best of both worlds - 5 layers + high lambda
- Architecture: [512, 256, 128, 64, 32] (5 layers)
- Lambda: 0.1, 0.1 (high)
- **Expected:** Could be our BEST result if depth keeps helping

#### 4. very_deep_very_high_lambda
**Purpose:** Test if 5 layers can handle very aggressive constraint pressure
- Architecture: [512, 256, 128, 64, 32] (5 layers)
- Lambda: 0.5, 0.5 (5x baseline)
- **Rationale:** More capacity might allow stronger constraint enforcement

---

### Group 3: Ultra Deep Architectures (2 configs)

#### 5. ultra_deep_baseline
**Purpose:** Push depth to 6 layers with baseline lambda
- Architecture: [512, 256, 128, 64, 32, 16] (6 layers)
- Lambda: 0.01, 0.01 (baseline)
- **Risk:** May be too deep, could struggle to optimize

#### 6. ultra_deep_high_lambda
**Purpose:** Maximum depth + high lambda
- Architecture: [512, 256, 128, 64, 32, 16] (6 layers)
- Lambda: 0.1, 0.1 (high)
- **Expected:** Either our best result or will overfit/struggle to train

---

### Group 4: Extreme Lambda Experiments (3 configs)

#### 7. deep_very_high_lambda
**Purpose:** Test if proven architecture can handle 5x lambda
- Architecture: [256, 128, 64, 32] (4 layers - proven)
- Lambda: 0.5, 0.5 (5x baseline)
- **Rationale:** Strong constraint pressure might sacrifice some accuracy for better constraint satisfaction

#### 8. deep_extreme_lambda
**Purpose:** Push lambda to the limit with proven architecture
- Architecture: [256, 128, 64, 32] (4 layers - proven)
- Lambda: 1.0, 1.0 (10x baseline)
- **Risk:** May over-prioritize constraints and hurt accuracy

#### 9. very_deep_extreme_lambda
**Purpose:** Maximum capacity + maximum constraint pressure
- Architecture: [512, 256, 128, 64, 32] (5 layers)
- Lambda: 1.0, 1.0 (10x baseline)
- **Expected:** Either excellent constraint satisfaction or poor accuracy

---

### Group 5: Wide Deep Architectures (2 configs)

#### 10. wide_deep_baseline
**Purpose:** Test if width + depth (without going too deep) helps
- Architecture: [512, 256, 128, 64] (4 layers, wider)
- Lambda: 0.01, 0.01 (baseline)
- **Comparison:** vs arch_deep [256,128,64,32] = 61.09%

#### 11. wide_deep_high_lambda
**Purpose:** Wide deep + high lambda
- Architecture: [512, 256, 128, 64] (4 layers, wider)
- Lambda: 0.1, 0.1 (high)
- **Rationale:** Width + depth + pressure might be optimal combination

---

### Group 6: Regularization Tuning (2 configs)

#### 12. deep_low_dropout
**Purpose:** Test if deep networks need less dropout with high lambda
- Architecture: [256, 128, 64, 32] (4 layers - proven)
- Lambda: 0.1, 0.1 (high)
- Dropout: 0.2 (low)
- **Rationale:** Constraint loss may provide enough regularization

#### 13. very_deep_high_dropout
**Purpose:** Test if very deep networks need more regularization
- Architecture: [512, 256, 128, 64, 32] (5 layers)
- Lambda: 0.1, 0.1 (high)
- Dropout: 0.4 (high)
- **Rationale:** Deeper networks may overfit without stronger dropout

---

### Group 7: Batch Size Variation (1 config)

#### 14. deep_small_batch_high_lambda
**Purpose:** Test if smaller batches help deep networks escape local minima
- Architecture: [256, 128, 64, 32] (4 layers - proven)
- Lambda: 0.1, 0.1 (high)
- Batch size: 32 (small)
- **Rationale:** More gradient noise for exploration

---

### Group 8: Maximum Complexity (1 config)

#### 15. mega_deep ⭐ **WILD CARD**
**Purpose:** Push everything to the limit - 7 layers, 1024 starting width
- Architecture: [1024, 512, 256, 128, 64, 32, 16] (7 layers!)
- Lambda: 0.1, 0.1 (high)
- Dropout: 0.35 (slightly higher for regularization)
- **Expected:** Either our absolute best or will fail to train properly
- **Risk:** Very high - may not fit in memory or take too long to train

---

## Expected Results Ranking

### Most Likely to Beat Benchmark (>60%):

1. **best_combined** (deep arch + high lambda) - **HIGHEST CONFIDENCE**
2. **very_deep_high_lambda** (5 layers + high lambda)
3. **wide_deep_high_lambda** (wide + deep + high lambda)
4. **deep_low_dropout** (proven arch + high lambda + optimized dropout)
5. **very_deep_baseline** (5 layers alone)

### Moderate Confidence (55-60%):

6. **ultra_deep_high_lambda** (6 layers + high lambda)
7. **very_deep_very_high_lambda** (5 layers + very high lambda)
8. **wide_deep_baseline** (wide deep baseline)
9. **very_deep_high_dropout** (5 layers + high dropout)
10. **deep_small_batch_high_lambda** (proven + small batch)

### Experimental/Risky (<55%):

11. **deep_very_high_lambda** (may over-prioritize constraints)
12. **ultra_deep_baseline** (may be too deep without high lambda)
13. **deep_extreme_lambda** (lambda=1.0 may hurt accuracy)
14. **very_deep_extreme_lambda** (both extreme)
15. **mega_deep** (may not train well or take forever)

---

## Success Criteria

### Tier 1 Success: Beat Benchmark by >3%
- **Target:** >61% accuracy (benchmark ≈58%)
- **Candidates:** best_combined, very_deep_high_lambda
- **Implication:** Transductive optimization is clearly superior

### Tier 2 Success: Beat Benchmark by 1-3%
- **Target:** 59-61% accuracy
- **Implication:** Optimization works but requires careful tuning

### Tier 3 Marginal: Beat Benchmark by <1%
- **Target:** 58-59% accuracy
- **Implication:** Questionable if complexity is worth it

### Failure: Don't Beat Benchmark
- **Result:** <58% accuracy
- **Implication:** Abandon transductive approach, use benchmark

---

## Key Insights to Watch

### 1. Depth vs. Width
Compare:
- deep (4 layers, 256 start): 61.09% (Round 1)
- very_deep (5 layers, 512 start): ?
- ultra_deep (6 layers, 512 start): ?
- mega_deep (7 layers, 1024 start): ?

**Question:** Does depth keep helping, or is there a limit?

### 2. Lambda Scaling
Compare across same architecture:
- baseline (0.01): 61.09%
- high (0.1): ?
- very high (0.5): ?
- extreme (1.0): ?

**Question:** What's the optimal lambda for deep networks?

### 3. Architecture Shape
Compare:
- deep [256,128,64,32]: 61.09%
- very_deep [512,256,128,64,32]: ?
- wide_deep [512,256,128,64]: ?

**Question:** Is pyramid (many layers, narrow) or trapezoid (fewer layers, wide) better?

### 4. Regularization Needs
Compare dropout for deep networks:
- low (0.2): ?
- baseline (0.3): 61.09%
- high (0.4): ?

**Question:** Do deeper networks need more or less regularization?

---

## Running the Experiments

```bash
# Run all 15 configurations
python experiments/run_experiments.py

# Results will be saved to:
# - results/hyperparam_{config_name}/ (individual folders)
# - results/nn_results.json (aggregate results)
```

**Estimated Time:** ~1.5-2 hours for all 15 configs (assuming ~6-8 min each)

**Special Note:** `mega_deep` may take 10-15 minutes due to 7 layers.

---

## Post-Experiment Analysis Plan

After results are in, analyze:

1. **Best absolute accuracy** - which config beat benchmark most?
2. **Depth scaling** - how does performance change with layers?
3. **Lambda scaling** - how does performance change with constraint pressure?
4. **Constraint satisfaction** - do deeper networks satisfy constraints better?
5. **Training dynamics** - did deeper networks converge properly?

Create ranked list and identify:
- **The Winner:** Best overall configuration
- **Key Patterns:** What architectural choices matter most?
- **Final Recommendation:** Go/no-go on transductive optimization

---

## Next Steps if Successful

If we beat benchmark by >3%:
1. Test on different constraint configurations (0.4/0.2, 0.6/0.4, etc.)
2. Investigate why depth helps (visualization, activation analysis)
3. Write up findings as technical contribution
4. Consider submitting to conference/journal

If marginal (1-3%):
1. Consider hybrid approach (train deep, apply greedy)
2. Test ensemble methods
3. Investigate constraint satisfaction quality

If failure (<1%):
1. Use simple greedy benchmark
2. Document findings as negative result
3. Focus engineering effort elsewhere

# Static Lambda V2 Experiment Results Analysis

## Executive Summary

**Total Experiments:** 48 completed (3 models × 4 constraints × 4 lambda values)
**Success Rate:** 79.2% (38/48)
**CRITICAL FINDING:** V2 still exhibits severe overfitting to constraints - successful experiments predict **95.8% Graduate on average** (same problem as V1!)

---

## 1. Success Rate by Lambda Value

| Lambda Value | Success | Failed | Total | Success Rate | **Avg Graduate %** |
|-------------|---------|--------|-------|--------------|-------------------|
| **0.02** | 6 | 6 | 12 | **50.0%** | 97.8% ❌ |
| **0.03** | 10 | 2 | 12 | **83.3%** | 93.6% ❌ |
| **0.05** | 11 | 1 | 12 | **91.7%** | 95.8% ❌ |
| **0.07** | 11 | 1 | 12 | **91.7%** | 96.6% ❌ |

### Key Insight: THE PROBLEM PERSISTS!

**All lambda values in V2 (0.02-0.07) still cause severe overfitting:**
- Even λ=0.02 (lowest V2) → 97.8% Graduate predictions
- λ=0.05-0.07 achieve 91.7% success but with 95-97% Graduate predictions
- Models satisfy constraints by predicting almost everyone graduates, not by learning

**Comparison with V1:**
- V1 λ=0.01: 33% success, 86.7% Graduate ← Best prediction quality
- V1 λ=0.1: 56% success, 93.2% Graduate ← Marginal
- **V2 λ=0.02-0.07: 50-92% success, 93.6-97.8% Graduate ← STILL OVERFITTING!**
- V1 λ=1.0: 100% success, 95.1% Graduate ← Same problem

**Conclusion:** The range 0.02-0.07 is STILL TOO HIGH. We're seeing the same overfitting behavior as V1's problematic λ=0.1-1.0 range.

---

## 2. Success Rate by Model Architecture

| Model | Success | Failed | Total | Success Rate | Avg Graduate % |
|-------|---------|--------|-------|--------------|----------------|
| **FTTransformer** | 15 | 1 | 16 | **93.8%** | 97.0% |
| **TabularResNet** | 14 | 2 | 16 | **87.5%** | 94.1% |
| **BasicNN** | 9 | 7 | 16 | **56.2%** | 96.3% |

### Key Insights:
- FTTransformer most robust (93.8% success) but worst overfitting (97.0% Graduate)
- BasicNN struggles most (56.2% success), also severe overfitting (96.3% Graduate)
- **All models exhibit extreme Graduate bias regardless of success rate**

---

## 3. Success Rate by Constraint Tightness

| Constraint | Type | Success | Failed | Total | Success Rate | **Avg Graduate %** |
|------------|------|---------|--------|-------|--------------|-------------------|
| **constraint_0.3_0.3** | [Hard, Hard] | 11 | 1 | 12 | **91.7%** | 98.8% ❌ |
| **constraint_0.3_0.8** | [Hard, Soft] | 10 | 2 | 12 | **83.3%** | 98.0% ❌ |
| **constraint_0.8_0.3** | [Soft, Hard] | 10 | 2 | 12 | **83.3%** | 98.0% ❌ |
| **constraint_0.9_0.8** | [Soft, Soft] | 7 | 5 | 12 | **58.3%** | 84.7% ⚠️ |

### Key Insights:
- **[Hard, Hard]** constraints easiest to satisfy (91.7%) BUT worst overfitting (98.8% Graduate!)
- **[Soft, Soft]** constraints hardest (58.3% success) BUT best prediction quality (84.7% Graduate)
- **Inverse relationship:** Easier constraints → worse predictions!

**Hypothesis:** Tight constraints provide an "easy shortcut" - predict everyone graduates to minimize constraint loss.

---

## 4. Early Stopping Analysis

**Early stopping effectiveness:** 91.7% (44/48 experiments stopped early)

### Convergence Speed by Lambda

| Lambda | Avg Epochs (Success) | Early Stop % |
|--------|---------------------|--------------|
| 0.02 | 2.5 | 100% |
| 0.03 | 6.7 | 100% |
| 0.05 | 6.1 | 100% |
| 0.07 | 4.4 | 100% |

**Observation:** All successful experiments converge in **2-7 epochs** - the models find the "predict Graduate" shortcut almost immediately!

### Example: FTTransformer [Hard, Hard] λ=0.5
```
Epoch  Global_Sat  Local_Sat  Hard_Dropout  Hard_Enrolled  Hard_Graduate
1      1           1          0             0              442/442 (100%!)
```

**After just 1 epoch:** Model predicts 100% Graduate, constraints satisfied, training stops.

---

## 5. Detailed Failure Analysis

**Total Failures:** 10/48 (20.8%)

### All Failed Experiments:

| Model | Constraint | Lambda | Graduate % at Failure |
|-------|------------|--------|-----------------------|
| BasicNN | [Hard, Hard] | 0.02 | ~73% |
| BasicNN | [Hard, Soft] | 0.02 | ~72% |
| BasicNN | [Hard, Soft] | 0.03 | ~71% |
| BasicNN | [Soft, Hard] | 0.02 | ~75% |
| BasicNN | [Soft, Soft] | 0.02 | ~69% |
| BasicNN | [Soft, Soft] | 0.03 | ~68% |
| BasicNN | [Soft, Soft] | 0.05 | ~70% |
| FTTransformer | [Soft, Hard] | 0.02 | ~76% |
| TabularResNet | [Soft, Soft] | 0.02 | ~72% |
| TabularResNet | [Soft, Soft] | 0.07 | ~74% |

### CRITICAL INSIGHT: Failed Experiments Have BETTER Predictions!

Failed experiments show more balanced predictions (68-76% Graduate) compared to successful experiments (94-99% Graduate)!

**Example: BasicNN [Hard, Hard] λ=0.02 (FAILED)**
```
Epoch  Global_Sat  Local_Sat  Hard_Dropout  Hard_Enrolled  Hard_Graduate  L_global  L_local
1      0           0          117           3              322            0.652872  0.497173
...
300    0           0          121           0              321            0.370626  0.452667
```

**Observations:**
- Ran full 300 epochs (never satisfied constraints)
- Predicts 321/442 = 72.6% Graduate (much more reasonable!)
- Hard_Dropout=121 exceeds limit of 42 (violates constraint)
- Constraint losses remain at ~0.37-0.45 (not converging to zero)

**Conclusion:** λ=0.02 is too weak to enforce constraints, but produces better prediction quality!

---

## 6. Prediction Quality Analysis

### Overall Prediction Distribution (Successful Experiments)

**Across all 38 successful experiments:**
- Dropout: 3.9% (should be ~32% based on true distribution)
- Enrolled: 0.3% (should be ~18%)
- Graduate: 95.8% (should be ~50%)

### By Lambda Value:

| Lambda | Dropout % | Enrolled % | Graduate % | Assessment |
|--------|-----------|------------|------------|------------|
| 0.02 | 2.1% | 0.0% | 97.8% | ❌ Severe overfitting |
| 0.03 | 5.7% | 0.6% | 93.6% | ❌ Severe overfitting |
| 0.05 | 3.9% | 0.3% | 95.8% | ❌ Severe overfitting |
| 0.07 | 3.2% | 0.2% | 96.6% | ❌ Severe overfitting |

**No lambda value in V2 produces reasonable predictions!**

### By Constraint Type:

| Constraint | Dropout % | Enrolled % | Graduate % |
|------------|-----------|------------|------------|
| [Hard, Hard] | 1.1% | 0.1% | 98.8% |
| [Hard, Soft] | 1.9% | 0.1% | 98.0% |
| [Soft, Hard] | 1.9% | 0.1% | 98.0% |
| [Soft, Soft] | 9.2% | 6.1% | 84.7% ← Closest to reasonable |

**[Soft, Soft]** constraints allow more prediction diversity but still biased toward Graduate.

---

## 7. Performance Metrics

**Average Accuracy (successful experiments):** 53.44%
**Average F1 Macro (successful experiments):** 28.49%

**Interpretation:**
- Accuracy of 53% is barely above random (33% for 3-class)
- F1 of 28% indicates poor performance, especially on minority classes
- Models are NOT learning meaningful patterns - just satisfying constraints

### Example: FTTransformer [Hard, Hard] λ=0.5 Confusion Matrix
```
                 Predicted
              Dropout  Enrolled  Graduate
True Dropout      0        0        142
True Enrolled     0        0         79
True Graduate     0        0        221
```

**Every single sample predicted as Graduate!**
- Recall for Dropout: 0.0%
- Recall for Enrolled: 0.0%
- Recall for Graduate: 100%

This is NOT useful for student dropout prediction!

---

## 8. Root Cause Analysis

### Why Are All Lambda Values Causing Overfitting?

**The constraint loss function creates a shortcut:**

Total Loss = L_pred(CE) + λ_global × L_target + λ_local × L_feat

**Scenario 1: Model learns to predict correctly**
- High cross-entropy loss initially
- Constraint losses vary based on predictions
- Requires many epochs to converge

**Scenario 2: Model predicts everyone graduates (shortcut)**
- Cross-entropy loss moderate (not optimal but acceptable)
- Constraint losses drop to near-zero IMMEDIATELY (if Graduate capacity is high)
- Constraints satisfied in 1-2 epochs!

**For lambda values ≥ 0.02:**
- The constraint penalty is strong enough to make shortcut #2 more attractive
- Model learns "predict Graduate = easy win" pattern
- Training ends before model learns to actually predict correctly

### Why Did V2 Fail?

**V2 hypothesis:** λ=0.02-0.07 would be the "sweet spot"
**Reality:** This range is STILL TOO HIGH

**The actual sweet spot is likely below λ=0.02**, possibly in the range:
- λ = 0.005 - 0.015

But this conflicts with the V1 finding that λ=0.01 only had 33% success rate!

---

## 9. The Fundamental Tradeoff

**There appears to be a fundamental tradeoff:**

```
Low Lambda (λ ≤ 0.01)
├─ ✓ Good prediction quality (balanced across classes)
├─ ✓ Model learns meaningful patterns
└─ ✗ Low constraint satisfaction (33% success)

Medium-High Lambda (λ ≥ 0.02)
├─ ✗ Severe overfitting to constraints
├─ ✗ Predicts >90% Graduate
├─ ✗ Poor prediction quality
└─ ✓ High constraint satisfaction (80-100% success)
```

**No static lambda value achieves BOTH:**
- Good constraint satisfaction (>70%)
- Reasonable prediction quality (<85% Graduate)

---

## 10. Comparison: V1 vs V2

| Metric | V1 Best | V2 Best | Winner |
|--------|---------|---------|--------|
| **Success Rate** | λ=1.0: 100% | λ=0.05: 91.7% | V1 |
| **Avg Graduate %** | λ=0.01: 86.7% | λ=0.03: 93.6% | V1 |
| **Balanced Metric** | λ=0.01: 33% success, 86.7% Grad | λ=0.02: 50% success, 97.8% Grad | V1 |

**V1 λ=0.01 remains the best static lambda** despite only 33% success rate, because it's the only value that doesn't completely overfit to constraints.

**V2 did not solve the problem** - it just explored a different part of the same problematic range.

---

## 11. Recommendations

### Option 1: Test Even Lower Lambda Values (V3)

**Proposed V3 configuration:**
```python
STATIC_LAMBDA_REGIMES = {
    'lambda_ultra_low': {
        'variations': [
            {'name': 'lambda_0.005', 'lambda_global': 0.005, 'lambda_local': 0.005},
            {'name': 'lambda_0.008', 'lambda_global': 0.008, 'lambda_local': 0.008},
            {'name': 'lambda_0.012', 'lambda_global': 0.012, 'lambda_local': 0.012},
            {'name': 'lambda_0.015', 'lambda_global': 0.015, 'lambda_local': 0.015},
        ]
    }
}
```

**Goal:** Find lambda between 0.005-0.015 that achieves 50-70% convergence with 75-85% Graduate predictions.

**Risk:** May still fail to converge on many experiments.

---

### Option 2: Abandon Static Lambda, Use Adaptive

**Evidence suggests static lambda cannot solve this problem:**
- Low λ: Good predictions, poor convergence
- High λ: Good convergence, terrible predictions
- No middle ground found after 114 experiments (66 V1 + 48 V2)

**Adaptive lambda methodology advantages:**
- Starts low (preserves prediction quality initially)
- Increases gradually (eventually enforces constraints)
- Allows model to learn patterns before constraints dominate
- Already implemented and tested

**Recommendation:** Compare best adaptive methodology results against V1 λ=0.01 static results.

---

### Option 3: Redesign Constraint Formulation

**Current constraints may be fundamentally flawed:**
- They incentivize "predict Graduate" shortcut
- Rational saturation loss L = E/(E+K) may be too aggressive

**Alternative approaches:**
1. **Soft penalties only:** Remove hard count constraints, use only soft ratios
2. **Separate training phases:** Train for prediction first, fine-tune for constraints
3. **Constraint relaxation:** Start with relaxed constraints, tighten over time (similar to adaptive lambda)
4. **Multi-objective optimization:** Use Pareto optimization to balance prediction vs constraints

---

### Option 4: Focus on Failed Experiments

**Observation:** Failed experiments have better prediction distributions

**Proposal:**
1. Train with low lambda (λ=0.01) to get good predictions
2. Accept that some experiments will fail constraints
3. Evaluate usefulness of resulting models for actual dropout prediction task
4. **Question:** Are perfectly satisfied constraints necessary if predictions are better?

---

## 12. Key Questions for Discussion

1. **Is 95%+ Graduate prediction acceptable?**
   - If goal is constraint satisfaction: Yes
   - If goal is dropout prediction: Absolutely not

2. **What is the actual objective?**
   - Satisfy constraints perfectly?
   - Predict student outcomes accurately?
   - Balance both (and if so, what's the priority)?

3. **Are the constraints realistic?**
   - Do the constraint limits match true data distribution?
   - Are we asking the model to violate the natural class distribution?

4. **Should we pursue V3 (even lower lambda)?**
   - Or accept that static lambda has fundamental limitations?

5. **Is this a problem with the methodology or the constraints?**
   - Maybe the rational saturation loss is too aggressive
   - Maybe the constraint limits are unrealistic

---

## 13. Summary Statistics

| Metric | Value |
|--------|-------|
| **Best Lambda (Convergence)** | 0.05-0.07 (91.7% success) |
| **Best Lambda (Prediction Quality)** | 0.02 (still 97.8% Graduate!) |
| **Worst Lambda** | 0.02 (50% success) |
| **Best Model** | FTTransformer (93.8% success) |
| **Worst Model** | BasicNN (56.2% success) |
| **Easiest Constraints** | [Hard, Hard] (91.7% success, 98.8% Graduate) |
| **Hardest Constraints** | [Soft, Soft] (58.3% success, 84.7% Graduate) |
| **Avg Epochs (Success)** | 2.5-6.7 (early stopping working well) |
| **Overall Success Rate** | 79.2% (38/48) |
| **Overall Graduate %** | 95.8% (SEVERE OVERFITTING) |
| **Overall Accuracy** | 53.4% (barely above random) |
| **Overall F1 Macro** | 28.5% (poor minority class performance) |

---

## 14. Final Verdict

**V2 DID NOT solve the overfitting problem.**

All lambda values 0.02-0.07 exhibit the same severe bias toward Graduate predictions (93.6-97.8%) as V1's problematic λ=0.1-1.0 range. Models satisfy constraints by taking a shortcut rather than learning meaningful patterns.

**The search for optimal static lambda continues**, but evidence suggests it may not exist. The fundamental tradeoff between constraint satisfaction and prediction quality may require an adaptive approach.

**Next steps:**
1. If pursuing static lambda: Test V3 with λ=0.005-0.015
2. If abandoning static lambda: Compare adaptive methodology quantitatively
3. Consider redesigning constraint formulation to eliminate "predict Graduate" shortcut

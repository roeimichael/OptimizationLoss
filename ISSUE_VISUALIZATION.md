# Visual Explanation of Issues

## Current Results: (0.4, 0.2)

```
Optimized:  Acc=55.4%  (WORSE)
Benchmark:  Acc=73.5%  (BETTER) ← This seems wrong!
```

---

## WHY THIS HAPPENS

### Constraint Formula Bug

```
Input percentage: 0.2 (expecting 20%)
Actual formula:   count * 0.2 / 10 = count * 0.02
Result:          2% (not 20%!)
```

### Example with 1000 Students

```
Class       True Count    Expected (20%)    Actual (2%!)
Dropout     400           80                8
Enrolled    350           70                7
Graduate    250           Unlimited         Unlimited
```

---

## Benchmark Algorithm Flow

```
Step 1: Get probabilities from model (trained 250 epochs, no constraints)
        Model is ~70% accurate at this point

Step 2: Greedy selector:
        ┌─────────────────────────────────────┐
        │ For each course:                    │
        │   Dropout:  Assign top 8 samples    │  Only ~4% of samples!
        │   Enrolled: Assign top 7 samples    │  Only ~2% of samples!
        │   Graduate: SKIP (unlimited)        │  0% assigned here
        └─────────────────────────────────────┘

Step 3: Fallback for unassigned (~94% of samples):
        final_predictions[i] = argmax(probabilities[i])
        ↑
        This is just NORMAL prediction!

Step 4: Result
        4% forced to Dropout (top probability samples - likely correct!)
        2% forced to Enrolled (top probability samples - likely correct!)
        94% get normal predictions (model is ~70% accurate)
        ────────────────────────────────────────────────────
        Overall: ~0.04*high% + 0.02*high% + 0.94*70% ≈ 73%
```

---

## Optimized Algorithm Flow

```
Step 1-250: Train normally (no constraint pressure)
            Model reaches ~70% accuracy

Step 251+:  Add constraint losses
            ┌────────────────────────────────────────┐
            │ Loss = CrossEntropy                    │
            │      + λ_global * GlobalConstraintLoss │
            │      + λ_local * LocalConstraintLoss   │
            └────────────────────────────────────────┘

            GlobalConstraintLoss says:
            "Predict ≤8 Dropouts globally!"

            LocalConstraintLoss says:
            "Predict ≤8 Dropouts per course!"

            Model learns:
            - Reduce Dropout predictions
            - Reduce Enrolled predictions
            - Shift toward Graduate (unlimited)

Step 2000+: Model has been pulled away from correct predictions
            to satisfy tiny constraints

            Result: ~55% accuracy (model distorted!)
```

---

## The Unfair Comparison

```
Benchmark:                      Optimized:
┌──────────────────────┐       ┌──────────────────────┐
│ Epoch 250           │       │ Epoch 2000          │
│ Normal Training     │       │ + Constraint Losses  │
│                     │       │                      │
│ Then:               │       │ Model State:         │
│ Greedy Selection    │       │ - Fewer Dropouts     │
│ - 4% Dropout        │       │ - Fewer Enrolled     │
│ - 2% Enrolled       │       │ - More Graduate      │
│ - 94% Normal!       │       │ - Lower Accuracy     │
└──────────────────────┘       └──────────────────────┘
      73.5% ✓                        55.4% ✗
```

---

## What SHOULD Happen (Fair Comparison)

```
Both use SAME model at Epoch 250:
┌──────────────────────────────────────┐
│         Model @ Epoch 250            │
│    (Trained Normally, ~70% Acc)      │
└───────────┬──────────────────────┬───┘
            │                      │
            ↓                      ↓
    ┌──────────────┐      ┌──────────────┐
    │  Benchmark   │      │  Optimized   │
    │  ────────    │      │  ─────────   │
    │  Greedy      │      │  Continue    │
    │  Selection   │      │  Training    │
    │  on Probs    │      │  with Loss   │
    └──────────────┘      └──────────────┘
```

---

## Visual: Greedy Selector with Proper Constraints (20% not 2%)

```
If constraints were 20% (as intended):

1000 Samples:
    400 Dropout (True)
    350 Enrolled (True)
    250 Graduate (True)

Greedy would assign:
    200 Dropout  (20% of 1000)  ← Much more restrictive!
    200 Enrolled (20% of 1000)
    600 Graduate (remainder)

This would force significant changes and likely lower accuracy too!
```

---

## Summary

The benchmark appears better because:
1. **Constraints are 100x smaller than intended** (2% vs 20%)
2. **96% of predictions are just normal** (not constrained)
3. **Model hasn't been distorted** (evaluated at epoch 250)

The optimized appears worse because:
1. **Model trained longer with harmful constraints**
2. **Constraints pull model away from correct predictions**
3. **Entire prediction distribution is shifted**

It's like comparing:
- A normal model with a Band-Aid (benchmark)
- vs. A model that had surgery (optimized)

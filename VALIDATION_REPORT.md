# Complete Codebase Validation Report

## Executive Summary

Found **5 CRITICAL BUGS** that completely invalidate current results:

1. ⚠️ Constraint formula error (100x too restrictive)
2. ⚠️ Greedy benchmark algorithm fundamentally flawed
3. ⚠️ Unfair comparison (different model states)
4. ⚠️ Graduate class always unlimited (hardcoded)
5. ⚠️ Course 1 always skipped (hardcoded)

---

## CRITICAL BUG #1: Constraint Computation

**File:** `src/training/constraints.py:8, 24`

**Bug:**
```python
constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)
```

**Impact:**
With `percentage = 0.2`, you get `count * 0.02` (2%) not `count * 0.2` (20%)!

**Example:**
- If there are 1000 Dropout samples
- percentage = 0.2 (expecting 20% = 200)
- **Actual constraint:** 1000 * 0.2 / 10 = 20 (only 2%!)

This makes constraints **100x more restrictive** than intended.

---

## CRITICAL BUG #2: Greedy Selector Algorithm

**File:** `src/benchmark/greedy_constraint_selector.py:49-73`

**Current Behavior:**
```python
for class_id in range(3):
    constraint = local_cons[class_id]
    if constraint >= 1e9:  # If unlimited
        continue  # SKIP THIS CLASS!
    # ... assign top N samples to this class

# Later:
for i in range(n_samples):
    if final_predictions[i] == -1:  # Unassigned
        final_predictions[i] = np.argmax(test_proba[i])  # Normal prediction
```

**What Actually Happens:**

With constraints (0.4, 0.2):
- Dropout constraint: 0.04 (4%) of count
- Enrolled constraint: 0.02 (2%) of count
- Graduate constraint: Unlimited (1e10)

1. Greedy selector assigns ~4% to Dropout (top probability samples)
2. Greedy selector assigns ~2% to Enrolled (top probability samples)
3. Graduate is SKIPPED (constraint >= 1e9)
4. **96% of samples get argmax prediction (normal prediction!)**

**Why Benchmark Has Higher Accuracy:**
- Model at epoch 250 is reasonably good
- 96% of predictions are normal (argmax)
- Only 4% forced into constraints
- **Essentially a normal model with tiny tweaks!**

**Why Optimized Has Lower Accuracy:**
- Model trained 250+ epochs with constraint losses
- Constraint losses force model to satisfy 2-4% limits
- Model learns to predict fewer Dropouts/Enrolled
- **Model distorted to satisfy constraints**

---

## CRITICAL BUG #3: Unfair Comparison

**Benchmark:**
- Model state: Epoch 250 (warmup complete)
- Trained: 250 epochs, NO constraint pressure
- Evaluation: Normal model + tiny greedy adjustments

**Optimized:**
- Model state: Epoch 250-10,000 (whenever constraints satisfied)
- Trained: 250 epochs normal + X epochs with constraint pressure
- Evaluation: Model warped by constraint losses

**They're comparing:**
- A good normal model (benchmark)
- vs. A distorted model (optimized)

**NOT a fair comparison!**

---

## CRITICAL BUG #4: Graduate Always Unlimited

**File:** `src/training/constraints.py:9, 25`

```python
constraint[2] = 1e10  # Hardcoded!
```

Class 2 (Graduate) is ALWAYS unlimited, regardless of percentage parameter.

**Impact:**
- Only Dropout and Enrolled are constrained
- Graduate predictions are never limited
- Not clear why this is hardcoded

---

## CRITICAL BUG #5: Course 1 Always Skipped

**File:** `src/training/constraints.py:16-17`

```python
if group == 1:
    continue  # Hardcoded!
```

Course 1 never gets local constraints.

**Impact:**
- Course 1 students can be predicted without local constraints
- Unclear why this is special-cased

---

## Detailed Trace: Why Results Don't Make Sense

### Scenario: (0.4, 0.2) constraints

**Expected:**
- 40% local constraint per course
- 20% global constraint

**Actual (with bugs):**
- 4% local constraint (due to /10 bug)
- 2% global constraint (due to /10 bug)
- Graduate unlimited (hardcoded)

**Benchmark Flow:**
1. Model trained 250 epochs normally → ~70%+ accuracy
2. Greedy selector:
   - Assigns ~4% of samples to Dropout (top probability)
   - Assigns ~2% of samples to Enrolled (top probability)
   - Skips Graduate (unlimited)
   - **Assigns ~94% via argmax (normal prediction)**
3. Result: ~73.5% accuracy (mostly normal predictions!)

**Optimized Flow:**
1. Model trained 250 epochs normally → ~70%+ accuracy
2. Model trained 250-10,000 epochs WITH constraint losses
   - Constraint losses push to satisfy 4% Dropout, 2% Enrolled limits
   - Model learns to predict FEWER Dropouts and Enrolled
   - Overall predictions shift to satisfy constraints
   - **Predictions become less accurate**
3. Result: ~55.4% accuracy (model distorted by constraints!)

---

## Data Flow Verification

### run_experiments.py:
```python
# Line 41-43: Load data
X_train, X_test, y_train, y_test, train_df, test_df = load_presplit_data(...)

# Line 81-95: Train
model, scaler, training_time, history, metrics = train_model_transductive(
    X_train_clean, y_train,
    X_test_clean, groups_test, y_test,  # ✓ y_test passed
    ...
)

# Line 97-98: Evaluate (using final trained model)
y_test_pred = predict(model, scaler, X_test_clean, device)
accuracy = evaluate_accuracy(y_test.values, y_test_pred)
```

### trainer.py:
```python
# Line 202-215: Benchmark at epoch 249
if epoch == WARMUP_EPOCHS - 1 and not benchmark_done:
    greedy_constraint_selection(
        model, X_test_tensor, group_ids_test, y_test,  # ✓ Using same test data
        ...
    )

# Line 245-249: Optimized evaluation (after training ends)
y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
y_true_np = y_test.values
metrics = compute_metrics(y_true_np, y_pred)
```

**Verification:** ✓ Both use same y_test, but different model states!

---

## RECOMMENDATIONS

### Option 1: Fix Everything (Proper Benchmark)
1. Fix constraint formula (remove /10)
2. Fix greedy selector to properly enforce constraints
3. Save model at epoch 250 and evaluate both from same state
4. Make Graduate constraint configurable
5. Make Course 1 skip configurable

### Option 2: Document Current Behavior
1. Document that constraints are percentage/10
2. Document that benchmark is mostly normal predictions
3. Document comparison is unfair (different training lengths)
4. Explain Graduate/Course1 hardcoding

### Option 3: Redesign Benchmark
Create a fair baseline that:
1. Uses SAME model state (epoch 250)
2. Applies constraints in a comparable way
3. Compares constraint satisfaction, not just accuracy

---

## Files Requiring Changes

If fixing properly:

1. **src/training/constraints.py**
   - Remove `/10` from lines 8, 24
   - Make `constraint[2]` configurable
   - Make Course 1 skip configurable

2. **src/benchmark/greedy_constraint_selector.py**
   - Rewrite to properly enforce constraints
   - Don't skip unlimited classes
   - Ensure final predictions satisfy constraints

3. **src/training/trainer.py**
   - Save model state at epoch 250
   - Evaluate both from same model state
   - OR clearly document the difference

4. **experiments/run_experiments.py**
   - Update to handle fair comparison
   - Document which model state is used when

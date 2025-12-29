# Critical Issues Found in Codebase

## Issue 1: CONSTRAINT COMPUTATION FORMULA ERROR ⚠️ **CRITICAL**

**Location:** `src/training/constraints.py`, lines 8 and 24

**Current Code:**
```python
constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)
```

**Problem:**
The formula divides by 10, so:
- percentage=0.2 → actual constraint = count * 0.02 (2% not 20%)
- percentage=0.4 → actual constraint = count * 0.04 (4% not 40%)

This makes constraints EXTREMELY restrictive (100x more restrictive than intended).

**Expected Formula:**
```python
constraint[int(class_id)] = np.round(items[class_id] * percentage)
```

---

## Issue 2: GRADUATE CLASS ALWAYS UNLIMITED

**Location:** `src/training/constraints.py`, lines 9 and 25

**Current Code:**
```python
constraint[2] = 1e10  # Graduate always unlimited
```

**Problem:**
Class 2 (Graduate) is hardcoded to unlimited, regardless of the percentage parameter.

**Impact:** Only Dropout (class 0) and Enrolled (class 1) are constrained.

---

## Issue 3: GREEDY SELECTOR FUNDAMENTALLY FLAWED ⚠️ **CRITICAL**

**Location:** `src/benchmark/greedy_constraint_selector.py`, lines 49-73

**Current Algorithm:**
1. For each course, for each class with a constraint:
   - Assign top N samples (by probability) to that class
2. All unassigned samples get their argmax prediction

**Problems:**

### 3a. Only Constrained Classes Get Greedy Selection
```python
if constraint >= 1e9:  # Line 52
    continue  # Skip unconstrained classes
```

This means Graduate (class 2) is NEVER assigned during greedy selection. It only gets assigned via argmax fallback.

### 3b. Argmax Fallback Ignores Constraints
```python
for i in range(n_samples):  # Lines 71-73
    if final_predictions[i] == -1:
        final_predictions[i] = np.argmax(test_proba[i])
```

Unassigned samples get their highest probability class WITHOUT checking if this violates constraints.

### 3c. No Constraint Enforcement
The greedy selector doesn't actually enforce constraints properly. It:
- Assigns a few samples to constrained classes
- Lets everything else be predicted normally
- Doesn't verify final predictions satisfy constraints

---

## Issue 4: WHY BENCHMARK HAS HIGHER ACCURACY

With the current implementation:

**Benchmark (Greedy Selector):**
1. Model trained for 250 epochs normally (no constraint pressure)
2. Greedy selector assigns ~2-4% of samples to Dropout/Enrolled
3. Rest (~96%) get argmax (normal prediction)
4. If model is decent, high accuracy on the ~96% normal predictions

**Optimized:**
1. Model trained for 250 epochs normally
2. Then trained MORE with constraint losses added
3. Constraint losses pull model to satisfy tiny (2-4%) constraints
4. Model forced to make bad predictions to satisfy constraints
5. Lower overall accuracy

**Result:** Benchmark appears better because it's mostly just normal predictions!

---

## Issue 5: EVALUATION DISCREPANCY

**Benchmark Evaluation:**
- Uses model at epoch 250 (warmup complete)
- Line 202-215 in trainer.py

**Optimized Evaluation:**
- Uses model after ALL epochs complete (could be 10,000 epochs)
- Lines 224-249 in trainer.py

**Different model states being compared!**

---

## Issue 6: COURSE 1 SKIPPED IN LOCAL CONSTRAINTS

**Location:** `src/training/constraints.py`, line 16-17

```python
if group == 1:
    continue  # Course 1 always skipped
```

Course 1 never gets local constraints applied.

---

## RECOMMENDED FIXES

1. **Fix constraint formula** - Remove `/10`
2. **Fix greedy selector** - Implement proper constraint-aware selection
3. **Ensure fair comparison** - Both should use same model state OR document difference
4. **Fix Graduate constraint** - Make it configurable
5. **Fix Course 1 skip** - Remove hardcoded skip or document why it's needed

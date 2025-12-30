# BENCHMARK ALGORITHM VALIDATION

## What SHOULD Happen (Your Requirements):
1. ✅ Train for 250 warmup epochs
2. ✅ Predict on test dataset (same as constraints)
3. ✅ Iterate course by course with local constraints
4. ❌ **CRITICAL BUG:** Assign top predicted values AND respect constraints
5. ❌ **CRITICAL BUG:** Argmax fallback VIOLATES constraints

---

## What ACTUALLY Happens (Code Trace):

### Phase 1: Constrained Assignment (Lines 42-69)
```python
for course_id in sorted(unique_courses):
    for class_id in range(3):
        constraint = local_cons[class_id]

        if constraint >= 1e9:  # ⚠️ SKIP unlimited classes!
            continue

        # Assign top N samples for THIS class
        for sample in sorted_by_probability:
            if not already_assigned and global_count[class] < global_limit:
                assign to class_id
```

**What gets assigned in Phase 1:**
- Dropout: Top N students with highest Dropout probability
- Enrolled: Top N students with highest Enrolled probability
- Graduate: **SKIPPED** (because constraint is 1e10)

### Phase 2: Argmax Fallback (Lines 71-73)
```python
for i in range(n_samples):
    if final_predictions[i] == -1:  # Unassigned
        final_predictions[i] = np.argmax(test_proba[i])  # ⚠️ NO CONSTRAINT CHECK!
```

**This assigns ALL remaining samples via argmax WITHOUT checking constraints!**

---

## Concrete Example:

Let's say you have 100 students in a course with constraints:
- Dropout constraint: 4 students (4%)
- Enrolled constraint: 2 students (2%)
- Graduate constraint: unlimited

**Phase 1 (Constrained):**
- Assigns 4 students to Dropout (top 4 by Dropout probability)
- Assigns 2 students to Enrolled (top 2 by Enrolled probability)
- Skips Graduate (unlimited)
- **6 students assigned** (6% of course)

**Phase 2 (Argmax):**
- 94 students remain unassigned
- For each, assign `argmax(probabilities)`
- **NO constraint checking!**

**Possible outcome:**
```
Final predictions:
- Dropout: 15 students (VIOLATED! Constraint was 4)
- Enrolled: 8 students (VIOLATED! Constraint was 2)
- Graduate: 77 students

Constraints violated because argmax added:
- 11 more Dropouts
- 6 more Enrolled
```

---

## Why Benchmark Shows ~70% Accuracy:

1. Model at epoch 250 is reasonably good (~70% base accuracy)
2. Only 6% of samples get "forced" assignments (Dropout + Enrolled constraints)
3. **94% get normal argmax predictions** (no constraint pressure)
4. If model is decent, these 94% are mostly correct
5. Result: High accuracy even though constraints are violated!

---

## The Fix:

Replace lines 71-73 with constraint-aware assignment:

```python
# Current (WRONG):
for i in range(n_samples):
    if final_predictions[i] == -1:
        final_predictions[i] = np.argmax(test_proba[i])  # VIOLATES CONSTRAINTS!

# Should be (CORRECT):
for i in range(n_samples):
    if final_predictions[i] == -1:
        sample_course = course_ids[i]
        sample_proba = test_proba[i]

        # Try classes in order of probability
        for class_id in np.argsort(sample_proba)[::-1]:
            # Check if assigning this class would violate constraints
            if global_counts[class_id] < global_constraints[class_id]:
                local_cons = local_constraints_dict.get(sample_course, [float('inf')] * 3)

                # Count current assignments for this course/class
                course_mask = (course_ids == sample_course)
                current_course_class_count = np.sum(
                    final_predictions[course_mask] == class_id
                )

                if current_course_class_count < local_cons[class_id]:
                    # Safe to assign!
                    final_predictions[i] = class_id
                    global_counts[class_id] += 1
                    break

        # If all classes violate constraints, assign to unlimited class
        if final_predictions[i] == -1:
            final_predictions[i] = 2  # Graduate (unlimited)
```

---

## Verification:

Check your `benchmark_constraint_comparison.csv` files. You'll likely see:
- **Status: OVER** for many course/class combinations
- Predictions exceed constraints
- Confirms argmax is violating limits

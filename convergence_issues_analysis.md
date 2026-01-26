# Convergence Issues Analysis

## Issue 1: Logging vs Checking Discrepancy

### The Problem
Looking at conv_10_5 training log, we only see **1 epoch** with Local_Satisfied=1 (epoch 90), but the sustained convergence checker requires **5 out of 10** epochs to be satisfied. How did it converge?

### Root Cause
**Logging happens every 3 epochs, but checking happens EVERY epoch.**

In `trainer.py`:
- **Line 203-206**: Convergence checker is updated EVERY single epoch
- **Line 185-200**: CSV logging only happens every 3 epochs (`if (epoch + 1) % 3 == 0`)

### What Actually Happened in conv_10_5:

**Logged epochs (visible in CSV):**
- Epoch 66, 69, 72, 75, 78, 81, 84, 87, 90 (every 3rd epoch)

**Actual checking (internally):**
- Epochs 81-90 (10-epoch window for conv_10_5)
- Checker tracked satisfaction flags for ALL 10 epochs
- Only epochs 81, 84, 87, 90 appear in the log
- But the checker saw satisfaction for 5+ epochs within 81-90 range (including unlogged epochs like 82, 83, 85, 86, 88, 89)

### Impact:
The sustained convergence logic **works correctly**, but the training log gives a **misleading view** because it only shows every 3rd epoch.

### Verification:
For conv_10_5 (window=10, required=5):
- Final epoch: 90
- Window: epochs 81-90
- Logged epochs in window: 81, 84, 87, 90 (4 logged)
- Actual epochs checked: 81, 82, 83, 84, 85, 86, 87, 88, 89, 90 (10 checked)
- At least 5 of those 10 satisfied → convergence ✓

---

## Issue 2: Post-Convergence Degradation

### The Problem
After the model first satisfies constraints, instead of "climbing back up" to improve CE loss (accuracy) while maintaining constraints at their upper limit, the model keeps getting heavily punished and drops lower in accuracy.

### Example from conv_10_5:
```
Epoch 66: First Global_Satisfied=1, CE=0.249577, lambda_local=0.18
Epoch 90: Final convergence, CE=0.159649, lambda_local=0.25

CE loss dropped by 36%, but constraints are already satisfied!
```

### Root Cause: Lambda Strategy Never Decreases

Looking at `lambda_adjusting.py` (LinearLambdaAdjuster, lines 86-102):

```python
# Increase global lambda if not satisfied
if global_loss > threshold:
    new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)

# Increase local lambda if not satisfied
if local_loss > threshold:
    new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)

# Case 4: Both satisfied - no changes (lambda stays at high value!)
```

**The problem:**
1. Lambda increases when constraints violated (correct)
2. Lambda stays constant when constraints satisfied (incorrect for optimization!)
3. High lambda values keep heavily penalizing constraint violations
4. Model cannot explore the constraint boundary to improve accuracy
5. Result: Model gets "locked in" at high lambda, preventing accuracy optimization

### Training Dynamics Breakdown:

**Phase 1: Pre-Convergence (epochs 51-66)**
- Lambda increases from 0.11 → 0.18 (local)
- Constraint violations decrease
- CE loss relatively stable around 0.25-0.32

**Phase 2: Post-First-Satisfaction (epochs 66-90)**
- Lambda continues increasing: 0.18 → 0.25 (local)
- Constraints already satisfied but lambda keeps rising
- CE loss drops: 0.249 → 0.159 (worse!)
- Model is "over-punished" for staying at constraint boundary

### What Should Happen:

After constraints are satisfied, the model should:
1. **Reduce lambda** to allow exploring constraint boundaries
2. **Maximize CE loss** (improve accuracy) while staying within constraints
3. **Balance** between constraint satisfaction and prediction quality

### Current Behavior:
```
                    Accuracy
                       ↑
                       |
Epoch 66: Satisfied ───┤   ← Should climb back up here
                       |
                       |
Epoch 90: Final ───────┤   ← Instead, drops down (over-punished)
                       ↓
```

### Desired Behavior:
```
                    Accuracy
                       ↑
                       |
Epoch 90: Optimal ─────┤   ← Maximize accuracy at constraint boundary
                       |
                       |
Epoch 66: Satisfied ───┤   ← First satisfaction
                       |
```

---

## Proposed Solutions

### Solution 1: Fix Logging Frequency (Minor Issue)
Make logging frequency configurable or log every epoch during convergence checking.

**Implementation:**
```python
# Option A: Log every epoch when close to convergence
if convergence_checker.get_satisfaction_rate() > 0.3:
    log_every = 1
else:
    log_every = 3

if (epoch + 1) % log_every == 0:
    log_progress_to_csv(...)
```

**Impact:** Better visibility into convergence process, no change to actual behavior.

---

### Solution 2: Lambda Decay After Satisfaction (Major Issue)

Add lambda decay when constraints are satisfied to allow the model to explore and optimize accuracy.

**Implementation Options:**

#### Option A: Simple Decay
```python
class LinearLambdaAdjuster:
    def __init__(self, lambda_step=0.005, lambda_max=50.0, lambda_decay=0.95):
        self.lambda_step = lambda_step
        self.lambda_max = lambda_max
        self.lambda_decay = lambda_decay  # NEW: decay factor

    def adjust_lambdas(self, lambda_global, lambda_local,
                       global_satisfied, local_satisfied,
                       global_loss, local_loss, threshold):
        # Increase if not satisfied
        if global_loss > threshold:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)
        else:
            # NEW: Decay when satisfied (but keep minimum value)
            new_lambda_global = max(lambda_global * self.lambda_decay, 0.01)

        if local_loss > threshold:
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)
        else:
            # NEW: Decay when satisfied
            new_lambda_local = max(lambda_local * self.lambda_decay, 0.01)

        return new_lambda_global, new_lambda_local
```

**Pros:**
- Simple to implement
- Allows model to explore constraint boundaries
- Improves accuracy after constraint satisfaction

**Cons:**
- Might oscillate if decay is too aggressive
- Need to tune decay rate (0.95 = 5% decay per epoch)

#### Option B: Sustained Satisfaction Decay
Only decay lambda after constraints have been satisfied for N consecutive epochs:

```python
class SustainedDecayLambdaAdjuster:
    def __init__(self, lambda_step=0.005, lambda_max=50.0,
                 lambda_decay=0.95, decay_wait_epochs=5):
        self.lambda_step = lambda_step
        self.lambda_max = lambda_max
        self.lambda_decay = lambda_decay
        self.decay_wait_epochs = decay_wait_epochs
        self.global_satisfied_count = 0
        self.local_satisfied_count = 0

    def adjust_lambdas(self, lambda_global, lambda_local,
                       global_satisfied, local_satisfied,
                       global_loss, local_loss, threshold):
        # Track consecutive satisfaction
        if global_satisfied:
            self.global_satisfied_count += 1
        else:
            self.global_satisfied_count = 0

        if local_satisfied:
            self.local_satisfied_count += 1
        else:
            self.local_satisfied_count = 0

        # Adjust global lambda
        if global_loss > threshold:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)
        elif self.global_satisfied_count >= self.decay_wait_epochs:
            # Decay after sustained satisfaction
            new_lambda_global = max(lambda_global * self.lambda_decay, 0.01)
        else:
            new_lambda_global = lambda_global  # Keep constant

        # Same for local lambda
        if local_loss > threshold:
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)
        elif self.local_satisfied_count >= self.decay_wait_epochs:
            new_lambda_local = max(lambda_local * self.lambda_decay, 0.01)
        else:
            new_lambda_local = lambda_local

        return new_lambda_global, new_lambda_local
```

**Pros:**
- More stable, waits for sustained satisfaction before decaying
- Prevents oscillation from premature decay
- Better for noisy convergence

**Cons:**
- More complex
- Requires tuning decay_wait_epochs

#### Option C: Constraint Utilization Target
Adjust lambda to push model toward using more of the constraint budget:

```python
class UtilizationTargetLambdaAdjuster:
    def __init__(self, lambda_step=0.005, lambda_max=50.0,
                 target_utilization=0.95):
        self.lambda_step = lambda_step
        self.lambda_max = lambda_max
        self.target_utilization = target_utilization  # Use 95% of constraint budget

    def adjust_lambdas(self, lambda_global, lambda_local,
                       global_satisfied, local_satisfied,
                       global_loss, local_loss, threshold,
                       current_count, constraint_value):
        """
        Args:
            current_count: Current soft prediction count
            constraint_value: Constraint limit
        """
        if global_loss > threshold:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)
        else:
            # Calculate utilization
            utilization = current_count / constraint_value

            if utilization < self.target_utilization:
                # Under-utilizing constraint, decay lambda to push higher
                new_lambda_global = max(lambda_global * 0.98, 0.01)
            else:
                # At target utilization, keep lambda
                new_lambda_global = lambda_global

        return new_lambda_global, new_lambda_local
```

**Pros:**
- Explicitly targets constraint utilization
- Maximizes use of constraint budget
- More interpretable behavior

**Cons:**
- More complex, needs access to prediction counts
- Requires tuning target_utilization parameter

---

## Recommendations

### Immediate Actions:
1. **Fix Issue 1**: Add logging frequency adjustment or document the behavior
2. **Test Option A (Simple Decay)**: Easy to implement, likely to improve results
   - Start with decay=0.98 (2% per epoch)
   - Set minimum lambda = 0.01

### Testing Plan:
1. Run conv_10_5 with Simple Decay lambda strategy
2. Compare:
   - Convergence speed (should be similar)
   - Final test accuracy (should improve)
   - CE loss trajectory (should improve after first satisfaction)
   - Constraint satisfaction stability (should remain stable)

### Expected Results:
- **Before**: CE loss drops from 0.25 → 0.16 after first satisfaction
- **After**: CE loss improves from 0.25 → 0.15 (or better) after first satisfaction
- Test accuracy should improve by 1-3%

### Long-term:
If Simple Decay works well, consider implementing Option B (Sustained Decay) or Option C (Utilization Target) for more sophisticated control.

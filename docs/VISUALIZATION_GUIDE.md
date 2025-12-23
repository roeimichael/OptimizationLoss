# Training Visualization Guide

## Overview

The training system now automatically generates comprehensive visualizations showing constraint satisfaction progress throughout training.

## Generated Visualizations

After training completes, three visualization files are automatically created in the `results/` directory:

### 1. **global_constraints.png**
Shows global constraint satisfaction over training epochs.

**Features:**
- Line plot for each class (Dropout, Enrolled, Graduate)
- Predicted student counts per class on Y-axis
- Epochs on X-axis
- **Dotted horizontal lines** showing constraint limits
- Clear visualization of when predictions cross below constraints

**Use Case:** Identify which classes are hardest to constrain and when global constraints are satisfied.

### 2. **local_constraints.png**
Shows per-course constraint satisfaction for selected courses.

**Features:**
- Subplots for up to 6 courses (prioritizes courses with most violations)
- Each subplot shows all 3 classes for that course
- Dotted lines for each class constraint
- Helps identify problematic courses

**Use Case:** Find which specific courses cause constraint violations and track their progress.

### 3. **losses.png**
Comprehensive loss tracking in 4 panels.

**Panels:**
- **L_target**: Global constraint loss (with 1e-6 threshold line)
- **L_feat**: Local constraint loss (with 1e-6 threshold line)
- **L_pred**: Cross-entropy prediction loss
- **L_total**: Combined total loss

**Use Case:** Monitor loss component balance and convergence.

## Code Structure

### Helper Functions (trainer.py)

```python
compute_prediction_statistics(model, X_test_tensor, group_ids_test)
```
- Computes global and local prediction counts
- Returns clean dictionaries for tracking
- Called every 50 epochs

```python
print_progress(epoch, losses, counts, criterion_constraint)
```
- Handles all console output formatting
- Shows global/local constraint tables
- Highlights violations clearly

### Visualization Module (visualization.py)

```python
plot_global_constraints(history, global_constraints, save_path)
```
- Creates global constraint visualization
- Handles unconstrained classes gracefully
- Saves as high-resolution PNG

```python
plot_local_constraints(history, local_constraints, save_path, max_courses=6)
```
- Selects most interesting courses (by violation frequency)
- Creates grid of subplots
- Shows constraint crossing points

```python
plot_losses(history, save_path)
```
- 4-panel loss visualization
- Threshold lines for constraint losses
- Clear titles and labels

```python
create_all_visualizations(history, g_cons, l_cons, output_dir)
```
- **Main entry point** - call this to generate all plots
- Automatically called at end of training
- Creates output directory if needed

## Training History Tracking

The trainer now maintains a `history` dictionary:

```python
history = {
    'epochs': [50, 100, 150, ...],
    'loss_total': [...],
    'loss_ce': [...],
    'loss_global': [...],
    'loss_local': [...],
    'global_predictions': [{0: 102, 1: 45, 2: 296}, ...],
    'local_predictions': [{course_id: {0: c0, 1: c1, 2: c2}}, ...]
}
```

**Updated every 50 epochs** to balance:
- Sufficient data points for smooth plots
- Minimal overhead during training

## Example Output

After training with constraint (0.9, 0.8), you'll see:

**Console Output (every 50 epochs):**
```
================================================================================
Epoch 150
================================================================================
L_target (Global):  0.000234
L_feat (Local):     0.012456
L_pred (CE):        0.456789
L_total:            0.469479

────────────────────────────────────────────────────────────────────────────────
GLOBAL CONSTRAINTS vs PREDICTIONS
────────────────────────────────────────────────────────────────────────────────
Class           Constraint      Predicted       Status
────────────────────────────────────────────────────────────────────────────────
Dropout         113             110             ✓ OK
Enrolled        64              62              ✓ OK
Graduate        None (unconstrained)  271       N/A
────────────────────────────────────────────────────────────────────────────────
Total                           443

────────────────────────────────────────────────────────────────────────────────
LOCAL CONSTRAINTS vs PREDICTIONS (Per Course)
────────────────────────────────────────────────────────────────────────────────
Courses with VIOLATIONS:
  Course 5: Drop:8/7 (✗+1), Enrl:3/3 (✓)
  Course 12: Drop:13/12 (✗+1), Enrl:5/4 (✗+1)

Courses SATISFYING constraints: 15 courses (all OK)
```

**Visualization Files Created:**
```
results/
├── global_constraints.png    # Global constraint tracking
├── local_constraints.png     # Per-course tracking (6 worst courses)
└── losses.png                # Loss components over training
```

## Benefits

1. **Clean Code**: Helper functions keep training loop readable
2. **Full Tracking**: Every 50 epochs captures complete state
3. **Automatic Viz**: No manual plotting needed
4. **Problem Identification**: Quickly see which constraints are hardest to satisfy
5. **Progress Monitoring**: Visual confirmation of constraint satisfaction
6. **Publication Ready**: High-quality plots for papers/reports

## Customization

### Change Tracking Frequency
In `trainer.py`, line 258:
```python
if (epoch + 1) % 50 == 0:  # Change 50 to desired frequency
```

### Adjust Course Count in Local Plots
In `visualization.py`, when calling:
```python
plot_local_constraints(history, local_constraints, save_path, max_courses=12)
```

### Change Output Directory
```python
create_all_visualizations(history, g_cons, l_cons, output_dir='./custom_results')
```

## Technical Details

- **Format**: PNG at 300 DPI (publication quality)
- **Colors**: Consistent across all plots (Dropout=red, Enrolled=blue, Graduate=green)
- **Memory**: History stored every 50 epochs to avoid memory bloat
- **Performance**: Minimal overhead (~0.1s per evaluation)

## Next Steps

After reviewing the visualizations, you can:
1. Identify which classes converge fastest
2. See which courses need special attention
3. Determine if constraints are achievable
4. Adjust constraint percentages if needed
5. Monitor loss balance between L_pred and constraint losses

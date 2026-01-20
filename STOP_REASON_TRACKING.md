# Stop Reason Tracking System

This document explains the comprehensive stop reason tracking system that documents why each training run ended.

## Overview

Every training run now tracks **exactly why it stopped** with detailed information saved in two places:
1. **`run_status.json`** - Lightweight status file in each experiment directory
2. **`config.json`** - Full configuration with `run_completion` field (for new runs)

## Stop Reason Categories

### 1. Converged ✓
**Status:** `converged`
**Reason:** Both global and local constraints were satisfied

**Example:**
```json
{
  "status": "converged",
  "reason": "Normal convergence: Both global and local constraints satisfied at epoch 267",
  "exception_type": null,
  "final_epoch": 267,
  "global_constraint_satisfied": true,
  "local_constraint_satisfied": true
}
```

### 2. Failed ✗
**Status:** `failed`
**Reason:** Reached maximum epochs (1000) without satisfying both constraints

**Example:**
```json
{
  "status": "failed",
  "reason": "Reached 1000 epochs with only Global constraint satisfied (Local constraint not satisfied)",
  "exception_type": null,
  "final_epoch": 1000,
  "global_constraint_satisfied": true,
  "local_constraint_satisfied": false
}
```

### 3. Interrupted ?
**Status:** `interrupted`
**Reason:** Stopped early due to external factors

#### Interruption Types Tracked:

**User Interruption (Ctrl+C)**
```json
{
  "status": "interrupted",
  "reason": "User manually interrupted training with Ctrl+C (KeyboardInterrupt)",
  "exception_type": "KeyboardInterrupt",
  "final_epoch": 423
}
```

**Out of Memory (RAM)**
```json
{
  "status": "interrupted",
  "reason": "Out of Memory (OOM) error - system ran out of RAM during training",
  "exception_type": "MemoryError",
  "final_epoch": 651
}
```

**CUDA Out of Memory (GPU)**
```json
{
  "status": "interrupted",
  "reason": "CUDA Out of Memory error - GPU ran out of memory during training. Error: CUDA out of memory. Tried to allocate 512.00 MiB",
  "exception_type": "RuntimeError (CUDA OOM)",
  "final_epoch": 543
}
```

**System Exit / Process Killed**
```json
{
  "status": "interrupted",
  "reason": "Process exited with code 143 - possibly killed by system, timeout, or external signal",
  "exception_type": "SystemExit",
  "final_epoch": 789
}
```

**Unknown Exception**
```json
{
  "status": "interrupted",
  "reason": "Unexpected exception during training: ValueError: invalid value encountered in loss calculation",
  "exception_type": "ValueError",
  "final_epoch": 234
}
```

## File Locations

### For Each Experiment:
```
results/our_approach/TabularResNet/constraint_0.5_0.3/lr_lambda_test/lr_0.0001_lambda_transfer/
├── training_log.csv           # Detailed epoch-by-epoch logs
├── run_status.json            # Status tracking (always present)
├── config.json                # Full config with run_completion (new runs only)
├── evaluation_metrics.csv     # Final metrics
└── final_predictions.csv      # Predictions
```

### run_status.json Structure:
```json
{
  "status": "converged|failed|interrupted",
  "final_epoch": 267,
  "global_constraint_satisfied": true,
  "local_constraint_satisfied": true,
  "details": "Detailed explanation of stop reason",
  "timestamp": "2026-01-20T12:30:45.123456",
  "retroactive": false
}
```

### config.json run_completion Field:
```json
{
  "run_completion": {
    "status": "converged|failed|interrupted",
    "reason": "Human-readable explanation",
    "exception_type": "KeyboardInterrupt|MemoryError|RuntimeError|null",
    "final_epoch": 267,
    "global_constraint_satisfied": true,
    "local_constraint_satisfied": true,
    "completed_at": "2026-01-20T12:30:45.123456"
  }
}
```

## Analysis Scripts

### 1. `analyze_stop_reasons.py`
Comprehensive analysis of why runs stopped:
```bash
python3 analyze_stop_reasons.py
```

**Output:**
- Summary statistics by status
- Breakdown of interruption causes
- Examples of each stop reason type
- Exports `stop_reasons_analysis.csv` for Excel/spreadsheet analysis

### 2. `analyze_run_statuses.py`
Status-based reporting:
```bash
python3 analyze_run_statuses.py
```

### 3. `retroactive_status_tagging.py`
Tags existing runs with status (already run on your 72 experiments):
```bash
python3 retroactive_status_tagging.py
```

## Current Results Summary (72 Experiments)

| Status | Count | Percentage |
|--------|-------|------------|
| ✓ Converged | 24 | 33.3% |
| ✗ Failed | 0 | 0.0% |
| ? Interrupted | 48 | 66.7% |

### Interrupted Runs Breakdown:
- **24 runs** - Stopped with only Global constraint satisfied
- **18 runs** - Stopped with only Local constraint satisfied
- **6 runs** - Stopped with neither constraint satisfied

**Note:** All 48 interrupted runs appear to have been stopped externally (likely process kills, timeouts, or OOM), as there are only 2 programmatic stop conditions in the code:
1. Both constraints satisfied (early convergence)
2. Reaching max epochs (1000)

## For Future Runs

All new training runs will automatically:
1. Save detailed stop reason to `run_status.json`
2. Save stop reason to `config.json` in the `run_completion` field
3. Capture specific exception types for interrupted runs
4. Track final constraint satisfaction state

## Viewing Stop Reasons

**Quick check of a single run:**
```bash
cat results/path/to/experiment/run_status.json
```

**Bulk analysis:**
```bash
python3 analyze_stop_reasons.py
# Opens stop_reasons_analysis.csv in Excel/spreadsheet for detailed analysis
```

**Search for specific exception types:**
```bash
grep -r "MemoryError" results/*/run_status.json
grep -r "KeyboardInterrupt" results/*/run_status.json
```

## Code Implementation

The stop reason tracking is implemented in:
- `src/training/logging.py` - `save_run_status()` function
- `src/training/trainer.py` - Saves status on convergence/failure
- `src/experiments/run_experiment.py` - Catches and categorizes exceptions
- `src/utils/filesystem_manager.py` - `save_stop_reason()` for config.json updates

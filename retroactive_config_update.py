#!/usr/bin/env python3
"""Retroactively update config.json files with stop reasons from run_status.json."""

import json
from pathlib import Path
from datetime import datetime

def update_config_with_stop_reason(experiment_dir):
    """
    Read run_status.json and update config.json with stop reason.

    Returns:
        bool: True if updated, False if skipped
    """
    status_file = experiment_dir / 'run_status.json'
    config_file = experiment_dir / 'config.json'

    if not status_file.exists():
        return False

    if not config_file.exists():
        return False

    # Read status
    with open(status_file) as f:
        status_data = json.load(f)

    # Read config
    with open(config_file) as f:
        config = json.load(f)

    # Skip if already has run_completion
    if 'run_completion' in config:
        return False

    # Extract data from status
    status = status_data['status']
    final_epoch = status_data['final_epoch']
    global_sat = status_data['global_constraint_satisfied']
    local_sat = status_data['local_constraint_satisfied']
    details = status_data['details']

    # Determine exception type and reason based on status and details
    exception_type = None
    reason = details

    if status == 'converged':
        reason = f"Normal convergence: Both global and local constraints satisfied at epoch {final_epoch}"
        exception_type = None

    elif status == 'failed':
        if global_sat and not local_sat:
            reason = f"Reached {final_epoch} epochs with only Global constraint satisfied (Local constraint not satisfied)"
        elif not global_sat and local_sat:
            reason = f"Reached {final_epoch} epochs with only Local constraint satisfied (Global constraint not satisfied)"
        elif not global_sat and not local_sat:
            reason = f"Reached {final_epoch} epochs without satisfying either Global or Local constraints"
        else:
            reason = f"Reached {final_epoch} epochs (unexpected state)"
        exception_type = None

    elif status == 'interrupted':
        # Try to infer exception type from details
        details_lower = details.lower()
        if 'keyboardinterrupt' in details_lower or 'ctrl+c' in details_lower or 'user' in details_lower:
            exception_type = 'KeyboardInterrupt'
            reason = "User manually interrupted training with Ctrl+C (KeyboardInterrupt)"
        elif 'out of memory' in details_lower or 'oom' in details_lower:
            if 'cuda' in details_lower:
                exception_type = 'RuntimeError (CUDA OOM)'
                reason = f"CUDA Out of Memory error - GPU ran out of memory during training"
            else:
                exception_type = 'MemoryError'
                reason = f"Out of Memory (OOM) error - system ran out of RAM during training"
        else:
            exception_type = 'Unknown'
            reason = f"Process interrupted unexpectedly at epoch {final_epoch}. Original details: {details}"

    # Add run_completion to config
    config['run_completion'] = {
        'status': status,
        'reason': reason,
        'exception_type': exception_type,
        'final_epoch': final_epoch,
        'global_constraint_satisfied': global_sat,
        'local_constraint_satisfied': local_sat,
        'completed_at': status_data.get('timestamp', datetime.now().isoformat()),
        'retroactive': True
    }

    # Update top-level status
    config['status'] = 'completed' if status == 'converged' else 'pending'

    # Write updated config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

    return True

def main():
    results_dir = Path('results')

    # Find all run_status.json files
    status_files = list(results_dir.rglob('run_status.json'))

    print(f"Found {len(status_files)} status files\n")

    updated = 0
    skipped = 0

    for status_file in status_files:
        experiment_dir = status_file.parent
        rel_path = experiment_dir.relative_to(results_dir)

        if update_config_with_stop_reason(experiment_dir):
            updated += 1
            print(f"[UPDATED] {rel_path}")
        else:
            skipped += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {len(status_files)}")
    print(f"  Updated: {updated}")
    print(f"  Skipped (already had run_completion): {skipped}")
    print("\nNote: All config.json files now include 'run_completion' field with:")
    print("  - status: converged/failed/interrupted")
    print("  - reason: Human-readable explanation")
    print("  - exception_type: Type of error if interrupted")
    print("  - final_epoch: Last epoch reached")
    print("  - global/local_constraint_satisfied: Final state")

if __name__ == '__main__':
    main()

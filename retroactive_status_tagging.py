#!/usr/bin/env python3
"""Retroactively tag existing experiment runs with status based on their training logs."""

import csv
import json
from pathlib import Path
from datetime import datetime

def analyze_and_tag_run(training_log_path):
    """
    Analyze a training log and create a run_status.json file.

    Returns:
        tuple: (status, epoch, global_sat, local_sat, details)
    """
    try:
        with open(training_log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            return ('interrupted', 0, False, False, "Empty training log")

        last_row = rows[-1]
        last_epoch = int(last_row['Epoch'])
        global_sat = int(last_row['Global_Satisfied']) == 1
        local_sat = int(last_row['Local_Satisfied']) == 1

        # Determine status based on conditions
        if global_sat and local_sat:
            status = 'converged'
            details = f"Both constraints satisfied at epoch {last_epoch}"
        elif last_epoch >= 1000:
            status = 'failed'
            if global_sat:
                details = f"Reached max epochs with only Global constraint satisfied"
            elif local_sat:
                details = f"Reached max epochs with only Local constraint satisfied"
            else:
                details = f"Reached max epochs without satisfying constraints"
        else:
            status = 'interrupted'
            constraints_status = []
            if global_sat:
                constraints_status.append("Global satisfied")
            if local_sat:
                constraints_status.append("Local satisfied")

            if constraints_status:
                details = f"Stopped at epoch {last_epoch} with {', '.join(constraints_status)}"
            else:
                details = f"Stopped at epoch {last_epoch} without satisfying constraints"

        return (status, last_epoch, global_sat, local_sat, details)

    except Exception as e:
        return ('interrupted', 0, False, False, f"Error reading log: {str(e)}")

def create_status_file(experiment_dir, status, epoch, global_sat, local_sat, details):
    """Create a run_status.json file for the experiment."""
    status_path = experiment_dir / 'run_status.json'

    status_data = {
        'status': status,
        'final_epoch': epoch,
        'global_constraint_satisfied': global_sat,
        'local_constraint_satisfied': local_sat,
        'details': details,
        'timestamp': datetime.now().isoformat(),
        'retroactive': True  # Mark that this was added retroactively
    }

    with open(status_path, 'w') as f:
        json.dump(status_data, f, indent=2)

def main():
    results_dir = Path('results')

    # Find all training_log.csv files
    log_files = list(results_dir.rglob('training_log.csv'))

    print(f"Found {len(log_files)} training log files\n")

    converged = 0
    failed = 0
    interrupted = 0

    for log_file in log_files:
        experiment_dir = log_file.parent

        # Check if status file already exists
        status_file = experiment_dir / 'run_status.json'
        if status_file.exists():
            print(f"[SKIP] {experiment_dir.relative_to(results_dir)} - status file already exists")
            continue

        # Analyze and tag
        status, epoch, global_sat, local_sat, details = analyze_and_tag_run(log_file)
        create_status_file(experiment_dir, status, epoch, global_sat, local_sat, details)

        if status == 'converged':
            converged += 1
        elif status == 'failed':
            failed += 1
        elif status == 'interrupted':
            interrupted += 1

        # Print summary for this run
        rel_path = experiment_dir.relative_to(results_dir)
        print(f"[{status.upper()}] {rel_path}")
        print(f"  Epoch: {epoch}, Global: {global_sat}, Local: {local_sat}")
        print(f"  Details: {details}\n")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total runs tagged: {converged + failed + interrupted}")
    print(f"  Converged: {converged}")
    print(f"  Failed: {failed}")
    print(f"  Interrupted: {interrupted}")

if __name__ == '__main__':
    main()

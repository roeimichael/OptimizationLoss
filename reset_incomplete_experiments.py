#!/usr/bin/env python3
"""
Reset incomplete experiments to clean state for re-running.

For converged experiments (both constraints satisfied):
  - Leave ALL files untouched

For incomplete experiments (interrupted/failed):
  - Keep config.json but reset status to 'pending'
  - Remove run_completion field from config.json
  - DELETE all output files (training logs, results, plots, run_status.json)
"""

import csv
import json
import shutil
from pathlib import Path

def check_if_converged(experiment_dir):
    """
    Check if an experiment converged by examining the training log.

    Returns:
        tuple: (converged: bool, has_training_log: bool)
    """
    training_log = experiment_dir / 'training_log.csv'

    if not training_log.exists():
        return False, False

    try:
        with open(training_log, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            return False, True

        last_row = rows[-1]
        global_satisfied = int(last_row['Global_Satisfied']) == 1
        local_satisfied = int(last_row['Local_Satisfied']) == 1

        return (global_satisfied and local_satisfied), True

    except Exception as e:
        print(f"  [WARNING] Error reading training log: {e}")
        return False, True

def reset_config(config_path):
    """
    Reset config.json to pending status and remove run_completion field.

    Returns:
        bool: True if successful
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Reset status to pending
        config['status'] = 'pending'

        # Remove run_completion if it exists
        if 'run_completion' in config:
            del config['run_completion']

        # Remove results if they exist
        if 'results' in config:
            del config['results']

        # Write back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to reset config: {e}")
        return False

def delete_output_files(experiment_dir):
    """
    Delete all output files except config.json.

    Returns:
        list: Names of deleted files
    """
    deleted_files = []

    # Files to delete
    files_to_delete = [
        'training_log.csv',
        'evaluation_metrics.csv',
        'final_predictions.csv',
        'run_status.json',
        'plot_confusion_matrix.png',
        'plot_loss_functions.png',
        'plot_predictions_by_class.png'
    ]

    for filename in files_to_delete:
        file_path = experiment_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files.append(filename)
            except Exception as e:
                print(f"  [WARNING] Could not delete {filename}: {e}")

    return deleted_files

def main():
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    # Find all experiment directories by looking for training_log.csv files
    training_logs = list(results_dir.rglob('training_log.csv'))

    if len(training_logs) == 0:
        print("[INFO] No training logs found - no experiments to clean")
        return

    print(f"Found {len(training_logs)} experiment directories\n")
    print("=" * 100)
    print("ANALYZING AND CLEANING EXPERIMENTS")
    print("=" * 100)

    converged_count = 0
    reset_count = 0

    converged_experiments = []
    reset_experiments = []

    for training_log in training_logs:
        experiment_dir = training_log.parent
        rel_path = experiment_dir.relative_to(results_dir)

        # Check if converged
        is_converged, _ = check_if_converged(experiment_dir)

        if is_converged:
            # Leave converged experiments completely untouched
            converged_count += 1
            converged_experiments.append(rel_path)
            print(f"\n[CONVERGED] {rel_path}")
            print(f"  → Leaving all files untouched")

        else:
            # Didn't converge - reset it
            reset_count += 1
            reset_experiments.append(rel_path)
            print(f"\n[RESETTING] {rel_path}")

            # Reset config if it exists
            config_file = experiment_dir / 'config.json'
            if config_file.exists():
                if reset_config(config_file):
                    print(f"  → Config reset to 'pending' status")
            else:
                print(f"  → No config.json found (will be created when re-run)")

            # Delete output files
            deleted = delete_output_files(experiment_dir)
            if deleted:
                print(f"  → Deleted {len(deleted)} files: {', '.join(deleted)}")
            else:
                print(f"  → No output files to delete")

    # Summary
    print("\n" + "=" * 100)
    print("CLEANUP SUMMARY")
    print("=" * 100)
    print(f"\nTotal experiments: {len(training_logs)}")
    print(f"  ✓ Converged (left untouched): {converged_count}")
    print(f"  ↻ Reset for re-running: {reset_count}")

    if converged_experiments:
        print(f"\n{'-' * 100}")
        print(f"CONVERGED EXPERIMENTS ({converged_count} total) - All files preserved:")
        print(f"{'-' * 100}")
        for exp in converged_experiments:
            print(f"  {exp}")

    if reset_experiments:
        print(f"\n{'-' * 100}")
        print(f"RESET EXPERIMENTS ({reset_count} total) - Ready to re-run:")
        print(f"{'-' * 100}")
        for exp in reset_experiments:
            print(f"  {exp}")

    print("\n" + "=" * 100)
    print("READY TO RE-RUN")
    print("=" * 100)
    print(f"\nYou can now run: python main.py")
    print(f"It will skip the {converged_count} converged experiments")
    print(f"and re-run the {reset_count} incomplete experiments from scratch")

if __name__ == '__main__':
    main()

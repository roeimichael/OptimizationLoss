#!/usr/bin/env python3
"""Analyze convergence results from training logs."""

import csv
from pathlib import Path

def analyze_training_log(log_path):
    """
    Analyze a single training log file.

    Returns:
        tuple: (status, last_epoch, global_sat, local_sat, path)
        status: 'not_converged', 'converged', 'unexpected_stop'
    """
    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            return ('empty', 0, None, None, log_path)

        last_row = rows[-1]
        last_epoch = int(last_row['Epoch'])
        global_sat = int(last_row['Global_Satisfied'])
        local_sat = int(last_row['Local_Satisfied'])

        # Check convergence conditions
        if global_sat == 1 and local_sat == 1:
            status = 'converged'
        elif last_epoch >= 1000:
            status = 'not_converged'
        else:
            status = 'unexpected_stop'

        return (status, last_epoch, global_sat, local_sat, log_path)

    except Exception as e:
        return ('error', None, None, None, f"{log_path} - Error: {str(e)}")

def main():
    results_dir = Path('results')

    # Find all training_log.csv files
    log_files = list(results_dir.rglob('training_log.csv'))

    print(f"Found {len(log_files)} training log files\n")

    # Analyze each file
    converged = []
    not_converged = []
    unexpected_stop = []
    errors = []
    empty = []

    for log_file in log_files:
        status, last_epoch, global_sat, local_sat, path = analyze_training_log(log_file)

        if status == 'converged':
            converged.append((path, last_epoch))
        elif status == 'not_converged':
            not_converged.append((path, last_epoch, global_sat, local_sat))
        elif status == 'unexpected_stop':
            unexpected_stop.append((path, last_epoch, global_sat, local_sat))
        elif status == 'error':
            errors.append(path)
        elif status == 'empty':
            empty.append(path)

    # Print summary
    print("=" * 80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal runs analyzed: {len(log_files)}")
    print(f"\n✓ CONVERGED (both constraints satisfied): {len(converged)}")
    print(f"✗ NOT CONVERGED (reached 1000 epochs): {len(not_converged)}")
    print(f"? UNEXPECTED STOP (stopped early without convergence): {len(unexpected_stop)}")

    if errors:
        print(f"⚠ ERRORS (failed to read): {len(errors)}")
    if empty:
        print(f"⚠ EMPTY FILES: {len(empty)}")

    # Detailed breakdown
    if converged:
        print("\n" + "=" * 80)
        print("CONVERGED RUNS:")
        print("=" * 80)
        for path, epoch in sorted(converged):
            rel_path = path.relative_to(results_dir)
            print(f"  {rel_path} (epoch: {epoch})")

    if not_converged:
        print("\n" + "=" * 80)
        print("NOT CONVERGED RUNS (reached 1000 epochs):")
        print("=" * 80)
        for path, epoch, g_sat, l_sat in sorted(not_converged):
            rel_path = path.relative_to(results_dir)
            print(f"  {rel_path}")
            print(f"    Final epoch: {epoch}, Global_Satisfied: {g_sat}, Local_Satisfied: {l_sat}")

    if unexpected_stop:
        print("\n" + "=" * 80)
        print("UNEXPECTED STOPS (stopped early without convergence):")
        print("=" * 80)
        for path, epoch, g_sat, l_sat in sorted(unexpected_stop):
            rel_path = path.relative_to(results_dir)
            print(f"  {rel_path}")
            print(f"    Stopped at epoch: {epoch}, Global_Satisfied: {g_sat}, Local_Satisfied: {l_sat}")

    if errors:
        print("\n" + "=" * 80)
        print("ERRORS:")
        print("=" * 80)
        for path_info in errors:
            print(f"  {path_info}")

    if empty:
        print("\n" + "=" * 80)
        print("EMPTY FILES:")
        print("=" * 80)
        for path in empty:
            rel_path = path.relative_to(results_dir)
            print(f"  {rel_path}")

if __name__ == '__main__':
    main()

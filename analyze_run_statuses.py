#!/usr/bin/env python3
"""Analyze run statuses from status files."""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path('results')

    # Find all run_status.json files
    status_files = list(results_dir.rglob('run_status.json'))

    print(f"Found {len(status_files)} status files\n")

    # Categorize by status
    by_status = {
        'converged': [],
        'failed': [],
        'interrupted': []
    }

    for status_file in status_files:
        with open(status_file) as f:
            status_data = json.load(f)

        experiment_dir = status_file.parent
        rel_path = experiment_dir.relative_to(results_dir)

        status = status_data['status']
        if status in by_status:
            by_status[status].append((rel_path, status_data))

    # Print summary
    print("=" * 80)
    print("RUN STATUS SUMMARY")
    print("=" * 80)
    print(f"\nTotal runs: {len(status_files)}")
    print(f"  ✓ Converged (both constraints satisfied): {len(by_status['converged'])}")
    print(f"  ✗ Failed (reached max epochs without convergence): {len(by_status['failed'])}")
    print(f"  ? Interrupted (stopped early externally): {len(by_status['interrupted'])}")

    # Analyze interrupted runs by what was satisfied
    interrupted_both_none = []
    interrupted_only_global = []
    interrupted_only_local = []
    interrupted_near_max = []

    for path, data in by_status['interrupted']:
        epoch = data['final_epoch']
        g_sat = data['global_constraint_satisfied']
        l_sat = data['local_constraint_satisfied']

        if epoch >= 999:
            interrupted_near_max.append((path, data))
        elif g_sat and not l_sat:
            interrupted_only_global.append((path, data))
        elif not g_sat and l_sat:
            interrupted_only_local.append((path, data))
        else:
            interrupted_both_none.append((path, data))

    print("\n" + "-" * 80)
    print("INTERRUPTED RUNS BREAKDOWN:")
    print("-" * 80)
    print(f"  Reached ~999 epochs (effectively failed): {len(interrupted_near_max)}")
    print(f"  Stopped with ONLY Global satisfied: {len(interrupted_only_global)}")
    print(f"  Stopped with ONLY Local satisfied: {len(interrupted_only_local)}")
    print(f"  Stopped with NEITHER satisfied: {len(interrupted_both_none)}")

    # Show converged runs
    if by_status['converged']:
        print("\n" + "=" * 80)
        print("CONVERGED RUNS (24 total):")
        print("=" * 80)
        for path, data in sorted(by_status['converged'], key=lambda x: x[1]['final_epoch']):
            print(f"  {path} (epoch: {data['final_epoch']})")

    # Show failed runs
    if by_status['failed']:
        print("\n" + "=" * 80)
        print("FAILED RUNS:")
        print("=" * 80)
        for path, data in sorted(by_status['failed']):
            print(f"  {path}")
            print(f"    Final epoch: {data['final_epoch']}, "
                  f"Global: {data['global_constraint_satisfied']}, "
                  f"Local: {data['local_constraint_satisfied']}")

    # Show some interrupted runs
    if interrupted_near_max:
        print("\n" + "=" * 80)
        print("INTERRUPTED AT ~999 EPOCHS (effectively failed):")
        print("=" * 80)
        for path, data in interrupted_near_max:
            print(f"  {path}")
            print(f"    Epoch: {data['final_epoch']}, "
                  f"Global: {data['global_constraint_satisfied']}, "
                  f"Local: {data['local_constraint_satisfied']}")

    # Statistics
    if by_status['converged']:
        epochs = [data['final_epoch'] for _, data in by_status['converged']]
        avg_epoch = sum(epochs) / len(epochs)
        min_epoch = min(epochs)
        max_epoch = max(epochs)

        print("\n" + "=" * 80)
        print("CONVERGENCE STATISTICS:")
        print("=" * 80)
        print(f"  Average convergence epoch: {avg_epoch:.1f}")
        print(f"  Fastest convergence: {min_epoch} epochs")
        print(f"  Slowest convergence: {max_epoch} epochs")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze stop reasons across all experiments."""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path('results')
    status_files = list(results_dir.rglob('run_status.json'))

    print(f"Found {len(status_files)} experiments with status tracking\n")

    # Categorize by status and reason
    by_status = defaultdict(list)
    by_reason_type = defaultdict(list)

    for status_file in status_files:
        with open(status_file) as f:
            data = json.load(f)

        experiment_dir = status_file.parent
        rel_path = experiment_dir.relative_to(results_dir)

        status = data['status']
        details = data['details']
        epoch = data['final_epoch']
        global_sat = data['global_constraint_satisfied']
        local_sat = data['local_constraint_satisfied']

        by_status[status].append((rel_path, data))

        # Categorize interrupted runs by likely cause
        if status == 'interrupted':
            details_lower = details.lower()
            if 'user' in details_lower or 'keyboard' in details_lower:
                by_reason_type['User Interruption (Ctrl+C)'].append((rel_path, data))
            elif 'out of memory' in details_lower or 'oom' in details_lower:
                by_reason_type['Out of Memory (OOM)'].append((rel_path, data))
            elif 'global satisfied' in details_lower:
                by_reason_type['Stopped with only Global satisfied'].append((rel_path, data))
            elif 'local satisfied' in details_lower:
                by_reason_type['Stopped with only Local satisfied'].append((rel_path, data))
            else:
                by_reason_type['Unknown/Other reason'].append((rel_path, data))

    # Print summary
    print("=" * 100)
    print("STOP REASON ANALYSIS - OVERALL SUMMARY")
    print("=" * 100)
    print(f"\nTotal experiments: {len(status_files)}")
    print(f"  ✓ Converged: {len(by_status['converged'])} ({len(by_status['converged'])/len(status_files)*100:.1f}%)")
    print(f"  ✗ Failed (max epochs): {len(by_status['failed'])} ({len(by_status['failed'])/len(status_files)*100:.1f}%)")
    print(f"  ? Interrupted: {len(by_status['interrupted'])} ({len(by_status['interrupted'])/len(status_files)*100:.1f}%)")

    # Interrupted breakdown
    if by_status['interrupted']:
        print("\n" + "-" * 100)
        print("INTERRUPTED RUNS - BREAKDOWN BY LIKELY CAUSE:")
        print("-" * 100)
        for reason_type, runs in sorted(by_reason_type.items(), key=lambda x: -len(x[1])):
            print(f"\n{reason_type}: {len(runs)} runs")

    # Show examples of each reason type
    print("\n" + "=" * 100)
    print("DETAILED STOP REASONS BY CATEGORY")
    print("=" * 100)

    # Converged examples
    if by_status['converged']:
        print(f"\n{'='*100}")
        print(f"CONVERGED RUNS ({len(by_status['converged'])} total)")
        print(f"{'='*100}")
        print("\nShowing first 5 examples:")
        for path, data in list(by_status['converged'])[:5]:
            print(f"\n  Path: {path}")
            print(f"  Epoch: {data['final_epoch']}")
            print(f"  Reason: {data['details']}")

    # Failed examples
    if by_status['failed']:
        print(f"\n{'='*100}")
        print(f"FAILED RUNS ({len(by_status['failed'])} total)")
        print(f"{'='*100}")
        for path, data in by_status['failed']:
            print(f"\n  Path: {path}")
            print(f"  Epoch: {data['final_epoch']}")
            print(f"  Global satisfied: {data['global_constraint_satisfied']}")
            print(f"  Local satisfied: {data['local_constraint_satisfied']}")
            print(f"  Reason: {data['details']}")

    # Interrupted - show examples by type
    if by_status['interrupted']:
        for reason_type, runs in sorted(by_reason_type.items(), key=lambda x: -len(x[1])):
            print(f"\n{'='*100}")
            print(f"{reason_type.upper()} ({len(runs)} runs)")
            print(f"{'='*100}")
            print(f"\nShowing first 3 examples:")
            for path, data in runs[:3]:
                print(f"\n  Path: {path}")
                print(f"  Epoch: {data['final_epoch']}")
                print(f"  Global satisfied: {data['global_constraint_satisfied']}")
                print(f"  Local satisfied: {data['local_constraint_satisfied']}")
                print(f"  Details: {data['details']}")

    # Summary statistics for interrupted runs
    if by_status['interrupted']:
        print("\n" + "=" * 100)
        print("INTERRUPTED RUNS - STATISTICS")
        print("=" * 100)

        epochs = [data['final_epoch'] for _, data in by_status['interrupted']]
        both_satisfied = sum(1 for _, data in by_status['interrupted']
                           if data['global_constraint_satisfied'] and data['local_constraint_satisfied'])
        only_global = sum(1 for _, data in by_status['interrupted']
                        if data['global_constraint_satisfied'] and not data['local_constraint_satisfied'])
        only_local = sum(1 for _, data in by_status['interrupted']
                       if not data['global_constraint_satisfied'] and data['local_constraint_satisfied'])
        neither = sum(1 for _, data in by_status['interrupted']
                    if not data['global_constraint_satisfied'] and not data['local_constraint_satisfied'])

        print(f"\nAverage stopping epoch: {sum(epochs)/len(epochs):.1f}")
        print(f"Earliest stop: {min(epochs)} epochs")
        print(f"Latest stop: {max(epochs)} epochs")
        print(f"\nConstraint satisfaction at stop:")
        print(f"  Both satisfied (would have converged): {both_satisfied}")
        print(f"  Only Global satisfied: {only_global}")
        print(f"  Only Local satisfied: {only_local}")
        print(f"  Neither satisfied: {neither}")

    # Export to CSV for further analysis
    print("\n" + "=" * 100)
    print("EXPORTING TO CSV")
    print("=" * 100)

    csv_path = Path('stop_reasons_analysis.csv')
    with open(csv_path, 'w') as f:
        f.write("experiment_path,status,final_epoch,global_satisfied,local_satisfied,stop_reason\n")
        for status_file in status_files:
            with open(status_file) as sf:
                data = json.load(sf)
            experiment_dir = status_file.parent
            rel_path = experiment_dir.relative_to(results_dir)

            # Escape commas and quotes in details
            details = data['details'].replace('"', '""')
            if ',' in details:
                details = f'"{details}"'

            f.write(f"{rel_path},{data['status']},{data['final_epoch']},"
                   f"{data['global_constraint_satisfied']},{data['local_constraint_satisfied']},"
                   f"{details}\n")

    print(f"\n✓ Exported detailed analysis to: {csv_path}")
    print("  You can open this in Excel or any spreadsheet tool for further analysis")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze patterns in converged runs."""

import csv
from pathlib import Path
from collections import defaultdict

def parse_path(path_str):
    """Extract experiment parameters from path."""
    parts = Path(path_str).parts
    # parts example: ('our_approach', 'BasicNN', 'constraint_0.5_0.3', 'lr_lambda_test', 'lr_0.0001_lambda_combined', 'training_log.csv')

    model = parts[1] if len(parts) > 1 else None

    # Extract constraint values
    constraint = None
    for part in parts:
        if part.startswith('constraint_'):
            constraint = part.replace('constraint_', '')
            break

    # Extract lr and lambda strategy from the last directory before filename
    lr = None
    lambda_strategy = None
    if len(parts) >= 2:
        lr_lambda_dir = parts[-2]  # e.g., 'lr_0.0001_lambda_combined'
        if lr_lambda_dir.startswith('lr_'):
            # Split by '_lambda_' to separate lr from lambda strategy
            split_parts = lr_lambda_dir.split('_lambda_')
            if len(split_parts) == 2:
                lr = split_parts[0].replace('lr_', '')
                lambda_strategy = split_parts[1]

    return model, constraint, lr, lambda_strategy

def main():
    results_dir = Path('results')
    log_files = list(results_dir.rglob('training_log.csv'))

    converged = []

    # Find all converged runs
    for log_file in log_files:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if len(rows) > 0:
                last_row = rows[-1]
                epoch = int(last_row['Epoch'])
                g_sat = int(last_row['Global_Satisfied'])
                l_sat = int(last_row['Local_Satisfied'])

                if g_sat == 1 and l_sat == 1:
                    rel_path = str(log_file.relative_to(results_dir))
                    model, constraint, lr, lambda_strategy = parse_path(rel_path)
                    converged.append({
                        'path': rel_path,
                        'epoch': epoch,
                        'model': model,
                        'constraint': constraint,
                        'lr': lr,
                        'lambda_strategy': lambda_strategy
                    })

    print(f"Total converged runs: {len(converged)}\n")

    # Analyze by model
    by_model = defaultdict(list)
    for run in converged:
        by_model[run['model']].append(run)

    print("=" * 80)
    print("CONVERGENCE BY MODEL:")
    print("=" * 80)
    for model, runs in sorted(by_model.items()):
        print(f"\n{model}: {len(runs)} converged runs")
        avg_epoch = sum(r['epoch'] for r in runs) / len(runs)
        print(f"  Average convergence epoch: {avg_epoch:.1f}")

    # Analyze by learning rate
    by_lr = defaultdict(list)
    for run in converged:
        by_lr[run['lr']].append(run)

    print("\n" + "=" * 80)
    print("CONVERGENCE BY LEARNING RATE:")
    print("=" * 80)
    for lr, runs in sorted(by_lr.items()):
        print(f"\n{lr}: {len(runs)} converged runs")
        avg_epoch = sum(r['epoch'] for r in runs) / len(runs)
        print(f"  Average convergence epoch: {avg_epoch:.1f}")

    # Analyze by lambda strategy
    by_lambda = defaultdict(list)
    for run in converged:
        by_lambda[run['lambda_strategy']].append(run)

    print("\n" + "=" * 80)
    print("CONVERGENCE BY LAMBDA STRATEGY:")
    print("=" * 80)
    for strategy, runs in sorted(by_lambda.items()):
        print(f"\n{strategy}: {len(runs)} converged runs")
        avg_epoch = sum(r['epoch'] for r in runs) / len(runs)
        print(f"  Average convergence epoch: {avg_epoch:.1f}")

    # Analyze by constraint
    by_constraint = defaultdict(list)
    for run in converged:
        by_constraint[run['constraint']].append(run)

    print("\n" + "=" * 80)
    print("CONVERGENCE BY CONSTRAINT:")
    print("=" * 80)
    for constraint, runs in sorted(by_constraint.items()):
        print(f"\n{constraint}: {len(runs)} converged runs")
        avg_epoch = sum(r['epoch'] for r in runs) / len(runs)
        print(f"  Average convergence epoch: {avg_epoch:.1f}")

    # Best combinations
    print("\n" + "=" * 80)
    print("TOP 5 FASTEST CONVERGENCES:")
    print("=" * 80)
    for run in sorted(converged, key=lambda x: x['epoch'])[:5]:
        print(f"\nEpoch {run['epoch']}: {run['model']} | constraint_{run['constraint']}")
        print(f"  lr={run['lr']}, lambda={run['lambda_strategy']}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Comprehensive analysis of experiment results
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

def analyze_all_results():
    results_dir = Path('results/our_approach')

    # Data structures
    all_experiments = []
    status_counts = defaultdict(int)
    model_stats = defaultdict(list)
    constraint_stats = defaultdict(list)
    lambda_strategy_stats = defaultdict(list)
    lr_stats = defaultdict(list)

    # Find all run_status.json files
    for run_status_file in results_dir.rglob('run_status.json'):
        exp_dir = run_status_file.parent
        config_file = exp_dir / 'config.json'

        if not config_file.exists():
            continue

        # Read status and config
        with open(run_status_file) as f:
            status_data = json.load(f)

        with open(config_file) as f:
            config = json.load(f)

        # Extract key information
        status = status_data.get('status', 'unknown')
        final_epoch = status_data.get('final_epoch', 0)
        global_satisfied = status_data.get('global_constraint_satisfied', False)
        local_satisfied = status_data.get('local_constraint_satisfied', False)

        model_name = config.get('model_name', 'unknown')
        constraint_key = f"{config.get('global_constraint', [0,0,0])}"
        lambda_strategy = config['hyperparams'].get('lambda_strategy', 'unknown')
        lr = config['hyperparams'].get('lr', 0)

        # Aggregate data
        exp_info = {
            'path': str(exp_dir.relative_to(results_dir)),
            'status': status,
            'final_epoch': final_epoch,
            'global_satisfied': global_satisfied,
            'local_satisfied': local_satisfied,
            'model': model_name,
            'constraint': constraint_key,
            'lambda_strategy': lambda_strategy,
            'lr': lr
        }

        all_experiments.append(exp_info)
        status_counts[status] += 1

        if status == 'converged':
            model_stats[model_name].append(final_epoch)
            constraint_stats[constraint_key].append(final_epoch)
            lambda_strategy_stats[lambda_strategy].append(final_epoch)
            lr_stats[lr].append(final_epoch)

    # Print summary
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments: {len(all_experiments)}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        pct = (count / len(all_experiments)) * 100
        print(f"  {status:12s}: {count:3d} ({pct:5.1f}%)")

    # Converged experiments analysis
    converged_exps = [e for e in all_experiments if e['status'] == 'converged']
    if converged_exps:
        print(f"\n" + "=" * 80)
        print(f"CONVERGED EXPERIMENTS ANALYSIS ({len(converged_exps)} total)")
        print("=" * 80)

        # By model
        print("\nBy Model:")
        for model, epochs in sorted(model_stats.items()):
            avg_epoch = sum(epochs) / len(epochs)
            print(f"  {model:20s}: {len(epochs):2d} converged, avg epoch {avg_epoch:6.1f}")

        # By constraint
        print("\nBy Constraint Level:")
        for constraint, epochs in sorted(constraint_stats.items()):
            avg_epoch = sum(epochs) / len(epochs)
            print(f"  {constraint:30s}: {len(epochs):2d} converged, avg epoch {avg_epoch:6.1f}")

        # By lambda strategy
        print("\nBy Lambda Strategy:")
        for strategy, epochs in sorted(lambda_strategy_stats.items()):
            avg_epoch = sum(epochs) / len(epochs)
            print(f"  {strategy:12s}: {len(epochs):2d} converged, avg epoch {avg_epoch:6.1f}")

        # By learning rate
        print("\nBy Learning Rate:")
        for lr, epochs in sorted(lr_stats.items()):
            avg_epoch = sum(epochs) / len(epochs)
            print(f"  {lr:.5f}: {len(epochs):2d} converged, avg epoch {avg_epoch:6.1f}")

        # Fastest convergence
        print("\nFastest Convergence:")
        fastest = min(converged_exps, key=lambda x: x['final_epoch'])
        print(f"  Path: {fastest['path']}")
        print(f"  Model: {fastest['model']}, Strategy: {fastest['lambda_strategy']}, LR: {fastest['lr']}")
        print(f"  Converged at epoch: {fastest['final_epoch']}")

        # Slowest convergence
        print("\nSlowest Convergence:")
        slowest = max(converged_exps, key=lambda x: x['final_epoch'])
        print(f"  Path: {slowest['path']}")
        print(f"  Model: {slowest['model']}, Strategy: {slowest['lambda_strategy']}, LR: {slowest['lr']}")
        print(f"  Converged at epoch: {slowest['final_epoch']}")

    # Failed experiments analysis
    failed_exps = [e for e in all_experiments if e['status'] == 'failed']
    if failed_exps:
        print(f"\n" + "=" * 80)
        print(f"FAILED EXPERIMENTS ({len(failed_exps)} total)")
        print("=" * 80)

        global_only = [e for e in failed_exps if e['global_satisfied'] and not e['local_satisfied']]
        local_only = [e for e in failed_exps if not e['global_satisfied'] and e['local_satisfied']]
        neither = [e for e in failed_exps if not e['global_satisfied'] and not e['local_satisfied']]

        print(f"\n  Only Global satisfied: {len(global_only)}")
        print(f"  Only Local satisfied:  {len(local_only)}")
        print(f"  Neither satisfied:     {len(neither)}")

    # Interrupted experiments
    interrupted_exps = [e for e in all_experiments if e['status'] == 'interrupted']
    if interrupted_exps:
        print(f"\n" + "=" * 80)
        print(f"INTERRUPTED EXPERIMENTS ({len(interrupted_exps)} total)")
        print("=" * 80)
        for exp in interrupted_exps:
            print(f"  {exp['path']}")

    print("\n" + "=" * 80)

    # Save detailed CSV
    if all_experiments:
        with open('experiment_summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_experiments[0].keys())
            writer.writeheader()
            writer.writerows(all_experiments)
        print(f"\nDetailed results saved to: experiment_summary.csv")

if __name__ == '__main__':
    analyze_all_results()

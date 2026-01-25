#!/usr/bin/env python3
"""
Generate comprehensive comparison CSV combining all three approaches:
- Optimization (our_approach)
- Heuristic (heuristic)
- Saturated (saturated_approach)

Output: comparison_results.csv with all experiments and their accuracies.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def extract_experiment_key(config):
    """Create a unique key for matching experiments across approaches."""
    model = config.get('model_name', 'unknown')
    constraint = config.get('constraint', [])
    constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"
    variation = config.get('variation_name', 'unknown')

    return f"{model}||{constraint_str}||{variation}"


def load_all_experiments():
    """Load experiments from all three approaches."""

    print("=" * 80)
    print("LOADING EXPERIMENTS FROM ALL APPROACHES")
    print("=" * 80)

    # Master list from our_approach
    our_approach_dir = Path('../../results/our_approach')
    heuristic_dir = Path('../../results/heuristic')
    saturated_dir = Path('../../results/saturated_approach')

    experiments = {}

    # 1. Load our_approach (optimization) - this is the master list
    print(f"\n[1/3] Loading optimization results from: {our_approach_dir}")
    our_configs = list(our_approach_dir.rglob('config.json'))
    print(f"  Found {len(our_configs)} experiments")

    for config_path in our_configs:
        with open(config_path) as f:
            config = json.load(f)

        key = extract_experiment_key(config)

        # Extract parameters
        experiments[key] = {
            'model_name': config.get('model_name', 'unknown'),
            'base_model_id': config.get('base_model_id', 'unknown'),
            'constraint_global': config['constraint'][1] if len(config.get('constraint', [])) == 2 else None,
            'constraint_local': config['constraint'][0] if len(config.get('constraint', [])) == 2 else None,
            'learning_rate': config['hyperparams'].get('lr', None),
            'lambda_global': config['hyperparams'].get('lambda_global', None),
            'lambda_local': config['hyperparams'].get('lambda_local', None),
            'lambda_strategy': config['hyperparams'].get('lambda_strategy', 'unknown'),
            'lambda_step': config['hyperparams'].get('lambda_step', None),
            'warmup_epochs': config['hyperparams'].get('warmup_epochs', None),
            'batch_size': config['hyperparams'].get('batch_size', None),
            'dropout': config['hyperparams'].get('dropout', None),
            'variation_name': config.get('variation_name', 'unknown'),
            'optimized_acc': None,
            'heuristic_acc': None,
            'saturated_acc': None,
            'optimized_status': config.get('status', 'unknown'),
            'heuristic_status': None,
            'saturated_status': None,
        }

        # Get optimization accuracy
        if 'results' in config and 'accuracy' in config['results']:
            experiments[key]['optimized_acc'] = config['results']['accuracy']

    # 2. Load heuristic results
    print(f"\n[2/3] Loading heuristic results from: {heuristic_dir}")
    if heuristic_dir.exists():
        heur_configs = list(heuristic_dir.rglob('config.json'))
        print(f"  Found {len(heur_configs)} experiments")

        matched = 0
        for config_path in heur_configs:
            with open(config_path) as f:
                config = json.load(f)

            key = extract_experiment_key(config)

            if key in experiments:
                if 'results' in config and 'accuracy' in config['results']:
                    experiments[key]['heuristic_acc'] = config['results']['accuracy']
                experiments[key]['heuristic_status'] = config.get('status', 'unknown')
                matched += 1

        print(f"  Matched {matched}/{len(heur_configs)} experiments")
    else:
        print(f"  ⚠️  Directory not found - run run_all_heuristics.py first")

    # 3. Load saturated results
    print(f"\n[3/3] Loading saturated results from: {saturated_dir}")
    if saturated_dir.exists():
        sat_configs = list(saturated_dir.rglob('config.json'))
        print(f"  Found {len(sat_configs)} experiments")

        matched = 0
        for config_path in sat_configs:
            with open(config_path) as f:
                config = json.load(f)

            key = extract_experiment_key(config)

            if key in experiments:
                if 'results' in config and 'accuracy' in config['results']:
                    experiments[key]['saturated_acc'] = config['results']['accuracy']
                experiments[key]['saturated_status'] = config.get('status', 'unknown')
                matched += 1

        print(f"  Matched {matched}/{len(sat_configs)} experiments")
    else:
        print(f"  ⚠️  Directory not found - run run_all_saturations.py first")

    return experiments


def generate_comparison_csv(experiments, output_path='comparison_results.csv'):
    """Generate comprehensive comparison CSV."""

    print(f"\n" + "=" * 80)
    print("GENERATING COMPARISON CSV")
    print("=" * 80)

    # Define columns
    fieldnames = [
        'model_name',
        'base_model_id',
        'constraint_global',
        'constraint_local',
        'learning_rate',
        'lambda_strategy',
        'lambda_global',
        'lambda_local',
        'lambda_step',
        'warmup_epochs',
        'batch_size',
        'dropout',
        'variation_name',
        'optimized_acc',
        'heuristic_acc',
        'saturated_acc',
        'optimized_status',
        'heuristic_status',
        'saturated_status',
    ]

    # Sort experiments by model, constraint, learning rate
    sorted_experiments = sorted(
        experiments.items(),
        key=lambda x: (
            x[1]['model_name'],
            x[1]['constraint_global'] or 0,
            x[1]['constraint_local'] or 0,
            x[1]['learning_rate'] or 0,
            x[1]['lambda_strategy']
        )
    )

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key, exp in sorted_experiments:
            writer.writerow(exp)

    print(f"✓ Saved: {output_path}")
    print(f"  Total experiments: {len(experiments)}")

    # Calculate statistics
    stats = {
        'complete_all_3': 0,
        'missing_heuristic': 0,
        'missing_saturated': 0,
        'heuristic_wins': 0,
        'optimized_wins': 0,
        'saturated_wins': 0,
        'heuristic_better_than_optimized': 0,
        'saturated_better_than_optimized': 0,
        'saturated_better_than_heuristic': 0,
    }

    for exp in experiments.values():
        opt_acc = exp['optimized_acc']
        heur_acc = exp['heuristic_acc']
        sat_acc = exp['saturated_acc']

        # Count complete experiments
        if opt_acc is not None and heur_acc is not None and sat_acc is not None:
            stats['complete_all_3'] += 1

            # Find winner
            best_acc = max(opt_acc, heur_acc, sat_acc)
            if heur_acc == best_acc:
                stats['heuristic_wins'] += 1
            elif sat_acc == best_acc:
                stats['saturated_wins'] += 1
            elif opt_acc == best_acc:
                stats['optimized_wins'] += 1

            # Pairwise comparisons
            if heur_acc > opt_acc:
                stats['heuristic_better_than_optimized'] += 1
            if sat_acc > opt_acc:
                stats['saturated_better_than_optimized'] += 1
            if sat_acc > heur_acc:
                stats['saturated_better_than_heuristic'] += 1

        if heur_acc is None:
            stats['missing_heuristic'] += 1
        if sat_acc is None:
            stats['missing_saturated'] += 1

    # Print summary
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total = len(experiments)
    print(f"\nData Completeness:")
    print(f"  Total experiments: {total}")
    print(f"  Complete (all 3 approaches): {stats['complete_all_3']}")
    print(f"  Missing heuristic: {stats['missing_heuristic']}")
    print(f"  Missing saturated: {stats['missing_saturated']}")

    if stats['complete_all_3'] > 0:
        complete = stats['complete_all_3']
        print(f"\nBest Approach (wins most experiments):")
        print(f"  Heuristic: {stats['heuristic_wins']}/{complete} ({stats['heuristic_wins']/complete*100:.1f}%)")
        print(f"  Saturated: {stats['saturated_wins']}/{complete} ({stats['saturated_wins']/complete*100:.1f}%)")
        print(f"  Optimized: {stats['optimized_wins']}/{complete} ({stats['optimized_wins']/complete*100:.1f}%)")

        print(f"\nPairwise Comparisons:")
        print(f"  Heuristic > Optimized: {stats['heuristic_better_than_optimized']}/{complete} "
              f"({stats['heuristic_better_than_optimized']/complete*100:.1f}%)")
        print(f"  Saturated > Optimized: {stats['saturated_better_than_optimized']}/{complete} "
              f"({stats['saturated_better_than_optimized']/complete*100:.1f}%)")
        print(f"  Saturated > Heuristic: {stats['saturated_better_than_heuristic']}/{complete} "
              f"({stats['saturated_better_than_heuristic']/complete*100:.1f}%)")

        # Calculate average differences
        diffs_heur_opt = []
        diffs_sat_opt = []
        diffs_sat_heur = []

        for exp in experiments.values():
            if exp['optimized_acc'] and exp['heuristic_acc'] and exp['saturated_acc']:
                diffs_heur_opt.append(exp['heuristic_acc'] - exp['optimized_acc'])
                diffs_sat_opt.append(exp['saturated_acc'] - exp['optimized_acc'])
                diffs_sat_heur.append(exp['saturated_acc'] - exp['heuristic_acc'])

        if diffs_heur_opt:
            import numpy as np
            print(f"\nAverage Accuracy Differences:")
            print(f"  Heuristic - Optimized: {np.mean(diffs_heur_opt):+.4f} (σ={np.std(diffs_heur_opt):.4f})")
            print(f"  Saturated - Optimized: {np.mean(diffs_sat_opt):+.4f} (σ={np.std(diffs_sat_opt):.4f})")
            print(f"  Saturated - Heuristic: {np.mean(diffs_sat_heur):+.4f} (σ={np.std(diffs_sat_heur):.4f})")

    print("=" * 80)


def main():
    # Load all experiments
    experiments = load_all_experiments()

    # Generate CSV
    generate_comparison_csv(experiments)

    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("  1. Open comparison_results.csv in Excel/spreadsheet")
    print("  2. Analyze which approach performs best")
    print("  3. Check if constraint training consistently underperforms")
    print("  4. See if saturation bridges the gap")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze V2 Static Lambda Experiment Results.

This script analyzes the fine-tuned lambda experiments (0.02, 0.03, 0.05, 0.07)
to determine if we've found the sweet spot between:
- Constraint satisfaction (convergence rate)
- Prediction quality (avoiding overfitting to Graduate predictions)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Base results directory
RESULTS_DIR = Path('results/static_lambda')

# Lambda value mapping (variation_name -> actual lambda value)
LAMBDA_MAP = {
    'lambda_0.2': 0.02,
    'lambda_0.3': 0.03,
    'lambda_0.5': 0.05,
    'lambda_0.7': 0.07
}

# Constraint type descriptions
CONSTRAINT_DESC = {
    '0.9_0.8': '[Soft, Soft]',
    '0.3_0.8': '[Hard, Soft]',
    '0.8_0.3': '[Soft, Hard]',
    '0.3_0.3': '[Hard, Hard]'
}


def find_all_v2_experiments() -> List[Tuple[Path, str, str, str]]:
    """Find all V2 experiment directories.

    Returns:
        List of tuples: (experiment_path, model, constraint, lambda_variation)
    """
    experiments = []

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for constraint_dir in model_dir.iterdir():
            if not constraint_dir.is_dir():
                continue
            constraint = constraint_dir.name.replace('constraint_', '')

            lambda_regime_dir = constraint_dir / 'lambda_fine_tune'
            if not lambda_regime_dir.exists():
                continue

            for lambda_dir in lambda_regime_dir.iterdir():
                if not lambda_dir.is_dir():
                    continue
                lambda_var = lambda_dir.name

                experiments.append((lambda_dir, model, constraint, lambda_var))

    return experiments


def analyze_experiment(exp_path: Path) -> Dict:
    """Analyze a single experiment.

    Args:
        exp_path: Path to experiment directory

    Returns:
        Dictionary with analysis results
    """
    result = {
        'success': False,
        'converged': False,
        'early_stopped': False,
        'final_epoch': None,
        'prediction_dist': None,
        'graduate_pct': None,
        'accuracy': None,
        'f1_macro': None,
        'recall_per_class': None
    }

    # Check if experiment succeeded (has final_predictions.csv)
    predictions_file = exp_path / 'final_predictions.csv'
    if predictions_file.exists():
        result['success'] = True

        # Analyze prediction distribution
        df_pred = pd.read_csv(predictions_file)
        pred_counts = df_pred['Predicted_Label'].value_counts().sort_index()
        total = len(df_pred)

        result['prediction_dist'] = {
            0: pred_counts.get(0, 0),
            1: pred_counts.get(1, 0),
            2: pred_counts.get(2, 0)
        }
        result['graduate_pct'] = (pred_counts.get(2, 0) / total) * 100
        result['accuracy'] = (df_pred['True_Label'] == df_pred['Predicted_Label']).mean()

        # Get evaluation metrics if available
        metrics_file = exp_path / 'evaluation_metrics.csv'
        if metrics_file.exists():
            try:
                # Read the file manually to handle irregular structure
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()

                # Parse F1 macro
                for line in lines:
                    if 'F1-Score (Macro)' in line:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            result['f1_macro'] = float(parts[1])

                # Parse per-class recall
                result['recall_per_class'] = {}
                for i, line in enumerate(lines):
                    for class_name in ['Dropout', 'Enrolled', 'Graduate']:
                        if line.startswith(class_name + ','):
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                result['recall_per_class'][class_name] = float(parts[2])
            except Exception as e:
                # If parsing fails, skip metrics
                pass

    # Check training log for convergence info
    log_file = exp_path / 'training_log.csv'
    if log_file.exists():
        df_log = pd.read_csv(log_file)

        if not df_log.empty:
            result['final_epoch'] = df_log['Epoch'].max()

            # Check if converged (last epoch has both constraints satisfied)
            last_row = df_log.iloc[-1]
            result['converged'] = (last_row['Global_Satisfied'] == 1 and
                                  last_row['Local_Satisfied'] == 1)

            # Check if early stopped (final epoch < 300)
            result['early_stopped'] = result['final_epoch'] < 300

    return result


def main():
    print("=" * 80)
    print("V2 STATIC LAMBDA RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Find all experiments
    experiments = find_all_v2_experiments()
    print(f"Found {len(experiments)} V2 experiments")
    print()

    # Analyze each experiment
    results = {}
    for exp_path, model, constraint, lambda_var in experiments:
        key = (model, constraint, lambda_var)
        results[key] = analyze_experiment(exp_path)

    # Aggregate statistics
    print("=" * 80)
    print("1. SUCCESS RATE BY LAMBDA VALUE")
    print("=" * 80)
    print()

    lambda_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'graduate_pcts': []})

    for (model, constraint, lambda_var), result in results.items():
        lambda_val = LAMBDA_MAP[lambda_var]
        lambda_stats[lambda_val]['total'] += 1
        if result['success']:
            lambda_stats[lambda_val]['success'] += 1
            if result['graduate_pct'] is not None:
                lambda_stats[lambda_val]['graduate_pcts'].append(result['graduate_pct'])

    print(f"{'Lambda':<10} {'Success':<10} {'Rate':<10} {'Avg Graduate %':<20}")
    print("-" * 80)

    for lambda_val in sorted(lambda_stats.keys()):
        stats = lambda_stats[lambda_val]
        success_rate = (stats['success'] / stats['total']) * 100
        avg_grad = np.mean(stats['graduate_pcts']) if stats['graduate_pcts'] else 0
        print(f"{lambda_val:<10.2f} {stats['success']}/{stats['total']:<8} "
              f"{success_rate:>6.1f}%    {avg_grad:>6.1f}%")

    print()

    # Model performance
    print("=" * 80)
    print("2. SUCCESS RATE BY MODEL")
    print("=" * 80)
    print()

    model_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'graduate_pcts': []})

    for (model, constraint, lambda_var), result in results.items():
        model_stats[model]['total'] += 1
        if result['success']:
            model_stats[model]['success'] += 1
            if result['graduate_pct'] is not None:
                model_stats[model]['graduate_pcts'].append(result['graduate_pct'])

    print(f"{'Model':<20} {'Success':<10} {'Rate':<10} {'Avg Graduate %':<20}")
    print("-" * 80)

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        success_rate = (stats['success'] / stats['total']) * 100
        avg_grad = np.mean(stats['graduate_pcts']) if stats['graduate_pcts'] else 0
        print(f"{model:<20} {stats['success']}/{stats['total']:<8} "
              f"{success_rate:>6.1f}%    {avg_grad:>6.1f}%")

    print()

    # Constraint performance
    print("=" * 80)
    print("3. SUCCESS RATE BY CONSTRAINT TYPE")
    print("=" * 80)
    print()

    constraint_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'graduate_pcts': []})

    for (model, constraint, lambda_var), result in results.items():
        constraint_stats[constraint]['total'] += 1
        if result['success']:
            constraint_stats[constraint]['success'] += 1
            if result['graduate_pct'] is not None:
                constraint_stats[constraint]['graduate_pcts'].append(result['graduate_pct'])

    print(f"{'Constraint':<20} {'Type':<15} {'Success':<10} {'Rate':<10} {'Avg Graduate %':<20}")
    print("-" * 80)

    for constraint in sorted(constraint_stats.keys()):
        stats = constraint_stats[constraint]
        success_rate = (stats['success'] / stats['total']) * 100
        avg_grad = np.mean(stats['graduate_pcts']) if stats['graduate_pcts'] else 0
        desc = CONSTRAINT_DESC.get(constraint, constraint)
        print(f"{constraint:<20} {desc:<15} {stats['success']}/{stats['total']:<8} "
              f"{success_rate:>6.1f}%    {avg_grad:>6.1f}%")

    print()

    # Early stopping analysis
    print("=" * 80)
    print("4. EARLY STOPPING ANALYSIS")
    print("=" * 80)
    print()

    early_stopped = sum(1 for r in results.values() if r['early_stopped'])
    ran_full = sum(1 for r in results.values() if r['final_epoch'] == 300)

    print(f"Experiments with early stopping: {early_stopped}/{len(results)} "
          f"({early_stopped/len(results)*100:.1f}%)")
    print(f"Experiments running full 300 epochs: {ran_full}/{len(results)} "
          f"({ran_full/len(results)*100:.1f}%)")

    # Show average epochs for successful experiments by lambda
    print()
    print(f"{'Lambda':<10} {'Avg Epochs (Success)':<25} {'Early Stop %':<20}")
    print("-" * 80)

    for lambda_val in sorted(LAMBDA_MAP.values()):
        lambda_var_name = [k for k, v in LAMBDA_MAP.items() if v == lambda_val][0]

        epochs = []
        early_stops = 0
        for (model, constraint, lv), result in results.items():
            if lv == lambda_var_name and result['success']:
                if result['final_epoch']:
                    epochs.append(result['final_epoch'])
                    if result['early_stopped']:
                        early_stops += 1

        if epochs:
            avg_epochs = np.mean(epochs)
            early_pct = (early_stops / len(epochs)) * 100
            print(f"{lambda_val:<10.2f} {avg_epochs:>6.1f}                   {early_pct:>6.1f}%")

    print()

    # Detailed failure analysis
    print("=" * 80)
    print("5. FAILURE ANALYSIS")
    print("=" * 80)
    print()

    failures = [(model, constraint, lambda_var)
                for (model, constraint, lambda_var), result in results.items()
                if not result['success']]

    print(f"Total failures: {len(failures)}/{len(results)} ({len(failures)/len(results)*100:.1f}%)")
    print()

    if failures:
        print("Failed experiments:")
        print(f"{'Model':<20} {'Constraint':<15} {'Lambda':<10}")
        print("-" * 80)
        for model, constraint, lambda_var in sorted(failures):
            lambda_val = LAMBDA_MAP[lambda_var]
            constraint_desc = CONSTRAINT_DESC.get(constraint, constraint)
            print(f"{model:<20} {constraint_desc:<15} {lambda_val:<10.2f}")

    print()

    # Prediction quality analysis
    print("=" * 80)
    print("6. PREDICTION QUALITY ANALYSIS")
    print("=" * 80)
    print()
    print("Average prediction distribution across all successful experiments:")
    print()

    all_pred_dists = [result['prediction_dist'] for result in results.values()
                      if result['prediction_dist'] is not None]

    if all_pred_dists:
        avg_dropout = np.mean([d[0] for d in all_pred_dists])
        avg_enrolled = np.mean([d[1] for d in all_pred_dists])
        avg_graduate = np.mean([d[2] for d in all_pred_dists])
        total = avg_dropout + avg_enrolled + avg_graduate

        print(f"  Dropout:  {avg_dropout:>6.1f} ({avg_dropout/total*100:>5.1f}%)")
        print(f"  Enrolled: {avg_enrolled:>6.1f} ({avg_enrolled/total*100:>5.1f}%)")
        print(f"  Graduate: {avg_graduate:>6.1f} ({avg_graduate/total*100:>5.1f}%)")
        print()
        print(f"  Graduate percentage: {avg_graduate/total*100:.1f}%")

    print()
    print("By lambda value:")
    print()
    print(f"{'Lambda':<10} {'Dropout %':<12} {'Enrolled %':<12} {'Graduate %':<12}")
    print("-" * 80)

    for lambda_val in sorted(LAMBDA_MAP.values()):
        lambda_var_name = [k for k, v in LAMBDA_MAP.items() if v == lambda_val][0]

        pred_dists = []
        for (model, constraint, lv), result in results.items():
            if lv == lambda_var_name and result['prediction_dist'] is not None:
                pred_dists.append(result['prediction_dist'])

        if pred_dists:
            avg_dropout = np.mean([d[0] for d in pred_dists])
            avg_enrolled = np.mean([d[1] for d in pred_dists])
            avg_graduate = np.mean([d[2] for d in pred_dists])
            total = avg_dropout + avg_enrolled + avg_graduate

            print(f"{lambda_val:<10.2f} {avg_dropout/total*100:>6.1f}%      "
                  f"{avg_enrolled/total*100:>6.1f}%      {avg_graduate/total*100:>6.1f}%")

    print()

    # Performance metrics
    print("=" * 80)
    print("7. PERFORMANCE METRICS")
    print("=" * 80)
    print()

    accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    f1_scores = [r['f1_macro'] for r in results.values() if r['f1_macro'] is not None]

    if accuracies:
        print(f"Average accuracy across successful experiments: {np.mean(accuracies):.4f}")
    if f1_scores:
        print(f"Average F1 (macro) across successful experiments: {np.mean(f1_scores):.4f}")

    print()

    # Summary and recommendations
    print("=" * 80)
    print("8. SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find best lambda value
    best_lambda = None
    best_score = 0

    for lambda_val in sorted(LAMBDA_MAP.values()):
        stats = lambda_stats[lambda_val]
        if stats['total'] > 0:
            success_rate = stats['success'] / stats['total']
            avg_grad = np.mean(stats['graduate_pcts']) if stats['graduate_pcts'] else 0

            # Score: prioritize success rate, penalize extreme Graduate %
            # Ideal Graduate % is ~75-85% (based on true distribution)
            grad_penalty = abs(avg_grad - 80) / 100  # Penalty for being far from 80%
            score = success_rate - grad_penalty

            if score > best_score:
                best_score = score
                best_lambda = lambda_val

    print(f"Best lambda value: {best_lambda:.2f}")
    print()

    # Compare with V1 results
    print("Comparison with V1 results:")
    print("  V1 λ=0.01: 33% success, 86.7% Graduate (too low lambda)")
    print("  V1 λ=0.1:  56% success, 93.2% Graduate (marginal)")
    print("  V1 λ=1.0:  100% success, 95.1% Graduate (overfitting!)")
    print()

    if best_lambda:
        best_stats = lambda_stats[best_lambda]
        best_success = (best_stats['success'] / best_stats['total']) * 100
        best_grad = np.mean(best_stats['graduate_pcts']) if best_stats['graduate_pcts'] else 0
        print(f"  V2 λ={best_lambda}: {best_success:.1f}% success, {best_grad:.1f}% Graduate")
        print()

        if best_grad < 90:
            print("✓ SUCCESS: V2 has significantly reduced Graduate overfitting!")
        else:
            print("⚠ WARNING: Graduate percentage still high - may be overfitting to constraints")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()

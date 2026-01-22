#!/usr/bin/env python3
"""
Quick summary of analysis results - prints key findings to console.
"""

import csv
from collections import defaultdict
import numpy as np

def load_data():
    """Load master results CSV."""
    experiments = []
    with open('comparison_evaluations/master_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['learning_rate'] = float(row['learning_rate'])
            row['final_epoch'] = int(row['final_epoch']) if row['final_epoch'] else 0
            row['converged'] = row['converged'] == 'True'
            row['test_accuracy'] = float(row['test_accuracy']) if row['test_accuracy'] else None
            experiments.append(row)
    return experiments

def print_top_configurations(experiments, n=10):
    """Print top N configurations by test accuracy."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] is not None]
    sorted_exps = sorted(converged, key=lambda x: x['test_accuracy'], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP {n} CONFIGURATIONS BY TEST ACCURACY")
    print('='*100)
    print(f"{'Rank':<6}{'Accuracy':<12}{'Epochs':<10}{'Model':<18}{'LR':<12}{'Strategy':<12}{'Constraint':<12}")
    print('-'*100)

    for i, exp in enumerate(sorted_exps[:n], 1):
        print(f"{i:<6}{exp['test_accuracy']:<12.4f}{exp['final_epoch']:<10}"
              f"{exp['model']:<18}{exp['learning_rate']:<12.6f}"
              f"{exp['lambda_strategy']:<12}{exp['constraint']:<12}")

def print_fastest_configurations(experiments, n=10):
    """Print fastest N configurations (lowest epochs to convergence)."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] is not None]
    sorted_exps = sorted(converged, key=lambda x: x['final_epoch'])

    print(f"\n{'='*100}")
    print(f"TOP {n} FASTEST CONVERGING CONFIGURATIONS")
    print('='*100)
    print(f"{'Rank':<6}{'Epochs':<10}{'Accuracy':<12}{'Model':<18}{'LR':<12}{'Strategy':<12}{'Constraint':<12}")
    print('-'*100)

    for i, exp in enumerate(sorted_exps[:n], 1):
        print(f"{i:<6}{exp['final_epoch']:<10}{exp['test_accuracy']:<12.4f}"
              f"{exp['model']:<18}{exp['learning_rate']:<12.6f}"
              f"{exp['lambda_strategy']:<12}{exp['constraint']:<12}")

def print_best_by_model(experiments):
    """Print best configuration for each model."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] is not None]

    model_best = {}
    for exp in converged:
        model = exp['model']
        if model not in model_best or exp['test_accuracy'] > model_best[model]['test_accuracy']:
            model_best[model] = exp

    print(f"\n{'='*100}")
    print("BEST CONFIGURATION PER MODEL")
    print('='*100)
    print(f"{'Model':<18}{'Accuracy':<12}{'Epochs':<10}{'LR':<12}{'Strategy':<12}{'Constraint':<12}")
    print('-'*100)

    for model in sorted(model_best.keys()):
        exp = model_best[model]
        print(f"{model:<18}{exp['test_accuracy']:<12.4f}{exp['final_epoch']:<10}"
              f"{exp['learning_rate']:<12.6f}{exp['lambda_strategy']:<12}{exp['constraint']:<12}")

def print_performance_by_hyperparameter(experiments):
    """Print average performance grouped by different hyperparameters."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] is not None]

    print(f"\n{'='*100}")
    print("AVERAGE PERFORMANCE BY HYPERPARAMETER")
    print('='*100)

    # By Learning Rate
    lr_stats = defaultdict(lambda: {'acc': [], 'epochs': []})
    for exp in converged:
        lr_stats[exp['learning_rate']]['acc'].append(exp['test_accuracy'])
        lr_stats[exp['learning_rate']]['epochs'].append(exp['final_epoch'])

    print("\nBy Learning Rate:")
    print(f"{'LR':<15}{'Avg Accuracy':<15}{'Avg Epochs':<15}{'Count':<10}")
    print('-'*55)
    for lr in sorted(lr_stats.keys()):
        avg_acc = np.mean(lr_stats[lr]['acc'])
        avg_epochs = np.mean(lr_stats[lr]['epochs'])
        count = len(lr_stats[lr]['acc'])
        print(f"{lr:<15.6f}{avg_acc:<15.4f}{avg_epochs:<15.1f}{count:<10}")

    # By Lambda Strategy
    strat_stats = defaultdict(lambda: {'acc': [], 'epochs': []})
    for exp in converged:
        strat_stats[exp['lambda_strategy']]['acc'].append(exp['test_accuracy'])
        strat_stats[exp['lambda_strategy']]['epochs'].append(exp['final_epoch'])

    print("\nBy Lambda Strategy:")
    print(f"{'Strategy':<15}{'Avg Accuracy':<15}{'Avg Epochs':<15}{'Count':<10}")
    print('-'*55)
    for strat in sorted(strat_stats.keys()):
        avg_acc = np.mean(strat_stats[strat]['acc'])
        avg_epochs = np.mean(strat_stats[strat]['epochs'])
        count = len(strat_stats[strat]['acc'])
        print(f"{strat:<15}{avg_acc:<15.4f}{avg_epochs:<15.1f}{count:<10}")

    # By Model
    model_stats = defaultdict(lambda: {'acc': [], 'epochs': []})
    for exp in converged:
        model_stats[exp['model']]['acc'].append(exp['test_accuracy'])
        model_stats[exp['model']]['epochs'].append(exp['final_epoch'])

    print("\nBy Model:")
    print(f"{'Model':<15}{'Avg Accuracy':<15}{'Avg Epochs':<15}{'Count':<10}")
    print('-'*55)
    for model in sorted(model_stats.keys()):
        avg_acc = np.mean(model_stats[model]['acc'])
        avg_epochs = np.mean(model_stats[model]['epochs'])
        count = len(model_stats[model]['acc'])
        print(f"{model:<15}{avg_acc:<15.4f}{avg_epochs:<15.1f}{count:<10}")

    # By Constraint
    constraint_stats = defaultdict(lambda: {'acc': [], 'epochs': []})
    for exp in converged:
        constraint_stats[exp['constraint']]['acc'].append(exp['test_accuracy'])
        constraint_stats[exp['constraint']]['epochs'].append(exp['final_epoch'])

    print("\nBy Constraint Level:")
    print(f"{'Constraint':<15}{'Avg Accuracy':<15}{'Avg Epochs':<15}{'Count':<10}")
    print('-'*55)
    for constraint in sorted(constraint_stats.keys()):
        avg_acc = np.mean(constraint_stats[constraint]['acc'])
        avg_epochs = np.mean(constraint_stats[constraint]['epochs'])
        count = len(constraint_stats[constraint]['acc'])
        print(f"{constraint:<15}{avg_acc:<15.4f}{avg_epochs:<15.1f}{count:<10}")

def print_failed_experiments(experiments):
    """Print details of failed experiments."""
    failed = [e for e in experiments if not e['converged']]

    if not failed:
        print("\n🎉 NO FAILED EXPERIMENTS!")
        return

    print(f"\n{'='*100}")
    print(f"FAILED EXPERIMENTS ({len(failed)} total)")
    print('='*100)
    print(f"{'Model':<18}{'LR':<12}{'Strategy':<12}{'Constraint':<12}{'Reason':<40}")
    print('-'*100)

    for exp in failed:
        reason = exp['details'][:40] + '...' if len(exp['details']) > 40 else exp['details']
        print(f"{exp['model']:<18}{exp['learning_rate']:<12.6f}"
              f"{exp['lambda_strategy']:<12}{exp['constraint']:<12}{reason:<40}")

def main():
    """Main execution."""
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS SUMMARY")
    print("="*100)

    experiments = load_data()

    print(f"\nTotal Experiments: {len(experiments)}")
    converged_count = sum(1 for e in experiments if e['converged'])
    print(f"Converged: {converged_count} ({converged_count/len(experiments)*100:.1f}%)")
    print(f"Failed: {len(experiments) - converged_count} ({(len(experiments)-converged_count)/len(experiments)*100:.1f}%)")

    print_top_configurations(experiments, n=10)
    print_fastest_configurations(experiments, n=10)
    print_best_by_model(experiments)
    print_performance_by_hyperparameter(experiments)
    print_failed_experiments(experiments)

    print(f"\n{'='*100}")
    print("RECOMMENDATIONS")
    print('='*100)

    # Find overall best
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] is not None]
    best = max(converged, key=lambda x: x['test_accuracy'])
    fastest = min(converged, key=lambda x: x['final_epoch'])

    print(f"\n🏆 HIGHEST ACCURACY CONFIGURATION:")
    print(f"   Model: {best['model']}, LR: {best['learning_rate']}, Strategy: {best['lambda_strategy']}, Constraint: {best['constraint']}")
    print(f"   Accuracy: {best['test_accuracy']:.4f}, Converged at epoch: {best['final_epoch']}")

    print(f"\n⚡ FASTEST CONVERGENCE CONFIGURATION:")
    print(f"   Model: {fastest['model']}, LR: {fastest['learning_rate']}, Strategy: {fastest['lambda_strategy']}, Constraint: {fastest['constraint']}")
    print(f"   Converged at epoch: {fastest['final_epoch']}, Accuracy: {fastest['test_accuracy']:.4f}")

    # Best balanced (good accuracy + fast)
    balanced = min(converged, key=lambda x: x['final_epoch'] / (x['test_accuracy'] + 0.01))
    print(f"\n⚖️  BEST BALANCED CONFIGURATION (Accuracy/Speed):")
    print(f"   Model: {balanced['model']}, LR: {balanced['learning_rate']}, Strategy: {balanced['lambda_strategy']}, Constraint: {balanced['constraint']}")
    print(f"   Accuracy: {balanced['test_accuracy']:.4f}, Converged at epoch: {balanced['final_epoch']}")

    print(f"\n{'='*100}\n")

if __name__ == '__main__':
    main()

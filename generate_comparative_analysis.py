#!/usr/bin/env python3
"""
Comprehensive comparative analysis script for all experiment results.
Generates CSV summary and comparison graphs.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

def extract_experiment_data():
    """Extract all experiment data into a structured format."""
    results_dir = Path('results/our_approach')
    all_experiments = []

    for run_status_file in sorted(results_dir.rglob('run_status.json')):
        exp_dir = run_status_file.parent
        config_file = exp_dir / 'config.json'
        metrics_file = exp_dir / 'evaluation_metrics.csv'

        if not config_file.exists():
            continue

        # Read run status
        with open(run_status_file) as f:
            status_data = json.load(f)

        # Read config
        with open(config_file) as f:
            config = json.load(f)

        # Read evaluation metrics if converged
        test_accuracy = None
        if metrics_file.exists() and status_data.get('status') == 'converged':
            try:
                with open(metrics_file) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2 and row[0] == 'Overall Accuracy':
                            test_accuracy = float(row[1])
                            break
            except:
                pass

        # Extract constraint from path
        path_parts = str(exp_dir).split('/')
        constraint_key = 'unknown'
        for part in path_parts:
            if part.startswith('constraint_'):
                constraint_key = part.replace('constraint_', '')
                break

        # Build experiment record
        exp_record = {
            'path': str(exp_dir.relative_to(results_dir)),
            'model': config.get('model_name', 'unknown'),
            'constraint': constraint_key,
            'learning_rate': config['hyperparams'].get('lr', 0),
            'lambda_strategy': config['hyperparams'].get('lambda_strategy', 'unknown'),
            'lambda_global': config['hyperparams'].get('lambda_global', 0),
            'lambda_local': config['hyperparams'].get('lambda_local', 0),
            'lambda_step': config['hyperparams'].get('lambda_step', 0),
            'batch_size': config['hyperparams'].get('batch_size', 0),
            'warmup_epochs': config['hyperparams'].get('warmup_epochs', 0),
            'max_epochs': config['hyperparams'].get('epochs', 0),
            'constraint_threshold': config['hyperparams'].get('constraint_threshold', 0),
            'status': status_data.get('status', 'unknown'),
            'converged': status_data.get('status') == 'converged',
            'final_epoch': status_data.get('final_epoch', 0),
            'global_constraint_satisfied': status_data.get('global_constraint_satisfied', False),
            'local_constraint_satisfied': status_data.get('local_constraint_satisfied', False),
            'test_accuracy': test_accuracy if test_accuracy is not None else '',
            'details': status_data.get('details', '')
        }

        all_experiments.append(exp_record)

    return all_experiments


def save_master_csv(experiments, output_path='comparison_evaluations/master_results.csv'):
    """Save all experiment data to a comprehensive CSV file."""
    if not experiments:
        print("No experiments found!")
        return

    fieldnames = experiments[0].keys()

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiments)

    print(f"Master CSV saved to: {output_path}")
    print(f"Total experiments: {len(experiments)}")


def plot_accuracy_by_learning_rate(experiments):
    """Graph 1: Average test accuracy by learning rate for each model."""
    # Filter converged experiments with test_accuracy
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] != '']

    if not converged:
        print("No converged experiments with test accuracy found!")
        return

    # Group by model and learning rate
    data = defaultdict(lambda: defaultdict(list))
    for exp in converged:
        model = exp['model']
        lr = exp['learning_rate']
        acc = float(exp['test_accuracy'])
        data[model][lr].append(acc)

    # Calculate averages
    models = sorted(data.keys())
    learning_rates = sorted(set(lr for model_data in data.values() for lr in model_data.keys()))

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(learning_rates))
    width = 0.25

    for i, model in enumerate(models):
        avg_accuracies = []
        for lr in learning_rates:
            if lr in data[model]:
                avg_acc = np.mean(data[model][lr])
                avg_accuracies.append(avg_acc)
            else:
                avg_accuracies.append(0)

        ax.bar(x + i * width, avg_accuracies, width, label=model)

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Average Test Accuracy', fontsize=12)
    ax.set_title('Average Test Accuracy by Learning Rate (Converged Experiments)', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{lr:.5f}' for lr in learning_rates], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/accuracy_by_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/accuracy_by_learning_rate.png")


def plot_accuracy_by_lambda_strategy(experiments):
    """Graph 2: Average test accuracy by lambda strategy for each model."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] != '']

    if not converged:
        return

    # Group by model and strategy
    data = defaultdict(lambda: defaultdict(list))
    for exp in converged:
        model = exp['model']
        strategy = exp['lambda_strategy']
        acc = float(exp['test_accuracy'])
        data[model][strategy].append(acc)

    models = sorted(data.keys())
    strategies = sorted(set(strat for model_data in data.values() for strat in model_data.keys()))

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(strategies))
    width = 0.25

    for i, model in enumerate(models):
        avg_accuracies = []
        for strategy in strategies:
            if strategy in data[model]:
                avg_acc = np.mean(data[model][strategy])
                avg_accuracies.append(avg_acc)
            else:
                avg_accuracies.append(0)

        ax.bar(x + i * width, avg_accuracies, width, label=model)

    ax.set_xlabel('Lambda Strategy', fontsize=12)
    ax.set_ylabel('Average Test Accuracy', fontsize=12)
    ax.set_title('Average Test Accuracy by Lambda Strategy (Converged Experiments)', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/accuracy_by_lambda_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/accuracy_by_lambda_strategy.png")


def plot_convergence_rate_by_factors(experiments):
    """Graph 3: Convergence rate by model type, learning rate, and strategy."""
    # Create three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. By Model Type
    model_data = defaultdict(lambda: {'converged': 0, 'total': 0})
    for exp in experiments:
        model = exp['model']
        model_data[model]['total'] += 1
        if exp['converged']:
            model_data[model]['converged'] += 1

    models = sorted(model_data.keys())
    conv_rates = [model_data[m]['converged'] / model_data[m]['total'] * 100 for m in models]

    axes[0].bar(models, conv_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Convergence Rate (%)', fontsize=11)
    axes[0].set_title('Convergence Rate by Model', fontsize=12)
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (model, rate) in enumerate(zip(models, conv_rates)):
        axes[0].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10)

    # 2. By Learning Rate
    lr_data = defaultdict(lambda: {'converged': 0, 'total': 0})
    for exp in experiments:
        lr = exp['learning_rate']
        lr_data[lr]['total'] += 1
        if exp['converged']:
            lr_data[lr]['converged'] += 1

    lrs = sorted(lr_data.keys())
    conv_rates_lr = [lr_data[lr]['converged'] / lr_data[lr]['total'] * 100 for lr in lrs]

    axes[1].bar(range(len(lrs)), conv_rates_lr, color='#d62728')
    axes[1].set_ylabel('Convergence Rate (%)', fontsize=11)
    axes[1].set_title('Convergence Rate by Learning Rate', fontsize=12)
    axes[1].set_xticks(range(len(lrs)))
    axes[1].set_xticklabels([f'{lr:.5f}' for lr in lrs], rotation=45, ha='right')
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)

    for i, rate in enumerate(conv_rates_lr):
        axes[1].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10)

    # 3. By Lambda Strategy
    strat_data = defaultdict(lambda: {'converged': 0, 'total': 0})
    for exp in experiments:
        strategy = exp['lambda_strategy']
        strat_data[strategy]['total'] += 1
        if exp['converged']:
            strat_data[strategy]['converged'] += 1

    strategies = sorted(strat_data.keys())
    conv_rates_strat = [strat_data[s]['converged'] / strat_data[s]['total'] * 100 for s in strategies]

    axes[2].bar(strategies, conv_rates_strat, color='#9467bd')
    axes[2].set_ylabel('Convergence Rate (%)', fontsize=11)
    axes[2].set_title('Convergence Rate by Lambda Strategy', fontsize=12)
    axes[2].set_xticklabels(strategies, rotation=45, ha='right')
    axes[2].set_ylim([0, 105])
    axes[2].grid(axis='y', alpha=0.3)

    for i, (strat, rate) in enumerate(zip(strategies, conv_rates_strat)):
        axes[2].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/convergence_rate_by_factors.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/convergence_rate_by_factors.png")


def plot_convergence_speed_by_constraint(experiments):
    """Additional Graph: Average convergence epoch by constraint difficulty."""
    converged = [e for e in experiments if e['converged']]

    if not converged:
        return

    # Group by constraint
    constraint_data = defaultdict(list)
    for exp in converged:
        constraint = exp['constraint']
        epoch = exp['final_epoch']
        constraint_data[constraint].append(epoch)

    constraints = sorted(constraint_data.keys())
    avg_epochs = [np.mean(constraint_data[c]) for c in constraints]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(constraints, avg_epochs, color=['#2ca02c', '#ff7f0e', '#d62728'])
    ax.set_xlabel('Constraint Level (Global, Local)', fontsize=12)
    ax.set_ylabel('Average Convergence Epoch', fontsize=12)
    ax.set_title('Convergence Speed by Constraint Difficulty', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (constraint, avg_ep) in enumerate(zip(constraints, avg_epochs)):
        ax.text(i, avg_ep + 10, f'{avg_ep:.0f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/convergence_speed_by_constraint.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/convergence_speed_by_constraint.png")


def plot_heatmap_convergence_model_constraint(experiments):
    """Additional Graph: Heatmap of convergence rate by model x constraint."""
    # Group by model and constraint
    data = defaultdict(lambda: defaultdict(lambda: {'converged': 0, 'total': 0}))
    for exp in experiments:
        model = exp['model']
        constraint = exp['constraint']
        data[model][constraint]['total'] += 1
        if exp['converged']:
            data[model][constraint]['converged'] += 1

    models = sorted(data.keys())
    constraints = sorted(set(c for model_data in data.values() for c in model_data.keys()))

    # Build heatmap matrix
    matrix = []
    for model in models:
        row = []
        for constraint in constraints:
            if constraint in data[model]:
                rate = data[model][constraint]['converged'] / data[model][constraint]['total'] * 100
                row.append(rate)
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(constraints)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(constraints)
    ax.set_yticklabels(models)

    ax.set_xlabel('Constraint Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Convergence Rate Heatmap: Model x Constraint', fontsize=14)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(constraints)):
            text = ax.text(j, i, f'{matrix[i][j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Convergence Rate (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/heatmap_model_constraint.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/heatmap_model_constraint.png")


def plot_accuracy_vs_convergence_speed(experiments):
    """Additional Graph: Scatter plot of test accuracy vs convergence speed."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] != '']

    if not converged:
        return

    # Group by model
    model_data = defaultdict(lambda: {'epochs': [], 'accuracies': []})
    for exp in converged:
        model = exp['model']
        model_data[model]['epochs'].append(exp['final_epoch'])
        model_data[model]['accuracies'].append(float(exp['test_accuracy']))

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'BasicNN': '#1f77b4', 'FTTransformer': '#ff7f0e', 'TabularResNet': '#2ca02c'}

    for model, data in model_data.items():
        ax.scatter(data['epochs'], data['accuracies'],
                  label=model, alpha=0.6, s=80, color=colors.get(model, 'gray'))

    ax.set_xlabel('Convergence Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy vs Convergence Speed', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_evaluations/accuracy_vs_convergence_speed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/accuracy_vs_convergence_speed.png")


def plot_learning_rate_vs_convergence_epochs(experiments):
    """Additional Graph: Box plot of convergence epochs by learning rate."""
    converged = [e for e in experiments if e['converged']]

    if not converged:
        return

    # Group by learning rate
    lr_data = defaultdict(list)
    for exp in converged:
        lr = exp['learning_rate']
        lr_data[lr].append(exp['final_epoch'])

    lrs = sorted(lr_data.keys())
    data_to_plot = [lr_data[lr] for lr in lrs]

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(data_to_plot, labels=[f'{lr:.5f}' for lr in lrs], patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.6)

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Convergence Epoch', fontsize=12)
    ax.set_title('Distribution of Convergence Epochs by Learning Rate', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('comparison_evaluations/lr_vs_convergence_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graph saved: comparison_evaluations/lr_vs_convergence_epochs.png")


def print_summary_statistics(experiments):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("=" * 80)

    total = len(experiments)
    converged_exps = [e for e in experiments if e['converged']]
    failed_exps = [e for e in experiments if e['status'] == 'failed']

    print(f"\nTotal experiments: {total}")
    print(f"Converged: {len(converged_exps)} ({len(converged_exps)/total*100:.1f}%)")
    print(f"Failed: {len(failed_exps)} ({len(failed_exps)/total*100:.1f}%)")

    if converged_exps:
        epochs = [e['final_epoch'] for e in converged_exps]
        print(f"\nConvergence Epochs Statistics:")
        print(f"  Mean: {np.mean(epochs):.1f}")
        print(f"  Median: {np.median(epochs):.1f}")
        print(f"  Min: {min(epochs)}")
        print(f"  Max: {max(epochs)}")
        print(f"  Std Dev: {np.std(epochs):.1f}")

    # Accuracy statistics
    with_accuracy = [e for e in converged_exps if e['test_accuracy'] != '']
    if with_accuracy:
        accuracies = [float(e['test_accuracy']) for e in with_accuracy]
        print(f"\nTest Accuracy Statistics ({len(with_accuracy)} experiments):")
        print(f"  Mean: {np.mean(accuracies):.4f}")
        print(f"  Median: {np.median(accuracies):.4f}")
        print(f"  Min: {min(accuracies):.4f}")
        print(f"  Max: {max(accuracies):.4f}")
        print(f"  Std Dev: {np.std(accuracies):.4f}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    print("Starting comprehensive comparative analysis...")
    print("=" * 80)

    # Extract all experiment data
    print("\n[1/9] Extracting experiment data...")
    experiments = extract_experiment_data()

    # Save master CSV
    print("\n[2/9] Generating master CSV...")
    save_master_csv(experiments)

    # Print summary statistics
    print("\n[3/9] Computing summary statistics...")
    print_summary_statistics(experiments)

    # Generate graphs
    print("\n[4/9] Generating accuracy by learning rate graph...")
    plot_accuracy_by_learning_rate(experiments)

    print("\n[5/9] Generating accuracy by lambda strategy graph...")
    plot_accuracy_by_lambda_strategy(experiments)

    print("\n[6/9] Generating convergence rate graphs...")
    plot_convergence_rate_by_factors(experiments)

    print("\n[7/9] Generating convergence speed by constraint graph...")
    plot_convergence_speed_by_constraint(experiments)

    print("\n[8/9] Generating model x constraint heatmap...")
    plot_heatmap_convergence_model_constraint(experiments)

    print("\n[9/9] Generating additional analysis graphs...")
    plot_accuracy_vs_convergence_speed(experiments)
    plot_learning_rate_vs_convergence_epochs(experiments)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll results saved to: comparison_evaluations/")
    print("  - master_results.csv: Complete dataset")
    print("  - accuracy_by_learning_rate.png")
    print("  - accuracy_by_lambda_strategy.png")
    print("  - convergence_rate_by_factors.png")
    print("  - convergence_speed_by_constraint.png")
    print("  - heatmap_model_constraint.png")
    print("  - accuracy_vs_convergence_speed.png")
    print("  - lr_vs_convergence_epochs.png")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

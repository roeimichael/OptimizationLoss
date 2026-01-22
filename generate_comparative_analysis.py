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

def save_constraint_csv(experiments, constraint, output_dir):
    """Save constraint-specific CSV."""
    constraint_exps = [e for e in experiments if e['constraint'] == constraint]

    if not constraint_exps:
        return

    output_path = output_dir / f'results_{constraint}.csv'
    fieldnames = constraint_exps[0].keys()

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(constraint_exps)

    print(f"  Constraint-specific CSV saved: {output_path}")


def plot_accuracy_by_learning_rate(experiments, output_dir, constraint_name='All'):
    """Graph 1: Average test accuracy by learning rate for each model."""
    # Filter converged experiments with test_accuracy
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] != '']

    if not converged:
        print(f"  No converged experiments with test accuracy found for {constraint_name}!")
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
    title = f'Average Test Accuracy by Learning Rate - {constraint_name}'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{lr:.5f}' for lr in learning_rates], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'accuracy_by_learning_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_accuracy_by_lambda_strategy(experiments, output_dir, constraint_name='All'):
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
    title = f'Average Test Accuracy by Lambda Strategy - {constraint_name}'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'accuracy_by_lambda_strategy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_convergence_rate_by_factors(experiments, output_dir, constraint_name='All'):
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
    conv_rates = [model_data[m]['converged'] / model_data[m]['total'] * 100 if model_data[m]['total'] > 0 else 0 for m in models]

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
    conv_rates_lr = [lr_data[lr]['converged'] / lr_data[lr]['total'] * 100 if lr_data[lr]['total'] > 0 else 0 for lr in lrs]

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
    conv_rates_strat = [strat_data[s]['converged'] / strat_data[s]['total'] * 100 if strat_data[s]['total'] > 0 else 0 for s in strategies]

    axes[2].bar(range(len(strategies)), conv_rates_strat, color='#9467bd')
    axes[2].set_ylabel('Convergence Rate (%)', fontsize=11)
    axes[2].set_title('Convergence Rate by Lambda Strategy', fontsize=12)
    axes[2].set_xticks(range(len(strategies)))
    axes[2].set_xticklabels(strategies, rotation=45, ha='right')
    axes[2].set_ylim([0, 105])
    axes[2].grid(axis='y', alpha=0.3)

    for i, (strat, rate) in enumerate(zip(strategies, conv_rates_strat)):
        axes[2].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=10)

    fig.suptitle(f'Convergence Rates - {constraint_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'convergence_rate_by_factors.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_model_comparison(experiments, output_dir, constraint_name='All'):
    """Graph: Model performance comparison for specific constraint."""
    converged = [e for e in experiments if e['converged']]

    if not converged:
        return

    # Group by model
    model_data = defaultdict(lambda: {'epochs': [], 'accuracies': []})
    for exp in converged:
        if exp['test_accuracy'] != '':
            model = exp['model']
            model_data[model]['epochs'].append(exp['final_epoch'])
            model_data[model]['accuracies'].append(float(exp['test_accuracy']))

    if not model_data:
        return

    models = sorted(model_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Average epochs
    avg_epochs = [np.mean(model_data[m]['epochs']) for m in models]
    ax1.bar(models, avg_epochs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Average Convergence Epoch', fontsize=11)
    ax1.set_title('Average Convergence Speed by Model', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    for i, (model, avg_ep) in enumerate(zip(models, avg_epochs)):
        ax1.text(i, avg_ep + 5, f'{avg_ep:.0f}', ha='center', fontsize=10)

    # Average accuracy
    avg_accs = [np.mean(model_data[m]['accuracies']) for m in models]
    ax2.bar(models, avg_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Average Test Accuracy', fontsize=11)
    ax2.set_title('Average Test Accuracy by Model', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    for i, (model, avg_acc) in enumerate(zip(models, avg_accs)):
        ax2.text(i, avg_acc + 0.01, f'{avg_acc:.3f}', ha='center', fontsize=10)

    fig.suptitle(f'Model Performance Comparison - {constraint_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_heatmap_lr_strategy(experiments, output_dir, constraint_name='All'):
    """Graph: Heatmap of accuracy by learning rate x lambda strategy."""
    converged = [e for e in experiments if e['converged'] and e['test_accuracy'] != '']

    if not converged:
        return

    # Group by LR and strategy
    data = defaultdict(lambda: defaultdict(list))
    for exp in converged:
        lr = exp['learning_rate']
        strategy = exp['lambda_strategy']
        acc = float(exp['test_accuracy'])
        data[lr][strategy].append(acc)

    lrs = sorted(data.keys())
    strategies = sorted(set(s for lr_data in data.values() for s in lr_data.keys()))

    # Build heatmap matrix (average accuracies)
    matrix = []
    for lr in lrs:
        row = []
        for strategy in strategies:
            if strategy in data[lr] and len(data[lr][strategy]) > 0:
                avg_acc = np.mean(data[lr][strategy])
                row.append(avg_acc)
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')

    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(lrs)))
    ax.set_xticklabels(strategies)
    ax.set_yticklabels([f'{lr:.5f}' for lr in lrs])

    ax.set_xlabel('Lambda Strategy', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(f'Accuracy Heatmap: LR × Strategy - {constraint_name}', fontsize=14)

    # Add text annotations
    for i in range(len(lrs)):
        for j in range(len(strategies)):
            if matrix[i][j] > 0:
                text = ax.text(j, i, f'{matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'heatmap_lr_strategy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_accuracy_vs_convergence_speed(experiments, output_dir, constraint_name='All'):
    """Graph: Scatter plot of test accuracy vs convergence speed."""
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
    ax.set_title(f'Test Accuracy vs Convergence Speed - {constraint_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'accuracy_vs_convergence_speed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_learning_rate_vs_convergence_epochs(experiments, output_dir, constraint_name='All'):
    """Graph: Box plot of convergence epochs by learning rate."""
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

    bp = ax.boxplot(data_to_plot, tick_labels=[f'{lr:.5f}' for lr in lrs], patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.6)

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Convergence Epoch', fontsize=12)
    ax.set_title(f'Distribution of Convergence Epochs by Learning Rate - {constraint_name}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'lr_vs_convergence_epochs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {output_path}")


def plot_cross_constraint_comparison(experiments, output_dir):
    """Generate graphs comparing performance across different constraints."""
    converged = [e for e in experiments if e['converged']]

    if not converged:
        return

    # 1. Convergence speed by constraint
    constraint_data = defaultdict(list)
    for exp in converged:
        constraint = exp['constraint']
        epoch = exp['final_epoch']
        constraint_data[constraint].append(epoch)

    constraints = sorted(constraint_data.keys())
    avg_epochs = [np.mean(constraint_data[c]) for c in constraints]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.bar(constraints, avg_epochs, color=['#2ca02c', '#ff7f0e', '#d62728'])
    ax1.set_xlabel('Constraint Level (Global, Local)', fontsize=12)
    ax1.set_ylabel('Average Convergence Epoch', fontsize=12)
    ax1.set_title('Convergence Speed by Constraint Difficulty', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (constraint, avg_ep) in enumerate(zip(constraints, avg_epochs)):
        ax1.text(i, avg_ep + 10, f'{avg_ep:.0f}', ha='center', fontsize=11)

    # 2. Accuracy by constraint
    acc_data = defaultdict(list)
    for exp in converged:
        if exp['test_accuracy'] != '':
            constraint = exp['constraint']
            acc_data[constraint].append(float(exp['test_accuracy']))

    avg_accs = [np.mean(acc_data[c]) if c in acc_data else 0 for c in constraints]

    bars2 = ax2.bar(constraints, avg_accs, color=['#2ca02c', '#ff7f0e', '#d62728'])
    ax2.set_xlabel('Constraint Level (Global, Local)', fontsize=12)
    ax2.set_ylabel('Average Test Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy by Constraint Difficulty', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (constraint, avg_acc) in enumerate(zip(constraints, avg_accs)):
        if avg_acc > 0:
            ax2.text(i, avg_acc + 0.01, f'{avg_acc:.3f}', ha='center', fontsize=11)

    fig.suptitle('Cross-Constraint Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'cross_constraint_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph saved: {output_path}")

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
    print("\n[1/4] Extracting experiment data...")
    experiments = extract_experiment_data()

    # Save master CSV
    print("\n[2/4] Generating master CSV...")
    save_master_csv(experiments)

    # Print summary statistics
    print("\n[3/4] Computing summary statistics...")
    print_summary_statistics(experiments)

    # Get unique constraints
    constraints = sorted(set(e['constraint'] for e in experiments))
    print(f"\n[4/4] Generating constraint-specific analyses for {len(constraints)} constraints...")

    # Create directories and generate analyses for each constraint
    base_dir = Path('comparison_evaluations')

    for i, constraint in enumerate(constraints, 1):
        print(f"\n--- Constraint {i}/{len(constraints)}: {constraint} ---")

        # Create directory
        constraint_dir = base_dir / f'constraint_{constraint}'
        constraint_dir.mkdir(parents=True, exist_ok=True)

        # Filter experiments for this constraint
        constraint_exps = [e for e in experiments if e['constraint'] == constraint]

        print(f"  Total experiments: {len(constraint_exps)}")
        converged_count = sum(1 for e in constraint_exps if e['converged'])
        print(f"  Converged: {converged_count}/{len(constraint_exps)} ({converged_count/len(constraint_exps)*100:.1f}%)")

        # Save constraint-specific CSV
        save_constraint_csv(experiments, constraint, constraint_dir)

        # Generate graphs for this constraint
        constraint_name = f"Constraint {constraint}"

        print(f"  Generating graphs...")
        plot_accuracy_by_learning_rate(constraint_exps, constraint_dir, constraint_name)
        plot_accuracy_by_lambda_strategy(constraint_exps, constraint_dir, constraint_name)
        plot_convergence_rate_by_factors(constraint_exps, constraint_dir, constraint_name)
        plot_model_comparison(constraint_exps, constraint_dir, constraint_name)
        plot_heatmap_lr_strategy(constraint_exps, constraint_dir, constraint_name)
        plot_accuracy_vs_convergence_speed(constraint_exps, constraint_dir, constraint_name)
        plot_learning_rate_vs_convergence_epochs(constraint_exps, constraint_dir, constraint_name)

    # Generate cross-constraint comparison graphs
    print(f"\n--- Cross-Constraint Analysis ---")
    print(f"  Generating cross-constraint comparison graphs...")
    plot_cross_constraint_comparison(experiments, base_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nResults organized by constraint:")
    for constraint in constraints:
        print(f"  - comparison_evaluations/constraint_{constraint}/")
    print("\nCross-constraint comparison:")
    print(f"  - comparison_evaluations/cross_constraint_comparison.png")
    print("\nMaster dataset:")
    print(f"  - comparison_evaluations/master_results.csv")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

"""
Top 5 Experiments Analysis and Comparison
==========================================

This script analyzes and compares the top 5 performing configurations
by generating comprehensive visualizations and comparison reports.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("white")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def load_experiment_data(config_name, results_dir=None):
    """Load all data for a specific experiment configuration."""
    if results_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        results_dir = project_root / "results"
    else:
        results_dir = Path(results_dir)

    hyperparam_dir = results_dir / f"hyperparam_{config_name}"

    if not hyperparam_dir.exists():
        print(f"    Warning: Folder not found - {hyperparam_dir}")
        return None

    data = {
        'config_name': config_name,
        'training_log': None,
        'evaluation_metrics': None,
        'benchmark_metrics': None,
        'constraint_comparison': None
    }

    files_found = []

    training_log_path = hyperparam_dir / "training_log.csv"
    if training_log_path.exists():
        data['training_log'] = pd.read_csv(training_log_path)
        files_found.append("training_log")

    eval_metrics_path = hyperparam_dir / "evaluation_metrics.csv"
    if eval_metrics_path.exists():
        eval_data = []
        with open(eval_metrics_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2 and parts[0] and parts[1]:
                    eval_data.append(parts)
        if eval_data:
            data['evaluation_metrics'] = pd.DataFrame(eval_data[1:], columns=eval_data[0])
            files_found.append("evaluation_metrics")

    benchmark_metrics_path = hyperparam_dir / "benchmark_metrics.csv"
    if benchmark_metrics_path.exists():
        bench_data = []
        with open(benchmark_metrics_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2 and parts[0] and parts[1]:
                    bench_data.append(parts)
        if bench_data:
            data['benchmark_metrics'] = pd.DataFrame(bench_data[1:], columns=bench_data[0])
            files_found.append("benchmark_metrics")

    constraint_comp_path = hyperparam_dir / "constraint_comparison.csv"
    if constraint_comp_path.exists():
        data['constraint_comparison'] = pd.read_csv(constraint_comp_path)
        files_found.append("constraint_comparison")

    if files_found:
        print(f"    Found: {', '.join(files_found)}")
    else:
        print(f"    Warning: No data files found in {hyperparam_dir}")

    return data


def create_accuracy_comparison(all_data, output_dir):
    """Create bar chart comparing optimized vs benchmark accuracy for all 5 configs."""
    configs = []
    optimized_acc = []
    benchmark_acc = []
    improvements = []

    for data in all_data:
        if data is None:
            continue

        config_name = data['config_name']
        configs.append(config_name)

        opt_acc = None
        if data['evaluation_metrics'] is not None:
            opt_row = data['evaluation_metrics'][data['evaluation_metrics']['Metric'] == 'Overall Accuracy']
            if not opt_row.empty:
                opt_acc = float(opt_row['Value'].values[0])

        bench_acc = None
        if data['benchmark_metrics'] is not None:
            bench_row = data['benchmark_metrics'][data['benchmark_metrics']['Metric'] == 'Overall Accuracy']
            if not bench_row.empty:
                bench_acc = float(bench_row['Value'].values[0])

        optimized_acc.append(opt_acc if opt_acc else 0)
        benchmark_acc.append(bench_acc if bench_acc else 0)
        improvements.append((opt_acc - bench_acc) * 100 if (opt_acc and bench_acc) else 0)

    if not configs:
        print("ERROR: No valid data found for any configurations")
        print("Please ensure experiments have been run and results exist in results/hyperparam_* folders")
        return [], [], [], []

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.38

    colors_opt = ['#1f77b4' if imp > 0 else '#d62728' for imp in improvements]
    colors_bench = ['#7f7f7f'] * len(configs)

    bars1 = ax.bar(x - width/2, optimized_acc, width, label='Optimized', color=colors_opt, alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, benchmark_acc, width, label='Benchmark', color=colors_bench, alpha=0.7, edgecolor='white', linewidth=1.5)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        h1, h2 = bar1.get_height(), bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.005, f'{h1:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.005, f'{h2:.3f}',
               ha='center', va='bottom', fontsize=9)

        if improvements[i] != 0:
            color = '#1f77b4' if improvements[i] > 0 else '#d62728'
            y_pos = max(h1, h2) + 0.025
            ax.text(i, y_pos, f'{improvements[i]:+.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5))

    min_val = min(min(optimized_acc), min(benchmark_acc))
    max_val = max(max(optimized_acc), max(benchmark_acc))
    margin = (max_val - min_val) * 0.15
    ax.set_ylim(min_val - margin, max_val + margin + 0.04)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison: Optimized vs Benchmark', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center')
    ax.legend(fontsize=10, frameon=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)

    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

    return configs, optimized_acc, benchmark_acc, improvements


def create_training_curves_comparison(all_data, output_dir):
    """Create line plots comparing training curves for all 5 configs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    has_any_data = False

    ax = axes[0, 0]
    for i, data in enumerate(all_data):
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'total_loss' in log.columns:
            ax.plot(log['epoch'], log['total_loss'], label=data['config_name'],
                   linewidth=2.5, color=colors[i % len(colors)], alpha=0.9)
            has_any_data = True
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Loss (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Total Loss Convergence', fontsize=12, fontweight='bold', pad=10)
    if has_any_data:
        ax.legend(fontsize=9, frameon=True, shadow=False, fancybox=False)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    for i, data in enumerate(all_data):
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'class_loss' in log.columns:
            ax.plot(log['epoch'], log['class_loss'], label=data['config_name'],
                   linewidth=2.5, color=colors[i % len(colors)], alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classification Loss (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Classification Loss Convergence', fontsize=12, fontweight='bold', pad=10)
    if has_any_data:
        ax.legend(fontsize=9, frameon=True, shadow=False, fancybox=False)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    for i, data in enumerate(all_data):
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'global_loss' in log.columns:
            ax.plot(log['epoch'], log['global_loss'], label=data['config_name'],
                   linewidth=2.5, color=colors[i % len(colors)], alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Global Constraint Loss (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Global Constraint Satisfaction', fontsize=12, fontweight='bold', pad=10)
    if has_any_data:
        ax.legend(fontsize=9, frameon=True, shadow=False, fancybox=False)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    for i, data in enumerate(all_data):
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'local_loss' in log.columns:
            ax.plot(log['epoch'], log['local_loss'], label=data['config_name'],
                   linewidth=2.5, color=colors[i % len(colors)], alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Local Constraint Loss (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Local Constraint Satisfaction', fontsize=12, fontweight='bold', pad=10)
    if has_any_data:
        ax.legend(fontsize=9, frameon=True, shadow=False, fancybox=False)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Training Dynamics Comparison', fontsize=15, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_heatmap(all_data, output_dir):
    """Create heatmap of various metrics across all 5 configs."""
    metrics_names = ['Overall Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    short_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    opt_matrix = []
    bench_matrix = []
    configs = []

    for data in all_data:
        if data is None:
            continue

        configs.append(data['config_name'])

        opt_row = []
        bench_row = []

        for metric_name in metrics_names:
            if data['evaluation_metrics'] is not None:
                metric_row = data['evaluation_metrics'][data['evaluation_metrics']['Metric'] == metric_name]
                opt_row.append(float(metric_row['Value'].values[0]) if not metric_row.empty else 0)
            else:
                opt_row.append(0)

            if data['benchmark_metrics'] is not None:
                metric_row = data['benchmark_metrics'][data['benchmark_metrics']['Metric'] == metric_name]
                bench_row.append(float(metric_row['Value'].values[0]) if not metric_row.empty else 0)
            else:
                bench_row.append(0)

        opt_matrix.append(opt_row)
        bench_matrix.append(bench_row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    opt_df = pd.DataFrame(opt_matrix, columns=short_names, index=configs)
    sns.heatmap(opt_df, annot=True, fmt='.3f', cmap='Blues', ax=ax1,
               vmin=0.55, vmax=0.65, cbar_kws={'label': 'Score'},
               linewidths=1, linecolor='white', annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Optimized Model Performance', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Configuration', fontsize=11, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    bench_df = pd.DataFrame(bench_matrix, columns=short_names, index=configs)
    sns.heatmap(bench_df, annot=True, fmt='.3f', cmap='Oranges', ax=ax2,
               vmin=0.55, vmax=0.65, cbar_kws={'label': 'Score'},
               linewidths=1, linecolor='white', annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Benchmark Performance', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='center')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    plt.suptitle('Performance Metrics Comparison', fontsize=15, fontweight='bold', y=0.98)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_improvement_breakdown(configs, improvements, output_dir):
    """Create horizontal bar chart showing improvement breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
    sorted_configs = [configs[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]

    colors = ['#1f77b4' if imp > 0 else '#d62728' if imp < 0 else '#7f7f7f'
             for imp in sorted_improvements]

    bars = ax.barh(sorted_configs, sorted_improvements, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
        width = bar.get_width()
        label_x = width + (0.1 if width > 0 else -0.1)
        color = '#1f77b4' if imp > 0 else '#d62728' if imp < 0 else '#7f7f7f'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{imp:+.2f}%',
               ha='left' if width > 0 else 'right', va='center',
               fontsize=11, fontweight='bold', color=color)

    ax.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2, alpha=0.4)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Ranking', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_improvements.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_constraint_satisfaction_comparison(all_data, output_dir):
    """Compare constraint satisfaction rates across configs."""
    configs = []
    global_satisfaction = []
    local_satisfaction = []

    for data in all_data:
        if data is None or data['constraint_comparison'] is None:
            continue

        configs.append(data['config_name'])
        comp = data['constraint_comparison']

        if 'prediction_dropout_count' in comp.columns and 'target_dropout_count' in comp.columns:
            final_pred = comp['prediction_dropout_count'].iloc[-1]
            target = comp['target_dropout_count'].iloc[-1]
            global_sat = 1.0 - abs(final_pred - target) / max(target, 1)
            global_satisfaction.append(max(0, global_sat))
        else:
            global_satisfaction.append(0)

        if 'local_violations' in comp.columns:
            avg_violations = comp['local_violations'].mean()
            local_sat = 1.0 - min(avg_violations / 10.0, 1.0)
            local_satisfaction.append(max(0, local_sat))
        else:
            local_satisfaction.append(0)

    if not configs or (sum(global_satisfaction) == 0 and sum(local_satisfaction) == 0):
        print("    Skipping constraint satisfaction plot (no meaningful data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(configs))
    bars1 = ax1.bar(x, global_satisfaction, color='#3498db', alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Satisfaction Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Global Constraint', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=0, ha='center')
    ax1.axhline(y=1.0, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5, label='Target')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, local_satisfaction, color='#9b59b6', alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Satisfaction Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Local Constraint (Average)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=0, ha='center')
    ax2.axhline(y=1.0, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5, label='Target')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Constraint Satisfaction Comparison', fontsize=15, fontweight='bold', y=1.02)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_constraint_satisfaction.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_hyperparameter_summary(output_dir):
    """Create visualization of hyperparameter configurations."""
    from config.experiment_config import NN_CONFIGS

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    configs = [c['name'] for c in NN_CONFIGS]
    x = np.arange(len(configs))

    ax = axes[0, 0]
    lambda_global = [c['lambda_global'] for c in NN_CONFIGS]
    lambda_local = [c['lambda_local'] for c in NN_CONFIGS]
    width = 0.38
    ax.bar(x - width/2, lambda_global, width, label='Global λ', color='#e74c3c',
          alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.bar(x + width/2, lambda_local, width, label='Local λ', color='#3498db',
          alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Lambda Value (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Constraint Lambda Values', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center')
    ax.legend(fontsize=9, frameon=True)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    depths = [len(c['hidden_dims']) for c in NN_CONFIGS]
    bars = ax.bar(x, depths, color='#9b59b6', alpha=0.85,
                 edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Number of Hidden Layers', fontsize=11, fontweight='bold')
    ax.set_title('Network Architecture Depth', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center')
    ax.set_ylim(0, max(depths) + 1)
    for bar, depth in zip(bars, depths):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
               f'{depth}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    dropout_rates = [c['dropout'] for c in NN_CONFIGS]
    bars = ax.bar(x, dropout_rates, color='#f39c12', alpha=0.85,
                 edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Dropout Rate', fontsize=11, fontweight='bold')
    ax.set_title('Dropout Regularization', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center')
    ax.set_ylim(0, max(dropout_rates) + 0.1)
    for bar, dropout in zip(bars, dropout_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
               f'{dropout:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    batch_sizes = [c['batch_size'] for c in NN_CONFIGS]
    bars = ax.bar(x, batch_sizes, color='#1abc9c', alpha=0.85,
                 edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Batch Size', fontsize=11, fontweight='bold')
    ax.set_title('Training Batch Size', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center')
    ax.set_ylim(0, max(batch_sizes) + 10)
    for bar, batch in zip(bars, batch_sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{batch}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Hyperparameter Configuration Summary', fontsize=15, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "comparison_hyperparameters.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_report(all_data, configs, optimized_acc, benchmark_acc, improvements, output_dir):
    """Generate a comprehensive text report comparing all configurations."""
    report_path = output_dir / "TOP5_COMPARISON_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Top 5 Configurations - Comprehensive Comparison Report\n\n")
        f.write("## Executive Summary\n\n")

        best_idx = improvements.index(max(improvements))
        f.write(f"**BEST PERFORMER:** `{configs[best_idx]}`\n")
        f.write(f"- **Optimized Accuracy:** {optimized_acc[best_idx]:.4f}\n")
        f.write(f"- **Benchmark Accuracy:** {benchmark_acc[best_idx]:.4f}\n")
        f.write(f"- **Improvement:** {improvements[best_idx]:+.2f}%\n\n")

        f.write("---\n\n")
        f.write("## Performance Rankings\n\n")
        f.write("| Rank | Config | Optimized | Benchmark | Improvement |\n")
        f.write("|------|--------|-----------|-----------|-------------|\n")

        sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
        for rank, idx in enumerate(sorted_indices, 1):
            f.write(f"| {rank} | {configs[idx]} | {optimized_acc[idx]:.4f} | "
                   f"{benchmark_acc[idx]:.4f} | {improvements[idx]:+.2f}% |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Metrics Comparison\n\n")

        for i, data in enumerate(all_data):
            if data is None:
                continue

            f.write(f"### {i+1}. {data['config_name']}\n\n")

            # Configuration details
            from config.experiment_config import NN_CONFIGS
            config = next((c for c in NN_CONFIGS if c['name'] == data['config_name']), None)
            if config:
                f.write("**Configuration:**\n")
                f.write(f"- Lambda: global={config['lambda_global']}, local={config['lambda_local']}\n")
                f.write(f"- Architecture: {config['hidden_dims']} ({len(config['hidden_dims'])} layers)\n")
                f.write(f"- Learning rate: {config['lr']}\n")
                f.write(f"- Dropout: {config['dropout']}\n")
                f.write(f"- Batch size: {config['batch_size']}\n\n")

            # Optimized metrics
            if data['evaluation_metrics'] is not None:
                f.write("**Optimized Model:**\n")
                for _, row in data['evaluation_metrics'].iterrows():
                    f.write(f"- {row['Metric']}: {float(row['Value']):.4f}\n")
                f.write("\n")

            # Benchmark metrics
            if data['benchmark_metrics'] is not None:
                f.write("**Benchmark:**\n")
                for _, row in data['benchmark_metrics'].iterrows():
                    f.write(f"- {row['Metric']}: {float(row['Value']):.4f}\n")
                f.write("\n")

            # Improvement summary
            f.write(f"**Improvement:** {improvements[i]:+.2f}%\n")
            f.write("\n---\n\n")

        f.write("## Key Insights\n\n")

        # Insight 1: Best lambda strategy
        f.write("### 1. Lambda Strategy\n\n")
        best_lambda_idx = improvements.index(max(improvements))
        from config.experiment_config import NN_CONFIGS
        best_config = next((c for c in NN_CONFIGS if c['name'] == configs[best_lambda_idx]), None)
        if best_config:
            f.write(f"The best performing configuration uses:\n")
            f.write(f"- Lambda global: {best_config['lambda_global']}\n")
            f.write(f"- Lambda local: {best_config['lambda_local']}\n\n")

        # Insight 2: Depth analysis
        f.write("### 2. Architecture Depth\n\n")
        depth_perf = [(c['name'], len(c['hidden_dims']), improvements[i])
                     for i, c in enumerate(NN_CONFIGS)]
        depth_perf.sort(key=lambda x: x[2], reverse=True)
        f.write("Performance by depth:\n")
        for name, depth, imp in depth_perf:
            f.write(f"- {depth} layers ({name}): {imp:+.2f}%\n")
        f.write("\n")

        f.write("### 3. Recommendation\n\n")
        if max(improvements) > 2.0:
            f.write(f"**STRONG RECOMMENDATION:** Use `{configs[best_idx]}` for production.\n")
            f.write(f"The {improvements[best_idx]:+.2f}% improvement over benchmark is substantial.\n\n")
        elif max(improvements) > 0.5:
            f.write(f"**MODERATE RECOMMENDATION:** Use `{configs[best_idx]}` if accuracy matters.\n")
            f.write(f"The {improvements[best_idx]:+.2f}% improvement is modest but real.\n\n")
        else:
            f.write(f"**NOT RECOMMENDED:** Consider using simple benchmark instead.\n")
            f.write(f"The improvement ({max(improvements):+.2f}%) doesn't justify the complexity.\n\n")

        f.write("---\n\n")
        f.write("## Visualizations Generated\n\n")
        f.write("1. `comparison_accuracy.png` - Accuracy comparison bar chart\n")
        f.write("2. `comparison_training_curves.png` - Training dynamics over epochs\n")
        f.write("3. `comparison_metrics_heatmap.png` - Performance metrics heatmap\n")
        f.write("4. `comparison_improvements.png` - Improvement ranking\n")
        f.write("5. `comparison_constraint_satisfaction.png` - Constraint satisfaction rates\n")
        f.write("6. `comparison_hyperparameters.png` - Hyperparameter configuration summary\n\n")

        f.write("---\n\n")
        f.write("**Report Generated:** Automated analysis of top 5 configurations\n")

    print(f"\nComparison report saved to: {report_path}")


def main():
    """Main function to run all comparisons and generate visualizations."""
    print("="*80)
    print("Top 5 Configurations - Comprehensive Analysis")
    print("="*80)

    # Load experiment configuration
    from config.experiment_config import NN_CONFIGS

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "results" / "top5_comparison"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data for all 5 configs
    print("\nLoading experiment data...")
    all_data = []
    for config in NN_CONFIGS:
        config_name = config['name']
        print(f"  Loading {config_name}...")
        data = load_experiment_data(config_name)
        all_data.append(data)

    valid_data = [d for d in all_data if d is not None]
    if not valid_data:
        print("\nERROR: No valid experiment data found!")
        print("Please run the experiments first using: python experiments/run_experiments.py")
        return

    print("\nGenerating visualizations...")

    print("  1. Accuracy comparison bar chart...")
    configs, optimized_acc, benchmark_acc, improvements = create_accuracy_comparison(all_data, output_dir)

    if not configs:
        print("\nERROR: Failed to extract metrics from experiment data")
        return

    # 2. Training curves
    print("  2. Training dynamics curves...")
    create_training_curves_comparison(all_data, output_dir)

    # 3. Metrics heatmap
    print("  3. Performance metrics heatmap...")
    create_metrics_heatmap(all_data, output_dir)

    # 4. Improvement breakdown
    print("  4. Improvement ranking...")
    create_improvement_breakdown(configs, improvements, output_dir)

    # 5. Constraint satisfaction
    print("  5. Constraint satisfaction comparison...")
    create_constraint_satisfaction_comparison(all_data, output_dir)

    # 6. Hyperparameter summary
    print("  6. Hyperparameter configuration summary...")
    create_hyperparameter_summary(output_dir)

    # 7. Generate report
    print("\nGenerating comprehensive comparison report...")
    generate_comparison_report(all_data, configs, optimized_acc, benchmark_acc,
                              improvements, output_dir)

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"\nAll visualizations and reports saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - comparison_accuracy.png")
    print("  - comparison_training_curves.png")
    print("  - comparison_metrics_heatmap.png")
    print("  - comparison_improvements.png")
    print("  - comparison_constraint_satisfaction.png")
    print("  - comparison_hyperparameters.png")
    print("  - TOP5_COMPARISON_REPORT.md")

    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    best_idx = improvements.index(max(improvements))
    print(f"\nBEST: {configs[best_idx]}")
    print(f"   Optimized: {optimized_acc[best_idx]:.4f}")
    print(f"   Benchmark: {benchmark_acc[best_idx]:.4f}")
    print(f"   Improvement: {improvements[best_idx]:+.2f}%\n")

    print("Rankings:")
    sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {configs[idx]}: {improvements[idx]:+.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

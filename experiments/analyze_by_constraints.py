"""
Constraint-Grouped Analysis
============================

This script analyzes and compares the top 5 configurations for each
constraint setting separately, allowing you to see how different
constraint configurations affect the same hyperparameter settings.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.experiment_config import NN_CONFIGS, CONSTRAINTS

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
        return None

    data = {
        'config_name': config_name,
        'training_log': None,
        'evaluation_metrics': None,
        'benchmark_metrics': None,
        'constraint_comparison': None
    }

    training_log_path = hyperparam_dir / "training_log.csv"
    if training_log_path.exists():
        data['training_log'] = pd.read_csv(training_log_path)

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

    constraint_comp_path = hyperparam_dir / "constraint_comparison.csv"
    if constraint_comp_path.exists():
        data['constraint_comparison'] = pd.read_csv(constraint_comp_path)

    return data


def create_constraint_comparison_graph(all_constraint_results, output_dir):
    """Create comparison graph showing best performance across constraint settings."""
    constraint_labels = []
    best_accuracies = []
    best_configs = []
    improvements = []

    for constraint_key, results in sorted(all_constraint_results.items()):
        if not results:
            continue

        local, global_c = constraint_key
        constraint_labels.append(f"L{local}/G{global_c}")

        best_acc = max(results, key=lambda x: x['optimized_acc'])
        best_accuracies.append(best_acc['optimized_acc'])
        best_configs.append(best_acc['config_name'])
        improvements.append(best_acc['improvement'])

    if not constraint_labels:
        print("No valid constraint results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(constraint_labels))
    colors = ['#1f77b4' if imp > 0 else '#d62728' for imp in improvements]

    bars = ax1.bar(x, best_accuracies, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Constraint Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Best Performance by Constraint Setting', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(constraint_labels, rotation=0, ha='center')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for i, (bar, acc, config) in enumerate(zip(bars, best_accuracies, best_configs)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{acc:.4f}\n({config})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.barh(constraint_labels, improvements, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Constraint Configuration', fontsize=12, fontweight='bold')
    ax2.set_title('Improvement by Constraint Setting', fontsize=13, fontweight='bold', pad=12)
    ax2.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2, alpha=0.4)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        width = bar.get_width()
        label_x = width + (0.1 if width > 0 else -0.1)
        color = '#1f77b4' if imp > 0 else '#d62728'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, f'{imp:+.2f}%',
                ha='left' if width > 0 else 'right', va='center',
                fontsize=10, fontweight='bold', color=color)

    plt.suptitle('Performance Comparison Across Constraint Settings', fontsize=15, fontweight='bold', y=1.02)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / "constraint_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_constraint_group(local_percent, global_percent, output_dir):
    """Analyze all configurations for a specific constraint setting."""
    print(f"\n{'='*80}")
    print(f"Analyzing Constraint Configuration: Local={local_percent}, Global={global_percent}")
    print(f"{'='*80}\n")

    constraint_suffix = f"_c{local_percent}_{global_percent}"
    configs = []
    optimized_acc = []
    benchmark_acc = []
    improvements = []
    all_data = []

    for config in NN_CONFIGS:
        config_name = config['name']
        exp_name = f"{config_name}{constraint_suffix}"

        print(f"  Loading {exp_name}...")
        data = load_experiment_data(exp_name)

        if data is None:
            print(f"    Warning: No data found for {exp_name}")
            continue

        all_data.append(data)
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
        print(f"  No valid data found for constraint ({local_percent}, {global_percent})")
        return None

    constraint_output_dir = output_dir / f"constraint_{local_percent}_{global_percent}"
    constraint_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n  Creating visualizations...")
    create_accuracy_comparison(configs, optimized_acc, benchmark_acc, improvements, constraint_output_dir)
    generate_constraint_report(configs, optimized_acc, benchmark_acc, improvements,
                              local_percent, global_percent, constraint_output_dir)

    print(f"  Results saved to: {constraint_output_dir}")

    return [{'config_name': c, 'optimized_acc': o, 'benchmark_acc': b, 'improvement': i}
            for c, o, b, i in zip(configs, optimized_acc, benchmark_acc, improvements)]


def create_accuracy_comparison(configs, optimized_acc, benchmark_acc, improvements, output_dir):
    """Create accuracy comparison bar chart."""
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
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_constraint_report(configs, optimized_acc, benchmark_acc, improvements,
                               local_percent, global_percent, output_dir):
    """Generate text report for a specific constraint configuration."""
    report_path = output_dir / "CONSTRAINT_REPORT.md"

    with open(report_path, 'w') as f:
        f.write(f"# Constraint Configuration Report\n\n")
        f.write(f"**Constraint Setting:** Local={local_percent} (dropout {int(local_percent*100)}%), ")
        f.write(f"Global={global_percent} (dropout {int(global_percent*100)}%)\n\n")

        best_idx = improvements.index(max(improvements))
        f.write(f"## Best Configuration\n\n")
        f.write(f"**{configs[best_idx]}**\n")
        f.write(f"- Optimized Accuracy: {optimized_acc[best_idx]:.4f}\n")
        f.write(f"- Benchmark Accuracy: {benchmark_acc[best_idx]:.4f}\n")
        f.write(f"- Improvement: {improvements[best_idx]:+.2f}%\n\n")

        f.write("## All Configurations\n\n")
        f.write("| Rank | Config | Optimized | Benchmark | Improvement |\n")
        f.write("|------|--------|-----------|-----------|-------------|\n")

        sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
        for rank, idx in enumerate(sorted_indices, 1):
            f.write(f"| {rank} | {configs[idx]} | {optimized_acc[idx]:.4f} | "
                   f"{benchmark_acc[idx]:.4f} | {improvements[idx]:+.2f}% |\n")

    print(f"    Report saved: {report_path}")


def main():
    """Main function to analyze all constraint configurations."""
    print("="*80)
    print("Constraint-Grouped Analysis: Top 5 Configurations")
    print("="*80)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "results" / "constraint_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"\nAnalyzing {len(CONSTRAINTS)} constraint configurations...")

    all_constraint_results = {}

    for local_percent, global_percent in CONSTRAINTS:
        results = analyze_constraint_group(local_percent, global_percent, output_dir)
        if results:
            all_constraint_results[(local_percent, global_percent)] = results

    if len(all_constraint_results) > 1:
        print(f"\n{'='*80}")
        print("Creating cross-constraint comparison...")
        print(f"{'='*80}\n")
        create_constraint_comparison_graph(all_constraint_results, output_dir)

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nGenerated files:")
    for local_p, global_p in CONSTRAINTS:
        print(f"  - constraint_{local_p}_{global_p}/")
    if len(CONSTRAINTS) > 1:
        print(f"  - constraint_comparison_summary.png")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

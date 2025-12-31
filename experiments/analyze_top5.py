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
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_experiment_data(config_name, results_dir="./results"):
    """Load all data for a specific experiment configuration."""
    hyperparam_dir = Path(results_dir) / f"hyperparam_{config_name}"

    if not hyperparam_dir.exists():
        print(f"Warning: {hyperparam_dir} does not exist")
        return None

    data = {
        'config_name': config_name,
        'training_log': None,
        'evaluation_metrics': None,
        'benchmark_metrics': None,
        'constraint_comparison': None
    }

    # Load training log (loss curves, metrics over epochs)
    training_log_path = hyperparam_dir / "training_log.csv"
    if training_log_path.exists():
        data['training_log'] = pd.read_csv(training_log_path)

    # Load evaluation metrics
    eval_metrics_path = hyperparam_dir / "evaluation_metrics.csv"
    if eval_metrics_path.exists():
        data['evaluation_metrics'] = pd.read_csv(eval_metrics_path, header=None, names=['metric', 'value'])

    # Load benchmark metrics
    benchmark_metrics_path = hyperparam_dir / "benchmark_metrics.csv"
    if benchmark_metrics_path.exists():
        data['benchmark_metrics'] = pd.read_csv(benchmark_metrics_path, header=None, names=['metric', 'value'])

    # Load constraint comparison
    constraint_comp_path = hyperparam_dir / "constraint_comparison.csv"
    if constraint_comp_path.exists():
        data['constraint_comparison'] = pd.read_csv(constraint_comp_path)

    return data


def create_accuracy_comparison(all_data, output_dir):
    """Create bar chart comparing optimized vs benchmark accuracy for all 5 configs."""
    fig, ax = plt.subplots(figsize=(14, 8))

    configs = []
    optimized_acc = []
    benchmark_acc = []
    improvements = []

    for data in all_data:
        if data is None:
            continue

        config_name = data['config_name']
        configs.append(config_name)

        # Extract optimized accuracy
        opt_acc = None
        if data['evaluation_metrics'] is not None:
            opt_row = data['evaluation_metrics'][data['evaluation_metrics']['metric'] == 'Overall Accuracy']
            if not opt_row.empty:
                opt_acc = float(opt_row['value'].values[0])

        # Extract benchmark accuracy
        bench_acc = None
        if data['benchmark_metrics'] is not None:
            bench_row = data['benchmark_metrics'][data['benchmark_metrics']['metric'] == 'Overall Accuracy']
            if not bench_row.empty:
                bench_acc = float(bench_row['value'].values[0])

        optimized_acc.append(opt_acc if opt_acc else 0)
        benchmark_acc.append(bench_acc if bench_acc else 0)
        improvements.append((opt_acc - bench_acc) * 100 if (opt_acc and bench_acc) else 0)

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, optimized_acc, width, label='Optimized', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, benchmark_acc, width, label='Benchmark', color='#e74c3c', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

    # Add improvement percentage above pairs
    for i, imp in enumerate(improvements):
        y_pos = max(optimized_acc[i], benchmark_acc[i]) + 0.01
        color = '#2ecc71' if imp > 0 else '#e74c3c' if imp < 0 else '#95a5a6'
        ax.text(i, y_pos, f'{imp:+.2f}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Configurations: Optimized vs Benchmark Accuracy',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, max(max(optimized_acc), max(benchmark_acc)) + 0.05)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

    return configs, optimized_acc, benchmark_acc, improvements


def create_training_curves_comparison(all_data, output_dir):
    """Create line plots comparing training curves for all 5 configs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Total Loss
    ax = axes[0, 0]
    for data in all_data:
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'total_loss' in log.columns:
            ax.plot(log['epoch'], log['total_loss'], label=data['config_name'], linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Total Loss', fontweight='bold')
    ax.set_title('Total Loss Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Plot 2: Classification Loss
    ax = axes[0, 1]
    for data in all_data:
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'class_loss' in log.columns:
            ax.plot(log['epoch'], log['class_loss'], label=data['config_name'], linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Classification Loss', fontweight='bold')
    ax.set_title('Classification Loss Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Plot 3: Global Constraint Loss
    ax = axes[1, 0]
    for data in all_data:
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'global_loss' in log.columns:
            ax.plot(log['epoch'], log['global_loss'], label=data['config_name'], linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Global Constraint Loss', fontweight='bold')
    ax.set_title('Global Constraint Loss Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Plot 4: Local Constraint Loss
    ax = axes[1, 1]
    for data in all_data:
        if data is None or data['training_log'] is None:
            continue
        log = data['training_log']
        if 'epoch' in log.columns and 'local_loss' in log.columns:
            ax.plot(log['epoch'], log['local_loss'], label=data['config_name'], linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Local Constraint Loss', fontweight='bold')
    ax.set_title('Local Constraint Loss Over Epochs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    plt.suptitle('Training Dynamics Comparison: Top 5 Configurations',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_heatmap(all_data, output_dir):
    """Create heatmap of various metrics across all 5 configs."""
    metrics_names = ['Overall Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']

    # Collect optimized metrics
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
            # Optimized
            if data['evaluation_metrics'] is not None:
                metric_row = data['evaluation_metrics'][data['evaluation_metrics']['metric'] == metric_name]
                opt_row.append(float(metric_row['value'].values[0]) if not metric_row.empty else 0)
            else:
                opt_row.append(0)

            # Benchmark
            if data['benchmark_metrics'] is not None:
                metric_row = data['benchmark_metrics'][data['benchmark_metrics']['metric'] == metric_name]
                bench_row.append(float(metric_row['value'].values[0]) if not metric_row.empty else 0)
            else:
                bench_row.append(0)

        opt_matrix.append(opt_row)
        bench_matrix.append(bench_row)

    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Optimized heatmap
    opt_df = pd.DataFrame(opt_matrix, columns=metrics_names, index=configs)
    sns.heatmap(opt_df, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax1,
               vmin=0.3, vmax=0.7, cbar_kws={'label': 'Score'})
    ax1.set_title('Optimized Model Metrics', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('')
    ax1.set_ylabel('Configuration', fontweight='bold')

    # Benchmark heatmap
    bench_df = pd.DataFrame(bench_matrix, columns=metrics_names, index=configs)
    sns.heatmap(bench_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax2,
               vmin=0.3, vmax=0.7, cbar_kws={'label': 'Score'})
    ax2.set_title('Benchmark Metrics', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    plt.suptitle('Performance Metrics Heatmap: Top 5 Configurations',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_improvement_breakdown(configs, improvements, output_dir):
    """Create horizontal bar chart showing improvement breakdown."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by improvement
    sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
    sorted_configs = [configs[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]

    colors = ['#2ecc71' if imp > 0 else '#e74c3c' if imp < 0 else '#95a5a6'
             for imp in sorted_improvements]

    bars = ax.barh(sorted_configs, sorted_improvements, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
        width = bar.get_width()
        label_x = width + (0.05 if width > 0 else -0.05)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{imp:+.2f}%',
               ha='left' if width > 0 else 'right', va='center',
               fontsize=11, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('Improvement over Benchmark (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Improvement Ranking: Top 5 Configurations',
                fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_improvements.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_constraint_satisfaction_comparison(all_data, output_dir):
    """Compare constraint satisfaction rates across configs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    configs = []
    global_satisfaction = []
    local_satisfaction = []

    for data in all_data:
        if data is None or data['constraint_comparison'] is None:
            continue

        configs.append(data['config_name'])
        comp = data['constraint_comparison']

        # Calculate satisfaction rates
        if 'prediction_dropout_count' in comp.columns and 'target_dropout_count' in comp.columns:
            # Global constraint satisfaction (at final epoch)
            final_pred = comp['prediction_dropout_count'].iloc[-1]
            target = comp['target_dropout_count'].iloc[-1]
            global_sat = 1.0 - abs(final_pred - target) / max(target, 1)
            global_satisfaction.append(max(0, global_sat))
        else:
            global_satisfaction.append(0)

        # Local constraint satisfaction (average across courses)
        if 'local_violations' in comp.columns:
            avg_violations = comp['local_violations'].mean()
            # Assume 10 courses, satisfaction = 1 - (violations / courses)
            local_sat = 1.0 - min(avg_violations / 10.0, 1.0)
            local_satisfaction.append(max(0, local_sat))
        else:
            local_satisfaction.append(0)

    # Global constraint satisfaction
    bars1 = ax1.bar(configs, global_satisfaction, color='#3498db', alpha=0.8)
    ax1.set_ylabel('Satisfaction Rate', fontweight='bold')
    ax1.set_title('Global Constraint Satisfaction', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
    ax1.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Local constraint satisfaction
    bars2 = ax2.bar(configs, local_satisfaction, color='#9b59b6', alpha=0.8)
    ax2.set_ylabel('Satisfaction Rate', fontweight='bold')
    ax2.set_title('Local Constraint Satisfaction (Avg)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
    ax2.legend()

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('Constraint Satisfaction Comparison: Top 5 Configurations',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_constraint_satisfaction.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_hyperparameter_summary(output_dir):
    """Create visualization of hyperparameter configurations."""
    from config.experiment_config import NN_CONFIGS

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    configs = [c['name'] for c in NN_CONFIGS]

    # Lambda values
    ax = axes[0, 0]
    lambda_global = [c['lambda_global'] for c in NN_CONFIGS]
    lambda_local = [c['lambda_local'] for c in NN_CONFIGS]
    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width/2, lambda_global, width, label='Global λ', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, lambda_local, width, label='Local λ', color='#3498db', alpha=0.8)
    ax.set_ylabel('Lambda Value', fontweight='bold')
    ax.set_title('Lambda Hyperparameters', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')

    # Architecture depth
    ax = axes[0, 1]
    depths = [len(c['hidden_dims']) for c in NN_CONFIGS]
    colors_depth = plt.cm.viridis(np.linspace(0.3, 0.9, len(depths)))
    bars = ax.bar(configs, depths, color=colors_depth, alpha=0.8)
    ax.set_ylabel('Number of Layers', fontweight='bold')
    ax.set_title('Network Depth', fontsize=12, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    for bar, depth in zip(bars, depths):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{depth}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Dropout rates
    ax = axes[1, 0]
    dropout_rates = [c['dropout'] for c in NN_CONFIGS]
    bars = ax.bar(configs, dropout_rates, color='#f39c12', alpha=0.8)
    ax.set_ylabel('Dropout Rate', fontweight='bold')
    ax.set_title('Dropout Regularization', fontsize=12, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    for bar, dropout in zip(bars, dropout_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{dropout:.2f}',
               ha='center', va='bottom', fontsize=9)

    # Batch sizes
    ax = axes[1, 1]
    batch_sizes = [c['batch_size'] for c in NN_CONFIGS]
    bars = ax.bar(configs, batch_sizes, color='#1abc9c', alpha=0.8)
    ax.set_ylabel('Batch Size', fontweight='bold')
    ax.set_title('Batch Size Configuration', fontsize=12, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    for bar, batch in zip(bars, batch_sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{batch}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Hyperparameter Configuration Summary: Top 5',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_hyperparameters.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_report(all_data, configs, optimized_acc, benchmark_acc, improvements, output_dir):
    """Generate a comprehensive text report comparing all configurations."""
    report_path = output_dir / "TOP5_COMPARISON_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Top 5 Configurations - Comprehensive Comparison Report\n\n")
        f.write("## Executive Summary\n\n")

        # Find best performer
        best_idx = improvements.index(max(improvements))
        f.write(f"**🏆 BEST PERFORMER:** `{configs[best_idx]}`\n")
        f.write(f"- **Optimized Accuracy:** {optimized_acc[best_idx]:.4f}\n")
        f.write(f"- **Benchmark Accuracy:** {benchmark_acc[best_idx]:.4f}\n")
        f.write(f"- **Improvement:** {improvements[best_idx]:+.2f}%\n\n")

        f.write("---\n\n")
        f.write("## Performance Rankings\n\n")
        f.write("| Rank | Config | Optimized | Benchmark | Improvement |\n")
        f.write("|------|--------|-----------|-----------|-------------|\n")

        # Sort by improvement
        sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
        for rank, idx in enumerate(sorted_indices, 1):
            status = "🏆" if rank == 1 else "✅" if improvements[idx] > 0 else "⚠️"
            f.write(f"| {rank} {status} | {configs[idx]} | {optimized_acc[idx]:.4f} | "
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
                    f.write(f"- {row['metric']}: {float(row['value']):.4f}\n")
                f.write("\n")

            # Benchmark metrics
            if data['benchmark_metrics'] is not None:
                f.write("**Benchmark:**\n")
                for _, row in data['benchmark_metrics'].iterrows():
                    f.write(f"- {row['metric']}: {float(row['value']):.4f}\n")
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

        # Insight 3: Overall conclusion
        f.write("### 3. Recommendation\n\n")
        if max(improvements) > 2.0:
            f.write(f"✅ **STRONG RECOMMENDATION:** Use `{configs[best_idx]}` for production.\n")
            f.write(f"The {improvements[best_idx]:+.2f}% improvement over benchmark is substantial.\n\n")
        elif max(improvements) > 0.5:
            f.write(f"⚠️ **MODERATE RECOMMENDATION:** Use `{configs[best_idx]}` if accuracy matters.\n")
            f.write(f"The {improvements[best_idx]:+.2f}% improvement is modest but real.\n\n")
        else:
            f.write(f"❌ **NOT RECOMMENDED:** Consider using simple benchmark instead.\n")
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

    print(f"\n✅ Comparison report saved to: {report_path}")


def main():
    """Main function to run all comparisons and generate visualizations."""
    print("="*80)
    print("Top 5 Configurations - Comprehensive Analysis")
    print("="*80)

    # Load experiment configuration
    from config.experiment_config import NN_CONFIGS

    # Create output directory
    output_dir = Path("./results/top5_comparison")
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

    print("\nGenerating visualizations...")

    # 1. Accuracy comparison
    print("  1. Accuracy comparison bar chart...")
    configs, optimized_acc, benchmark_acc, improvements = create_accuracy_comparison(all_data, output_dir)

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
    print("✅ Analysis Complete!")
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

    # Print quick summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    best_idx = improvements.index(max(improvements))
    print(f"\n🏆 BEST: {configs[best_idx]}")
    print(f"   Optimized: {optimized_acc[best_idx]:.4f}")
    print(f"   Benchmark: {benchmark_acc[best_idx]:.4f}")
    print(f"   Improvement: {improvements[best_idx]:+.2f}%\n")

    print("Rankings:")
    sorted_indices = sorted(range(len(improvements)), key=lambda i: improvements[i], reverse=True)
    for rank, idx in enumerate(sorted_indices, 1):
        status = "🏆" if rank == 1 else "✅" if improvements[idx] > 0 else "⚠️"
        print(f"  {rank}. {status} {configs[idx]}: {improvements[idx]:+.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

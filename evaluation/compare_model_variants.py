#!/usr/bin/env python3
"""
Compare results across different model variants.
Analyzes all model variant folders and creates comparative visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def find_model_variants(results_dir):
    """Find all model variant folders in results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    variants = []
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if item.name not in ['constraint_analysis', 'comprehensive_analysis', 'constraint_comparison']:
                nn_results = item / 'nn_results.json'
                if nn_results.exists():
                    variants.append(item.name)

    return sorted(variants)


def load_variant_results(results_dir, variant_name):
    """Load results for a specific model variant."""
    variant_path = Path(results_dir) / variant_name
    results_file = variant_path / 'nn_results.json'

    if not results_file.exists():
        return {}

    with open(results_file, 'r') as f:
        return json.load(f)


def create_variant_comparison_table(all_variants_data):
    """Create a comparison table across all variants."""
    comparison_data = []

    for variant_name, results in all_variants_data.items():
        for config_name, config_data in results.items():
            for constraint, metrics in config_data.items():
                comparison_data.append({
                    'Variant': variant_name,
                    'Config': config_name,
                    'Constraint': constraint,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Benchmark': metrics.get('benchmark_accuracy', 0),
                    'Improvement': metrics.get('accuracy', 0) - metrics.get('benchmark_accuracy', 0),
                    'Training_Time': metrics.get('training_time', 0)
                })

    return pd.DataFrame(comparison_data)


def plot_variant_comparison(df, output_dir):
    """Create comparison plots across model variants."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax1 = axes[0, 0]
    variant_avg = df.groupby('Variant')['Accuracy'].mean().sort_values(ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(variant_avg)))
    bars = ax1.bar(range(len(variant_avg)), variant_avg.values * 100, color=colors)
    ax1.set_xticks(range(len(variant_avg)))
    ax1.set_xticklabels(variant_avg.index, rotation=45, ha='right')
    ax1.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax1.set_title('Average Accuracy by Model Variant', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, variant_avg.values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = axes[0, 1]
    variant_improvement = df.groupby('Variant')['Improvement'].mean().sort_values(ascending=False)
    colors_imp = ['green' if x > 0 else 'red' for x in variant_improvement.values]
    bars = ax2.bar(range(len(variant_improvement)), variant_improvement.values * 100, color=colors_imp, alpha=0.7)
    ax2.set_xticks(range(len(variant_improvement)))
    ax2.set_xticklabels(variant_improvement.index, rotation=45, ha='right')
    ax2.set_ylabel('Average Improvement over Benchmark (%)', fontweight='bold')
    ax2.set_title('Average Improvement by Model Variant', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, variant_improvement.values):
        height = bar.get_height()
        y_pos = height + 0.1 if height > 0 else height - 0.3
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val*100:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    ax3 = axes[1, 0]
    best_per_variant = df.groupby('Variant').apply(
        lambda x: x.nlargest(1, 'Accuracy')[['Accuracy', 'Constraint', 'Config']].iloc[0]
    )
    best_accuracies = best_per_variant['Accuracy'].sort_values(ascending=False)

    bars = ax3.barh(range(len(best_accuracies)), best_accuracies.values * 100, color='steelblue', alpha=0.8)
    ax3.set_yticks(range(len(best_accuracies)))
    ax3.set_yticklabels(best_accuracies.index)
    ax3.set_xlabel('Best Accuracy (%)', fontweight='bold')
    ax3.set_title('Best Result per Model Variant', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, best_accuracies.values)):
        width = bar.get_width()
        config = best_per_variant.loc[best_accuracies.index[i], 'Config']
        constraint = best_per_variant.loc[best_accuracies.index[i], 'Constraint']
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{val*100:.2f}%\n{config}@{constraint}',
                ha='left', va='center', fontsize=7)

    ax4 = axes[1, 1]
    variant_time = df.groupby('Variant')['Training_Time'].mean().sort_values()
    bars = ax4.barh(range(len(variant_time)), variant_time.values, color='coral', alpha=0.8)
    ax4.set_yticks(range(len(variant_time)))
    ax4.set_yticklabels(variant_time.index)
    ax4.set_xlabel('Average Training Time (seconds)', fontweight='bold')
    ax4.set_title('Training Time by Model Variant', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, variant_time.values):
        width = bar.get_width()
        ax4.text(width + 5, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}s', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'model_variant_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: model_variant_comparison.png")
    plt.close()

    constraints = sorted(df['Constraint'].unique())
    variants = sorted(df['Variant'].unique())

    pivot_accuracy = df.pivot_table(
        values='Accuracy',
        index='Variant',
        columns='Constraint',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_accuracy * 100, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=50, vmax=80, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('Model Variant Performance Heatmap Across Constraints', fontsize=14, fontweight='bold')
    ax.set_xlabel('Constraint (Local, Global)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Variant', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'variant_constraint_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: variant_constraint_heatmap.png")
    plt.close()


def generate_comparison_report(df, all_variants_data, output_dir):
    """Generate markdown comparison report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# Model Variant Comparison Report\n\n")
    report.append(f"**Total Variants Tested**: {df['Variant'].nunique()}\n")
    report.append(f"**Total Experiments**: {len(df)}\n\n")

    report.append("## Overall Rankings\n\n")
    variant_stats = df.groupby('Variant').agg({
        'Accuracy': ['mean', 'max'],
        'Improvement': 'mean',
        'Training_Time': 'mean'
    }).round(4)

    variant_stats.columns = ['Avg_Accuracy', 'Best_Accuracy', 'Avg_Improvement', 'Avg_Time']
    variant_stats = variant_stats.sort_values('Avg_Accuracy', ascending=False)

    report.append("| Rank | Variant | Avg Accuracy | Best Accuracy | Avg Improvement | Avg Time (s) |\n")
    report.append("|------|---------|--------------|---------------|-----------------|-------------|\n")

    for i, (variant, row) in enumerate(variant_stats.iterrows(), 1):
        report.append(f"| {i} | {variant} | {row['Avg_Accuracy']*100:.2f}% | {row['Best_Accuracy']*100:.2f}% | {row['Avg_Improvement']*100:+.2f}% | {row['Avg_Time']:.1f} |\n")

    report.append("\n## Best Result per Variant\n\n")
    report.append("| Variant | Config | Constraint | Accuracy | Benchmark | Improvement |\n")
    report.append("|---------|--------|------------|----------|-----------|-------------|\n")

    for variant in sorted(df['Variant'].unique()):
        best_row = df[df['Variant'] == variant].nlargest(1, 'Accuracy').iloc[0]
        report.append(f"| {variant} | {best_row['Config']} | {best_row['Constraint']} | "
                     f"{best_row['Accuracy']*100:.2f}% | {best_row['Benchmark']*100:.2f}% | "
                     f"{best_row['Improvement']*100:+.2f}% |\n")

    report.append("\n## Performance Comparison\n\n")

    baseline_present = 'baseline' in df['Variant'].values
    if baseline_present:
        baseline_avg = df[df['Variant'] == 'baseline']['Accuracy'].mean()
        report.append(f"**Baseline Average**: {baseline_avg*100:.2f}%\n\n")

        for variant in sorted(df['Variant'].unique()):
            if variant != 'baseline':
                variant_avg = df[df['Variant'] == variant]['Accuracy'].mean()
                delta = variant_avg - baseline_avg
                report.append(f"- **{variant}**: {variant_avg*100:.2f}% ({delta*100:+.2f}% vs baseline)\n")
    else:
        report.append("No baseline variant found for comparison.\n")

    report.append("\n## Key Insights\n\n")
    best_variant = variant_stats.index[0]
    best_avg = variant_stats.loc[best_variant, 'Avg_Accuracy']
    report.append(f"1. **Best Overall Variant**: {best_variant} with {best_avg*100:.2f}% average accuracy\n")

    overall_best = df.nlargest(1, 'Accuracy').iloc[0]
    report.append(f"2. **Best Single Result**: {overall_best['Variant']} - {overall_best['Config']} @ "
                 f"{overall_best['Constraint']} with {overall_best['Accuracy']*100:.2f}% accuracy\n")

    fastest_variant = variant_stats['Avg_Time'].idxmin()
    slowest_variant = variant_stats['Avg_Time'].idxmax()
    report.append(f"3. **Fastest Training**: {fastest_variant} ({variant_stats.loc[fastest_variant, 'Avg_Time']:.1f}s avg)\n")
    report.append(f"4. **Slowest Training**: {slowest_variant} ({variant_stats.loc[slowest_variant, 'Avg_Time']:.1f}s avg)\n")

    report_path = output_path / 'MODEL_VARIANT_COMPARISON.md'
    with open(report_path, 'w') as f:
        f.write(''.join(report))

    print(f"Saved: MODEL_VARIANT_COMPARISON.md")


def main():
    results_dir = 'results'

    print("="*80)
    print("MODEL VARIANT COMPARISON ANALYSIS")
    print("="*80)

    variants = find_model_variants(results_dir)

    if not variants:
        print(f"\nNo model variants found in {results_dir}/")
        print("Make sure you have run experiments with different variants.")
        return

    print(f"\nFound {len(variants)} model variant(s):")
    for variant in variants:
        print(f"  - {variant}")

    print("\nLoading results...")
    all_variants_data = {}
    for variant in variants:
        data = load_variant_results(results_dir, variant)
        if data:
            all_variants_data[variant] = data
            print(f"  Loaded: {variant}")

    if not all_variants_data:
        print("\nNo valid results found.")
        return

    print("\nCreating comparison table...")
    df = create_variant_comparison_table(all_variants_data)
    print(f"  Total records: {len(df)}")

    output_dir = Path(results_dir) / 'variant_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'variant_comparison_data.csv', index=False)
    print(f"\nSaved: variant_comparison_data.csv")

    print("\nGenerating visualizations...")
    plot_variant_comparison(df, output_dir)

    print("\nGenerating comparison report...")
    generate_comparison_report(df, all_variants_data, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated Files:")
    print("  - model_variant_comparison.png")
    print("  - variant_constraint_heatmap.png")
    print("  - variant_comparison_data.csv")
    print("  - MODEL_VARIANT_COMPARISON.md")
    print()


if __name__ == "__main__":
    main()

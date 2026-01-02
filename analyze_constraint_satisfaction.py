#!/usr/bin/env python3
"""
Analyze constraint satisfaction across all successful experiments.
Creates per-course visualizations and summaries.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Define constraint configurations
CONSTRAINTS = [
    "(0.4, 0.2)",
    "(0.5, 0.3)",
    "(0.6, 0.5)",
    "(0.7, 0.5)",
    "(0.8, 0.2)",
    "(0.8, 0.7)",
    "(0.9, 0.5)",
    "(0.9, 0.8)"
]

# Failed experiments to exclude
FAILED_EXPERIMENTS = [
    ("very_deep_extreme_lambda", "(0.8, 0.7)"),
    ("very_deep_extreme_lambda", "(0.8, 0.2)"),
    ("very_deep_extreme_lambda", "(0.7, 0.5)"),
    ("very_deep_extreme_lambda", "(0.4, 0.2)")
]

def load_results():
    """Load and filter the results JSON."""
    with open('./results/nn_results.json', 'r') as f:
        results = json.load(f)

    # Filter out failed experiments
    filtered_results = {}
    for config_name, config_data in results.items():
        filtered_results[config_name] = {}
        for constraint_key, metrics in config_data.items():
            if (config_name, constraint_key) not in FAILED_EXPERIMENTS:
                filtered_results[config_name][constraint_key] = metrics

    return filtered_results

def analyze_constraint_satisfaction():
    """Analyze how well each configuration satisfied constraints by course."""

    results_dir = Path('/home/user/OptimizationLoss/results')
    constraint_data = defaultdict(lambda: defaultdict(list))

    # Map constraint tuple to folder name
    constraint_folders = {
        "(0.4, 0.2)": "constraint_0.4_0.2",
        "(0.5, 0.3)": "constraint_0.5_0.3",
        "(0.6, 0.5)": "constraint_0.6_0.5",
        "(0.7, 0.5)": "constraint_0.7_0.5",
        "(0.8, 0.2)": "constraint_0.8_0.2",
        "(0.8, 0.7)": "constraint_0.8_0.7",
        "(0.9, 0.5)": "constraint_0.9_0.5",
        "(0.9, 0.8)": "constraint_0.9_0.8"
    }

    # Config name mappings
    config_mappings = {
        "arch_deep": "hyperparam_arch_deep",
        "dropout_high": "hyperparam_dropout_high",
        "lambda_high": "hyperparam_lambda_high",
        "very_deep_baseline": "hyperparam_very_deep_baseline",
        "very_deep_extreme_lambda": "hyperparam_very_deep_extreme_lambda"
    }

    # Iterate through all constraint folders
    for constraint_key, folder_name in constraint_folders.items():
        constraint_path = results_dir / folder_name

        if not constraint_path.exists():
            continue

        # Check each config
        for short_name, full_name in config_mappings.items():
            # Skip failed experiments
            if (short_name, constraint_key) in FAILED_EXPERIMENTS:
                continue

            config_path = constraint_path / full_name
            if not config_path.exists():
                continue

            # Read constraint comparison file
            constraint_file = config_path / "constraint_comparison.csv"
            if constraint_file.exists():
                df = pd.read_csv(constraint_file)

                # Store per-course data
                for course_id in df['Course_ID'].unique():
                    course_data = df[df['Course_ID'] == course_id]

                    # Calculate constraint satisfaction metrics
                    total_constraints = len(course_data[course_data['Constraint'] != 'Unlimited'])
                    satisfied = len(course_data[course_data['Status'] == 'OK'])
                    violated = len(course_data[course_data['Status'] == 'VIOLATED'])
                    overprediction = course_data['Overprediction'].sum()

                    constraint_data[constraint_key][course_id].append({
                        'config': short_name,
                        'total_constraints': total_constraints,
                        'satisfied': satisfied,
                        'violated': violated,
                        'satisfaction_rate': satisfied / total_constraints if total_constraints > 0 else 1.0,
                        'overprediction': overprediction
                    })

    return constraint_data

def create_course_summary(constraint_data):
    """Create a summary of constraint satisfaction by course."""

    summary = []

    for constraint_key in CONSTRAINTS:
        if constraint_key not in constraint_data:
            continue

        for course_id, configs in constraint_data[constraint_key].items():
            for config_info in configs:
                summary.append({
                    'Constraint': constraint_key,
                    'Course_ID': course_id,
                    'Config': config_info['config'],
                    'Satisfaction_Rate': config_info['satisfaction_rate'],
                    'Satisfied': config_info['satisfied'],
                    'Violated': config_info['violated'],
                    'Overprediction': config_info['overprediction']
                })

    return pd.DataFrame(summary)

def visualize_constraint_satisfaction(df_summary):
    """Create comprehensive visualizations of constraint satisfaction."""

    # Create output directory
    output_dir = Path('./results/constraint_analysis')
    output_dir.mkdir(exist_ok=True)

    # 1. Heatmap: Satisfaction rate by constraint and course
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Average satisfaction rate by constraint
    ax1 = axes[0, 0]
    constraint_avg = df_summary.groupby('Constraint')['Satisfaction_Rate'].mean().sort_index()
    constraint_avg.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Average Constraint Satisfaction Rate by Constraint Setting', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Constraint (Global, Local)', fontsize=12)
    ax1.set_ylabel('Average Satisfaction Rate', fontsize=12)
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Satisfaction')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Satisfaction rate by config
    ax2 = axes[0, 1]
    config_avg = df_summary.groupby('Config')['Satisfaction_Rate'].mean().sort_values(ascending=False)
    config_avg.plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Average Constraint Satisfaction Rate by Configuration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Average Satisfaction Rate', fontsize=12)
    ax2.set_ylim([0, 1.1])
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Satisfaction')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Overprediction by constraint
    ax3 = axes[1, 0]
    overpred_by_constraint = df_summary.groupby('Constraint')['Overprediction'].sum().sort_index()
    overpred_by_constraint.plot(kind='bar', ax=ax3, color='indianred')
    ax3.set_title('Total Overprediction by Constraint Setting', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Constraint (Global, Local)', fontsize=12)
    ax3.set_ylabel('Total Overprediction Count', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # Violation count by constraint
    ax4 = axes[1, 1]
    violation_data = df_summary.groupby('Constraint').agg({
        'Satisfied': 'sum',
        'Violated': 'sum'
    })
    violation_data.plot(kind='bar', ax=ax4, stacked=True, color=['lightgreen', 'salmon'])
    ax4.set_title('Constraint Satisfaction vs Violation Counts', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Constraint (Global, Local)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.legend(['Satisfied', 'Violated'])
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'constraint_satisfaction_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'constraint_satisfaction_overview.png'}")
    plt.close()

    # 2. Heatmap by course
    pivot_satisfaction = df_summary.pivot_table(
        values='Satisfaction_Rate',
        index='Course_ID',
        columns='Constraint',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_satisfaction, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Satisfaction Rate'})
    ax.set_title('Constraint Satisfaction Rate by Course and Constraint Setting',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Constraint (Global, Local)', fontsize=12)
    ax.set_ylabel('Course ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'constraint_satisfaction_heatmap_by_course.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'constraint_satisfaction_heatmap_by_course.png'}")
    plt.close()

    # 3. Per-course detailed view for best constraint
    best_constraint = constraint_avg.idxmax()
    best_df = df_summary[df_summary['Constraint'] == best_constraint]

    fig, ax = plt.subplots(figsize=(16, 8))
    course_pivot = best_df.pivot_table(
        values='Satisfaction_Rate',
        index='Course_ID',
        columns='Config',
        aggfunc='mean'
    )

    course_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'Constraint Satisfaction by Course for Best Setting: {best_constraint}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Course ID', fontsize=12)
    ax.set_ylabel('Satisfaction Rate', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f'constraint_satisfaction_best_{best_constraint.replace(", ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()

    return output_dir

def generate_report(df_summary, constraint_data, results):
    """Generate a detailed markdown report."""

    output_dir = Path('/home/user/OptimizationLoss/results/constraint_analysis')
    output_dir.mkdir(exist_ok=True)

    report = []
    report.append("# Constraint Satisfaction Analysis Report\n")
    report.append("## Executive Summary\n")
    report.append(f"- **Total Successful Experiments**: {len(df_summary.groupby(['Constraint', 'Config']))}\n")
    report.append(f"- **Failed Experiments Excluded**: {len(FAILED_EXPERIMENTS)}\n")
    report.append(f"- **Courses Analyzed**: {df_summary['Course_ID'].nunique()}\n")
    report.append(f"- **Constraint Settings Tested**: {len(CONSTRAINTS)}\n\n")

    # Failed experiments section
    report.append("## Failed Experiments (Excluded from Analysis)\n")
    report.append("| Configuration | Constraint | Reason |\n")
    report.append("|--------------|------------|--------|\n")
    for config, constraint in FAILED_EXPERIMENTS:
        report.append(f"| {config} | {constraint} | Benchmark baseline failed |\n")
    report.append("\n")

    # Overall satisfaction rates
    report.append("## Overall Constraint Satisfaction\n")
    constraint_stats = df_summary.groupby('Constraint').agg({
        'Satisfaction_Rate': 'mean',
        'Satisfied': 'sum',
        'Violated': 'sum',
        'Overprediction': 'sum'
    }).sort_values('Satisfaction_Rate', ascending=False)

    report.append("| Constraint | Avg Satisfaction | Total Satisfied | Total Violated | Total Overprediction |\n")
    report.append("|-----------|------------------|-----------------|----------------|---------------------|\n")
    for idx, row in constraint_stats.iterrows():
        report.append(f"| {idx} | {row['Satisfaction_Rate']:.1%} | {int(row['Satisfied'])} | {int(row['Violated'])} | {int(row['Overprediction'])} |\n")
    report.append("\n")

    # Best performers
    report.append("## Best Performing Configurations by Constraint Satisfaction\n")
    best_configs = df_summary.groupby('Config')['Satisfaction_Rate'].mean().sort_values(ascending=False)

    report.append("| Configuration | Avg Satisfaction Rate |\n")
    report.append("|--------------|----------------------|\n")
    for config, rate in best_configs.items():
        report.append(f"| {config} | {rate:.1%} |\n")
    report.append("\n")

    # Course-level insights
    report.append("## Course-Level Insights\n")
    course_stats = df_summary.groupby('Course_ID').agg({
        'Satisfaction_Rate': 'mean',
        'Violated': 'sum',
        'Overprediction': 'sum'
    }).sort_values('Satisfaction_Rate')

    report.append("### Most Challenging Courses (Lowest Satisfaction Rates)\n")
    report.append("| Course ID | Avg Satisfaction | Total Violations | Total Overprediction |\n")
    report.append("|----------|------------------|------------------|---------------------|\n")
    for idx, row in course_stats.head(5).iterrows():
        report.append(f"| {idx} | {row['Satisfaction_Rate']:.1%} | {int(row['Violated'])} | {int(row['Overprediction'])} |\n")
    report.append("\n")

    report.append("### Best Performing Courses (Highest Satisfaction Rates)\n")
    report.append("| Course ID | Avg Satisfaction | Total Violations | Total Overprediction |\n")
    report.append("|----------|------------------|------------------|---------------------|\n")
    for idx, row in course_stats.tail(5).iterrows():
        report.append(f"| {idx} | {row['Satisfaction_Rate']:.1%} | {int(row['Violated'])} | {int(row['Overprediction'])} |\n")
    report.append("\n")

    # Key findings
    report.append("## Key Findings\n\n")

    best_constraint = constraint_stats.index[0]
    worst_constraint = constraint_stats.index[-1]

    report.append(f"1. **Best Constraint Setting**: {best_constraint} with {constraint_stats.loc[best_constraint, 'Satisfaction_Rate']:.1%} average satisfaction\n")
    report.append(f"2. **Most Challenging Setting**: {worst_constraint} with {constraint_stats.loc[worst_constraint, 'Satisfaction_Rate']:.1%} average satisfaction\n")
    report.append(f"3. **Most Robust Configuration**: {best_configs.index[0]} consistently satisfies constraints best\n")
    report.append(f"4. **Total Violations Across All Experiments**: {int(df_summary['Violated'].sum())}\n")
    report.append(f"5. **Total Overpredictions**: {int(df_summary['Overprediction'].sum())}\n\n")

    # Recommendations
    report.append("## Recommendations\n\n")
    report.append(f"- For maximum constraint satisfaction, use **{best_constraint}** constraints\n")
    report.append(f"- Deploy **{best_configs.index[0]}** configuration for best robustness across courses\n")
    report.append(f"- Pay special attention to courses {', '.join(map(str, course_stats.head(3).index.tolist()))} which show lower satisfaction rates\n")

    # Write report
    report_path = output_dir / 'CONSTRAINT_SATISFACTION_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(''.join(report))

    print(f"\nSaved: {report_path}")
    return report_path

def main():
    print("="*80)
    print("CONSTRAINT SATISFACTION ANALYSIS")
    print("="*80)

    # Load and filter results
    print("\n1. Loading and filtering results...")
    results = load_results()
    print(f"   ✓ Loaded results, excluded {len(FAILED_EXPERIMENTS)} failed experiments")

    # Analyze constraint satisfaction
    print("\n2. Analyzing constraint satisfaction by course...")
    constraint_data = analyze_constraint_satisfaction()
    print(f"   ✓ Analyzed {len(constraint_data)} constraint settings")

    # Create summary dataframe
    print("\n3. Creating summary data...")
    df_summary = create_course_summary(constraint_data)
    print(f"   ✓ Created summary with {len(df_summary)} records")

    # Save raw data
    output_dir = Path('./results/constraint_analysis')
    output_dir.mkdir(exist_ok=True)
    df_summary.to_csv(output_dir / 'constraint_satisfaction_data.csv', index=False)
    print(f"   ✓ Saved raw data to {output_dir / 'constraint_satisfaction_data.csv'}")

    # Create visualizations
    print("\n4. Generating visualizations...")
    visualize_constraint_satisfaction(df_summary)
    print("   ✓ Created all visualizations")

    # Generate report
    print("\n5. Generating detailed report...")
    report_path = generate_report(df_summary, constraint_data, results)
    print("   ✓ Report generated")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("  - constraint_satisfaction_overview.png")
    print("  - constraint_satisfaction_heatmap_by_course.png")
    print("  - constraint_satisfaction_best_*.png")
    print("  - constraint_satisfaction_data.csv")
    print("  - CONSTRAINT_SATISFACTION_REPORT.md")

if __name__ == "__main__":
    main()

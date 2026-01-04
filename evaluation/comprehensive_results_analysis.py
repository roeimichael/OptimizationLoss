#!/usr/bin/env python3

import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

CONFIGS = {
    "arch_deep": "hyperparam_arch_deep",
    "dropout_high": "hyperparam_dropout_high",
    "lambda_high": "hyperparam_lambda_high",
    "very_deep_baseline": "hyperparam_very_deep_baseline",
    "very_deep_extreme_lambda": "hyperparam_very_deep_extreme_lambda"
}

CONSTRAINT_FOLDERS = {
    "(0.4, 0.2)": "constraint_0.4_0.2",
    "(0.5, 0.3)": "constraint_0.5_0.3",
    "(0.6, 0.5)": "constraint_0.6_0.5",
    "(0.7, 0.5)": "constraint_0.7_0.5",
    "(0.8, 0.2)": "constraint_0.8_0.2",
    "(0.8, 0.7)": "constraint_0.8_0.7",
    "(0.9, 0.5)": "constraint_0.9_0.5",
    "(0.9, 0.8)": "constraint_0.9_0.8"
}

FAILED_EXPERIMENTS = [
    ("very_deep_extreme_lambda", "(0.8, 0.7)"),
    ("very_deep_extreme_lambda", "(0.8, 0.2)"),
    ("very_deep_extreme_lambda", "(0.7, 0.5)"),
    ("very_deep_extreme_lambda", "(0.4, 0.2)")
]

CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

def read_accuracy(csv_file):
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == 'Overall Accuracy':
                    return float(row[1])
    except:
        pass
    return None

def load_all_results():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    all_results = []

    for config_short, config_full in CONFIGS.items():
        for constraint_key, folder_name in CONSTRAINT_FOLDERS.items():
            if (config_short, constraint_key) in FAILED_EXPERIMENTS:
                continue

            config_path = results_dir / folder_name / config_full
            if not config_path.exists():
                continue

            metrics_file = config_path / 'evaluation_metrics.csv'
            benchmark_file = config_path / 'benchmark_metrics.csv'

            if metrics_file.exists() and benchmark_file.exists():
                accuracy = read_accuracy(metrics_file)
                benchmark_accuracy = read_accuracy(benchmark_file)

                if accuracy is not None and benchmark_accuracy is not None:
                    all_results.append({
                        'config': config_short,
                        'constraint': constraint_key,
                        'accuracy': accuracy,
                        'benchmark_accuracy': benchmark_accuracy,
                        'improvement': accuracy - benchmark_accuracy,
                        'path': config_path
                    })

    return all_results

def load_predictions_and_constraints(results):
    detailed_results = []

    for result in results:
        config_path = result['path']
        predictions_file = config_path / 'final_predictions.csv'
        constraint_file = config_path / 'constraint_comparison.csv'

        if not predictions_file.exists() or not constraint_file.exists():
            continue

        predictions_df = pd.read_csv(predictions_file)
        constraints_df = pd.read_csv(constraint_file)
        class_counts = predictions_df['Predicted_Label'].value_counts().to_dict()
        global_max, local_max = eval(result['constraint'])
        total_samples = len(predictions_df)

        dropout_count = class_counts.get(0, 0)
        enrolled_count = class_counts.get(1, 0)
        graduate_count = class_counts.get(2, 0)

        global_dropout_limit = int(total_samples * global_max)
        global_enrolled_limit = int(total_samples * local_max)

        course_predictions = defaultdict(lambda: {'Dropout': 0, 'Enrolled': 0, 'Graduate': 0})
        for _, row in predictions_df.iterrows():
            course_id = row['Course_ID']
            pred_class = CLASS_NAMES[int(row['Predicted_Label'])]
            course_predictions[course_id][pred_class] += 1

        total_constraints = len(constraints_df[constraints_df['Constraint'] != 'Unlimited'])
        satisfied_constraints = len(constraints_df[constraints_df['Status'] == 'OK'])

        detailed_results.append({
            **result,
            'dropout_count': dropout_count,
            'enrolled_count': enrolled_count,
            'graduate_count': graduate_count,
            'global_dropout_limit': global_dropout_limit,
            'global_enrolled_limit': global_enrolled_limit,
            'dropout_within_limit': dropout_count <= global_dropout_limit,
            'enrolled_within_limit': enrolled_count <= global_enrolled_limit,
            'course_predictions': dict(course_predictions),
            'total_constraints': total_constraints,
            'satisfied_constraints': satisfied_constraints,
            'constraint_satisfaction_rate': satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
        })

    return detailed_results

def analyze_constraint_satisfaction_by_course():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    constraint_data = defaultdict(lambda: defaultdict(list))

    for constraint_key, folder_name in CONSTRAINT_FOLDERS.items():
        constraint_path = results_dir / folder_name
        if not constraint_path.exists():
            continue

        for short_name, full_name in CONFIGS.items():
            if (short_name, constraint_key) in FAILED_EXPERIMENTS:
                continue

            config_path = constraint_path / full_name
            if not config_path.exists():
                continue

            constraint_file = config_path / "constraint_comparison.csv"
            if constraint_file.exists():
                df = pd.read_csv(constraint_file)

                for course_id in df['Course_ID'].unique():
                    course_data = df[df['Course_ID'] == course_id]
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
    summary = []
    for constraint_key, course_dict in constraint_data.items():
        for course_id, configs in course_dict.items():
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

def create_output_directory():
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results' / 'comprehensive_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_top_5_performers(results, output_dir):
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(14, 8))
    x_labels = [f"{r['config']}\n{r['constraint']}" for r in sorted_results]
    accuracies = [r['accuracy'] * 100 for r in sorted_results]
    benchmarks = [r['benchmark_accuracy'] * 100 for r in sorted_results]
    improvements = [r['improvement'] * 100 for r in sorted_results]

    x = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Transductive Model', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, benchmarks, width, label='Benchmark', color='coral', alpha=0.8)

    for i, (bar, imp) in enumerate(zip(bars1, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'+{imp:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

    ax.set_xlabel('Configuration & Constraint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Performing Configurations Across All Constraints', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'top_5_performers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: top_5_performers.png")
    plt.close()

def plot_performance_by_constraint(results, output_dir):
    constraint_data = defaultdict(list)
    for r in results:
        constraint_data[r['constraint']].append(r)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1 = axes[0]
    constraints_sorted = sorted(constraint_data.keys())
    avg_accuracies = []
    avg_benchmarks = []
    avg_improvements = []

    for constraint in constraints_sorted:
        data = constraint_data[constraint]
        avg_acc = np.mean([d['accuracy'] for d in data]) * 100
        avg_bench = np.mean([d['benchmark_accuracy'] for d in data]) * 100
        avg_imp = np.mean([d['improvement'] for d in data]) * 100
        avg_accuracies.append(avg_acc)
        avg_benchmarks.append(avg_bench)
        avg_improvements.append(avg_imp)

    x = np.arange(len(constraints_sorted))
    width = 0.35

    bars1 = ax1.bar(x - width/2, avg_accuracies, width, label='Avg Transductive', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_benchmarks, width, label='Avg Benchmark', color='coral', alpha=0.8)

    ax1.set_xlabel('Constraint (Global, Local)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Performance by Constraint Setting', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(constraints_sorted)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in avg_improvements]
    bars = ax2.bar(x, avg_improvements, color=colors, alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Constraint (Global, Local)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Improvement over Benchmark (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Improvement by Constraint Setting', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(constraints_sorted)
    ax2.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars, avg_improvements):
        height = bar.get_height()
        y_pos = height + 0.2 if height > 0 else height - 0.4
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_constraint.png', dpi=300, bbox_inches='tight')
    print(f"Saved: performance_by_constraint.png")
    plt.close()

def plot_performance_by_model(results, output_dir):
    config_data = defaultdict(list)
    for r in results:
        config_data[r['config']].append(r)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax1 = axes[0]
    configs_sorted = sorted(config_data.keys())
    avg_accuracies = []
    avg_benchmarks = []

    for config in configs_sorted:
        data = config_data[config]
        avg_acc = np.mean([d['accuracy'] for d in data]) * 100
        avg_bench = np.mean([d['benchmark_accuracy'] for d in data]) * 100
        avg_accuracies.append(avg_acc)
        avg_benchmarks.append(avg_bench)

    x = np.arange(len(configs_sorted))
    width = 0.35

    bars1 = ax1.bar(x - width/2, avg_accuracies, width, label='Avg Transductive', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_benchmarks, width, label='Avg Benchmark', color='coral', alpha=0.8)

    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Performance by Model Configuration', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs_sorted, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    win_rates = []
    for config in configs_sorted:
        data = config_data[config]
        wins = sum(1 for d in data if d['improvement'] > 0.001)
        win_rate = (wins / len(data)) * 100 if data else 0
        win_rates.append(win_rate)

    bars = ax2.bar(x, win_rates, color='forestgreen', alpha=0.7)
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')

    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Win Rate by Model Configuration (% of constraints beaten)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs_sorted, rotation=15, ha='right')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_model.png', dpi=300, bbox_inches='tight')
    print(f"Saved: performance_by_model.png")
    plt.close()

def plot_constraint_satisfaction(detailed_results, df_course_summary, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    constraint_satisfaction = defaultdict(list)
    for r in detailed_results:
        constraint_satisfaction[r['constraint']].append(r['constraint_satisfaction_rate'])

    constraints_sorted = sorted(constraint_satisfaction.keys())
    avg_satisfaction = [np.mean(constraint_satisfaction[c]) * 100 for c in constraints_sorted]

    bars = ax1.bar(constraints_sorted, avg_satisfaction, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    ax1.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')
    ax1.set_xlabel('Constraint (Global, Local)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg Constraint Satisfaction (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Constraint Satisfaction Rate by Setting', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 110])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, avg_satisfaction):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    ax2 = axes[0, 1]
    config_satisfaction = defaultdict(list)
    for r in detailed_results:
        config_satisfaction[r['config']].append(r['constraint_satisfaction_rate'])

    configs_sorted = sorted(config_satisfaction.keys())
    avg_satisfaction_config = [np.mean(config_satisfaction[c]) * 100 for c in configs_sorted]

    bars = ax2.bar(configs_sorted, avg_satisfaction_config, color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (100%)')
    ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Avg Constraint Satisfaction (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Constraint Satisfaction Rate by Model', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(configs_sorted, rotation=15, ha='right')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, avg_satisfaction_config):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    ax3 = axes[1, 0]
    dropout_adherence = []
    labels = []
    for r in detailed_results[:10]:
        dropout_adherence.append([r['dropout_count'], r['global_dropout_limit'] - r['dropout_count']])
        labels.append(f"{r['config'][:10]}\n{r['constraint']}")

    dropout_adherence = np.array(dropout_adherence)
    x_pos = np.arange(len(labels))

    ax3.barh(x_pos, dropout_adherence[:, 0], label='Predicted Dropout', color='salmon', alpha=0.8)
    ax3.barh(x_pos, dropout_adherence[:, 1], left=dropout_adherence[:, 0],
             label='Remaining Capacity', color='lightgray', alpha=0.5)

    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Number of Students', fontsize=11, fontweight='bold')
    ax3.set_title('Global Dropout Constraint Adherence (Top 10)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    ax4 = axes[1, 1]
    enrolled_adherence = []
    for r in detailed_results[:10]:
        enrolled_adherence.append([r['enrolled_count'], r['global_enrolled_limit'] - r['enrolled_count']])

    enrolled_adherence = np.array(enrolled_adherence)

    ax4.barh(x_pos, enrolled_adherence[:, 0], label='Predicted Enrolled', color='skyblue', alpha=0.8)
    ax4.barh(x_pos, enrolled_adherence[:, 1], left=enrolled_adherence[:, 0],
             label='Remaining Capacity', color='lightgray', alpha=0.5)

    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('Number of Students', fontsize=11, fontweight='bold')
    ax4.set_title('Global Enrolled Constraint Adherence (Top 10)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'constraint_satisfaction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: constraint_satisfaction_analysis.png")
    plt.close()

def plot_per_course_constraint_satisfaction(df_summary, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

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

    ax3 = axes[1, 0]
    overpred_by_constraint = df_summary.groupby('Constraint')['Overprediction'].sum().sort_index()
    overpred_by_constraint.plot(kind='bar', ax=ax3, color='indianred')
    ax3.set_title('Total Overprediction by Constraint Setting', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Constraint (Global, Local)', fontsize=12)
    ax3.set_ylabel('Total Overprediction Count', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

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
    plt.savefig(output_dir / 'per_course_constraint_satisfaction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: per_course_constraint_satisfaction.png")
    plt.close()

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
    print(f"Saved: constraint_satisfaction_heatmap_by_course.png")
    plt.close()

def plot_class_distribution(detailed_results, output_dir):
    best_result = max(detailed_results, key=lambda x: x['accuracy'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    classes = ['Dropout', 'Enrolled', 'Graduate']
    counts = [best_result['dropout_count'], best_result['enrolled_count'], best_result['graduate_count']]
    colors_pie = ['#ff6b6b', '#4ecdc4', '#95e1d3']

    wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title(f'Class Distribution - Best Model\n{best_result["config"]} @ {best_result["constraint"]}\nAccuracy: {best_result["accuracy"]*100:.2f}%',
                  fontsize=12, fontweight='bold')

    ax2 = axes[0, 1]
    avg_dropout = np.mean([r['dropout_count'] for r in detailed_results])
    avg_enrolled = np.mean([r['enrolled_count'] for r in detailed_results])
    avg_graduate = np.mean([r['graduate_count'] for r in detailed_results])

    avg_counts = [avg_dropout, avg_enrolled, avg_graduate]
    wedges, texts, autotexts = ax2.pie(avg_counts, labels=classes, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Average Class Distribution\nAcross All Experiments',
                  fontsize=12, fontweight='bold')

    ax3 = axes[1, 0]
    constraint_class_data = defaultdict(lambda: {'Dropout': [], 'Enrolled': [], 'Graduate': []})
    for r in detailed_results:
        constraint_class_data[r['constraint']]['Dropout'].append(r['dropout_count'])
        constraint_class_data[r['constraint']]['Enrolled'].append(r['enrolled_count'])
        constraint_class_data[r['constraint']]['Graduate'].append(r['graduate_count'])

    constraints_sorted = sorted(constraint_class_data.keys())
    dropout_avgs = [np.mean(constraint_class_data[c]['Dropout']) for c in constraints_sorted]
    enrolled_avgs = [np.mean(constraint_class_data[c]['Enrolled']) for c in constraints_sorted]
    graduate_avgs = [np.mean(constraint_class_data[c]['Graduate']) for c in constraints_sorted]

    x = np.arange(len(constraints_sorted))
    width = 0.25

    ax3.bar(x - width, dropout_avgs, width, label='Dropout', color='#ff6b6b', alpha=0.8)
    ax3.bar(x, enrolled_avgs, width, label='Enrolled', color='#4ecdc4', alpha=0.8)
    ax3.bar(x + width, graduate_avgs, width, label='Graduate', color='#95e1d3', alpha=0.8)

    ax3.set_xlabel('Constraint (Global, Local)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Count', fontsize=11, fontweight='bold')
    ax3.set_title('Average Class Predictions by Constraint', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(constraints_sorted, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    ax4 = axes[1, 1]
    course_data = best_result['course_predictions']
    courses = sorted(course_data.keys())[:15]

    dropout_per_course = [course_data[c]['Dropout'] for c in courses]
    enrolled_per_course = [course_data[c]['Enrolled'] for c in courses]
    graduate_per_course = [course_data[c]['Graduate'] for c in courses]

    x_course = np.arange(len(courses))
    width = 0.25

    ax4.bar(x_course - width, dropout_per_course, width, label='Dropout', color='#ff6b6b', alpha=0.8)
    ax4.bar(x_course, enrolled_per_course, width, label='Enrolled', color='#4ecdc4', alpha=0.8)
    ax4.bar(x_course + width, graduate_per_course, width, label='Graduate', color='#95e1d3', alpha=0.8)

    ax4.set_xlabel('Course ID', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Prediction Count', fontsize=11, fontweight='bold')
    ax4.set_title(f'Per-Course Predictions - Best Model (First 15 Courses)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_course)
    ax4.set_xticklabels(courses)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: class_distribution_analysis.png")
    plt.close()

def plot_heatmap_performance(results, output_dir):
    configs = sorted(set(r['config'] for r in results))
    constraints = sorted(set(r['constraint'] for r in results))

    accuracy_matrix = np.zeros((len(configs), len(constraints)))
    improvement_matrix = np.zeros((len(configs), len(constraints)))

    for i, config in enumerate(configs):
        for j, constraint in enumerate(constraints):
            matching = [r for r in results if r['config'] == config and r['constraint'] == constraint]
            if matching:
                accuracy_matrix[i, j] = matching[0]['accuracy'] * 100
                improvement_matrix[i, j] = matching[0]['improvement'] * 100
            else:
                accuracy_matrix[i, j] = np.nan
                improvement_matrix[i, j] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax1 = axes[0]
    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=constraints, yticklabels=configs,
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax1, vmin=50, vmax=80)
    ax1.set_title('Accuracy Heatmap: Models x Constraints', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Constraint (Global, Local)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Configuration', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    sns.heatmap(improvement_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=constraints, yticklabels=configs,
                cbar_kws={'label': 'Improvement (%)'}, ax=ax2, vmin=-5, vmax=6)
    ax2.set_title('Improvement over Benchmark: Models x Constraints', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Constraint (Global, Local)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Configuration', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: performance_heatmaps.png")
    plt.close()

def generate_summary_report(results, detailed_results, df_course_summary, output_dir):
    report = []
    report.append("# Comprehensive Multi-Constraint Experiment Analysis\n\n")
    report.append(f"**Total Experiments Analyzed**: {len(results)}\n")
    report.append(f"**Configurations Tested**: {len(set(r['config'] for r in results))}\n")
    report.append(f"**Constraint Settings**: {len(set(r['constraint'] for r in results))}\n")
    report.append(f"**Failed/Excluded**: {len(FAILED_EXPERIMENTS)}\n\n")

    report.append("## Top 5 Performers\n\n")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:5]
    report.append("| Rank | Configuration | Constraint | Accuracy | Benchmark | Improvement |\n")
    report.append("|------|--------------|------------|----------|-----------|-------------|\n")
    for i, r in enumerate(sorted_results, 1):
        report.append(f"| {i} | {r['config']} | {r['constraint']} | {r['accuracy']*100:.2f}% | {r['benchmark_accuracy']*100:.2f}% | {r['improvement']*100:+.2f}% |\n")
    report.append("\n")

    report.append("## Best Performer by Constraint\n\n")
    report.append("| Constraint | Best Config | Accuracy | Improvement |\n")
    report.append("|-----------|-------------|----------|-------------|\n")
    constraints = sorted(set(r['constraint'] for r in results))
    for constraint in constraints:
        constraint_results = [r for r in results if r['constraint'] == constraint]
        best = max(constraint_results, key=lambda x: x['accuracy'])
        report.append(f"| {constraint} | {best['config']} | {best['accuracy']*100:.2f}% | {best['improvement']*100:+.2f}% |\n")
    report.append("\n")

    report.append("## Performance by Model Configuration\n\n")
    report.append("| Configuration | Avg Accuracy | Win Rate | Avg Improvement |\n")
    report.append("|--------------|--------------|----------|----------------|\n")
    configs = sorted(set(r['config'] for r in results))
    for config in configs:
        config_results = [r for r in results if r['config'] == config]
        avg_acc = np.mean([r['accuracy'] for r in config_results]) * 100
        wins = sum(1 for r in config_results if r['improvement'] > 0.001)
        win_rate = (wins / len(config_results)) * 100
        avg_imp = np.mean([r['improvement'] for r in config_results]) * 100
        report.append(f"| {config} | {avg_acc:.2f}% | {win_rate:.0f}% ({wins}/{len(config_results)}) | {avg_imp:+.2f}% |\n")
    report.append("\n")

    report.append("## Constraint Satisfaction Summary\n\n")
    avg_satisfaction = np.mean([r['constraint_satisfaction_rate'] for r in detailed_results]) * 100
    perfect_satisfaction = sum(1 for r in detailed_results if r['constraint_satisfaction_rate'] == 1.0)
    report.append(f"- **Average Constraint Satisfaction Rate**: {avg_satisfaction:.1f}%\n")
    report.append(f"- **Perfect Satisfaction (100%)**: {perfect_satisfaction}/{len(detailed_results)} experiments\n")
    report.append(f"- **Total Constraints Evaluated**: {sum(r['total_constraints'] for r in detailed_results)}\n")
    report.append(f"- **Total Satisfied**: {sum(r['satisfied_constraints'] for r in detailed_results)}\n\n")

    if df_course_summary is not None and len(df_course_summary) > 0:
        report.append("## Per-Course Constraint Analysis\n\n")
        total_violations = int(df_course_summary['Violated'].sum())
        total_overpredictions = int(df_course_summary['Overprediction'].sum())
        report.append(f"- **Total Violations**: {total_violations}\n")
        report.append(f"- **Total Overpredictions**: {total_overpredictions}\n\n")

    report.append("## Class Distribution Summary\n\n")
    best = max(detailed_results, key=lambda x: x['accuracy'])
    report.append(f"**Best Model** ({best['config']} @ {best['constraint']}):\n")
    report.append(f"- Dropout Predictions: {best['dropout_count']} (limit: {best['global_dropout_limit']})\n")
    report.append(f"- Enrolled Predictions: {best['enrolled_count']} (limit: {best['global_enrolled_limit']})\n")
    report.append(f"- Graduate Predictions: {best['graduate_count']} (unlimited)\n")
    report.append(f"- Dropout Within Limit: {'Yes' if best['dropout_within_limit'] else 'No'}\n")
    report.append(f"- Enrolled Within Limit: {'Yes' if best['enrolled_within_limit'] else 'No'}\n\n")

    report.append("## Key Insights\n\n")
    perfect = [c for c in configs if all(r['improvement'] > 0.001 for r in results if r['config'] == c)]
    if perfect:
        perfect_count = len([r for r in results if r['config'] == perfect[0]])
        report.append(f"1. **Perfect Performer**: {perfect[0]} beat the benchmark in ALL {perfect_count} tested constraints\n")

    best_constraint = max(constraints, key=lambda c: np.mean([r['accuracy'] for r in results if r['constraint'] == c]))
    avg_best = np.mean([r['accuracy'] for r in results if r['constraint'] == best_constraint]) * 100
    report.append(f"2. **Best Constraint Setting**: {best_constraint} with {avg_best:.2f}% average accuracy\n")

    consistency = [(c, np.std([r['accuracy'] for r in results if r['config'] == c])) for c in configs]
    most_consistent = min(consistency, key=lambda x: x[1])
    report.append(f"3. **Most Consistent Config**: {most_consistent[0]} with lowest variance (sigma={most_consistent[1]:.4f})\n")
    report.append(f"4. **Constraint Satisfaction**: {avg_satisfaction:.1f}% average satisfaction rate demonstrates effective transductive learning\n")

    report_path = output_dir / 'COMPREHENSIVE_ANALYSIS_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(''.join(report))

    print(f"Saved: COMPREHENSIVE_ANALYSIS_REPORT.md")
    return report_path

def main():
    print("="*80)
    print("COMPREHENSIVE MULTI-CONSTRAINT EXPERIMENT ANALYSIS")
    print("="*80)

    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir}\n")

    print("1. Loading experiment results...")
    results = load_all_results()
    print(f"   Loaded {len(results)} experiments")

    print("\n2. Loading predictions and constraint data...")
    detailed_results = load_predictions_and_constraints(results)
    print(f"   Loaded detailed data for {len(detailed_results)} experiments")

    print("\n3. Analyzing per-course constraint satisfaction...")
    constraint_data = analyze_constraint_satisfaction_by_course()
    df_course_summary = create_course_summary(constraint_data)
    print(f"   Analyzed {len(df_course_summary)} per-course records")

    df_course_summary.to_csv(output_dir / 'per_course_constraint_data.csv', index=False)

    print("\n4. Generating visualizations...")
    plot_top_5_performers(results, output_dir)
    plot_performance_by_constraint(results, output_dir)
    plot_performance_by_model(results, output_dir)
    plot_constraint_satisfaction(detailed_results, df_course_summary, output_dir)
    plot_per_course_constraint_satisfaction(df_course_summary, output_dir)
    plot_class_distribution(detailed_results, output_dir)
    plot_heatmap_performance(results, output_dir)

    print("\n5. Generating summary report...")
    generate_summary_report(results, detailed_results, df_course_summary, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated Files:")
    print("   - top_5_performers.png")
    print("   - performance_by_constraint.png")
    print("   - performance_by_model.png")
    print("   - constraint_satisfaction_analysis.png")
    print("   - per_course_constraint_satisfaction.png")
    print("   - constraint_satisfaction_heatmap_by_course.png")
    print("   - class_distribution_analysis.png")
    print("   - performance_heatmaps.png")
    print("   - per_course_constraint_data.csv")
    print("   - COMPREHENSIVE_ANALYSIS_REPORT.md")
    print()

if __name__ == "__main__":
    main()

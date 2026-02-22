"""Cross-experiment comparison charts: accuracy, F1, calibration, uncertainty."""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.training.metrics import compute_metrics

log = logging.getLogger(__name__)

COLORS = {
    'our_approach': '#2196F3',
    'heuristic': '#F44336',
}
METHOD_LABELS = {
    'our_approach': 'Our Approach (KL-Reg)',
    'heuristic': 'Heuristic Baseline',
}


def load_predictions(experiment_path):
    """Load final_predictions.csv and return (y_true, y_pred, y_proba, group_ids)."""
    df = pd.read_csv(Path(experiment_path) / 'final_predictions.csv')
    y_true = df['True_Label'].values
    y_pred = df['Predicted_Label'].values
    prob_cols = [c for c in df.columns if c.startswith('Prob_Class_')]
    y_proba = df[prob_cols].values
    group_ids = df['Group_ID'].values if 'Group_ID' in df.columns else None
    return y_true, y_pred, y_proba, group_ids


def collect_all_metrics(results_dir='results'):
    """Compute full metrics for all experiments."""
    results_path = Path(results_dir)
    records = []
    for method in ['our_approach', 'heuristic']:
        method_dir = results_path / 'binary' / method / 'FTTransformer'
        if not method_dir.exists():
            continue
        for exp_path in sorted(method_dir.glob('constraint_*/standard/default')):
            constraint_name = exp_path.parts[-3]
            parts = constraint_name.replace('constraint_', '').split('_')
            local_pct, global_pct = float(parts[0]), float(parts[1])
            y_true, y_pred, y_proba, _ = load_predictions(exp_path)
            metrics = compute_metrics(y_true, y_pred, y_proba)
            records.append({
                'method': method, 'constraint': constraint_name,
                'local_pct': local_pct, 'global_pct': global_pct,
                'label': f'L={parts[0]} G={parts[1]}',
                'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'],
                'precision_macro': metrics['precision_macro'], 'recall_macro': metrics['recall_macro'],
                'ece': metrics.get('ece'), 'brier_score': metrics.get('brier_score'),
                'mean_entropy': metrics.get('mean_entropy'), 'mean_confidence': metrics.get('mean_confidence'),
                'confidence_correct': metrics.get('confidence_correct'),
                'confidence_incorrect': metrics.get('confidence_incorrect'),
                'confidence_gap': metrics.get('confidence_gap'),
                'pct_high_confidence': metrics.get('pct_high_confidence'),
                'pct_low_confidence': metrics.get('pct_low_confidence'),
                'path': str(exp_path),
            })
    return pd.DataFrame(records)


def _get_paired_data(df):
    """Pair our_approach and heuristic by constraint, sorted by global_pct."""
    constraints = sorted(df['constraint'].unique(),
                         key=lambda c: (float(c.split('_')[2]), float(c.split('_')[1])))
    labels, ours_data, heur_data = [], {}, {}
    for c in constraints:
        c_df = df[df['constraint'] == c]
        labels.append(c.replace('constraint_', 'L=').replace('_', ' G='))
        ours = c_df[c_df['method'] == 'our_approach']
        heur = c_df[c_df['method'] == 'heuristic']
        for col in df.columns:
            if col not in ['method', 'constraint', 'label', 'path', 'local_pct', 'global_pct']:
                ours_data.setdefault(col, []).append(ours[col].values[0] if len(ours) > 0 else None)
                heur_data.setdefault(col, []).append(heur[col].values[0] if len(heur) > 0 else None)
    return labels, ours_data, heur_data


def plot_accuracy_f1_comparison(df, save_path):
    """Grouped bar chart: accuracy and F1 for both methods."""
    labels, ours, heur = _get_paired_data(df)
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Accuracy & F1 Comparison', fontsize=14, fontweight='bold')

    for ax, metric, title in [(axes[0], 'accuracy', 'Accuracy'), (axes[1], 'f1_macro', 'F1 (Macro)')]:
        bars_h = ax.bar(x - width/2, heur[metric], width, label=METHOD_LABELS['heuristic'],
                        color=COLORS['heuristic'], alpha=0.85)
        bars_o = ax.bar(x + width/2, ours[metric], width, label=METHOD_LABELS['our_approach'],
                        color=COLORS['our_approach'], alpha=0.85)
        for bar in list(bars_h) + list(bars_o):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_delta_chart(df, save_path):
    """Improvement delta of our_approach over heuristic."""
    labels, ours, heur = _get_paired_data(df)
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Our Approach Improvement Over Heuristic', fontsize=14, fontweight='bold')

    for i, (metric, label) in enumerate([('accuracy', 'Accuracy'), ('f1_macro', 'F1 (Macro)')]):
        deltas = [o - h for o, h in zip(ours[metric], heur[metric])]
        offset = (i - 0.5) * width
        colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
        bars = ax.bar(x + offset, [d * 100 for d in deltas], width, label=label,
                      color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'+{d*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Improvement (pp)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    avg_delta = np.mean([o - h for o, h in zip(ours['accuracy'], heur['accuracy'])]) * 100
    ax.axhline(y=avg_delta, color='#2196F3', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(n - 0.5, avg_delta + 0.3, f'Avg: +{avg_delta:.1f}%', fontsize=9, color='#2196F3', ha='right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_calibration_comparison(df, save_path):
    """ECE, Brier score, confidence metrics comparison."""
    labels, ours, heur = _get_paired_data(df)
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Calibration & Uncertainty', fontsize=14, fontweight='bold')

    panels = [
        (axes[0, 0], 'ece', 'ECE', True),
        (axes[0, 1], 'brier_score', 'Brier Score', True),
        (axes[1, 0], 'mean_confidence', 'Mean Confidence', False),
        (axes[1, 1], 'confidence_gap', 'Confidence Gap', False),
    ]
    for ax, metric, title, lower_better in panels:
        bars_h = ax.bar(x - width/2, heur[metric], width, label=METHOD_LABELS['heuristic'],
                        color=COLORS['heuristic'], alpha=0.85)
        bars_o = ax.bar(x + width/2, ours[metric], width, label=METHOD_LABELS['our_approach'],
                        color=COLORS['our_approach'], alpha=0.85)
        for bar in list(bars_h) + list(bars_o):
            val = bar.get_height()
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_uncertainty_breakdown(df, save_path):
    """Entropy, confidence distribution analysis."""
    labels, ours, heur = _get_paired_data(df)
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Uncertainty Analysis', fontsize=14, fontweight='bold')

    panels = [
        (axes[0, 0], 'mean_entropy', 'Mean Normalized Entropy'),
        (axes[0, 1], 'pct_high_confidence', 'Pct High Confidence (>0.8)'),
        (axes[1, 0], 'pct_low_confidence', 'Pct Low Confidence (<0.6)'),
        (axes[1, 1], 'confidence_incorrect', 'Confidence on Incorrect'),
    ]
    for ax, metric, title in panels:
        ax.bar(x - width/2, heur[metric], width, label=METHOD_LABELS['heuristic'],
               color=COLORS['heuristic'], alpha=0.85)
        ax.bar(x + width/2, ours[metric], width, label=METHOD_LABELS['our_approach'],
               color=COLORS['our_approach'], alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_table(df, save_path):
    """Summary table: Accuracy, F1, Precision, ECE per constraint pair."""
    labels, ours, heur = _get_paired_data(df)
    metrics_display = [
        ('accuracy', 'Accuracy', '{:.4f}'), ('f1_macro', 'F1 (Macro)', '{:.4f}'),
        ('precision_macro', 'Precision', '{:.4f}'), ('ece', 'ECE', '{:.4f}'),
    ]
    higher_better = {'accuracy', 'f1_macro', 'precision_macro'}

    col_headers = ['Constraint']
    for _, name, _ in metrics_display:
        col_headers.extend([f'H {name}', f'O {name}', 'Delta'])

    table_data = []
    for i, label in enumerate(labels):
        row = [label]
        for key, _, fmt in metrics_display:
            h_val, o_val = heur[key][i], ours[key][i]
            delta = o_val - h_val if o_val is not None and h_val is not None else None
            row.append(fmt.format(h_val) if h_val is not None else '-')
            row.append(fmt.format(o_val) if o_val is not None else '-')
            row.append(f'{delta:+.4f}' if delta is not None else '-')
        table_data.append(row)

    # Average row
    avg_row = ['AVERAGE']
    for key, _, fmt in metrics_display:
        h_avg = np.mean([v for v in heur[key] if v is not None])
        o_avg = np.mean([v for v in ours[key] if v is not None])
        avg_row.extend([fmt.format(h_avg), fmt.format(o_avg), f'{o_avg - h_avg:+.4f}'])
    table_data.append(avg_row)

    n_cols = len(col_headers)
    fig, ax = plt.subplots(figsize=(n_cols * 1.1, 1.2 + len(table_data) * 0.55))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=col_headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(n_cols):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for j in range(n_cols):
        table[(len(table_data), j)].set_facecolor('#ECEFF1')
        table[(len(table_data), j)].set_text_props(fontweight='bold')

    for i in range(len(table_data)):
        col_idx = 1
        for key, _, _ in metrics_display:
            delta_col = col_idx + 2
            cell_text = table_data[i][delta_col]
            if cell_text != '-':
                val = float(cell_text)
                is_good = (val > 0 and key in higher_better) or (val < 0 and key not in higher_better)
                table[(i + 1, delta_col)].set_facecolor('#E8F5E9' if is_good else '#FFEBEE')
            col_idx += 3

    fig.suptitle('Results Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_comparison_charts(results_dir='results'):
    """Generate all comparison charts."""
    output_dir = Path(results_dir) / 'figures' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all_metrics(results_dir)
    if len(df) == 0:
        log.warning("No experiment results found")
        return df

    log.info("Found %d experiments (%d methods, %d constraints)",
             len(df), df['method'].nunique(), df['constraint'].nunique())

    df.to_csv(Path(results_dir) / 'all_metrics.csv', index=False)

    plot_accuracy_f1_comparison(df, output_dir / 'accuracy_f1_comparison.png')
    plot_delta_chart(df, output_dir / 'improvement_delta.png')
    plot_calibration_comparison(df, output_dir / 'calibration_comparison.png')
    plot_uncertainty_breakdown(df, output_dir / 'uncertainty_analysis.png')
    plot_summary_table(df, output_dir / 'summary_table.png')

    log.info("Generated 5 comparison charts in %s", output_dir)
    return df


if __name__ == '__main__':
    generate_comparison_charts()

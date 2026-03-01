"""Cross-experiment comparison charts for DermaMNIST-C constraint optimization.

Generates publication-quality figures comparing all optimization configs
against heuristic baselines, with per-class breakdowns, parameter sensitivity
analysis, and constraint convergence analysis.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.training.metrics import compute_metrics

log = logging.getLogger(__name__)

# ── Styling ──────────────────────────────────────────────────────────────────

CLASS_NAMES = {0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC'}
OPT_COLOR = '#2196F3'
HEU_COLOR = '#F44336'
ACCENT = '#4CAF50'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_predictions(experiment_path):
    """Load final_predictions.csv → (y_true, y_pred, y_proba, group_ids)."""
    df = pd.read_csv(Path(experiment_path) / 'final_predictions.csv')
    y_true = df['True_Label'].values
    y_pred = df['Predicted_Label'].values
    prob_cols = [c for c in df.columns if c.startswith('Prob_Class_')]
    y_proba = df[prob_cols].values if prob_cols else None
    group_ids = df['Group_ID'].values if 'Group_ID' in df.columns else None
    return y_true, y_pred, y_proba, group_ids


def collect_all_experiments(results_dir='results'):
    """Scan results/ for completed experiments, compute full metrics.

    Uses rglob to find configs at any depth — works with the hierarchical
    structure: results/{methodology}/{model}/{constraint}/{experiment}/
    """
    results_path = Path(results_dir)
    records = []

    for config_path in sorted(results_path.rglob('config.json')):
        exp_dir = config_path.parent
        pred_path = exp_dir / 'final_predictions.csv'
        if not pred_path.exists():
            continue

        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get('status') != 'completed':
            continue

        method = cfg.get('methodology', 'unknown')
        hp = cfg.get('hyperparams', {})
        res = cfg.get('results', {})

        y_true, y_pred, y_proba, group_ids = load_predictions(exp_dir)
        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Per-class TP/FP
        constrained_class = cfg.get('dataset_config', {}).get('constrained_class', 4)
        pred_c = int((y_pred == constrained_class).sum())
        tp_c = int(((y_pred == constrained_class) & (y_true == constrained_class)).sum())
        fp_c = pred_c - tp_c

        records.append({
            'method': method,
            'model_name': cfg.get('model_name', 'unknown'),
            'constraint': str(cfg.get('constraint', [])),
            'name': cfg.get('exp_name', exp_dir.name),
            'path': str(exp_dir),
            # Hyperparams
            'warmup_epochs': hp.get('warmup_epochs', 50),
            'constraint_epochs': hp.get('constraint_epochs', 500),
            'lr_constraint': hp.get('lr_constraint', 5e-6),
            'lambda_global': hp.get('lambda_global', 0.01),
            'lambda_step': hp.get('lambda_step', 0.002),
            'initial_rho': hp.get('initial_rho', 1.0),
            'alpha_kl': hp.get('alpha_kl', 0.0),
            'pretrained': hp.get('pretrained', False),
            'class_weighted_ce': hp.get('class_weighted_ce', False),
            'kl_temperature': hp.get('kl_temperature', 1.0),
            # Overall metrics
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics.get('f1_weighted', 0),
            # Per-class metrics (constrained class)
            'mel_precision': metrics['precision_per_class'][constrained_class],
            'mel_recall': metrics['recall_per_class'][constrained_class],
            'mel_f1': metrics['f1_per_class'][constrained_class],
            'mel_tp': tp_c,
            'mel_fp': fp_c,
            'mel_pred': pred_c,
            # All per-class
            **{f'prec_c{c}': metrics['precision_per_class'][c] for c in range(7)},
            **{f'rec_c{c}': metrics['recall_per_class'][c] for c in range(7)},
            **{f'f1_c{c}': metrics['f1_per_class'][c] for c in range(7)},
            **{f'support_c{c}': int(metrics['support_per_class'][c]) for c in range(7)},
            # Calibration
            'ece': metrics.get('ece'),
            'brier_score': metrics.get('brier_score'),
            'mean_confidence': metrics.get('mean_confidence'),
            'confidence_gap': metrics.get('confidence_gap'),
            # Training
            'training_time': res.get('training_time', 0),
            'posthoc_adj': res.get('samples_adjusted', 0),
        })

    return pd.DataFrame(records)


# ── Figure 1: Main Comparison Bar Chart ──────────────────────────────────────

def plot_main_comparison(df, save_path):
    """Ranked bar chart: all configs sorted by mel TP, with heuristic baselines."""
    df_sorted = df.sort_values('mel_tp', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(df_sorted) * 0.4)))
    fig.suptitle('Experiment Comparison — Sorted by Melanoma True Positives',
                 fontsize=15, fontweight='bold', y=1.02)

    y_pos = np.arange(len(df_sorted))
    colors = [HEU_COLOR if r['method'] == 'heuristic' else OPT_COLOR
              for _, r in df_sorted.iterrows()]
    alphas = [0.7 if r['method'] == 'heuristic' else 0.9
              for _, r in df_sorted.iterrows()]

    labels = []
    for _, r in df_sorted.iterrows():
        prefix = '[H] ' if r['method'] == 'heuristic' else ''
        labels.append(f"{prefix}{r['name']}")

    # Panel 1: mel TP / FP
    ax = axes[0]
    bars_tp = ax.barh(y_pos, df_sorted['mel_tp'], color=colors, alpha=0.9, label='TP')
    bars_fp = ax.barh(y_pos, df_sorted['mel_fp'], left=df_sorted['mel_tp'].values,
                      color='#FFCDD2', alpha=0.7, label='FP')
    for i, (tp, fp) in enumerate(zip(df_sorted['mel_tp'], df_sorted['mel_fp'])):
        ax.text(tp + fp + 0.5, i, f'{tp}TP / {fp}FP', va='center', fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predictions (Class 4 - Melanoma)')
    ax.set_title('Melanoma: TP vs FP', fontsize=12, fontweight='bold')
    ax.axvline(x=67, color=ACCENT, linestyle='--', linewidth=2, label='Constraint limit (67)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)

    # Panel 2: mel Precision
    ax = axes[1]
    ax.barh(y_pos, df_sorted['mel_precision'], color=colors, alpha=0.9)
    for i, v in enumerate(df_sorted['mel_precision']):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['' for _ in y_pos])
    ax.set_xlabel('Precision')
    ax.set_title('Melanoma Precision', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.85)
    ax.grid(True, axis='x', alpha=0.3)

    # Panel 3: Overall metrics
    ax = axes[2]
    x_inner = np.arange(3)
    bar_width = 0.8
    for i, (_, r) in enumerate(df_sorted.iterrows()):
        c = HEU_COLOR if r['method'] == 'heuristic' else OPT_COLOR
        vals = [r['accuracy'], r['f1_macro'], r['f1_weighted']]
        for j, v in enumerate(vals):
            ax.plot(v, i, 'o', color=c, markersize=6, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['' for _ in y_pos])
    ax.set_xlabel('Score')
    ax.set_title('Overall: Acc / F1-Macro / F1-Weighted', fontsize=12, fontweight='bold')
    ax.legend(['Accuracy', 'F1 Macro', 'F1 Weighted'], fontsize=8, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0.4, 0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 2: Per-Class F1 Heatmap ──────────────────────────────────────────

def plot_perclass_heatmap(df, save_path):
    """Heatmap of per-class F1 scores across all experiments."""
    df_sorted = df.sort_values('mel_tp', ascending=False)
    labels = []
    for _, r in df_sorted.iterrows():
        prefix = '[H] ' if r['method'] == 'heuristic' else ''
        labels.append(f"{prefix}{r['name']}")

    f1_matrix = np.array([[r[f'f1_c{c}'] for c in range(7)] for _, r in df_sorted.iterrows()])
    class_labels = [f'{CLASS_NAMES[c]} (n={df_sorted.iloc[0][f"support_c{c}"]})' for c in range(7)]

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.45)))
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=class_labels, yticklabels=labels,
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'F1 Score', 'shrink': 0.8})

    ax.set_title('Per-Class F1 Score Across All Experiments', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.tick_params(axis='y', labelsize=9)

    # Highlight constrained class column
    ax.add_patch(plt.Rectangle((4, 0), 1, len(df_sorted), fill=False,
                               edgecolor='red', linewidth=2.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 3: Parameter Sensitivity ─────────────────────────────────────────

def plot_parameter_sensitivity(df, save_path):
    """How each hyperparameter affects melanoma TP and overall accuracy."""
    opt = df[df['method'] == 'our_approach'].copy()
    if len(opt) < 3:
        return

    params = [
        ('initial_rho', 'Initial Rho'),
        ('alpha_kl', 'KL Alpha'),
        ('lambda_step', 'Lambda Step'),
        ('lambda_global', 'Lambda Global (initial)'),
    ]

    fig, axes = plt.subplots(2, len(params), figsize=(5 * len(params), 9))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=15, fontweight='bold', y=1.02)

    for j, (param, label) in enumerate(params):
        unique_vals = sorted(opt[param].unique())
        if len(unique_vals) < 2:
            axes[0, j].text(0.5, 0.5, 'Single value\n(no variation)',
                            ha='center', va='center', transform=axes[0, j].transAxes)
            axes[1, j].text(0.5, 0.5, 'Single value\n(no variation)',
                            ha='center', va='center', transform=axes[1, j].transAxes)
            axes[0, j].set_title(f'{label} → mel TP')
            axes[1, j].set_title(f'{label} → Accuracy')
            continue

        # mel TP
        ax = axes[0, j]
        ax.scatter(opt[param], opt['mel_tp'], c=OPT_COLOR, s=60, alpha=0.8, edgecolors='white')
        for _, r in opt.iterrows():
            ax.annotate(r['name'], (r[param], r['mel_tp']),
                        fontsize=6, alpha=0.6, rotation=30,
                        textcoords='offset points', xytext=(3, 3))
        ax.set_xlabel(label)
        ax.set_ylabel('mel TP')
        ax.set_title(f'{label} vs mel TP')
        ax.grid(True, alpha=0.3)
        # Best heuristic line
        best_heu_tp = df[df['method'] == 'heuristic']['mel_tp'].max()
        ax.axhline(y=best_heu_tp, color=HEU_COLOR, linestyle='--', alpha=0.5,
                   label=f'Best heuristic ({best_heu_tp})')
        ax.legend(fontsize=7)

        # Accuracy
        ax = axes[1, j]
        ax.scatter(opt[param], opt['accuracy'], c=OPT_COLOR, s=60, alpha=0.8, edgecolors='white')
        ax.set_xlabel(label)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{label} vs Accuracy')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 4: Confusion Matrices ────────────────────────────────────────────

def plot_confusion_matrices(df, save_path):
    """Side-by-side confusion matrices: best optimization vs best heuristic."""
    best_opt = df[df['method'] == 'our_approach'].sort_values('mel_tp', ascending=False).iloc[0]
    best_heu = df[df['method'] == 'heuristic'].sort_values('mel_tp', ascending=False).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Confusion Matrices — Best Optimization vs Best Heuristic',
                 fontsize=14, fontweight='bold')

    for ax, row, title_prefix, cmap in [
        (axes[0], best_opt, f'Optimization: {best_opt["name"]}', 'Blues'),
        (axes[1], best_heu, f'Heuristic: {best_heu["name"]}', 'Reds'),
    ]:
        y_true, y_pred, _, _ = load_predictions(row['path'])
        cm_matrix = confusion_matrix(y_true, y_pred, labels=range(7))

        # Normalize by row (recall-oriented)
        cm_norm = cm_matrix.astype(float) / cm_matrix.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        class_labels = [CLASS_NAMES[c] for c in range(7)]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=class_labels, yticklabels=class_labels,
                    vmin=0, vmax=1, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8})

        # Overlay raw counts in smaller text
        for i in range(7):
            for j in range(7):
                if cm_matrix[i, j] > 0:
                    ax.text(j + 0.5, i + 0.75, f'({cm_matrix[i, j]})',
                            ha='center', va='center', fontsize=7, color='gray')

        ax.set_title(f'{title_prefix}\n(acc={row["accuracy"]:.4f}, mel TP={row["mel_tp"]})',
                     fontsize=11)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 5: TP vs Precision Tradeoff ───────────────────────────────────────

def plot_tp_precision_tradeoff(df, save_path):
    """Scatter: mel TP vs mel Precision, bubble size = overall accuracy."""
    fig, ax = plt.subplots(figsize=(12, 8))

    opt = df[df['method'] == 'our_approach']
    heu = df[df['method'] == 'heuristic']

    # Size proportional to accuracy
    size_scale = 800

    # Optimization configs
    scatter_opt = ax.scatter(opt['mel_tp'], opt['mel_precision'],
                             s=opt['accuracy'] * size_scale, c=OPT_COLOR,
                             alpha=0.7, edgecolors='white', linewidth=1.5,
                             label='Optimization', zorder=5)
    for _, r in opt.iterrows():
        ax.annotate(r['name'], (r['mel_tp'], r['mel_precision']),
                    fontsize=7, alpha=0.7, textcoords='offset points', xytext=(5, 5))

    # Heuristic baselines
    scatter_heu = ax.scatter(heu['mel_tp'], heu['mel_precision'],
                             s=heu['accuracy'] * size_scale, c=HEU_COLOR,
                             alpha=0.7, edgecolors='white', linewidth=1.5,
                             marker='s', label='Heuristic', zorder=5)
    for _, r in heu.iterrows():
        ax.annotate(f'[H] {r["name"]}', (r['mel_tp'], r['mel_precision']),
                    fontsize=7, alpha=0.7, textcoords='offset points', xytext=(5, -10))

    ax.set_xlabel('Melanoma True Positives (TP)', fontsize=12)
    ax.set_ylabel('Melanoma Precision', fontsize=12)
    ax.set_title('TP vs Precision Tradeoff\n(bubble size = overall accuracy)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Quadrant lines at median
    ax.axhline(y=opt['mel_precision'].median(), color='gray', linestyle=':', alpha=0.4)
    ax.axvline(x=opt['mel_tp'].median(), color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 6: Training Time vs Quality ───────────────────────────────────────

def plot_time_vs_quality(df, save_path):
    """Training time vs mel TP — are longer runs worth it?"""
    opt = df[df['method'] == 'our_approach'].copy()
    if len(opt) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Training Time vs Quality', fontsize=14, fontweight='bold')

    # Time vs mel TP
    ax = axes[0]
    ax.scatter(opt['training_time'] / 60, opt['mel_tp'], c=OPT_COLOR, s=80, alpha=0.8,
               edgecolors='white')
    for _, r in opt.iterrows():
        ax.annotate(r['name'], (r['training_time'] / 60, r['mel_tp']),
                    fontsize=7, alpha=0.6, textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Melanoma TP')
    ax.set_title('Time vs mel TP')
    ax.grid(True, alpha=0.3)

    # Time vs accuracy
    ax = axes[1]
    ax.scatter(opt['training_time'] / 60, opt['accuracy'], c=OPT_COLOR, s=80, alpha=0.8,
               edgecolors='white')
    for _, r in opt.iterrows():
        ax.annotate(r['name'], (r['training_time'] / 60, r['accuracy']),
                    fontsize=7, alpha=0.6, textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Time vs Accuracy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 7: Summary Table ─────────────────────────────────────────────────

def plot_summary_table(df, save_path):
    """Publication-quality summary table as a figure."""
    df_sorted = df.sort_values(['method', 'mel_tp'], ascending=[True, False])

    cols = ['Name', 'Acc', 'F1-Mac', 'F1-Wt', 'mel-P', 'mel-R', 'mel-F1',
            'TP', 'FP', 'Adj', 'Time', 'rho', 'kl']
    table_data = []
    for _, r in df_sorted.iterrows():
        prefix = '[H] ' if r['method'] == 'heuristic' else ''
        table_data.append([
            f"{prefix}{r['name']}",
            f"{r['accuracy']:.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['f1_weighted']:.4f}",
            f"{r['mel_precision']:.3f}",
            f"{r['mel_recall']:.3f}",
            f"{r['mel_f1']:.3f}",
            str(r['mel_tp']),
            str(r['mel_fp']),
            str(r['posthoc_adj']),
            f"{r['training_time']/60:.1f}m" if r['training_time'] > 0 else '-',
            f"{r['initial_rho']:.1f}",
            f"{r['alpha_kl']:.1f}" if r['alpha_kl'] > 0 else '-',
        ])

    n_rows = len(table_data)
    fig_height = max(5, 0.4 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis('off')

    table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(cols)):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=9)

    # Row styling
    for i in range(1, n_rows + 1):
        is_heu = table_data[i - 1][0].startswith('[H]')
        for j in range(len(cols)):
            if is_heu:
                table[(i, j)].set_facecolor('#FFEBEE')
            elif i <= 3:  # top 3 optimization
                table[(i, j)].set_facecolor('#E3F2FD')

    fig.suptitle('Full Results Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Figure 8: Optimization vs Heuristic Delta ───────────────────────────────

def plot_opt_vs_heuristic_delta(df, save_path):
    """Bar chart: improvement of each optimization config over best heuristic."""
    heu_df = df[df['method'] == 'heuristic']
    if len(heu_df) == 0:
        log.warning("No heuristic baselines found, skipping opt_vs_heuristic_delta")
        return
    best_heu = heu_df.sort_values('mel_tp', ascending=False).iloc[0]
    opt = df[df['method'] == 'our_approach'].sort_values('mel_tp', ascending=False).copy()
    if len(opt) == 0:
        return

    n_configs = len(opt)
    fig_height = max(6, n_configs * 0.5 + 2)
    fig, axes = plt.subplots(1, 3, figsize=(22, fig_height),
                             gridspec_kw={'wspace': 0.4})
    fig.suptitle(f'Improvement Over Best Heuristic ({best_heu["name"]})',
                 fontsize=14, fontweight='bold', y=1.02)

    y_pos = np.arange(n_configs)
    names = opt['name'].values

    metrics = [
        ('mel_tp', 'mel TP Delta'),
        ('mel_precision', 'mel Precision Delta'),
        ('accuracy', 'Accuracy Delta'),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        deltas = opt[metric].values - best_heu[metric]
        colors = [ACCENT if d > 0 else HEU_COLOR for d in deltas]
        ax.barh(y_pos, deltas, color=colors, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels with safe offsets
        max_abs = max(abs(d) for d in deltas) if len(deltas) > 0 else 1
        offset = max_abs * 0.05

        for i, d in enumerate(deltas):
            if metric == 'mel_tp':
                fmt = f'+{d:.0f}' if d >= 0 else f'{d:.0f}'
            else:
                fmt = f'+{d:.3f}' if d >= 0 else f'{d:.3f}'
            ax.text(d + (offset if d >= 0 else -offset), i, fmt,
                    va='center', fontsize=7.5,
                    ha='left' if d >= 0 else 'right')

        # Add some padding to x limits
        ax.set_xlim(min(deltas) - max_abs * 0.2, max(deltas) + max_abs * 0.3)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ── Entry Point ──────────────────────────────────────────────────────────────

def generate_comparison_charts(results_dir='results'):
    """Generate all comparison charts from completed experiments."""
    output_dir = Path(results_dir) / 'figures' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all_experiments(results_dir)
    if len(df) == 0:
        log.warning("No completed experiments found in %s", results_dir)
        return df

    n_opt = len(df[df['method'] == 'our_approach'])
    n_heu = len(df[df['method'] == 'heuristic'])
    log.info("Found %d experiments (%d optimization, %d heuristic)", len(df), n_opt, n_heu)

    # Save raw metrics CSV
    df.to_csv(Path(results_dir) / 'all_metrics.csv', index=False)
    log.info("Saved: %s/all_metrics.csv", results_dir)

    # Clean old comparison charts
    for old in output_dir.glob('*.png'):
        old.unlink()

    # Generate comparison figures
    plot_main_comparison(df, output_dir / '01_main_comparison.png')
    plot_perclass_heatmap(df, output_dir / '02_perclass_f1_heatmap.png')
    plot_confusion_matrices(df, output_dir / '03_confusion_matrices.png')
    plot_summary_table(df, output_dir / '04_summary_table.png')
    # NOTE: opt_vs_heuristic_delta is available but skipped for now (fixed layout)
    # plot_opt_vs_heuristic_delta(df, output_dir / '05_opt_vs_heuristic_delta.png')

    log.info("Generated 4 comparison charts in %s", output_dir)
    return df

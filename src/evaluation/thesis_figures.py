# Thesis-quality analysis figures for constraint optimization experiments.
# Generates publication-ready PDF figures comparing optimization vs heuristic,
# hyperparameter sensitivity, model comparisons, and statistical significance.
# Usage: python -c "from src.evaluation.thesis_figures import generate_all; generate_all()"

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from src.training.metrics import compute_metrics
from src.evaluation.experiment_comparison import load_predictions, collect_all_experiments
from src.evaluation.training_curves import load_training_log

log = logging.getLogger(__name__)

OPT_COLOR = '#2196F3'
HEU_COLOR = '#F44336'
ACCENT = '#4CAF50'

CLASS_NAMES = {0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC'}

CONSTRAINT_MAP = {
    'c04_02': (0.4, 0.2), 'c05_03': (0.5, 0.3), 'c06_05': (0.6, 0.5),
    'c07_05': (0.7, 0.5), 'c08_02': (0.8, 0.2), 'c08_07': (0.8, 0.7),
    'c09_05': (0.9, 0.5), 'c09_08': (0.9, 0.8),
}

CONSTRAINTS_SORTED = ['c04_02', 'c05_03', 'c06_05', 'c07_05', 'c08_02', 'c08_07', 'c09_05', 'c09_08']

MODEL_COLORS = {'ResNet18': '#2196F3', 'MobileNetV3': '#FF9800', 'ResNet50': '#9C27B0'}


def _setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _constraint_label(cname):
    g, l = CONSTRAINT_MAP.get(cname, (0, 0))
    return f'({g}, {l})'


def _get_constraint_folder(constraint_list):
    if isinstance(constraint_list, str):
        constraint_list = json.loads(constraint_list.replace("'", '"'))
    g, l = constraint_list[0], constraint_list[1]
    for name, (gv, lv) in CONSTRAINT_MAP.items():
        if abs(g - gv) < 0.01 and abs(l - lv) < 0.01:
            return name
    return None


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", path)


def collect_dermmnist_experiments(base_dir):
    dermmnist_dir = Path(base_dir) / 'dermmnist'
    if not dermmnist_dir.exists():
        log.warning("Directory not found: %s", dermmnist_dir)
        return pd.DataFrame()

    df = collect_all_experiments(str(dermmnist_dir))
    if len(df) == 0:
        return df

    df['constraint_name'] = df['constraint'].apply(_get_constraint_folder)
    df['variation'] = df['path'].apply(lambda p: Path(p).name)

    log.info("Collected %d completed experiments from %s", len(df), dermmnist_dir)
    return df


def fig1_opt_vs_heuristic(df, output_dir):
    mask = (df['model_name'] == 'ResNet18') & (df['constraint_name'] == 'c05_03')
    sub = df[mask].copy()

    opt = sub[sub['method'] == 'our_approach']
    heu = sub[sub['method'] == 'heuristic']

    if len(heu) == 0:
        log.warning("Fig1: No completed heuristic experiments found, skipping")
        return
    if len(opt) == 0:
        log.warning("Fig1: No completed optimization experiments found, skipping")
        return

    opt_baseline = opt[opt['variation'] == 'baseline']
    if len(opt_baseline) == 0:
        opt_baseline = opt.iloc[:1]
    else:
        opt_baseline = opt_baseline.iloc[:1]

    configs = pd.concat([opt_baseline, heu], ignore_index=True)
    configs = configs.sort_values('accuracy', ascending=False)

    sample_path = configs.iloc[0]['path']
    tlog = load_training_log(sample_path)
    if tlog is not None and 'Limit_Class4' in tlog.columns:
        limit_val = float(tlog['Limit_Class4'].dropna().iloc[-1])
    else:
        limit_val = None

    labels = []
    for _, r in configs.iterrows():
        prefix = 'Heuristic: ' if r['method'] == 'heuristic' else 'Optimization: '
        labels.append(prefix + r['variation'])

    n = len(configs)
    x = np.arange(3)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (_, r) in enumerate(configs.iterrows()):
        color = OPT_COLOR if r['method'] == 'our_approach' else HEU_COLOR
        alpha = 0.9 if r['method'] == 'our_approach' else 0.7
        vals = [r['accuracy'], r['mel_f1'], 1.0 if (limit_val and r['mel_pred'] <= limit_val) else 0.0]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=labels[i], color=color, alpha=alpha,
                       edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.3f}' if v < 1 else 'Yes',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'MEL F1', 'Constraint\nSatisfied'])
    ax.set_ylabel('Score')
    ax.set_title('Optimization vs Heuristic Baseline (ResNet18, c05\\_03)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.15)

    if limit_val:
        ax.annotate(f'MEL limit: {int(limit_val)}', xy=(0.02, 0.95),
                    xycoords='axes fraction', fontsize=9, style='italic',
                    color='gray')

    _savefig(fig, Path(output_dir) / 'fig1_opt_vs_heuristic.pdf')


def fig2_lr_sensitivity(df, output_dir):
    mask = (df['model_name'] == 'ResNet18') & (df['method'] == 'our_approach')
    sub = df[mask].copy()

    lr_variations = ['baseline', 'lr5e-05', 'lr0.0002', 'lr0.0005']
    lr_labels = {
        'baseline': '1e-4 (default)',
        'lr5e-05': '5e-5',
        'lr0.0002': '2e-4',
        'lr0.0005': '5e-4',
    }
    lr_colors = {
        'baseline': '#2196F3',
        'lr5e-05': '#4CAF50',
        'lr0.0002': '#FF9800',
        'lr0.0005': '#F44336',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for var in lr_variations:
        var_data = sub[sub['variation'] == var].copy()
        if len(var_data) == 0:
            continue
        var_data = var_data.sort_values('constraint_name',
                                         key=lambda s: s.map({c: i for i, c in enumerate(CONSTRAINTS_SORTED)}))

        x_labels = [_constraint_label(c) for c in var_data['constraint_name']]
        x = np.arange(len(x_labels))

        label = lr_labels.get(var, var)
        color = lr_colors.get(var, '#999999')

        ax1.plot(x, var_data['accuracy'].values, marker='o', label=f'lr={label}',
                 color=color, linewidth=1.8, markersize=5)
        ax2.plot(x, var_data['mel_pred'].values, marker='s', label=f'lr={label}',
                 color=color, linewidth=1.8, markersize=5)

    ax1.set_xticks(np.arange(len(CONSTRAINTS_SORTED)))
    ax1.set_xticklabels([_constraint_label(c) for c in CONSTRAINTS_SORTED], rotation=45, ha='right')
    ax1.set_xlabel('Constraint (global, local)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Accuracy vs Constraint Tightness')
    ax1.legend()

    ax2.set_xticks(np.arange(len(CONSTRAINTS_SORTED)))
    ax2.set_xticklabels([_constraint_label(c) for c in CONSTRAINTS_SORTED], rotation=45, ha='right')
    ax2.set_xlabel('Constraint (global, local)')
    ax2.set_ylabel('MEL Prediction Count')
    ax2.set_title('(b) MEL Predictions vs Constraint Limit')

    limits = []
    for cname in CONSTRAINTS_SORTED:
        base_row = sub[(sub['constraint_name'] == cname) & (sub['variation'] == 'baseline')]
        if len(base_row) > 0:
            tlog = load_training_log(base_row.iloc[0]['path'])
            if tlog is not None and 'Limit_Class4' in tlog.columns:
                limits.append(float(tlog['Limit_Class4'].dropna().iloc[-1]))
            else:
                limits.append(None)
        else:
            limits.append(None)

    valid_limits = [(i, l) for i, l in enumerate(limits) if l is not None]
    if valid_limits:
        lx, ly = zip(*valid_limits)
        ax2.plot(lx, ly, 'k--', linewidth=2, label='Constraint limit', alpha=0.7)
        ax2.legend()
    else:
        ax2.legend()

    fig.suptitle('Learning Rate Sensitivity (ResNet18)', fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig2_lr_sensitivity.pdf')


def fig3_hp_ablation_heatmap(df, output_dir):
    mask = (df['model_name'] == 'ResNet18') & (df['constraint_name'] == 'c05_03') & \
           (df['method'] == 'our_approach')
    sub = df[mask].copy()

    if len(sub) < 5:
        log.warning("Fig3: Too few experiments for ablation heatmap (%d), skipping", len(sub))
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    kl_vals = sorted(sub['alpha_kl'].unique())
    if len(kl_vals) > 1:
        kl_groups = sub.groupby('alpha_kl').agg(
            accuracy=('accuracy', 'mean'),
            mel_f1=('mel_f1', 'mean'),
            count=('accuracy', 'count')
        ).reset_index()
        bars = ax.bar(range(len(kl_groups)), kl_groups['accuracy'], color=OPT_COLOR, alpha=0.8)
        for i, (_, r) in enumerate(kl_groups.iterrows()):
            ax.text(i, r['accuracy'] + 0.002, f'F1={r["mel_f1"]:.3f}\nn={int(r["count"])}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(kl_groups)))
        ax.set_xticklabels([f'{v:.1f}' for v in kl_groups['alpha_kl']])
        ax.set_xlabel('KL Alpha')
    else:
        ax.text(0.5, 0.5, 'Single KL value', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('(a) KL Alpha Effect')

    ax = axes[0, 1]
    lam_vals = sorted(sub['lambda_global'].unique())
    if len(lam_vals) > 1:
        lam_groups = sub.groupby('lambda_global').agg(
            accuracy=('accuracy', 'mean'),
            mel_f1=('mel_f1', 'mean'),
            count=('accuracy', 'count')
        ).reset_index()
        bars = ax.bar(range(len(lam_groups)), lam_groups['accuracy'], color='#FF9800', alpha=0.8)
        for i, (_, r) in enumerate(lam_groups.iterrows()):
            ax.text(i, r['accuracy'] + 0.002, f'F1={r["mel_f1"]:.3f}\nn={int(r["count"])}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(lam_groups)))
        ax.set_xticklabels([f'{v:.3f}' for v in lam_groups['lambda_global']])
        ax.set_xlabel('Lambda Global')
    else:
        ax.text(0.5, 0.5, 'Single lambda value', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('(b) Lambda Scale Effect')

    ax = axes[1, 0]
    rho_vals = sorted(sub['initial_rho'].unique())
    if len(rho_vals) > 1:
        rho_groups = sub.groupby('initial_rho').agg(
            accuracy=('accuracy', 'mean'),
            mel_f1=('mel_f1', 'mean'),
            count=('accuracy', 'count')
        ).reset_index()
        bars = ax.bar(range(len(rho_groups)), rho_groups['accuracy'], color='#9C27B0', alpha=0.8)
        for i, (_, r) in enumerate(rho_groups.iterrows()):
            ax.text(i, r['accuracy'] + 0.002, f'F1={r["mel_f1"]:.3f}\nn={int(r["count"])}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(rho_groups)))
        ax.set_xticklabels([f'{v:.1f}' for v in rho_groups['initial_rho']])
        ax.set_xlabel('Initial Rho')
    else:
        ax.text(0.5, 0.5, 'Single rho value', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('(c) Rho Effect')

    ax = axes[1, 1]
    sub_sorted = sub.sort_values('accuracy', ascending=False).head(15)
    scatter = ax.scatter(sub_sorted['mel_f1'], sub_sorted['accuracy'],
                         c=sub_sorted['alpha_kl'], cmap='viridis',
                         s=80, edgecolors='white', linewidth=0.8, zorder=5)
    for _, r in sub_sorted.iterrows():
        short_name = r['variation']
        if len(short_name) > 20:
            short_name = short_name[:18] + '..'
        ax.annotate(short_name, (r['mel_f1'], r['accuracy']),
                    fontsize=6, alpha=0.7, textcoords='offset points', xytext=(4, 4))
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('KL Alpha')
    ax.set_xlabel('MEL F1')
    ax.set_ylabel('Accuracy')
    ax.set_title('(d) Top Configs (color = KL alpha)')

    fig.suptitle('Hyperparameter Ablation (ResNet18, c05\\_03)', fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig3_hp_ablation.pdf')


def fig4_pretrained_impact(df, output_dir):
    mask = (df['model_name'] == 'ResNet18') & (df['method'] == 'our_approach')
    sub = df[mask].copy()

    pairs = [
        ('kl0.5', 'kl0.5_pretrained', 'kl0.5'),
        ('kl0.5_rho5.0', 'kl0.5_rho5.0_pretrained', 'kl0.5 + rho5.0'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for pair_idx, (no_pt, with_pt, pair_label) in enumerate(pairs):
        y_positions = []
        acc_no, acc_yes = [], []
        f1_no, f1_yes = [], []
        constraint_labels = []

        for cname in CONSTRAINTS_SORTED:
            row_no = sub[(sub['constraint_name'] == cname) & (sub['variation'] == no_pt)]
            row_yes = sub[(sub['constraint_name'] == cname) & (sub['variation'] == with_pt)]
            if len(row_no) == 0 or len(row_yes) == 0:
                continue

            y_positions.append(len(constraint_labels))
            constraint_labels.append(_constraint_label(cname))
            acc_no.append(row_no.iloc[0]['accuracy'])
            acc_yes.append(row_yes.iloc[0]['accuracy'])
            f1_no.append(row_no.iloc[0]['mel_f1'])
            f1_yes.append(row_yes.iloc[0]['mel_f1'])

        if not y_positions:
            continue

        y = np.array(y_positions) + pair_idx * 0.3 - 0.15

        for i in range(len(y)):
            ax1.plot([acc_no[i], acc_yes[i]], [y[i], y[i]], 'k-', linewidth=1, alpha=0.4)
        ax1.scatter(acc_no, y, c=HEU_COLOR, s=50, zorder=5,
                    label=f'{pair_label} (no pretrain)' if pair_idx == 0 else None,
                    marker='o', edgecolors='white')
        ax1.scatter(acc_yes, y, c=ACCENT, s=50, zorder=5,
                    label=f'{pair_label} (pretrained)' if pair_idx == 0 else None,
                    marker='D', edgecolors='white')

        for i in range(len(y)):
            ax2.plot([f1_no[i], f1_yes[i]], [y[i], y[i]], 'k-', linewidth=1, alpha=0.4)
        ax2.scatter(f1_no, y, c=HEU_COLOR, s=50, zorder=5,
                    label=f'{pair_label} (no pretrain)' if pair_idx == 0 else None,
                    marker='o', edgecolors='white')
        ax2.scatter(f1_yes, y, c=ACCENT, s=50, zorder=5,
                    label=f'{pair_label} (pretrained)' if pair_idx == 0 else None,
                    marker='D', edgecolors='white')

    if constraint_labels:
        ax1.set_yticks(range(len(constraint_labels)))
        ax1.set_yticklabels(constraint_labels)
        ax2.set_yticks(range(len(constraint_labels)))
        ax2.set_yticklabels(constraint_labels)

    ax1.set_xlabel('Accuracy')
    ax1.set_title('(a) Accuracy')
    ax1.legend(fontsize=8, loc='lower right')

    ax2.set_xlabel('MEL F1')
    ax2.set_title('(b) MEL F1')
    ax2.legend(fontsize=8, loc='lower right')

    all_acc_diffs = []
    for pair_idx, (no_pt, with_pt, _) in enumerate(pairs):
        for cname in CONSTRAINTS_SORTED:
            row_no = sub[(sub['constraint_name'] == cname) & (sub['variation'] == no_pt)]
            row_yes = sub[(sub['constraint_name'] == cname) & (sub['variation'] == with_pt)]
            if len(row_no) > 0 and len(row_yes) > 0:
                all_acc_diffs.append(row_yes.iloc[0]['accuracy'] - row_no.iloc[0]['accuracy'])
    if all_acc_diffs:
        mean_diff = np.mean(all_acc_diffs) * 100
        ax1.annotate(f'Mean improvement: +{mean_diff:.1f}pp', xy=(0.02, 0.02),
                     xycoords='axes fraction', fontsize=9, style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Impact of ImageNet Pretraining (ResNet18)', fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig4_pretrained_impact.pdf')


def fig5_constraint_tightness(df, output_dir):
    mask = (df['model_name'] == 'ResNet18') & (df['method'] == 'our_approach')
    sub = df[mask].copy()

    target_var = 'kl0.5_rho5.0_pretrained'
    best = sub[sub['variation'] == target_var]
    if len(best) == 0:
        target_var = 'kl0.5_rho5.0'
        best = sub[sub['variation'] == target_var]
    if len(best) == 0:
        target_var = 'baseline'
        best = sub[sub['variation'] == target_var]
    if len(best) == 0:
        log.warning("Fig5: No suitable config found, skipping")
        return

    best = best.sort_values('constraint_name',
                             key=lambda s: s.map({c: i for i, c in enumerate(CONSTRAINTS_SORTED)}))

    x = np.arange(len(best))
    x_labels = [_constraint_label(c) for c in best['constraint_name']]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x, best['accuracy'].values, 'o-', color=OPT_COLOR,
                     linewidth=2, markersize=7, label='Accuracy')
    ax1.set_ylabel('Accuracy', color=OPT_COLOR)
    ax1.tick_params(axis='y', labelcolor=OPT_COLOR)

    line2 = ax2.plot(x, best['mel_f1'].values, 's--', color=HEU_COLOR,
                     linewidth=2, markersize=7, label='MEL F1')
    ax2.set_ylabel('MEL F1', color=HEU_COLOR)
    ax2.tick_params(axis='y', labelcolor=HEU_COLOR)
    ax2.spines['right'].set_visible(True)

    for i, (_, r) in enumerate(best.iterrows()):
        tlog = load_training_log(r['path'])
        if tlog is not None and 'Global_Satisfied' in tlog.columns:
            satisfied = int(tlog['Global_Satisfied'].iloc[-1]) == 1
        else:
            satisfied = None

        if satisfied is not None:
            marker = ACCENT if satisfied else HEU_COLOR
            symbol = 'v' if satisfied else 'x'
            ax1.scatter(i, best['accuracy'].values[i] + 0.01, marker=symbol,
                        c=marker, s=40, zorder=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_xlabel('Constraint (global, local)')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')

    ax1.scatter([], [], marker='v', c=ACCENT, s=40, label='Constraint satisfied')
    ax1.scatter([], [], marker='x', c=HEU_COLOR, s=40, label='Constraint violated')

    ax1.set_title(f'Constraint Tightness Sensitivity ({target_var})')
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig5_constraint_tightness.pdf')


def fig6_model_comparison(df, output_dir):
    mask = df['method'] == 'our_approach'
    sub = df[mask].copy()

    model_variations = {
        'ResNet18': 'baseline',
        'MobileNetV3': 'baseline',
        'ResNet50': 'kl0.5_chunk128',
    }

    models_present = []
    for model, var in model_variations.items():
        model_data = sub[(sub['model_name'] == model) & (sub['variation'] == var)]
        if len(model_data) > 0:
            models_present.append(model)

    if len(models_present) < 2:
        log.warning("Fig6: Fewer than 2 models available (%s), skipping", models_present)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_models = len(models_present)
    width = 0.8 / n_models

    constraints_with_data = []
    for cname in CONSTRAINTS_SORTED:
        has_any = False
        for model in models_present:
            var = model_variations[model]
            row = sub[(sub['model_name'] == model) & (sub['constraint_name'] == cname) &
                      (sub['variation'] == var)]
            if len(row) > 0:
                has_any = True
        if has_any:
            constraints_with_data.append(cname)

    x = np.arange(len(constraints_with_data))

    for m_idx, model in enumerate(models_present):
        var = model_variations[model]
        accs = []
        f1s = []
        for cname in constraints_with_data:
            row = sub[(sub['model_name'] == model) & (sub['constraint_name'] == cname) &
                      (sub['variation'] == var)]
            if len(row) > 0:
                accs.append(row.iloc[0]['accuracy'])
                f1s.append(row.iloc[0]['mel_f1'])
            else:
                accs.append(0)
                f1s.append(0)

        offset = (m_idx - n_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(model, '#999999')

        ax1.bar(x + offset, accs, width, label=model, color=color, alpha=0.85,
                edgecolor='white', linewidth=0.5)
        ax2.bar(x + offset, f1s, width, label=model, color=color, alpha=0.85,
                edgecolor='white', linewidth=0.5)

    for ax, metric_name in [(ax1, 'Accuracy'), (ax2, 'MEL F1')]:
        ax.set_xticks(x)
        ax.set_xticklabels([_constraint_label(c) for c in constraints_with_data],
                           rotation=45, ha='right')
        ax.set_xlabel('Constraint (global, local)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'({"a" if ax == ax1 else "b"}) {metric_name}')
        ax.legend()

    fig.suptitle('Model Architecture Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig6_model_comparison.pdf')


def fig7_training_dynamics(base_dir, output_dir):
    approach_dir = Path(base_dir) / 'dermmnist' / 'our_approach' / 'ResNet18' / 'c05_03'
    if not approach_dir.exists():
        log.warning("Fig7: Directory not found: %s", approach_dir)
        return

    targets = [
        ('baseline', 'Baseline (slow)'),
        ('kl0.5_rho5.0', 'kl0.5 + rho5.0 (fast)'),
        ('kl0.5_rho5.0_pretrained', 'kl0.5 + rho5.0 + pretrained'),
    ]

    logs = []
    labels = []
    for var_name, label in targets:
        var_dir = approach_dir / var_name
        if not var_dir.exists():
            continue
        tlog = load_training_log(str(var_dir))
        if tlog is not None and len(tlog) > 2:
            logs.append(tlog)
            labels.append(label)

    if len(logs) == 0:
        log.warning("Fig7: No training logs found, skipping")
        return

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, (tlog, label) in enumerate(zip(logs, labels)):
        epochs = tlog['Epoch']
        color = colors[i % len(colors)]

        ax1.plot(epochs, tlog['L_CE'], color=color, linewidth=1.5, alpha=0.8,
                 label=f'{label} (CE)', linestyle='-')

        mask = tlog['L_Global'] > 0
        if mask.any():
            ax1.plot(epochs[mask], tlog['L_Global'][mask], color=color,
                     linewidth=1.2, alpha=0.6, linestyle='--')

        if 'L_KL' in tlog.columns:
            kl = tlog['L_KL'].fillna(0)
            kl_mask = kl > 0
            if kl_mask.any():
                ax1.plot(epochs[kl_mask], kl[kl_mask], color=color,
                         linewidth=1, alpha=0.5, linestyle=':')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss Components')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_yscale('log')

    limit_val = None
    for i, (tlog, label) in enumerate(zip(logs, labels)):
        epochs = tlog['Epoch']
        color = colors[i % len(colors)]

        if 'Hard_Class4' in tlog.columns:
            mask = tlog['L_Global'] > 0
            if mask.any():
                ax2.plot(epochs[mask], tlog['Hard_Class4'][mask], color=color,
                         linewidth=2, label=label, marker='', linestyle='-')

        if limit_val is None and 'Limit_Class4' in tlog.columns:
            limit_val = float(tlog['Limit_Class4'].dropna().iloc[-1])

    if limit_val:
        ax2.axhline(y=limit_val, color='k', linewidth=2, linestyle='--',
                     label=f'Constraint limit ({int(limit_val)})', alpha=0.7)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MEL Prediction Count (hard)')
    ax2.set_title('(b) MEL Count Convergence')
    ax2.legend(fontsize=8)

    fig.suptitle('Training Dynamics (ResNet18, c05\\_03)', fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig7_training_dynamics.pdf')


def fig8_statistical_significance(base_dir, output_dir):
    sig_dir = Path(base_dir) / 'dermmnist' / 'statistical_significance' / 'ResNet18' / 'c05_03'
    if not sig_dir.exists():
        log.warning("Fig8: Directory not found: %s", sig_dir)
        return

    seed_dirs = sorted(sig_dir.glob('seed_*'))
    if len(seed_dirs) < 2:
        log.warning("Fig8: Need at least 2 seeds, found %d, skipping", len(seed_dirs))
        return

    records = []
    for sd in seed_dirs:
        pred_path = sd / 'final_predictions.csv'
        if not pred_path.exists():
            continue
        y_true, y_pred, y_proba, _ = load_predictions(str(sd))
        m = compute_metrics(y_true, y_pred, y_proba)
        records.append({
            'seed': sd.name,
            'Accuracy': m['accuracy'],
            'F1 Macro': m['f1_macro'],
            'MEL F1': m['f1_per_class'][4],
            'ECE': m.get('ece', 0),
            'Brier': m.get('brier_score', 0),
        })

    if len(records) < 2:
        log.warning("Fig8: Insufficient completed seeds, skipping")
        return

    seed_df = pd.DataFrame(records)
    metrics_cols = ['Accuracy', 'F1 Macro', 'MEL F1', 'ECE', 'Brier']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                     gridspec_kw={'width_ratios': [1.2, 1]})

    bp = ax1.boxplot([seed_df[m].values for m in metrics_cols],
                     labels=metrics_cols, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))

    box_colors = [OPT_COLOR, '#FF9800', HEU_COLOR, '#9C27B0', '#607D8B']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, m in enumerate(metrics_cols):
        jitter = np.random.normal(0, 0.04, len(seed_df))
        ax1.scatter(np.ones(len(seed_df)) * (i + 1) + jitter, seed_df[m].values,
                    c='black', s=20, zorder=5, alpha=0.7)

    ax1.set_ylabel('Score')
    ax1.set_title('(a) Metric Distribution Across Seeds')
    ax1.tick_params(axis='x', rotation=30)

    ax2.axis('off')
    table_data = []
    for m in metrics_cols:
        vals = seed_df[m].values
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci_95 = 1.96 * std / np.sqrt(len(vals))
        table_data.append([m, f'{mean:.4f}', f'{std:.4f}', f'{mean:.4f} +/- {ci_95:.4f}'])

    table = ax2.table(cellText=table_data,
                       colLabels=['Metric', 'Mean', 'Std', '95% CI'],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(4):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')

    ax2.set_title(f'(b) Summary ({len(seed_df)} seeds)', pad=20)

    fig.suptitle('Statistical Significance (ResNet18, c05\\_03, kl0.5 + rho5.0)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, Path(output_dir) / 'fig8_statistical_significance.pdf')


def generate_summary_csv(results_dir='results'):
    dermmnist_dir = Path(results_dir) / 'dermmnist'
    output_path = Path(results_dir) / 'dermmnist' / 'experiment_summary.csv'

    df = collect_dermmnist_experiments(results_dir)
    if len(df) > 0:
        df = df[~df['path'].str.contains('statistical_significance')].reset_index(drop=True)

    pending_records = []
    heuristic_dir = dermmnist_dir / 'heuristic'
    if heuristic_dir.exists():
        for config_path in sorted(heuristic_dir.rglob('config.json')):
            exp_dir = config_path.parent
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get('status') == 'completed':
                continue
            pending_records.append({
                'method': 'heuristic',
                'model_name': cfg.get('model_name', 'unknown'),
                'constraint': str(cfg.get('constraint', [])),
                'name': cfg.get('exp_name', exp_dir.name),
                'path': str(exp_dir),
                'variation': exp_dir.name,
                'constraint_name': _get_constraint_folder(cfg.get('constraint', [0, 0])),
                'status': 'pending',
            })

    sig_records = []
    sig_dir = dermmnist_dir / 'statistical_significance'
    if sig_dir.exists():
        for config_path in sorted(sig_dir.rglob('config.json')):
            exp_dir = config_path.parent
            pred_path = exp_dir / 'final_predictions.csv'
            if not pred_path.exists():
                continue
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get('status') != 'completed':
                continue

            hp = cfg.get('hyperparams', {})
            y_true, y_pred, y_proba, _ = load_predictions(str(exp_dir))
            metrics = compute_metrics(y_true, y_pred, y_proba)
            constrained_class = cfg.get('dataset_config', {}).get('constrained_class', 4)
            pred_c = int((y_pred == constrained_class).sum())
            tp_c = int(((y_pred == constrained_class) & (y_true == constrained_class)).sum())

            sig_records.append({
                'method': 'our_approach',
                'model_name': cfg.get('model_name', 'unknown'),
                'constraint': str(cfg.get('constraint', [])),
                'name': cfg.get('exp_name', exp_dir.name),
                'path': str(exp_dir),
                'variation': exp_dir.name,
                'constraint_name': _get_constraint_folder(cfg.get('constraint', [0, 0])),
                'status': 'completed',
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics.get('f1_weighted', 0),
                'mel_precision': metrics['precision_per_class'][constrained_class],
                'mel_recall': metrics['recall_per_class'][constrained_class],
                'mel_f1': metrics['f1_per_class'][constrained_class],
                'mel_pred': pred_c,
                'mel_tp': tp_c,
                'mel_fp': pred_c - tp_c,
                'ece': metrics.get('ece'),
                'brier_score': metrics.get('brier_score'),
                'alpha_kl': hp.get('alpha_kl', 0.0),
                'initial_rho': hp.get('initial_rho', 1.0),
                'lambda_global': hp.get('lambda_global', 0.01),
                'pretrained': hp.get('pretrained', False),
                'lr_constraint': hp.get('lr_constraint', 5e-6),
                'warmup_epochs': hp.get('warmup_epochs', 50),
                'training_time': cfg.get('results', {}).get('training_time', 0),
                'posthoc_adj': cfg.get('results', {}).get('samples_adjusted', 0),
                'source': 'stat_sig',
                'seed': hp.get('seed'),
            })

    if len(df) > 0:
        df['status'] = 'completed'
        df['source'] = 'main'
        df['seed'] = None

    frames = []
    if len(df) > 0:
        frames.append(df)
    if pending_records:
        frames.append(pd.DataFrame(pending_records))
    if sig_records:
        frames.append(pd.DataFrame(sig_records))

    if not frames:
        log.error("No experiments found at all. Aborting CSV generation.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    section_keys = []
    for _, r in all_df.iterrows():
        source = r.get('source', 'main')
        if source == 'stat_sig':
            section_keys.append((r['model_name'], r['constraint_name'], 'stat_sig'))
        else:
            section_keys.append((r['model_name'], r['constraint_name'], 'main'))
    all_df['_section_key'] = section_keys

    constraint_order = {c: i for i, c in enumerate(CONSTRAINTS_SORTED)}
    model_order = {'MobileNetV3': 0, 'ResNet18': 1, 'ResNet50': 2}

    unique_keys = sorted(set(section_keys), key=lambda k: (
        model_order.get(k[0], 99),
        constraint_order.get(k[1], 99),
        0 if k[2] == 'main' else 1,
    ))
    key_to_id = {k: i + 1 for i, k in enumerate(unique_keys)}
    all_df['section_id'] = all_df['_section_key'].map(key_to_id)

    def _sort_priority(row):
        method_rank = 0 if row.get('method') == 'heuristic' else 1
        status_rank = 1 if row.get('status') == 'pending' else 0
        acc = -(row.get('accuracy') or 0)
        return (row['section_id'], method_rank, status_rank, acc)

    all_df['_sort'] = all_df.apply(_sort_priority, axis=1)
    all_df = all_df.sort_values('_sort').reset_index(drop=True)

    output_cols = [
        'section_id', 'model_name', 'constraint_name', 'method', 'variation',
        'status', 'source', 'seed',
        'accuracy', 'f1_macro', 'f1_weighted',
        'mel_precision', 'mel_recall', 'mel_f1', 'mel_pred', 'mel_tp', 'mel_fp',
        'ece', 'brier_score',
        'alpha_kl', 'initial_rho', 'lambda_global', 'pretrained',
        'lr_constraint', 'warmup_epochs',
        'training_time', 'posthoc_adj',
    ]
    output_cols = [c for c in output_cols if c in all_df.columns]
    out = all_df[output_cols].copy()

    float_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'mel_precision', 'mel_recall',
                  'mel_f1', 'ece', 'brier_score', 'training_time']
    for c in float_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: round(v, 4) if pd.notna(v) else v)

    out.to_csv(output_path, index=False)

    n_sections = all_df['section_id'].nunique()
    n_completed = len(all_df[all_df['status'] == 'completed'])
    n_pending = len(all_df[all_df['status'] == 'pending'])
    n_heu = len(all_df[all_df['method'] == 'heuristic'])
    n_sig = len(all_df[all_df.get('source', '') == 'stat_sig']) if 'source' in all_df.columns else 0
    log.info("Saved summary CSV: %s", output_path)
    log.info("  %d sections, %d rows (%d completed, %d pending)", n_sections, len(out), n_completed, n_pending)
    log.info("  %d heuristic rows, %d stat_sig rows", n_heu, n_sig)
    return out


def generate_all(results_dir='results'):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    _setup_style()

    output_dir = Path(results_dir) / 'dermmnist' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob('*.pdf'):
        old.unlink()

    log.info("Collecting experiments from %s/dermmnist/", results_dir)
    df = collect_dermmnist_experiments(results_dir)

    if len(df) == 0:
        log.error("No completed experiments found. Aborting.")
        return

    n_opt = len(df[df['method'] == 'our_approach'])
    n_heu = len(df[df['method'] == 'heuristic'])
    log.info("Found %d experiments (%d optimization, %d heuristic)", len(df), n_opt, n_heu)

    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

    log.info("--- Figure 1: Optimization vs Heuristic ---")
    fig1_opt_vs_heuristic(df, output_dir)

    log.info("--- Figure 2: Learning Rate Sensitivity ---")
    fig2_lr_sensitivity(df, output_dir)

    log.info("--- Figure 3: Hyperparameter Ablation ---")
    fig3_hp_ablation_heatmap(df, output_dir)

    log.info("--- Figure 4: Pretrained Impact ---")
    fig4_pretrained_impact(df, output_dir)

    log.info("--- Figure 5: Constraint Tightness ---")
    fig5_constraint_tightness(df, output_dir)

    log.info("--- Figure 6: Model Comparison ---")
    fig6_model_comparison(df, output_dir)

    log.info("--- Figure 7: Training Dynamics ---")
    fig7_training_dynamics(results_dir, output_dir)

    log.info("--- Figure 8: Statistical Significance ---")
    fig8_statistical_significance(results_dir, output_dir)

    log.info("--- Summary CSV ---")
    generate_summary_csv(results_dir)

    generated = list(output_dir.glob('*.pdf'))
    log.info("Generated %d figures in %s", len(generated), output_dir)
    for f in sorted(generated):
        log.info("  %s", f.name)


if __name__ == '__main__':
    generate_all()

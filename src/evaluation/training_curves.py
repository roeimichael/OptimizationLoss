# Training curve plots for constraint optimization experiments.
# Generates per-experiment training curves (loss, constraint convergence)
# and multi-experiment overlays comparing convergence across configs.

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)

CLASS_NAMES = {0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC'}


def load_training_log(experiment_path):
    log_path = Path(experiment_path) / 'training_log.csv'
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None

    header_candidates = []
    for i, line in enumerate(lines):
        if 'Epoch' in line and 'Train_Acc' in line:
            header_candidates.append((i, line.strip().split(',')))

    if not header_candidates:
        return None

    best_header_idx, best_header = max(header_candidates, key=lambda x: len(x[1]))

    data_rows = []
    for i, line in enumerate(lines):
        if i in [h[0] for h in header_candidates]:
            continue
        fields = line.strip().split(',')
        if len(fields) < 3:
            continue
        while len(fields) < len(best_header):
            fields.append('')
        data_rows.append(fields[:len(best_header)])

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows, columns=best_header)
    for col in df.columns:
        if col in ['Global_Satisfied', 'Local_Satisfied']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
        elif 'Limit' in col:
            df[col] = df[col].replace({'inf': float('inf'), '': float('inf')})
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _find_constrained_class(df):
    for col in df.columns:
        if col.startswith('Limit_Class'):
            vals = df[col].replace('inf', float('inf'))
            if (vals < UNLIMITED).any():
                return int(col.replace('Limit_Class', ''))
    return None


def plot_training_summary(df, experiment_name, save_path):
    constrained_class = _find_constrained_class(df)
    group_cols = [c for c in df.columns if c.startswith('Group') and '_Hard_' in c]
    has_local = len(group_cols) > 0

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Training Curves -- {experiment_name}', fontsize=15, fontweight='bold', y=1.01)
    epochs = df['Epoch']

    warmup_end = epochs.iloc[0]
    for _, row in df.iterrows():
        if row.get('L_Global', 0) > 0 or row.get('L_Local', 0) > 0:
            warmup_end = row['Epoch']
            break

    ax = axes[0]
    ax.plot(epochs, df['L_CE'], color='#2196F3', linewidth=2, label='CE Loss', zorder=3)
    ax.plot(epochs, df['L_Global'], color='#F44336', linewidth=2, label='Global Constraint Loss', zorder=3)
    ax.plot(epochs, df['L_Local'], color='#FF9800', linewidth=2, label='Local Constraint Loss', zorder=3)
    if 'L_KL' in df.columns:
        kl_vals = df['L_KL'].fillna(0)
        if kl_vals.sum() > 0:
            ax.plot(epochs, kl_vals, color='#9C27B0', linewidth=1.8, linestyle='--',
                    label='KL Divergence Loss', alpha=0.85, zorder=2)
    ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.6, linewidth=1,
               label='Constraint phase start')
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Components Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    constraint_mask = epochs >= warmup_end

    if constrained_class is not None:
        class_name = CLASS_NAMES.get(constrained_class, f'Class {constrained_class}')
        limit_col = f'Limit_Class{constrained_class}'
        limit_val = float(df[limit_col].dropna().iloc[-1])

        hard_col = f'Hard_Class{constrained_class}'
        soft_col = f'Soft_Class{constrained_class}'
        ax.plot(epochs[constraint_mask], df[hard_col][constraint_mask],
                color='#F44336', linewidth=2.5, label=f'Global Hard Count',
                zorder=5, marker='', linestyle='-')
        ax.plot(epochs[constraint_mask], df[soft_col][constraint_mask],
                color='#FF8A80', linewidth=1.5, linestyle='--',
                label=f'Global Soft Count', alpha=0.7, zorder=4)

        ax.axhline(y=limit_val, color='#4CAF50', linewidth=3, linestyle='-',
                   label=f'Global Limit ({int(limit_val)})', zorder=6)
        ax.fill_between(epochs[constraint_mask], 0, limit_val, alpha=0.04, color='green')

        sat_epochs = df[constraint_mask & (df['Global_Satisfied'] == 1)]['Epoch']
        if len(sat_epochs) > 0:
            first_sat = sat_epochs.iloc[0]
            ax.axvline(x=first_sat, color='#4CAF50', linestyle=':', alpha=0.7,
                       linewidth=1.5, label=f'Global satisfied @ E{int(first_sat)}')

        if has_local:
            group_ids = set()
            for col in group_cols:
                gid = col.split('_')[0].replace('Group', '')
                class_id = int(col.split('Class')[1])
                if class_id == constrained_class:
                    group_ids.add(gid)

            group_colors = {'0': '#E91E63', '1': '#3F51B5'}
            group_labels = {0: 'Male', 1: 'Female'}
            group_markers = {'0': 's', '1': 'D'}

            for gid in sorted(group_ids, key=lambda g: float(g)):
                display_gid = int(float(gid))
                hard_col_g = f'Group{gid}_Hard_Class{constrained_class}'
                limit_col_g = f'Group{gid}_Limit_Class{constrained_class}'
                color = group_colors.get(gid, '#795548')
                grp_name = group_labels.get(display_gid, f'Group {display_gid}')

                if hard_col_g in df.columns:
                    mask = constraint_mask
                    ax.plot(epochs[mask], df[hard_col_g][mask], color=color,
                            linewidth=1.8, linestyle='-', alpha=0.85,
                            label=f'{grp_name} Hard Count', zorder=3)

                    if limit_col_g in df.columns:
                        lv = float(df[limit_col_g].dropna().iloc[-1])
                        if lv < UNLIMITED:
                            ax.axhline(y=lv, color=color, linewidth=2, linestyle='--',
                                       alpha=0.7,
                                       label=f'{grp_name} Limit ({int(lv)})', zorder=6)

        ax.set_ylabel(f'{class_name} Prediction Count', fontsize=12)
        ax.set_title(f'Constraint Convergence -- Global + Per-Group ({class_name})',
                     fontsize=12, fontweight='bold')

    ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_convergence_overlay(all_logs, save_path):
    if not all_logs:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = cm.get_cmap('tab20', len(all_logs))

    for i, (name, df) in enumerate(sorted(all_logs.items())):
        constrained_class = _find_constrained_class(df)
        if constrained_class is None:
            continue
        hard_col = f'Hard_Class{constrained_class}'
        mask = df['L_Global'] > 0
        if not mask.any():
            continue
        epochs = df.loc[mask, 'Epoch']
        hard = df.loc[mask, hard_col]
        color = cmap(i)
        ax.plot(epochs, hard, linewidth=1.5, color=color, label=name, alpha=0.85)
        ax.plot(epochs.iloc[-1], hard.iloc[-1], 'o', color=color, markersize=5)

    sample_df = list(all_logs.values())[0]
    cc = _find_constrained_class(sample_df)
    if cc is not None:
        limit_val = float(sample_df[f'Limit_Class{cc}'].dropna().iloc[-1])
        ax.axhline(y=limit_val, color='#4CAF50', linewidth=3, linestyle='-',
                   label=f'Constraint limit ({int(limit_val)})', zorder=10)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{CLASS_NAMES.get(cc, "Constrained Class")} Hard Count', fontsize=12)
    ax.set_title('Constraint Convergence -- All Optimization Configs',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_lambda_rho_evolution(all_logs, save_path):
    if not all_logs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Lambda & Rho Schedule Across Experiments', fontsize=14, fontweight='bold')
    cmap = cm.get_cmap('tab20', len(all_logs))

    for i, (name, df) in enumerate(sorted(all_logs.items())):
        mask = df['L_Global'] > 0
        if not mask.any():
            continue
        epochs = df.loc[mask, 'Epoch']
        color = cmap(i)

        if 'Lambda_Global' in df.columns:
            axes[0].plot(epochs, df.loc[mask, 'Lambda_Global'], linewidth=1.2,
                         color=color, label=name, alpha=0.8)

        if 'L_Global' in df.columns:
            axes[1].plot(epochs, df.loc[mask, 'L_Global'], linewidth=1.2,
                         color=color, label=name, alpha=0.8)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Lambda Global')
    axes[0].set_title('Lambda Schedule')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Global Constraint Loss')
    axes[1].set_title('Constraint Loss Over Time')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved: %s", save_path)


def generate_training_curves(results_dir='results'):
    results_path = Path(results_dir)

    output_dir = results_path / 'figures' / 'training_curves'
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob('*.png'):
        old.unlink()

    all_logs = {}
    generated = 0

    for config_path in sorted(results_path.rglob('config.json')):
        exp_dir = config_path.parent
        with open(config_path) as f:
            cfg = json.load(f)

        if cfg.get('methodology') != 'tralo':
            continue
        if cfg.get('status') != 'completed':
            continue

        df = load_training_log(exp_dir)
        if df is None or len(df) < 2:
            continue

        name = cfg.get('exp_name', exp_dir.name)
        plot_training_summary(df, name, output_dir / f'{name}.png')
        all_logs[name] = df
        generated += 1

    log.info("Generated %d per-experiment training curve plots", generated)

    overlay_dir = results_path / 'figures' / 'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for old in overlay_dir.glob('*.png'):
        old.unlink()

    if all_logs:
        plot_convergence_overlay(all_logs, overlay_dir / 'convergence_overlay.png')
        plot_lambda_rho_evolution(all_logs, overlay_dir / 'lambda_rho_evolution.png')
        log.info("Generated overlay plots in %s", overlay_dir)
    else:
        log.info("No completed optimization experiments found for overlays")

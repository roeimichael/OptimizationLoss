"""Training curve plots: loss components, global/local constraint counts per experiment."""

import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

log = logging.getLogger(__name__)


def load_training_log(experiment_path):
    """Load training_log.csv, handling inconsistent column counts between phases."""
    log_path = Path(experiment_path) / 'training_log.csv'
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None

    # Find the longest header (constraint header has more columns than warmup)
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


def plot_training_summary(df, constraint_label, save_path):
    """Three subplots: losses, global counts, per-group counts."""
    group_cols = [c for c in df.columns if c.startswith('Group') and '_Hard_' in c]
    has_local = len(group_cols) > 0
    n_subplots = 3 if has_local else 2

    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4.5 * n_subplots), sharex=True)
    fig.suptitle(f'Training Summary - {constraint_label}', fontsize=14, fontweight='bold')
    epochs = df['Epoch']

    # Detect warmup end
    warmup_end = epochs.iloc[0]
    for _, row in df.iterrows():
        if row['L_Global'] > 0 or row['L_Local'] > 0:
            warmup_end = row['Epoch']
            break

    # Loss components
    ax = axes[0]
    ax.plot(epochs, df['L_CE'], color='#2196F3', linewidth=1.8, label='CE Loss')
    ax.plot(epochs, df['L_Global'], color='#F44336', linewidth=1.8, label='Global Constraint Loss')
    ax.plot(epochs, df['L_Local'], color='#FF9800', linewidth=1.8, label='Local Constraint Loss')
    ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components', fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Global prediction counts vs limit
    ax = axes[1]
    limit_cols = [c for c in df.columns if c.startswith('Limit_Class')]
    constrained_class = None
    for col in limit_cols:
        vals = df[col].replace('inf', float('inf'))
        if (vals < 1e9).any():
            constrained_class = int(col.replace('Limit_Class', ''))
            break

    if constrained_class is not None:
        hard_col = f'Hard_Class{constrained_class}'
        soft_col = f'Soft_Class{constrained_class}'
        limit_val = float(df[f'Limit_Class{constrained_class}'].iloc[-1])

        ax.plot(epochs, df[hard_col], color='#F44336', linewidth=1.8, label='Hard Count (argmax)')
        ax.plot(epochs, df[soft_col], color='#FF9800', linewidth=1.8, linestyle='--', label='Soft Count (sum proba)')
        ax.axhline(y=limit_val, color='#4CAF50', linewidth=2, linestyle='-', label=f'Global Limit ({int(limit_val)})')
        ax.fill_between(epochs, 0, limit_val, alpha=0.05, color='green')
        ax.set_ylabel(f'Class {constrained_class} Count')
        ax.set_title(f'Global: Class {constrained_class} Predictions vs Constraint Limit', fontsize=11)

    ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Per-group local constraint counts
    if has_local and constrained_class is not None:
        ax = axes[2]
        group_ids = set()
        for col in group_cols:
            gid = col.split('_')[0].replace('Group', '')
            class_id = int(col.split('Class')[1])
            if class_id == constrained_class:
                group_ids.add(gid)

        group_ids = sorted(group_ids, key=lambda g: float(g))
        colors = cm.Set2(range(len(group_ids)))

        for i, gid in enumerate(group_ids):
            display_gid = int(float(gid))
            hard_col = f'Group{gid}_Hard_Class{constrained_class}'
            limit_col = f'Group{gid}_Limit_Class{constrained_class}'
            if hard_col in df.columns:
                color = colors[i % len(colors)]
                ax.plot(epochs, df[hard_col], color=color, linewidth=1.5, label=f'Group {display_gid} (hard)')
                if limit_col in df.columns:
                    lv = df[limit_col].iloc[-1]
                    lv = float(lv) if not isinstance(lv, str) else float(lv)
                    if lv < 1e9:
                        ax.axhline(y=lv, color=color, linewidth=1.5, linestyle='--',
                                   alpha=0.6, label=f'Group {display_gid} limit ({int(lv)})')

        ax.axvline(x=warmup_end, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'Class {constrained_class} Count')
        ax.set_title(f'Per-Group: Class {constrained_class} vs Local Limits', fontsize=11)
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Epoch')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_training_curves(results_dir='results'):
    """Generate one training summary plot per our_approach experiment."""
    results_path = Path(results_dir)
    our_dir = results_path / 'binary' / 'our_approach' / 'FTTransformer'
    if not our_dir.exists():
        log.warning("No our_approach results at %s", our_dir)
        return

    output_dir = results_path / 'figures' / 'training_curves'
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob('*.png'):
        old.unlink()

    generated = 0
    for exp_path in sorted(our_dir.glob('constraint_*/standard/default')):
        df = load_training_log(exp_path)
        if df is None or len(df) < 2:
            continue
        constraint_name = exp_path.parts[-3]
        label = constraint_name.replace('constraint_', 'L=').replace('_', ' G=')
        plot_training_summary(df, label, output_dir / f'{constraint_name}.png')
        generated += 1

    log.info("Generated %d training curve plots in %s", generated, output_dir)


if __name__ == '__main__':
    generate_training_curves()

"""Training convergence figures from actual experiment logs.

Shows how each loss component evolves during training and how
the model converges to satisfy constraints.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

BASE = os.path.join(SCRIPT_DIR, '..', '..',
                     'archive_experiments', 'dermmnist', 'our_approach', 'MobileNetV3')

# Three experiments at different constraint tightness
EXPERIMENTS = [
    ('c04_02/kl0.5', 'K=45 (tight)'),
    ('c05_03/kl0.5', 'K=67 (medium)'),
    ('c06_05/kl0.5', 'K=112 (loose)'),
]

COLORS = ['#E53935', '#FF9800', '#2196F3']


# ═══════════════════════════════════════════════════════════════
# Figure 1: Prediction count convergence
#   How the number of constrained-class predictions (hard count)
#   drops from the unconstrained warmup value toward the limit K
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

for (exp_path, label), color in zip(EXPERIMENTS, COLORS):
    df = pd.read_csv(os.path.join(BASE, exp_path, 'training_log.csv'))
    K = int(df['Limit_Class4'].iloc[0])
    ax.plot(df['Epoch'], df['Hard_Class4'], color=color, linewidth=2, label=label)
    ax.axhline(y=K, color=color, linewidth=1, linestyle='--', alpha=0.5)
    ax.text(df['Epoch'].iloc[-1] + 2, K, 'K=%d' % K, fontsize=9, color=color, va='center')

ax.set_xlabel('Epoch')
ax.set_ylabel('Predictions for constrained class (MEL)')
ax.set_title('Constraint Convergence: MEL Prediction Count During Training')
ax.legend(loc='upper right')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_convergence.png'), bbox_inches='tight')
print("Saved training_convergence.png")


# ═══════════════════════════════════════════════════════════════
# Figure 2: All loss components during training (single experiment)
#   4 panels: L_CE, L_Global, L_Local, L_KL
#   Using the medium-tightness experiment (K=67)
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(BASE, 'c05_03/kl0.5', 'training_log.csv'))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Panel (a): Cross-Entropy Loss
ax = axes[0, 0]
ax.plot(df['Epoch'], df['L_CE'], '#2196F3', linewidth=2)
ax.set_ylabel(r'$\mathcal{L}_{CE}$')
ax.set_title('(a) Cross-Entropy Loss')
ax.grid(alpha=0.15)

# Panel (b): Global Constraint Loss
ax = axes[0, 1]
ax.plot(df['Epoch'], df['L_Global'], '#4CAF50', linewidth=2)
ax.set_ylabel(r'$\mathcal{L}_{global}$')
ax.set_title('(b) Global Constraint Loss')
ax.grid(alpha=0.15)
# Mark when satisfied
sat_mask = df['Global_Satisfied'] == True
if sat_mask.any():
    first_sat = df.loc[sat_mask, 'Epoch'].iloc[0]
    ax.axvline(x=first_sat, color='green', linewidth=1, linestyle=':', alpha=0.6)
    ax.text(first_sat + 2, ax.get_ylim()[1] * 0.8, 'First\nsatisfied',
            fontsize=9, color='green')

# Panel (c): Local Constraint Loss
ax = axes[1, 0]
ax.plot(df['Epoch'], df['L_Local'], '#FF9800', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$\mathcal{L}_{local}$')
ax.set_title('(c) Local Constraint Loss')
ax.grid(alpha=0.15)
sat_mask_l = df['Local_Satisfied'] == True
if sat_mask_l.any():
    first_sat_l = df.loc[sat_mask_l, 'Epoch'].iloc[0]
    ax.axvline(x=first_sat_l, color='green', linewidth=1, linestyle=':', alpha=0.6)

# Panel (d): KL Divergence Loss
ax = axes[1, 1]
ax.plot(df['Epoch'], df['L_KL'], '#9C27B0', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$\mathcal{L}_{KL}$')
ax.set_title('(d) KL Divergence Loss')
ax.grid(alpha=0.15)

fig.suptitle('Training Dynamics — MobileNetV3, MEL Constrained (K=67, L50/G30)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_all_losses.png'), bbox_inches='tight')
print("Saved training_all_losses.png")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Lambda schedule and constraint satisfaction
#   Shows how lambda_global and lambda_local grow during training
#   and how satisfaction toggles
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# Top: lambda values
ax1.plot(df['Epoch'], df['Lambda_Global'], '#4CAF50', linewidth=2, label=r'$\lambda_{global}$')
ax1.plot(df['Epoch'], df['Lambda_Local'], '#FF9800', linewidth=2, label=r'$\lambda_{local}$')
ax1.set_ylabel('Lambda value')
ax1.set_title('Lambda Schedule and Constraint Satisfaction')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.15)

# Bottom: satisfaction state (binary)
both_sat = (df['Global_Satisfied'].astype(int) & df['Local_Satisfied'].astype(int))
ax2.fill_between(df['Epoch'], 0, both_sat, color='#4CAF50', alpha=0.4, step='mid',
                  label='Both satisfied')
ax2.fill_between(df['Epoch'], 0, 1 - both_sat, color='#E53935', alpha=0.2, step='mid',
                  label='Violated')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Satisfied')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['No', 'Yes'])
ax2.legend(loc='upper left')
ax2.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_lambda_satisfaction.png'), bbox_inches='tight')
print("Saved training_lambda_satisfaction.png")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Soft count vs Hard count convergence
#   Shows both soft count (differentiable) and hard count (discrete)
#   approaching the limit K — illustrating the soft/hard gap
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

df_tight = pd.read_csv(os.path.join(BASE, 'c04_02/kl0.5', 'training_log.csv'))
K = int(df_tight['Limit_Class4'].iloc[0])

ax.plot(df_tight['Epoch'], df_tight['Hard_Class4'], '#2196F3', linewidth=2.2,
        label='Hard count (argmax)', marker='o', markersize=3)
ax.plot(df_tight['Epoch'], df_tight['Soft_Class4'], '#FF9800', linewidth=2.2,
        label='Soft count (probability sum)', marker='s', markersize=3)
ax.axhline(y=K, color='#E53935', linewidth=2, linestyle='--', label='Limit K=%d' % K)

ax.fill_between(df_tight['Epoch'], K, df_tight['Hard_Class4'],
                where=(df_tight['Hard_Class4'] > K),
                alpha=0.1, color='red', label='Excess (violated)')
ax.fill_between(df_tight['Epoch'], K, df_tight['Hard_Class4'],
                where=(df_tight['Hard_Class4'] <= K),
                alpha=0.1, color='green', label='Under limit (satisfied)')

ax.set_xlabel('Epoch')
ax.set_ylabel('MEL prediction count')
ax.set_title('Soft Count vs Hard Count — The Differentiability Gap (K=45)')
ax.legend(loc='upper right')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_soft_hard_gap.png'), bbox_inches='tight')
print("Saved training_soft_hard_gap.png")


print("\nAll training curve figures saved to %s/" % OUT_DIR)

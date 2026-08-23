"""Simple figures for the research proposal.

Two figures only, matching the clean/moderate style of reference proposals:
  Figure 1: 1x2 — penalty overview (components + bounded vs unbounded)
  Figure 2: Single panel — training convergence (from real experiment data)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
EPS = 1e-8


def sat(E, K):
    """Rational saturation: E/(E+K)."""
    return E / (E + K + EPS)

def quad(E, K):
    """Bounded quadratic: (E/K)^2 / (1+(E/K)^2)."""
    r = (E / (K + EPS)) ** 2
    return r / (1 + r + EPS)

def penalty(E, K, rho):
    """Combined penalty."""
    return sat(E, K) + rho * quad(E, K)


# ═══════════════════════════════════════════════════════════════
# Figure 1: Penalty function overview (1x2)
#   (a) Combined penalty with feasible/infeasible regions
#   (b) Our bounded penalty vs unbounded quadratic
# ═══════════════════════════════════════════════════════════════
K = 70
rho = 5.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) Penalty as function of soft count
s = np.linspace(0, K * 3, 500)
E = np.maximum(0, s - K)
y_sat = sat(E, K)
y_quad = rho * quad(E, K)
y_total = y_sat + y_quad

ax1.fill_between(s, 0, y_sat, alpha=0.25, color='#4CAF50',
                  label=r'Saturation $\varphi$')
ax1.fill_between(s, y_sat, y_total, alpha=0.25, color='#FF9800',
                  label=r'Bounded quad. $\rho\psi$')
ax1.plot(s, y_total, '#1976D2', linewidth=2.2, label=r'Combined $\ell$')
ax1.axvline(x=K, color='#E53935', linewidth=1.5, linestyle='--', alpha=0.7)
ax1.fill_between(s, 0, max(y_total) * 1.1, where=(s <= K),
                  alpha=0.03, color='green')
ax1.text(K * 0.45, max(y_total) * 0.5, 'Feasible',
         fontsize=10, color='#2E7D32', ha='center')
ax1.text(K * 1.8, max(y_total) * 0.5, 'Violated',
         fontsize=10, color='#B71C1C', ha='center')
ax1.text(K, max(y_total) * 1.02, '$K$', ha='center',
         fontsize=10, color='#E53935')
ax1.set_xlabel('Soft count $s$')
ax1.set_ylabel(r'Penalty $\ell(s, K)$')
ax1.set_title(r'(a) Constraint Penalty ($K=%d,\ \rho=%g$)' % (K, rho))
ax1.legend(fontsize=9, loc='center right')
ax1.grid(alpha=0.12)
ax1.set_ylim(-0.3, max(y_total) * 1.15)

# (b) Ours vs unbounded quadratic
E_range = np.linspace(0, K * 2.5, 500)
ax2.plot(E_range, penalty(E_range, K, rho), '#1976D2', linewidth=2.2,
         label='Ours (bounded)')
ax2.plot(E_range, (E_range / K) ** 2, '#E53935', linewidth=1.8,
         linestyle='--', label=r'Quadratic $(E/K)^2$')
bound = 1 + rho
ax2.axhline(y=bound, color='#1976D2', linewidth=0.8, linestyle=':',
            alpha=0.5)
ax2.text(E_range[-1] * 0.65, bound + 0.35,
         'bound $= 1 + \\rho = %g$' % bound,
         fontsize=9, color='#1976D2')
ax2.set_xlabel('Excess $E = \\max(0, s - K)$')
ax2.set_ylabel('Penalty')
ax2.set_title('(b) Bounded vs Unbounded Penalty')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.12)
ax2.set_ylim(0, 12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'proposal_fig1_penalty.png'),
            bbox_inches='tight')
print("Saved proposal_fig1_penalty.png")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Training convergence (from real experiment logs)
#   Single panel: prediction count converging toward K
# ═══════════════════════════════════════════════════════════════
import pandas as pd

BASE = os.path.join(SCRIPT_DIR, '..', '..',
                     'archive_experiments', 'dermmnist', 'our_approach',
                     'MobileNetV3')

EXPERIMENTS = [
    ('c04_02/kl0.5', 'K=45 (tight)', '#E53935'),
    ('c05_03/kl0.5', 'K=67 (medium)', '#FF9800'),
    ('c06_05/kl0.5', 'K=112 (loose)', '#2196F3'),
]

fig, ax = plt.subplots(figsize=(8, 4.5))

for exp_path, label, color in EXPERIMENTS:
    csv_path = os.path.join(BASE, exp_path, 'training_log.csv')
    if not os.path.exists(csv_path):
        print("Skipping %s (file not found)" % csv_path)
        continue
    df = pd.read_csv(csv_path)
    K_val = int(df['Limit_Class4'].iloc[0])
    ax.plot(df['Epoch'], df['Hard_Class4'], color=color,
            linewidth=2, label=label)
    ax.axhline(y=K_val, color=color, linewidth=1, linestyle='--', alpha=0.4)
    ax.text(df['Epoch'].iloc[-1] + 3, K_val,
            'K=%d' % K_val, fontsize=9, color=color, va='center')

ax.set_xlabel('Epoch')
ax.set_ylabel('Predictions for constrained class')
ax.set_title('Prediction Count Convergence During Training')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'proposal_fig2_convergence.png'),
            bbox_inches='tight')
print("Saved proposal_fig2_convergence.png")

print("\nDone.")

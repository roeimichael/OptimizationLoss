"""Loss function analysis figures for research proposal.

Matches ACTUAL code in src/losses/transductive_loss.py:
  loss = E/(E+K) + rho * (E/K)^2 / (1 + (E/K)^2)
         -------   ----------------------------------
         rational   bounded quadratic
         saturation

  rho increases LINEARLY from initial_rho to rho_target over training.

NO ALM terminology — that's not what we implement.
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

# Actual project values
INITIAL_RHO = 5.0
RHO_TARGET = 100.0


def sat(E, K):
    """Rational saturation: E/(E+K). Bounded in [0,1)."""
    return E / (E + K + EPS)

def quad(E, K):
    """Bounded quadratic: (E/K)^2 / (1+(E/K)^2). Bounded in [0,1)."""
    r = (E / (K + EPS)) ** 2
    return r / (1 + r + EPS)

def penalty(E, K, rho):
    """Full constraint penalty as implemented in code."""
    return sat(E, K) + rho * quad(E, K)

def sat_grad(E, K):
    return K / (E + K + EPS) ** 2

def quad_grad(E, K):
    r = E / (K + EPS)
    return 2 * r / (K * (1 + r ** 2) ** 2 + EPS)


# ═══════════════════════════════════════════════════════════════
# Figure 1: Penalty for different budget sizes K
#   4 subplots — each a realistic K from our experiments
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
K_list = [10, 30, 70, 110]
titles = [
    '(a) K=10 (very tight, e.g. VASC)',
    '(b) K=30 (tight, e.g. BCC at 30%)',
    '(c) K=70 (medium, e.g. MEL at 30%)',
    '(d) K=110 (standard, e.g. MEL at 50%)',
]

for ax, K, title in zip(axes.flat, K_list, titles):
    s = np.linspace(0, K * 3, 500)
    E = np.maximum(0, s - K)
    y = penalty(E, K, INITIAL_RHO)

    ax.plot(s, y, '#1976D2', linewidth=2.2)
    ax.axvline(x=K, color='#E53935', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.fill_between(s, 0, max(y) * 1.1, where=(s <= K), alpha=0.04, color='green')
    ax.fill_between(s, 0, max(y) * 1.1, where=(s > K), alpha=0.04, color='red')
    ax.text(K * 0.4, max(y) * 0.6, 'Feasible', fontsize=10, color='#2E7D32', ha='center')
    ax.text(K * 1.8, max(y) * 0.6, 'Violated', fontsize=10, color='#B71C1C', ha='center')
    ax.text(K, max(y) * 1.02, 'K=%d' % K, ha='center', fontsize=9, color='#E53935')
    ax.set_xlabel('Soft count s')
    ax.set_ylabel(r'$\ell(s, K)$')
    ax.set_title(title)
    ax.grid(alpha=0.12)
    ax.set_ylim(-0.2, max(y) * 1.15)

fig.suptitle(r'Constraint Penalty for Different Budget Sizes ($\rho=%.0f$)' % INITIAL_RHO,
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_penalty_by_K.png'), bbox_inches='tight')
print("Saved fig1_penalty_by_K.png")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Our bounded penalty vs unbounded quadratic
#   4 subplots at different K values
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, K, title in zip(axes.flat, K_list, titles):
    E = np.linspace(0, K * 2.5, 500)
    ax.plot(E, penalty(E, K, INITIAL_RHO), '#1976D2', linewidth=2.2, label='Ours (bounded)')
    ax.plot(E, (E / K) ** 2, '#E53935', linewidth=1.8, linestyle='--', label=r'Quadratic $(E/K)^2$')
    bound = 1 + INITIAL_RHO
    ax.axhline(y=bound, color='#1976D2', linewidth=0.8, linestyle=':', alpha=0.4)
    ax.text(E[-1] * 0.95, bound + 0.2, 'bound=%.0f' % bound, fontsize=8,
            color='#1976D2', ha='right')
    ax.set_xlabel('Excess E')
    ax.set_ylabel('Penalty')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.12)
    ax.set_ylim(0, min(15, max(penalty(E, K, INITIAL_RHO)) * 1.3))

fig.suptitle('Bounded Penalty (Ours) vs Unbounded Quadratic',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_ours_vs_quadratic.png'), bbox_inches='tight')
print("Saved fig2_ours_vs_quadratic.png")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Function and gradient decomposition
#   Left: the two components (saturation + bounded quadratic)
#   Right: their gradients
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
K = 70
E = np.linspace(0, 250, 500)

ax1.plot(E, sat(E, K), '#4CAF50', linewidth=2,
         label=r'Saturation: $\frac{E}{E+K}$')
ax1.plot(E, INITIAL_RHO * quad(E, K), '#FF9800', linewidth=2,
         label=r'Bounded quad.: $\rho \cdot \frac{(E/K)^2}{1+(E/K)^2}$')
ax1.plot(E, penalty(E, K, INITIAL_RHO), '#1976D2', linewidth=2.5,
         label='Combined')
ax1.axhline(y=1, color='#4CAF50', linewidth=0.5, linestyle=':', alpha=0.4)
ax1.axhline(y=INITIAL_RHO, color='#FF9800', linewidth=0.5, linestyle=':', alpha=0.4)
ax1.set_xlabel('Excess E')
ax1.set_ylabel('Penalty value')
ax1.set_title(r'(a) Penalty Components (K=%d, $\rho$=%.0f)' % (K, INITIAL_RHO))
ax1.legend(loc='center right', fontsize=9)
ax1.grid(alpha=0.12)

ax2.plot(E, sat_grad(E, K), '#4CAF50', linewidth=2, label='Saturation gradient')
ax2.plot(E, INITIAL_RHO * quad_grad(E, K), '#FF9800', linewidth=2, label='Bounded quad. gradient')
ax2.plot(E, sat_grad(E, K) + INITIAL_RHO * quad_grad(E, K), '#1976D2', linewidth=2.5,
         label='Combined gradient')
peak_idx = np.argmax(quad_grad(E, K))
ax2.axvline(x=E[peak_idx], color='#FF9800', linewidth=1, linestyle='--', alpha=0.5)
ax2.text(E[peak_idx] + 3, max(sat_grad(E, K) + INITIAL_RHO * quad_grad(E, K)) * 0.85,
         'Peak at\nE=%d' % int(E[peak_idx]), fontsize=9, color='#E65100')
ax2.set_xlabel('Excess E')
ax2.set_ylabel('Gradient')
ax2.set_title('(b) Gradient Decays for Large Violations')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.12)

fig.suptitle('Penalty Components and Gradient Behavior',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_function_and_gradient.png'), bbox_inches='tight')
print("Saved fig3_function_and_gradient.png")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Rho effect during training
#   4 subplots showing penalty at different rho values
#   from our ACTUAL linear schedule: 5 -> 100
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
K = 70
rho_list = [5.0, 25.0, 50.0, 100.0]
rho_titles = [
    r'(a) $\rho=5$ (training start)',
    r'(b) $\rho=25$ (early training)',
    r'(c) $\rho=50$ (mid-training)',
    r'(d) $\rho=100$ (training end / target)',
]

s = np.linspace(0, K * 3.5, 500)
E_range = np.maximum(0, s - K)

for ax, rho, title in zip(axes.flat, rho_list, rho_titles):
    y_sat = sat(E_range, K)
    y_quad = rho * quad(E_range, K)
    y_total = y_sat + y_quad

    ax.fill_between(s, 0, y_sat, alpha=0.25, color='#4CAF50', label='Saturation')
    ax.fill_between(s, y_sat, y_total, alpha=0.25, color='#FF9800', label='Bounded quadratic')
    ax.plot(s, y_total, '#1976D2', linewidth=2, label='Combined')
    ax.axvline(x=K, color='#E53935', linewidth=1.2, linestyle='--', alpha=0.6)
    bound = 1 + rho
    ax.set_xlabel('Soft count s')
    ax.set_ylabel(r'$\ell(s, K)$')
    ax.set_title(title + ' (bound=%.0f)' % bound)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.12)
    ax.set_ylim(-0.3, min(bound * 1.1, max(y_total) * 1.15))

fig.suptitle(r'Effect of $\rho$ During Linear Schedule ($\rho$: 5 $\rightarrow$ 100, K=%d)' % K,
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_rho_sensitivity.png'), bbox_inches='tight')
print("Saved fig4_rho_sensitivity.png")


print("\nAll figures saved to %s/" % OUT_DIR)
print("\nTerminology used (matching code):")
print("  - 'sat' = E/(E+K)  [rational saturation]")
print("  - 'quad' = (E/K)^2 / (1+(E/K)^2)  [bounded quadratic]")
print("  - 'rho' increases linearly via increment_rho(step)")
print("  - NO 'ALM' terminology anywhere")

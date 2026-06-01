"""Loss function analysis figures for research proposal.

4 figures, each focused on one component of the total loss:
  1. Global constraint loss vs excess samples for different K
  2. Local constraint loss — effect of K and number of violating classes
  3. Cross-entropy loss vs predicted probability
  4. KL divergence loss as prediction shifts from warmup
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-8

# Constraint sizes relevant to our datasets:
#   DermMNIST: VASC~8, BCC~31, MEL at 30%~67, MEL at 50%~112, MEL at 80%~178
#   TissueMNIST: GE at 20%~34, GE at 50%~86, GE at 80%~137
K_VALUES = [10, 30, 70, 110, 180]
K_LABELS = ['K=10\n(very tight)', 'K=30\n(tight)', 'K=70\n(medium)',
            'K=110\n(standard)', 'K=180\n(loose)']
K_COLORS = ['#E53935', '#FF9800', '#4CAF50', '#2196F3', '#7B1FA2']

# Actual rho used in project
RHO = 5.0  # initial_rho (we show the penalty at training start)


def single_class_penalty(E, K, rho=RHO):
    """Penalty for one (class, scope) pair: phi + rho * psi."""
    sat = E / (E + K + EPS)
    r = (E / (K + EPS)) ** 2
    quad = r / (1 + r + EPS)
    return sat + rho * quad


# ═══════════════════════════════════════════════════════════════
# Figure 1: Global Constraint Loss
#   x-axis: number of excess samples (how many predictions over limit)
#   y-axis: L_global for a single constrained class
#   One curve per K value
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

excess = np.linspace(0, 200, 500)

for K, label, color in zip(K_VALUES, K_LABELS, K_COLORS):
    y = single_class_penalty(excess, K)
    ax.plot(excess, y, color=color, linewidth=2.2, label=label)

ax.set_xlabel('Excess predictions beyond limit (E = soft_count - K)')
ax.set_ylabel(r'$\ell_{global}(E, K)$')
ax.set_title(r'Global Constraint Loss — Single Constrained Class ($\rho=%.0f$)' % RHO)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(alpha=0.15)
ax.set_xlim(0, 200)
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss_global.png'), bbox_inches='tight')
print("Saved loss_global.png")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Local Constraint Loss
#   Shows how total local loss accumulates when multiple classes
#   violate their constraints simultaneously.
#   Panel (a): varying K with 1 violating class
#   Panel (b): fixed K, varying number of violating classes (1, 3, 5)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

excess = np.linspace(0, 150, 500)

# Panel (a): Different K, single violating class
for K, label, color in zip(K_VALUES[:4], K_LABELS[:4], K_COLORS[:4]):
    y = single_class_penalty(excess, K)
    ax1.plot(excess, y, color=color, linewidth=2.2, label=label)

ax1.set_xlabel('Excess per class (E)')
ax1.set_ylabel(r'$\ell_{local}(E, K)$  (single class)')
ax1.set_title('(a) Local Loss — One Violating Class')
ax1.legend(loc='lower right', framealpha=0.95)
ax1.grid(alpha=0.15)
ax1.set_xlim(0, 150)

# Panel (b): Fixed K=70, multiple violating classes
# Each class contributes its own penalty, so total = sum of individual penalties
# We assume each violating class has the same excess for simplicity
K_fixed = 70
n_classes_list = [1, 3, 5]
class_colors = ['#4CAF50', '#FF9800', '#E53935']

for n_classes, color in zip(n_classes_list, class_colors):
    # Total local loss = sum of penalties from n_classes, each with excess E
    y = n_classes * single_class_penalty(excess, K_fixed)
    ax2.plot(excess, y, color=color, linewidth=2.2,
             label='%d class%s violating' % (n_classes, 'es' if n_classes > 1 else ''))

ax2.set_xlabel('Excess per class (E)')
ax2.set_ylabel(r'$\mathcal{L}_{local} = \sum_{c} \ell(E_c, K_c)$')
ax2.set_title('(b) Local Loss — Multiple Violating Classes (K=%d)' % K_fixed)
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(alpha=0.15)
ax2.set_xlim(0, 150)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss_local.png'), bbox_inches='tight')
print("Saved loss_local.png")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Cross-Entropy Loss
#   x-axis: predicted probability for the TRUE class p(y=true)
#   y-axis: CE loss = -log(p)
#   Simple textbook curve — shows the asymmetry
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

p = np.linspace(0.01, 1.0, 500)
ce = -np.log(p)

ax.plot(p, ce, 'b-', linewidth=2.5)

# Annotate key points
ax.annotate('High confidence,\ncorrect prediction\n(low loss)',
            xy=(0.95, -np.log(0.95)), xytext=(0.7, 1.5),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', linewidth=1.5),
            fontsize=10, color='#2E7D32', ha='center')

ax.annotate('Low confidence\nin true class\n(high loss)',
            xy=(0.1, -np.log(0.1)), xytext=(0.35, 3.5),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', linewidth=1.5),
            fontsize=10, color='#B71C1C', ha='center')

ax.set_xlabel(r'Predicted probability for true class  $p(y = y_{true})$')
ax.set_ylabel(r'$\mathcal{L}_{CE} = -\log(p)$')
ax.set_title('Cross-Entropy Loss')
ax.grid(alpha=0.15)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss_ce.png'), bbox_inches='tight')
print("Saved loss_ce.png")


# ═══════════════════════════════════════════════════════════════
# Figure 4: KL Divergence Loss
#   Shows how KL(current || warmup) grows as the current model's
#   predictions diverge from the warmup model's predictions.
#
#   Scenario: warmup predicts p_warmup for the constrained class,
#   current model shifts that probability. KL measures the divergence.
#
#   Panel (a): binary case — warmup says p=0.8 for class A,
#              current shifts to different values
#   Panel (b): effect of warmup confidence — what happens when
#              warmup was confident vs uncertain
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): KL divergence as current prediction shifts
# Simplified to 2-class case: warmup=[q, 1-q], current=[p, 1-p]
# KL(current || warmup) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
q_warmup = 0.8  # warmup says 80% class A
p_range = np.linspace(0.01, 0.99, 500)

kl = p_range * np.log(p_range / q_warmup + EPS) + \
     (1 - p_range) * np.log((1 - p_range) / (1 - q_warmup) + EPS)

ax1.plot(p_range, kl, 'b-', linewidth=2.5)
ax1.axvline(x=q_warmup, color='#E53935', linewidth=1.5, linestyle='--')
ax1.text(q_warmup + 0.02, max(kl) * 0.85,
         'Warmup\nprediction\np=%.1f' % q_warmup,
         fontsize=10, color='#E53935')

ax1.annotate('No divergence\n(matches warmup)',
             xy=(q_warmup, 0), xytext=(0.55, 1.0),
             arrowprops=dict(arrowstyle='->', color='#2E7D32', linewidth=1.5),
             fontsize=10, color='#2E7D32', ha='center')

ax1.set_xlabel('Current model prediction  p(class A)')
ax1.set_ylabel(r'$D_{KL}(p_{current} \| p_{warmup})$')
ax1.set_title('(a) KL Divergence as Prediction Shifts')
ax1.grid(alpha=0.15)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, max(kl) * 1.05)

# Panel (b): Effect of warmup confidence
# Show KL curves for different warmup confidences
for q, label, color in [
    (0.5, 'Warmup uncertain (p=0.5)', '#9E9E9E'),
    (0.7, 'Warmup moderate (p=0.7)', '#FF9800'),
    (0.9, 'Warmup confident (p=0.9)', '#E53935'),
]:
    kl_q = p_range * np.log(p_range / q + EPS) + \
           (1 - p_range) * np.log((1 - p_range) / (1 - q) + EPS)
    ax2.plot(p_range, kl_q, color=color, linewidth=2.2, label=label)

ax2.set_xlabel('Current model prediction  p(class A)')
ax2.set_ylabel(r'$D_{KL}(p_{current} \| p_{warmup})$')
ax2.set_title('(b) Higher Warmup Confidence = Stronger Anchor')
ax2.legend(loc='upper center', framealpha=0.95)
ax2.grid(alpha=0.15)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss_kl.png'), bbox_inches='tight')
print("Saved loss_kl.png")


print("\nAll figures saved to %s/" % OUT_DIR)

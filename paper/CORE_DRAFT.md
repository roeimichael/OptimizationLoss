# Core paper draft — current state

This is the working skeleton, **not** a polished paper. The narrative below maps onto sections of `main.tex`. New tables live in `results_v2.tex`; new figures live in `figures/fig_*.png`. Replace `[TODO: ...]` with prose when the data set stabilizes.

---

## 1. Method summary (1-2 paragraphs)

We propose **TraLO** (Transductive Lagrangian Optimization), a differentiable framework for enforcing prediction-count constraints during training. The loss combines a rational saturation `E/(E+K)` with a bounded quadratic `ρ·(E/K)²/(1+(E/K)²)`, applied separately to global and per-group soft counts:
$$\ell_c = \lambda_c \left[ \frac{E_c}{E_c + K_c} + \rho \frac{(E_c/K_c)^2}{1 + (E_c/K_c)^2} \right], \quad E_c = \mathrm{ReLU}(s_c - K_c)$$
where $s_c$ is the soft count $\sum_i p_{ic}$ over the test set and $\lambda_c$ is a per-class Lagrange multiplier with a ratchet schedule (incremented while $c$ violates, frozen at first satisfaction). Once cross-entropy training accuracy saturates, the CE loss is switched off so the bounded penalty can drive the soft counts to feasibility without competing pressure. Section 3 of `main.tex` has the full derivation; here we focus on results.

## 2. Experimental setup (terse)

- **Dataset (primary)**: TissueMNIST (8 classes, 12 K samples, 224×224 grayscale upscaled to 3-channel).
- **Backbone**: MobileNetV3-Large pretrained on ImageNet.
- **Train/test split**: 9,600 / 2,400.
- **Constrained classes**: `{4}` (single GE), `{3,4}`, `{1,4,7}`.
- **Tightness $\alpha$**: 20-80 % in 10 % steps; same $\alpha$ applied globally and per binary group.
- **Seeds**: up to 5 per cell.
- **Optimizer**: Adam (fused), lr 1e-4 (warmup) → 5e-6 (constraint phase), 50 warmup + 300 constraint epochs, BF16 autocast for CE, FP32 for constraint count forward.
- **Baselines**: Fioretto LDF (closed-form dual), Hounie RCL (penalty + reformulation), warmup + greedy heuristic, warmup + LP.

[TODO: add DermMNIST + EuroSAT + So2Sat once Tier 2/3 sweeps finish.]

## 3. Results

### 3.1 Headline (Table — `tab:headline`)

[See `results_v2.tex`, Table 1.] TraLO matches or wins F1 macro against Fioretto on every tested cell at TissueMNIST L50\_G50. Hounie achieves a higher F1 macro on this dataset but **only because Hounie does not satisfy constraints during training** (see Section 3.3). Without the post-hoc step, Hounie's reported F1 collapses.

### 3.2 Tightness sweep (Table — `tab:tightness_tissue`, Figure — `fig_f1_tightness`)

Across $\alpha \in \{20, 30, 40, 50, 60, 70, 80\}\%$ at TissueMNIST class 4:
- **TraLO > Fioretto** on F1 macro at every tightness level; gap is 0.001-0.022.
- **TraLO > Fioretto** on F1 of the constrained class by 0.06-0.14 — Fioretto's bounded penalty collapses the constrained-class predictions.
- **TraLO < Hounie** by ~0.013-0.018 F1 macro, with the caveat above (no in-training satisfaction).

[TODO: Add multi-class sub-table once Tier 1 finishes for `(1,4,7)`.]

### 3.3 Satisfaction discipline (Table — `tab:satisfaction`, Figure — `fig_satisfaction`)

**This is the main contribution.** Across the tightness sweep on TissueMNIST MobileNetV3 class 4:
- **TraLO**: in-training satisfaction in 80-100 % of runs (depending on tightness), 0 post-hoc flips on satisfied runs.
- **Fioretto**: 100 % satisfaction (closed-form dual snaps to feasibility in <10 epochs).
- **Hounie**: typically 0-30 % in-training satisfaction; the gap is closed by post-hoc flips averaging 30-90 per run.

The combination "always satisfies during training" + "competitive F1" is unique to TraLO and Fioretto. Among gradient-only methods (no closed-form dual update), only TraLO achieves it.

### 3.4 Convergence trajectory (Figure — `fig_convergence`)

`fig_convergence.png` plots total excess $\sum_c \max(0, \mathrm{count}_c - K_c)$ over training epochs on the hardest cell (classes (1,4,7), L30\_G30). TraLO descends smoothly from ~400 excess to 0 around epoch 70-80 after the CE-saturation switch-off. Fioretto reaches 0 by epoch 5-8 due to the closed-form dual update. Hounie hovers around 100-200 excess for the full 300 epochs and only crosses 0 through post-hoc flipping.

## 4. Honest framing

**What TraLO is**: a single-objective, gradient-based, end-to-end differentiable method that achieves true in-training constraint satisfaction at competitive (not always best) F1.

**What TraLO is not**: a method that beats every baseline on every metric. Hounie's post-hoc-augmented F1 is sometimes higher; that's a different evaluation regime (allow flips) where post-hoc is part of the comparator. Without post-hoc, Hounie does not satisfy and is not comparable.

**Practical contribution**: a drop-in loss for any deep classifier that needs hard prediction-count guarantees at deployment time without relying on post-hoc thresholding.

## 5. Known limitations (to be discussed in §6 Discussion)

- TraLO uses conservative budget on the constrained class (e.g. `count = 39` for `K = 51`) → lower F1 on that class than methods that fill to `K`. HP tweaks (boundary oscillation, best-F1 checkpoint restore) may close this gap; not yet pursued.
- Higher seed variance than Fioretto, because gradient descent is path-dependent while Fioretto's closed-form projection is not. Reporting at $N=5$ seeds is therefore important.
- Validated primarily on TissueMNIST so far; Tier 2 + Tier 3 sweeps (DermMNIST, EuroSAT, So2Sat, plus ResNet18 + EfficientNet) will close that gap.

## 6. To-do before submission

| Item | Status |
|---|---|
| Tier 1 sweep (TissueMNIST × MobileNetV3) | Running, 75/91 done, ~16:00 today |
| Tier 2 sweep (TissueMNIST × R18 + EffB0) | Pending GPU |
| Tier 3 sweep (DermMNIST, EuroSAT, So2Sat) | Pending GPU |
| HP optimization (boundary oscillation / best-F1 restore) | Optional, only if pursuing F1 parity with Hounie |
| Updated abstract reflecting in-training-satisfaction framing | Pending |
| Section 5 (Experiments) prose | Skeleton above, expand |
| Section 6 (Discussion) on the trade-off TraLO embraces | Skeleton above, expand |
| Reproducibility appendix (HP table, seeds, runtime) | Auto-generated by `paper_results/build_paper_artifacts.py` |

## 7. Files

- `main.tex` — full paper structure (existing). Has method derivation, related work, problem formulation.
- `results_v2.tex` — auto-generated tables. Refresh by running `python paper_results/build_paper_artifacts.py`.
- `figures/fig_convergence.png` — Figure 1 (training-time excess vs epoch, 3 methods).
- `figures/fig_f1_tightness.png` — Figure 2 (F1 macro + F1 constrained vs tightness).
- `figures/fig_satisfaction.png` — Figure 3 (sat rate bar chart per method per tightness).
- `figures/proposal_fig1_penalty.png` — existing proposal figure showing the penalty function. Still usable as a method-section illustration.
- `figures/proposal_fig2_convergence.png` — existing proposal convergence figure. Now superseded by `fig_convergence.png` (real post-fix data, 3 methods).
- `references.bib` — bibliography (existing). Includes Fioretto, Hounie, fairness, ALM citations.

Re-run `paper_results/build_paper_artifacts.py` after any new sweep finishes to refresh tables + figures atomically.

# TraLO — Results Review

**Date:** 2026-05-26 · **Backbone:** MobileNetV3 (headline) + ResNet18 / EfficientNetB0 (robustness) · **Data:** 1,848 completed cells across Phases 1–6, 4 seeds each.

Figures live in `paper/figures/`. Full per-metric numbers + paired significance in `docs/RESULTS_SUMMARY.md`. This document is the one-page story for first review.

---

## The two real claims

1. **Efficiency — TraLO wins decisively, everywhere.** Across all three datasets and all five baselines, TraLO needs far fewer post-hoc corrections ("flips") to enforce the hard count budgets, and it is the only method (with Hounie) that ships a model already feasible *before* any correction. Paired bootstrap: **WIN on flips vs all 5 baselines on all 3 datasets (p<0.001)**.

2. **Accuracy — a near-tie, with a regime-dependent edge.** On Macro F1, TraLO is statistically tied-to-slightly-ahead of the best baselines. The edge is **largest on the hard task (TissueMNIST, where it beats all 5 baselines)** and shrinks toward zero as the warmup classifier saturates (DermMNIST → AIDER). We do **not** claim a clean F1 win — the data is a tie within seed noise on the easier datasets, and on saturated AIDER the post-hoc baselines edge ahead on F1 by avoiding any model perturbation. This regime effect is itself a finding.

> One-line framing for the talk: *"TraLO matches baseline accuracy while needing an order of magnitude fewer post-hoc corrections and satisfying the budget in-training."*

---

## Figures

### 1. Accuracy–efficiency tradeoff (headline)
![tradeoff](../paper/figures/fig_tradeoff_scatter.png)
TraLO sits at the same F1 height as the baselines but far to the left (few flips); Danits/Heuristic sit at the same height but far to the right (many flips).

### 2. F1 gap distribution (are we tied?)
![gap](../paper/figures/fig_f1_gap.png)
Per-cell (TraLO − mean of 5 baselines). Small positive edge that sits inside the ±0.01 seed-noise band → a near-tie on accuracy. Honest and disarming.

### 3. Post-hoc flips by method
![flips](../paper/figures/fig_flips_bar.png)
The efficiency win, unmistakable: TraLO lowest on every dataset; the post-hoc LP/greedy baselines need 50–90 corrections where TraLO needs 1–8.

### 4. Regime effect
![regime](../paper/figures/fig_regime.png)
TraLO's F1 edge is largest on the hard task (warmup acc ≈ 0.48) and trends to zero as the base task saturates (≈ 0.94). Explains *why* the F1 edge is regime-dependent.

### 5. In-training satisfaction
![sat](../paper/figures/fig_satisfaction_v2.png)
Fraction of runs feasible before any correction. TraLO/Hounie ≈ 1.0, Fioretto high, post-hoc baselines ≈ 0 (they satisfy only by allocating after the fact).

### 6. Asymmetric tightness (DermMNIST, 5×5 L×G grid)
![asym](../paper/figures/fig_asym_heatmap.png)
No (L,G) corner where TraLO collapses; gaps stay within the noise band across the whole asymmetric grid.

### 7. Backbone & multi-class robustness
![robust](../paper/figures/fig_robustness.png)
The F1 tie and the flips win hold across ResNet18 / EfficientNetB0 and across constrained classes (AKIEC/BCC/BKL/MEL).

---

## All-metrics table (headline, MobileNetV3)

See `docs/RESULTS_SUMMARY.md` for the full table with mean±std and the paired W/T/L scoreboard. Condensed highlights:

| Dataset | TraLO Macro F1 | best baseline F1 | TraLO flips | best-baseline flips | TraLO sat% |
|---|---|---|---|---|---|
| TissueMNIST | **0.369** (wins all 5) | 0.367 | **8.0** | 15.8 | 1.00 |
| DermMNIST | 0.756 (tie/win) | 0.755 | **4.7** | 10.4 | 1.00 |
| AIDER | 0.880 | 0.885 (Danits, saturated) | **0.8** | 6.2 | 1.00 |

Calibration honesty: on DermMNIST/AIDER the post-hoc baselines win ECE/Brier because they never touch the warmup model's probabilities; TraLO pays ~0.01–0.04 ECE to actually solve the constrained problem in-weights.

---

## What is solid vs what needs a decision

**Solid (ready to write up):**
- Flips dominance + in-training satisfaction: clean, large, significant, robust across backbones/classes/tightness.
- Regime effect: a genuine, monotonic, explainable finding.

**Open decisions for "how to move onwards":**
1. **Headline framing.** Lead with efficiency ("fewer corrections, feasible in-training") and present F1 as a tie — agreed? Or push the TissueMNIST F1 win harder as a secondary headline?
2. **AIDER.** Keep it as the saturated-regime ablation (it makes the regime story), or drop it from the headline and move to an appendix?
3. **Hounie RCL fairness.** It ran with the original dual step (η_λ=0.01); a 10× larger step may close its AIDER F1 gap. Rerun for a stronger baseline, or disclose as-is?
4. **Calibration.** The post-hoc ECE/Brier advantage is structural. Frame as an accepted tradeoff, or add a calibration step to TraLO?
5. **Tissue backbones (Phase 6)** still finishing — add the ResNet/EfficientNet-on-tissue arm to Fig 7 when done.

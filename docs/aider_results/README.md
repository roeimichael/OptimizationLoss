# Aider — frozen dataset experiments

**Status:** FROZEN as of 2026-05-24. No further aider experiments will be run.

**Why frozen:** the warmup model on aider saturates at 99.98 % training accuracy after only 2–3 epochs (out of 50). The trained constraint-satisfying methods (TraLO, TraLO-bounded, Fioretto LDF, Hounie RCL) cannot improve on this base model without disturbing the already-correct predictions on the unconstrained classes. The post-hoc heuristic, which leaves the model untouched and only flips the top-K most confident over-predictions on the constrained class, is therefore close to optimal on F1 macro.

---

## Dataset setup

| Property | Value |
|---|---|
| Source | Aerial disaster imagery (4 classes) |
| Classes | collapsed_building, fire, flooded_areas, normal |
| Class balance | 8.6 % / 8.7 % / 8.8 % / 73.9 % (test) |
| Constrained class | 0 = collapsed_building (8.6 %) — rescue-triage framing |
| Group column | `synth_group` (binary, near-balanced 7.5 % / 9.7 %) |
| Backbone | MobileNetV3 |
| Warmup epochs | 50 |
| Constraint epochs | 300 |
| Seeds | 1, 2, 3, 4 |
| Tightness | symmetric L20/L30/L50/L70/L80 |

## Structural finding: F1 on the constrained class is identical across methods

With warmup saturated, the trained methods cannot find a better set of collapsed_building predictions than the post-hoc heuristic. Every constraint-satisfying method ends up predicting the same top-K most confident collapsed_building instances:

| Tight | TraLO F1_c0 | Fior F1_c0 | Hounie F1_c0 | Danits F1_c0 | Heuristic F1_c0 |
|---|---|---|---|---|---|
| L20_G20 | 0.328 | 0.328 | 0.295 | 0.328 | 0.328 |
| L30_G30 | 0.466 | 0.466 | 0.414 | 0.466 | 0.466 |
| L50_G50 | 0.667 | 0.667 | 0.621 | 0.667 | 0.667 |
| L70_G70 | 0.821 | 0.821 | 0.818 | 0.821 | 0.821 |
| L80_G80 | 0.891 | 0.891 | 0.883 | 0.891 | 0.891 |

The differences across methods in macro F1 come entirely from collateral damage on the **unconstrained** classes (fire, flooded, normal). Trained methods perturb features used by adjacent classes; heuristic doesn't.

## Headline summary: TraLO vs best baseline

| Tight | TraLO F1m | Best-base F1m | F1m gap | TraLO Flips | Best-base Flips | Flips gap | TraLO ECE | Best-base ECE | ECE gap |
|---|---|---|---|---|---|---|---|---|---|
| L20_G20 | 0.7929 | 0.8024 | -0.0095 | 1.2 | 5.8 | +4.5 | 0.0781 | 0.0116 | -0.0665 |
| L30_G30 | 0.8362 | 0.8391 | -0.0029 | 2.0 | 8.0 | +6.0 | 0.0650 | 0.0116 | -0.0534 |
| L50_G50 | 0.8867 | 0.8931 | -0.0064 | 0.0 | 8.2 | +8.2 | 0.0486 | 0.0116 | -0.0370 |
| L70_G70 | 0.9320 | 0.9356 | -0.0037 | 0.5 | 2.2 | +1.8 | 0.0305 | 0.0116 | -0.0189 |
| L80_G80 | 0.9533 | 0.9563 | -0.0030 | 0.5 | 3.8 | +3.2 | 0.0233 | 0.0116 | -0.0117 |

**Interpretation:** TraLO ties or loses 0.003–0.010 on F1m, but wins the Flips comparison by 2–8 (and by 22–84 against the post-hoc baselines danits_lp / heuristic, since those need to flip every over-predicted collapsed instance after training). On ECE, post-hoc methods win because they leave the well-calibrated warmup model untouched.

## Possible paper framings (TBD with thesis advisor)

1. **Easy-task regime ablation.** Keep aider as evidence that on easy tasks the heuristic is hard to beat on F1m, but TraLO is still the only method that produces a deployable constraint-aware model end-to-end. The Flips gap is the deployability claim.

2. **Switch constrained class.** Try fire (cls=1) or flooded (cls=2) to see whether they trigger the same saturation. Most likely yes — the warmup is the bottleneck, not the class.

3. **Reconstruct aider as a harder benchmark** (reduce warmup epochs, add noise, real groups). Changes the experimental contract, so probably not.

## Files in this folder

- `aider_per_seed.csv` — 6 methods × 8 metrics, one row per (tightness, seed)
- `aider_head_to_head.csv` — TraLO vs best-baseline gap per tightness
- `aider_per_class.csv` — per-class F1, precision, recall for every (tight, method)

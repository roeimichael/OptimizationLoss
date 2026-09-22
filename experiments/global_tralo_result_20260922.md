# Global-only TraLO, null and Clipper: first matched screen

**Result: this TraLO recipe did not improve the mean results over either
control.** It enforced the global upper bounds, but suppressed the constrained
classes excessively. This is a finding about one small frozen-feature recipe,
not proof that global constraint training cannot work.

## What was tested

[Registered protocol](global_tralo_comparison.md),
[exact configuration](../examples/global_comparison.json).
Training commit: `0c355ba4aa817d5af6b88d32ba4cf0890c9b5ab8`.
Three seeds, nine completed fits, no hyperparameter search or replacements.
CIFAR-100: 10,000 training and 2,000 development images, fixed frozen ResNet-18
features. Each classifier was freshly initialized. Adam learning rate 0.001, ten supervised
epochs, batch 256. Quota: at most ten predictions for each of classes 0–9;
all other classes uncapped. The official test split was not evaluated.

| Arm | Same 400 supervised updates | Adam reset after epoch 5 | Extra global-penalty updates |
|---|---|---|---|
| Clipper | Yes | No | None |
| TraLO-null | Yes | Yes | None |
| TraLO | Yes | Yes | Up to one after each of epochs6–10 |

TraLO uses the development **features without labels** during constraint updates.
This is transductive evaluation. All three arms use identical initial weights
and supervised batch order for each seed; warm-up and batch hashes match.
There is no randomness reset at the phase boundary.

## Side-by-side results

Values are three-seed means. F1 is on a 0–100 scale. Macro-F1 averages 100
classes; constrained-class F1 averages only the ten capped classes. No metric
was selected afterward as the sole winner criterion.

| Final allocation | Training arm | Accuracy | Macro-F1 | Constrained-class F1 |
|---|---|---:|---:|---:|
| Upper-bound correction | Clipper | 60.47% | 60.26 | 50.85 |
| Upper-bound correction | TraLO-null | 60.67% | 60.42 | 50.71 |
| Upper-bound correction | TraLO | 57.70% | 54.96 | 5.45 |
| Capped-first | Clipper | 60.50% | 60.32 | 51.65 |
| Capped-first | TraLO-null | 60.77% | 60.58 | 52.05 |
| Capped-first | TraLO | 60.10% | 59.58 | 46.52 |

All allocated outputs satisfy the same caps. Upper-bound correction only
repairs excess predictions; it does not force the model to fill unused slots.
Capped-first explicitly fills constrained slots before allocating the rest.
It therefore restores many of the class assignments TraLO suppressed, but the
result still falls below the controls on average.

Raw diagnostic results, before any allocation:

| Training arm | Accuracy | Macro-F1 | Constrained-class F1 | All ten raw caps satisfied in every seed? |
|---|---:|---:|---:|---|
| Clipper | 61.75% | 61.54 | 58.74 | No |
| TraLO-null | 61.98% | 61.71 | 58.64 | No |
| TraLO | 57.70% | 54.96 | 5.45 | Yes |

Raw infeasible results are diagnostics, not feasible competitors.

## How consistent was the difference?

Each delta subtracts the other arm at the **same seed and same allocation**.
Accuracy differences are percentage points; F1 differences use the same 0–100 scale.

| Allocation | TraLO minus null metric | Seed 701 | Seed 702 | Seed 703 | Mean | Sample SD of paired differences |
|---|---|---:|---:|---:|---:|---:|
| Upper-bound correction | Accuracy | -3.55 | -3.30 | -2.05 | -2.97 | 0.80 |
| Upper-bound correction | Constrained-class F1 | -44.88 | -46.66 | -44.27 | -45.27 | 1.24 |
| Capped-first | Accuracy | -1.15 | -0.95 | +0.10 | -0.67 | 0.67 |
| Capped-first | Constrained-class F1 | -5.51 | -6.18 | -4.89 | -5.53 | 0.65 |

The small accuracy win at seed 703 under capped-first remains visible. With
only three seeds on one fixed split, these are descriptive differences, not
a general statistical-superiority claim. Every metric and all three pairwise
comparisons are saved in [the audited numerical record](global_tralo_result_20260922.json).

TraLO-null minus Clipper mean accuracy is +0.20 points for upper-bound correction
and +0.27 for capped-first. That small schedule effect does not establish a
general optimizer-reset benefit.

## What actually went wrong in this run?

The constraint implementation was active: gradient and parameter-displacement
logs show real updates. There were no skipped/nonfinite updates. However, the
final raw model assigned only **14, 6 and 9 images**, respectively, across all
ten constrained classes, versus a combined maximum capacity of 100. It satisfied
the rule partly by avoiding these classes. An upper bound does not reward using
the remaining capacity or selecting the correct images.

The declared reference controller increases rho from 0.5 toward 100 over a short
five-epoch phase. Observed active gradient norms grew from about 0.46–0.49 on the
first constraint update to as high as 209.32. This is consistent with excessive
constraint pressure; the logs do not isolate rho from the multiplier and shared
Adam-state effects.

There is an additional concrete observation: after the epoch 9 constraint step,
the total constrained raw predictions were 64, 86, 56. After the next supervised
epoch they were 14, 6, 9, and the final constraint loss was zero, so no final
constraint optimizer step ran. Thus the final suppression cannot be blamed on
a hidden zero-loss constraint step. The changed parameters and shared Adam
state persist into supervised training. A separate-optimizer ablation would
be required to distinguish those effects; it was not run here.

**Interpretation:** for this recipe the constraint intervention changed the
model substantially, but in an unhelpful direction. The count penalty controls
how much probability goes to a class; it does not itself tell the model which
unlabeled examples truly belong there. The supervised objective supplies that
information, but the balance/controller/optimizer interaction did not preserve
the useful class predictions here. This does not show every possible balance fails.

## Verification and reproduction

- 53 tests passed locally and natively on both DSI hosts: exact analytic gradient,
  finite-difference gradient, inactive gradients, controller freezing, exact
  zero-intervention/null parity and logging neutrality, plus prior regressions.
- New source committed/pushed to GitHub and DSI; deployed tracked hashes matched.
  dsisco01 GPU0, Quadro RTX 6000/Turing, FP32, TF32 off. Blackwell GPUs occupied by
  another user were untouched. UUID and launch command are saved in `launch.json`.
- Cache completion log pinned by SHA-256; its artifacts and current dataset
  bytes verified. No inherited runtime imports.
- All nine arms applied 400/400 supervised updates; TraLO applied 4, 3, 4 active
  constraint updates for seeds 701, 702, 703. Zero skipped updates. Inactive
  penalty opportunities were deliberately skipped, not numerical failures.
- Artifact hashes, final counts, all three metrics and all paired deltas audited.
  Independent scikit-learn metric recomputation agreed within 1e-12.
- Per-arm training/report times ranged roughly 1.29–3.60 seconds using cached
  features; the first arm includes startup overhead. These are not a fair
  end-to-end speed benchmark and exclude feature extraction.

Remote evidence:
`/home/dsi/michaer8/tralo-rebuild/runs/global-tralo-0c355ba4-20260922T135157Z/`.
Local complete copy:
`C:/Users/roeym/.codex/rebuild-audit-20260922/global-tralo-run/`.
The registered protocol contains the exact reproduction command.

No extra recipes were run to seek a better score. A targeted next experiment
would distinguish controller strength and shared optimizer-state effects,
keeping these negative results as the baseline rather than replacing them.

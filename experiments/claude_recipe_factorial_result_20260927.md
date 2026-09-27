# Result: which part of Yuval's pipeline carries its gain, and TraLO's step across eight recipes (knee, cap 76)

Preregistration: [claude_recipe_factorial_prereg_20260927.md](claude_recipe_factorial_prereg_20260927.md),
with its Amendment 1 (the dose relation), the erratum and the operational notes.

- **What ran.** A 2 x 2 x 2 factorial on ResNet18 inside Yuval's pipeline. Augmentation (A), the
  balanced sampler (S) and early stopping (E) are each on or off, and every cell carries pto,
  tralo_final and sham_final. 192 study jobs (seeds 4200-4223 x 8 cells) and 8 pilot jobs (seed
  4299), runner release 67ecde20.
- **n = 24 seeds.** Each contrast is formed within a seed, then tested across seeds.
- **Scorer, fixed before the data.** `analysis/score_recipe.py` as of commit 698e540e, the 13:25
  correction, made before any job completed. It ran on dsisco01 at 16:24 from release 7d8f7dd7, whose
  copy is byte-identical (sha256 3cff5bd5). Output: `analysis/recipe_factorial_score.txt`.

**Finding: augmentation carries the recipe gain. TraLO's step is attributable across the eight
recipes, but in every cell the snapshot ensemble gains more on its point estimate.**
- F-A is **+5.64 cc-F1 [+4.60, +6.68]** (Holm 0.000), the size of the whole +4.81 measured before.
- F-S is +0.72 [-0.24, +1.68] (Holm 0.135). F-E is **-0.95 [-1.77, -0.13]** (Holm 0.049).
- P2-pooled is **+0.72 [+0.46, +0.97]** (Holm 0.000), or +0.66 slots of 76 (converted, below).
- Amendment 1's dose relation holds prospectively, pooled and within seed.

## Integrity

- **Runs.** All 192 study jobs exited 0, the last at 16:21; the scorer found 24 complete seeds of 24.
- **Pilot.** The first pilot (12:29) died of CUDA out-of-memory when a B5 process took 10.6 GB on the
  same GPU. Its outputs are quarantined in `runs/claude-recipe/failed_pilot_oom_20260927_1229/`; no
  score was produced or read. The re-run pilot (same release, configs and gate) passed at 13:38:40.
- **Gate items, applied to every study seed.** The scorer raises unless a seed has all eight cells,
  one initialisation, a sampler order shared exactly by the cells with the same S, and augmentation
  draws shared exactly by the cells with the same (A, S). E-off cells must keep epoch 10 and match
  their E-on twin's epoch-10 snapshot bit for bit, wherever the twin reached epoch 10. Every applied
  step must reach hard count <= 76, with the sham radius equal to the target radius. It printed "192
  jobs pass every gate item". The runner also raises unless the restored best weights reproduce the
  best epoch's output.
- **Duplicates.** No pto prediction vector repeats across seeds or across (A, S) cells. The scorer
  raises on either, though its printed line names only the first. Three E-off cells equal their E-on
  twin, whose best epoch was 10: the real zero that the 13:25 correction allows.
- **Deployment and determinism.** Every arm fills exactly 76 grade-3 slots and matches its run's
  report (the scorer raises otherwise). Determinism was not re-checked by re-training.

## Primary: cc-F1 of grade 3 under capped_first

n = 24 paired seeds, Holm over the four. p and Holm are as printed, to three decimals.

| Contrast | Mean [95% t CI], points | sd | p | Holm |
|---|---|---|---|---|
| F-A augment | +5.64 [+4.60, +6.68] | 2.46 | 0.000 | 0.000 |
| F-S balanced | +0.72 [-0.24, +1.68] | 2.28 | 0.135 | 0.135 |
| F-E early_stop | -0.95 [-1.77, -0.13] | 1.93 | 0.024 | 0.049 |
| P2-pooled tralo_final - sham_final | +0.72 [+0.46, +0.97] | 0.60 | 0.000 | 0.000 |

- **P2-pooled in slots:** +0.66 [+0.42, +0.88] of 76. Converted here by dividing the printed points
  by 1.0989, the cc-F1 value of one slot (F1 = 2TP/(76 + 106)).
- **Pooling bought precision:** per-seed sd 0.60, against 1.54 for the ResNet18 block's single P2.

## Secondary (no family claim)

**Yuval's metrics for the primaries:**

| Contrast | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| F-A | +4.72 [+4.05, +5.39] | +5.22 [+4.49, +5.96] | +4.97 [+4.24, +5.70] |
| F-S | -1.64 [-2.47, -0.82] | +1.33 [+0.39, +2.28] | +0.04 [-0.83, +0.92] |
| F-E | -0.31 [-0.97, +0.35] | -1.69 [-2.31, -1.07] | -1.08 [-1.65, -0.51] |
| P2-pooled | +0.18 [+0.03, +0.33] | +0.01 [-0.20, +0.22] | -0.17 [-0.33, -0.01] |

The scorer also prints p and a Holm adjustment within each metric (for example, P2-pooled on
weighted-F1: p 0.041, Holm 0.082); the prereg makes no family claim for these. The sampler trades
accuracy for macro-F1; early stopping also costs macro-F1 and weighted-F1.

**Two-way interactions on pto cc-F1** (the first switch's effect, second switch on minus off):

| Interaction | Mean [95% t CI] | sd | p |
|---|---|---|---|
| A, by S | -1.21 [-2.54, +0.11] | 3.13 | 0.070 |
| A, by E | +2.40 [+0.91, +3.90] | 3.54 | 0.003 |
| S, by E | -0.11 [-1.60, +1.37] | 3.51 | 0.875 |

- Augmentation gains 2.40 points more with early stopping on. Equivalently, early stopping's effect
  is 2.40 points better with augmentation on.
- Post hoc, from the cell table below: the two lowest pto means, 63.74 and 62.96, are the E-on cells
  without augmentation, which restore a mean best epoch of 2.1 and 1.8. The effect of E within each
  level of A is not computed; it needs the per-seed E effect within the A-on and A-off cells.

**How P2 depends on each switch** (on minus off): A +0.08 [-0.46, +0.62] (p 0.760), S -0.31 [-0.80,
+0.19] (p 0.209), E +0.10 [-0.43, +0.64] (p 0.695). No switch detectably changes the step's gain.

**Each cell, P2 set beside ENS - best** (pto in %; epochs are means; 95% t intervals, unadjusted):

| A S E | pto cc-F1 / acc | Best, run | P2 | ENS - best |
|---|---|---|---|---|
| 1 1 1 (Yuval) | 70.47 / 62.13 | 9.1, 14.1 | +0.78 [+0.24, +1.32] | +2.20 [+0.89, +3.50] |
| 1 1 0 | 69.78 / 60.94 | 10.0, 10.0 | +0.96 [+0.41, +1.51] | +2.29 [+1.20, +3.37] |
| 1 0 1 | 69.92 / 63.88 | 7.7, 12.7 | +0.50 [-0.09, +1.10] | +1.92 [+0.92, +2.93] |
| 1 0 0 | 70.10 / 63.58 | 10.0, 10.0 | +0.78 [+0.16, +1.40] | +1.05 [+0.23, +1.88] |
| 0 1 1 | 63.74 / 56.11 | 2.1, 7.1 | +0.41 [-0.11, +0.94] | +3.98 [+2.67, +5.30] |
| 0 1 0 | 66.44 / 58.64 | 10.0, 10.0 | +0.09 [-0.49, +0.67] | +0.87 [-0.18, +1.92] |
| 0 0 1 | 62.96 / 58.36 | 1.8, 6.8 | +1.37 [+0.21, +2.54] | +3.75 [+2.51, +5.00] |
| 0 0 0 (all off) | 64.56 / 58.56 | 10.0, 10.0 | +0.82 [+0.14, +1.51] | +2.01 [+0.77, +3.26] |

- P2's point estimate is positive in all eight cells; its interval excludes 0 in five (111, 110,
  100, 001, 000). ENS - best's interval excludes 0 in seven, all but 010.
- ENS - best is larger than P2 on its point estimate in every cell.
- In E-off cells the best epoch is the last, so the window rule averages epochs 8-10, three
  snapshots. In E-on cells it runs from best - 2 to the stopping epoch.
- The all-on cell's P2, +0.78 [+0.24, +1.32], sits beside the ResNet18 block's +0.73 [+0.08, +1.38].

**Dose (Amendment 1).** P2 against the excess of pto's hard count over the cap, binding jobs only:
- Pooled: Spearman +0.340 over 175 jobs (p printed as 0.0000). This p treats the jobs as
  independent. They are not, since each seed contributes up to eight.
- Within seed: the mean of the 24 per-seed coefficients is +0.26 [+0.10, +0.43] (sd 0.39, p 0.003).

**The two unpaired references** (Welch, n 24 vs 24; the output prints no interval for them):

| Metric (%) | All-on cell | ResNet18 block pto | Diff (p) | All-off cell | v3 clipper | Diff (p) |
|---|---|---|---|---|---|---|
| cc-F1 | 70.47 | 69.78 | +0.69 (0.4433) | 64.56 | 64.97 | -0.41 (0.6597) |
| Accuracy | 62.13 | 62.45 | -0.32 (0.5210) | 58.56 | 58.85 | -0.29 (0.6424) |
| Macro-F1 | 64.01 | 62.83 | +1.18 (0.0517) | 59.19 | 59.48 | -0.28 (0.5907) |
| Weighted-F1 | 61.97 | 61.25 | +0.71 (0.1775) | 57.68 | 57.67 | +0.01 (0.9870) |

The ResNet18 block is seeds 4000-4023 of the Yuval-pipeline study; the v3 clipper is seeds 1801-1824.

**Exploratory (printed by the scorer, not in the prereg).** All on minus all off, pto cc-F1, paired:
+5.91 [+3.76, +8.06] (sd 5.09, p 0.000). The earlier +4.81, unpaired on other seeds, lies inside it.

## Readings, against the readings fixed before the data

- **F-A: "A main effect with its CI above 0."** It is. Augmentation carries the recipe gain. The
  reading expects "part" of the +4.81 total; the measured effect is the size of all of it (its
  interval contains +4.81). It is the largest of the three, as predicted from CUTPAIR.
- **F-S and F-E: "S or E main effect with its CI above 0 ... A CI below 0 means it costs cc-F1 at
  this cap."** F-S covers 0, so neither half fires: the sampler carries no detectable part of the
  gain on cc-F1. F-E is below 0 (Holm 0.049): early stopping costs cc-F1 at this cap, averaged over
  A and S. The A x E interaction shows that the cost depends on augmentation.
- **"All three CIs cover 0."** Does not apply. The comparison it names, the all-off cell against v3,
  is -0.41 cc-F1 (p 0.6597). It detects no part of the +4.81 in the components held fixed.
- **P2-pooled: "CI above 0."** It is. TraLO's step carries an attributable who-signal across the
  eight recipes: +0.66 [+0.42, +0.88] slots of 76 (converted). All eight recipes are on ResNet18.
- **"It clears the thesis bar only if it also beats the ensembled clipper."** Not shown, and the
  side-by-side comparison points the other way: P2 is below ENS - best in all eight cells, on point
  estimates. The paired test, tralo_final against the ensembled clipper within a seed, is not
  computed; it needs their per-seed difference in each cell and its per-seed mean over the cells. A
  pooled ENS - best over the eight cells, formed like P2-pooled, is not computed either.
- **Amendment 1: "A positive correlation means the step's who-signal grows with the number of items
  it must evict."** Both versions are positive. The within-seed version holds seed-level model
  quality fixed, though not recipe-level quality, so a recipe-level confound is not excluded.
- **"A recipe effect of any size is a gain for the post-hoc clipper too."** The +5.64 from
  augmentation is such a gain. It is not evidence for a constraint loss.
- **The pipeline level (no reading fixed).** The all-on cell differs from the ResNet18 block's pto
  by +0.69 cc-F1 (p 0.4433): it replicates within the noise of an unpaired test.

## What it means

- **On ResNet18, the gain in Yuval's pipeline is his augmentation:** +5.64 cc-F1 averaged over the
  other two switches, the size of the whole pipeline gain. His balanced sampler adds nothing
  detectable on cc-F1. The settings held fixed (optimiser, LR decay, normalisation, carve) add
  nothing detectable over our v3 clipper. His early stopping (patience 5 on the carve's loss, best
  weights restored) costs cc-F1 against a fixed 10 epochs, on average. Post hoc: without augmentation
  it restores a mean best epoch of 2.1 or 1.8, and those are the two lowest cells.
- **For the clipper this is a practical recipe:** plain training with augmentation, plus the cut. In
  our own recipe, augmentation alone gave +3.48 [+1.71, +5.25] (CUTPAIR, LEDGER PART 4).
- **TraLO's step, applied on top of each of the eight recipes, has a small attributable gain on
  average:** +0.66 slots of 76 (converted), and no switch detectably changes it. The ResNet18
  block's +0.73 missed Holm (0.088); here the pooled contrast is Holm-positive on fresh seeds.
- **The dose pattern** was retracted as "backbone-specific or noise" after B5 (-0.03). On ResNet18
  it now holds prospectively, so "noise" no longer fits there. B5 did not show it.
- **It is not shown to reach the post-hoc bar.** In every cell, on point estimates, the snapshot
  ensemble gains more over the best checkpoint than the step gains over its sham. Whether the step
  adds anything on top of the ensemble is not measured here; the step-ensemble study
  ([prereg](claude_stepens_prereg_20260927.md)) measures it.

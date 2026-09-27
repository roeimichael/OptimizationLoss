# Result: TraLO's step on MobileNetV3 and RegNetY inside Yuval's pipeline (knee, cap 76)

Preregistration: [claude_yuval_smallbb_prereg_20260927.md](claude_yuval_smallbb_prereg_20260927.md).

- **What ran.** Yuval's pipeline with PTO only (`max_retrains` 1) and the arms pto, tralo_final and
  sham_final, on `mobilenet_v3_large` (block mn3: seeds 4300-4323, pilot 4399) and `regnet_y_400mf`
  (block rgy: seeds 4400-4423, pilot 4499). Release 7d8f7dd7.
- **n = 24 paired seeds per block,** de-duplicated by prediction hash.
- **Scorer, fixed before the data.** `analysis/score_smallbb.py` was last changed in 7d8f7dd7, the
  launch release, and ran on dsisco01 at 18:23 from that release. Output:
  `analysis/yuval_smallbb_score.txt`. The ensemble confirmation is `analysis/yuval_ensemble.py`,
  unchanged since e45a1bca (`analysis/yuval_{mn3,rgy}_ensemble.txt`). The swaps are
  `analysis/yuval_swaps.py` (`analysis/yuval_{mn3,rgy}_swaps.txt`).

**Finding: the MobileNetV3 positive does not survive Yuval's pipeline, and RegNetY misses Holm. The
snapshot ensemble confirms on both, and gains more than the step on its point estimate.**
- MobileNetV3 P2 is **+0.37 [-0.30, +1.03]** (Holm 0.267). An effect the size of v3's +1.47 is
  excluded.
- RegNetY P2 is **+0.55 [+0.04, +1.06]** (Holm 0.073).
- ENS - best is **+1.24 [+0.51, +1.96]** on MobileNetV3 and **+1.56 [+0.45, +2.66]** on RegNetY.

## Integrity

- **Runs.** All 48 study seeds exited 0 on release 7d8f7dd7; the launcher finished starting them at
  17:34. In each block the scorer found 24 of 24 seeds, with none missing, incomplete or duplicated.
- **Pilot gate.** Both pilots passed: "seed4399: MobileNetV3, best 6/11, step applied 110 -> 76;
  seed4499: RegNet, best 12/17, step applied 116 -> 76; PILOT GATE PASSED". No pilot score was read.
- **Built-in checks.** The runner raises unless the restored best weights reproduce the best epoch's
  output, and unless the sham radius equals the target radius; all 48 seeds exited 0. The scorer
  raises unless each seed was trained on its block's backbone, and unless every arm fills exactly 76
  grade-3 slots and matches its report. It does not check where each study seed's applied step
  landed; the gate checked that on the two pilots.
- **Cap.** pto's hard count averages 91.3 (MobileNetV3) and 98.0 (RegNetY). The cap binds in 22 of 24
  and 23 of 24 seeds. The others (4300 at 76, 4320 at 67, 4414 at 71) have a step radius of 0.
- **Sham.** It changes cc-F1 in two MobileNetV3 seeds (4309: 71.43 to 72.53; 4313: 72.53 to 71.43)
  and one RegNetY seed (4418: 73.63 to 72.53).
- **Not measured.** Determinism was not re-checked by re-training. Whether these runs memorise is not
  measured: the outputs print no training-set accuracy.

## Primary: P2 on cc-F1 of grade 3 under capped_first

n = 24 paired seeds per block, intent to treat, Holm over the two blocks.

| Block | P2 tralo_final - sham_final, points [95% t CI] | sd | p | Holm |
|---|---|---|---|---|
| MobileNetV3 | +0.37 [-0.30, +1.03] | 1.58 | 0.267 | 0.267 |
| RegNetY | +0.55 [+0.04, +1.06] | 1.21 | 0.037 | 0.073 |

- **In slots of 76:** MobileNetV3 +0.34 [-0.27, +0.94], RegNetY +0.50 [+0.04, +0.96]. Converted here
  by dividing the printed points by 1.0989, the cc-F1 value of one slot (F1 = 2TP/(76 + 106)).
- **Precision.** The power statement used the v3 MobileNetV3 sd, about 2.45; here it is 1.58 (RegNetY 1.21).

**P2 in all four Yuval-pipeline blocks** (the prereg's one table, with no pooled claim):

| Block | Seeds | P2 in Yuval's pipeline [95% t CI] | Holm (family) | Same contrast, v3 recipe |
|---|---|---|---|---|
| ResNet18 | 4000-4023 | +0.73 [+0.08, +1.38] | 0.088 (P1-P3) | +0.14 [-1.21, +1.48] |
| EfficientNet-B5 | 4100-4123 | -0.50 [-1.35, +0.34] | 0.458 (P1-P3) | none cited |
| MobileNetV3 | 4300-4323 | +0.37 [-0.30, +1.03] | 0.267 (two blocks) | +1.47 [+0.43, +2.50], Holm 0.016 (0.023 in the rescore family) |
| RegNetY | 4400-4423 | +0.55 [+0.04, +1.06] | 0.073 (two blocks) | -0.09 |

The ResNet18 and B5 rows and the ResNet18 v3 figure are from [the Yuval-pipeline result](claude_yuval_pipeline_result_20260927.md).
Not one of the four blocks: the recipe factorial's all-on cell repeats the ResNet18 pipeline on seeds
4200-4223, with P2 +0.78 [+0.24, +1.32], unadjusted ([result](claude_recipe_factorial_result_20260927.md)).

## Secondary (no family claim)

**P3 and Yuval's metrics, all 24 seeds:**

| Contrast | cc-F1 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| MobileNetV3 P2 | +0.37 [-0.30, +1.03] | +0.10 [-0.12, +0.31] | -0.18 [-0.46, +0.09] | -0.25 [-0.52, +0.01] |
| MobileNetV3 P3 tralo_final - pto | +0.37 [-0.26, +0.99] | +0.09 [-0.13, +0.31] | -0.17 [-0.46, +0.11] | -0.26 [-0.54, +0.02] |
| RegNetY P2 | +0.55 [+0.04, +1.06] | +0.24 [-0.26, +0.74] | -0.39 [-0.85, +0.07] | -0.58 [-1.21, +0.05] |
| RegNetY P3 tralo_final - pto | +0.50 [+0.01, +1.00] | +0.25 [-0.25, +0.75] | -0.38 [-0.85, +0.09] | -0.57 [-1.20, +0.06] |

- Apart from the primary itself, only RegNetY's P3 on cc-F1 has an interval that excludes 0 (p 0.046).
- The step's point estimates on macro-F1 and weighted-F1 are negative on both backbones. In the
  Yuval-pipeline study they were -0.19 and -0.40 on ResNet18, and -1.87 and -1.97 on B5; only B5's
  intervals excluded 0.

**Binding seeds only,** cc-F1:

| Block | n | P2 | P3 |
|---|---|---|---|
| MobileNetV3 | 22 | +0.40 [-0.33, +1.13], p 0.268 | +0.40 [-0.28, +1.08], p 0.236 |
| RegNetY | 23 | +0.57 [+0.04, +1.11], p 0.036 | +0.53 [+0.01, +1.04], p 0.045 |

On the binding seeds, Yuval's metrics keep their all-seed signs, and every interval covers 0.

**Snapshot ensemble, confirmation sets 6 and 7.** capped_first on the mean of pto's snapshots from
epoch max(1, best - 2) to the last (8.0 snapshots on average in each block), against pto's restored
best epoch:

| Metric | MobileNetV3 ENS - best | p | RegNetY ENS - best | p |
|---|---|---|---|---|
| cc-F1 | +1.24 [+0.51, +1.96] | 0.002 | +1.56 [+0.45, +2.66] | 0.008 |
| Accuracy | -0.75 [-1.52, +0.02] | 0.057 | -0.06 [-0.52, +0.41] | 0.809 |
| Macro-F1 | +0.77 [+0.17, +1.37] | 0.015 | +1.43 [+0.87, +1.98] | 0.000 |

- In slots of 76 (converted as above): +1.13 [+0.46, +1.78] and +1.42 [+0.41, +2.42].
- On MobileNetV3 the ensemble does not raise accuracy (-0.75, interval covering 0). On ResNet18 and
  B5 in this pipeline it did (+1.32 and +2.85).

**Recipe: pto against the v3 clipper of the same backbone** (unpaired Welch, n 24 vs 24):

| Metric (%) | MobileNetV3 | v3, 3001-3024 | Diff | RegNetY | v3, 3101-3124 | Diff |
|---|---|---|---|---|---|---|
| cc-F1 | 71.98 (sd 2.02) | 64.70 (sd 3.64) | +7.28 | 71.70 (sd 2.94) | 66.90 (sd 2.28) | +4.81 |
| Accuracy | 62.65 (sd 1.27) | 56.53 (sd 2.88) | +6.12 | 61.56 (sd 1.60) | 56.98 (sd 2.30) | +4.58 |
| Macro-F1 | 64.04 (sd 1.64) | 58.24 (sd 2.12) | +5.80 | 62.78 (sd 1.60) | 58.35 (sd 2.00) | +4.43 |
| Weighted-F1 | 62.51 (sd 1.36) | 56.32 (sd 1.88) | +6.19 | 61.09 (sd 1.45) | 56.13 (sd 1.73) | +4.96 |

Every p prints as 0.0000. RegNetY's +4.81 is its own measurement; it matches ResNet18's +4.81 only
in value. Which component carries the gain on these backbones is not measured. The recipe factorial
split it on ResNet18 only.

**Diagnostics (listed in the prereg, which calls them post hoc).**
- **Swaps** of the top-76 set by p3, development labels read offline:

  | | MobileNetV3 | RegNetY |
  |---|---|---|
  | Slots the step swaps | 44 | 73 |
  | Brought in: true grade 3 | 31/44 = 0.705 | 50/73 = 0.685 |
  | Pushed out: true grade 3 | 23/44 = 0.523 | 39/73 = 0.534 |
  | Correct-direction share | 0.591 | 0.575 |
  | Net slots (per seed) | +8 (+0.33) | +11 (+0.46) |
  | Sham: slots swapped, net | 2, +0 | 4, -1 |

  On ResNet18 the share was 0.629 (net +16); on B5 it was 0.477 (net -9).
- **P2 against the excess over the cap,** binding seeds: Spearman +0.43 (p 0.047, n 22) on
  MobileNetV3 and +0.35 (p 0.103, n 23) on RegNetY. It was 0.49 (p 0.024) on ResNet18 and -0.03 on
  B5, both post hoc. The recipe factorial found it prospectively on ResNet18.
- **Epochs.** The best epoch averages 8.2 of 13.2 run on MobileNetV3, and 10.2 of 15.2 on RegNetY.

## Readings, against the readings fixed before the data

- **MobileNetV3: "P2 null."** +0.37 [-0.30, +1.03], Holm 0.267. By the prereg, the v3 positive is
  recipe-bound, and the claim "a small attributable who-signal on smaller backbones" is scoped to the
  memorising v3 recipe. The upper CI, +1.03, is below +1.47, so an effect the size of the v3 one is
  excluded here. The Holm-positive reading, with its check against ENS - best, does not arise. Set
  beside anyway, ENS - best is +1.24 against P2's +0.37.
- **RegNetY: "A null is the expected replication."** P2 is not Holm-positive (0.073), so it is not
  the new result the prereg describes. Its unadjusted interval is above 0, as the ResNet18 block's
  was (Holm 0.088). Under the prereg that is not a finding.
- **"Every case."** P2 in the four blocks is in the one table above, with no pooled claim. None is
  Holm-positive in its own family.
- **Snapshot ensemble: "a CI above 0 confirms."** Both cc-F1 intervals are above 0, so sets 6 and 7
  confirm.
- **Recipe.** The prereg fixes no reading. The pipeline's pto beats the v3 clipper of the same
  backbone on all four metrics (unpaired).

## What it means

- **The one v3 positive is recipe-bound.** In a pipeline that lifts MobileNetV3's clipper by +7.28
  cc-F1, the step adds +0.37 over its sham, and an effect the size of v3's +1.47 is excluded. So "a
  small attributable who-signal on smaller backbones" is scoped to the memorising v3 recipe. That
  these runs do not memorise is the prereg's premise, not measured here.
- **Where the step stands in Yuval's pipeline.** On ResNet18 it is attributable across eight
  recipes: the factorial's P2-pooled is +0.72 [+0.46, +0.97] (Holm 0.000). On RegNetY it is positive
  and misses Holm; on MobileNetV3 it is null; on B5 it is -0.50 and costs about 2 points of macro-F1
  and weighted-F1. Its point estimates on macro-F1 and weighted-F1 are negative in all four blocks
  (pooled over the factorial's eight recipes on ResNet18 they are +0.01 and -0.17).
- **The post-hoc bar is the ensembled clipper.** On both backbones ENS - best is larger than P2 on
  the point estimates (+1.24 against +0.37; +1.56 against +0.55). A paired test of tralo_final
  against the ensembled clipper is not computed; it needs per-seed tralo_final minus the ensembled
  clipper's cc-F1. Whether the step adds on top of the ensemble is not measured here; the step-ensemble
  study ([prereg](claude_stepens_prereg_20260927.md)) measures it on ResNet18.
- **The recipe gain is again a gain for the clipper:** +7.28 and +4.81 cc-F1 over v3 (unpaired). It
  is not evidence for a constraint loss.

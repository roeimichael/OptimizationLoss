# Result: TraLO and Yuval Kassif's PAO inside Yuval's pipeline (knee, cap 76)

Preregistration: [claude_yuval_pipeline_prereg_20260927.md](claude_yuval_pipeline_prereg_20260927.md).
Scorer: `analysis/score_yuval.py`, run on dsisco01 after each block's 24 seeds finished. Outputs:
`analysis/yuval_r18_score.txt` and `analysis/yuval_b5_score.txt`.

## Both blocks

- **Yuval's loss does not beat PTO on either backbone.** pao - pto is -0.09 [-2.85, +2.66] cc-F1 on
  ResNet18 and +0.96 [-0.28, +2.20] on EfficientNet-B5, his backbone (Holm 0.95 and 0.37).
- **TraLO's step is not attributable on either.** tralo_final - sham_final is +0.73 [+0.08, +1.38]
  (Holm 0.088) on ResNet18 and -0.50 [-1.35, +0.34] (Holm 0.46) on B5. On B5 it also costs about 2
  points of macro-F1 and weighted-F1.
- **His pipeline helps every method.** It adds about 5 cc-F1 points on either backbone. The
  snapshot ensemble adds a further +2.43 (ResNet18) and +4.08 (B5, preregistered). In a post-hoc
  comparison it beats both constraint losses on both backbones.

## ResNet18 block (seeds 4000-4023, release bcf5d010), scored 2026-09-27 11:07

**Finding: Yuval's loss does not beat PTO; his pipeline does.**
- pao - pto is **-0.09 cc-F1 [-2.85, +2.66]** (Holm 0.95), and accuracy -1.08 [-2.44, +0.28].
- PTO inside his pipeline beats our recipe's clipper by **+4.81 cc-F1 and +3.60 accuracy** (unpaired, p < 1e-4).
- TraLO's step is +0.73 [+0.08, +1.38] over its sham, which misses the Holm threshold (0.088).

### Integrity

- **Runs.** All 24 seeds exited 0 on release bcf5d010.
- **Built-in checks.** The runner raises unless each seed's retrains share the first-epoch sampler order and augmentation draws, and unless every restored best checkpoint reproduces its epoch output. All 21 applied steps landed exactly on 76. Each sham used the target radius and moved the count by at most 1. No two seeds share a prediction vector.
- **Helper queues.** Five seeds (4011, 4012, 4018, 4020, 4021) were listed a second time in two helper queues added on GPU0 at 10:18, with the same release and configs. A queue skips any seed whose directory exists. Across all queue logs there are 24 starts, 24 exit-0 ends and 5 skips, so each seed ran exactly once.
- **Determinism.** Seeds 4001, 4004 and 4006 were re-trained in separate processes and reproduced the study's PTO probabilities byte for byte (`analysis/grad_alignment_out/`).

### Primary: cc-F1 of grade 3 under capped_first

n = 24 paired seeds, intent to treat, Holm over P1-P3.

| Contrast | Mean [95% t CI], points | sd | p | Holm |
|---|---|---|---|---|
| P1 pao - pto | -0.09 [-2.85, +2.66] | 6.52 | 0.946 | 0.946 |
| P2 tralo_final - sham_final | +0.73 [+0.08, +1.38] | 1.54 | 0.029 | 0.088 |
| P3 tralo_final - pto | +0.73 [+0.08, +1.38] | 1.54 | 0.029 | 0.088 |

The sham step leaves cc-F1 unchanged in 24 of 24 seeds (it swaps one slot in one seed, seed 4016, with no
effect on cc-F1), so P2 equals P3. On the 21 seeds where the cap binds, P2 is +0.84 [+0.10, +1.58] and P1 is -0.10.

### Yuval's own metrics (secondary)

| Contrast | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| P1 pao - pto | -1.08 [-2.44, +0.28] | -0.19 [-1.87, +1.48] | -0.67 [-2.14, +0.80] |
| P2 tralo_final - sham_final | +0.12 [-0.18, +0.42] | -0.19 [-0.71, +0.33] | -0.40 [-0.85, +0.06] |
| P3 tralo_final - pto | +0.15 [-0.18, +0.48] | -0.17 [-0.71, +0.36] | -0.37 [-0.84, +0.11] |

### How PAO behaved

- **Live, not inert.** At PTO's best epoch there were 97-231 training false positives per epoch (median 162); these are the items PAO up-weights. The median training loss there was 0.63. The null is not a dead signal.
- **Converged 24/24 within 8 retrains.**

  | Retrains | 1 (cap not binding) | 2 | 3 | 4 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | Seeds | 3 | 9 | 3 | 5 | 2 | 1 | 1 |

- **It overshoots.** The pool grade-3 count falls from a PTO mean of 95.2 to a final mean of 65.8, against the cap of 76.
- **It steers on noise.** Within one retrain, the pool count's epoch-to-epoch sd (epochs 3 onward) has a median of 19.3 (range 8.7-32.0); consecutive retrains differ by a median of 12 (max 75). For example, seed 4003 needed 8 retrains, with counts 91, 96, 84, 86, 88, 79, 80, 68 while C rose from 1 to 4.84 (`analysis/yuval_r18_count_noise.txt`).
- **Each retrain is a fresh draw.** Per seed, PAO - PTO runs from +12.1 (seed 4012) to -25.3 (seed 4023); the paired mean is null.

### Recipe effect (secondary, unpaired)

This study's pto against the v3 ResNet18 clipper (seeds 1801-1824, same pool, same cap), Welch t:

| Metric (%) | Yuval's pipeline | Our v3 recipe | Difference | p |
|---|---|---|---|---|
| cc-F1 | 69.78 (sd 3.07) | 64.97 (sd 3.29) | +4.81 | < 1e-4 |
| Accuracy | 62.45 (sd 1.31) | 58.85 (sd 2.59) | +3.60 | < 1e-4 |
| Macro-F1 | 62.83 (sd 2.22) | 59.48 (sd 1.65) | +3.35 | < 1e-4 |
| Weighted-F1 | 61.25 (sd 1.82) | 57.67 (sd 1.75) | +3.58 | < 1e-4 |

Caveat: the comparison is unpaired and the recipes differ in several components at once:
augmentation, balanced sampling, early stopping with the best checkpoint restored, weight decay and
LR decay, normalisation, and a 10% early-stop carve. The recipe factorial
([prereg](claude_recipe_factorial_prereg_20260927.md), queued) splits the first three.

### Exploratory (not preregistered for this study)

**Snapshot ensemble on top of the pipeline.** capped_first on the mean of retrain 1's development
snapshots from epoch best-2 to the stopping epoch, against PTO's restored best epoch:

| Metric | ENS - best [95% t CI] | p |
|---|---|---|
| cc-F1 | +2.43 [+0.82, +4.03] (+2.2 slots of 76) | 0.005 |
| Accuracy | +1.32 [+0.65, +1.98] | < 0.001 |
| Macro-F1 | +3.12 [+2.04, +4.19] | < 0.001 |

Source: `analysis/yuval_r18_ensemble.txt`. This matches the ensemble finding in our recipe, which was
found post hoc on two seed sets and confirmed, preregistered, on four fresh ones across three backbones
and both caps (+1.0 to +2.9 slots for the clipper and tralo_null).
**The best system measured today is post hoc: Yuval's pipeline, then the snapshot ensemble, then
capped_first.**

**On ResNet18, TraLO's step gain grew with how far PTO is over the cap. B5 does not replicate
it** (Spearman -0.03, p 0.914; B5 block below). Per seed, P2 against PTO's excess (argmax grade-3
count - 76), over the 21 binding seeds:

| Excess | Seeds | Mean P2 | In slots |
|---|---|---|---|
| 1-19 | 10 | +0.11 | +0.1 |
| 20-29 | 4 | +0.27 | +0.25 |
| 30-48 | 7 | +2.20 | +2.0 |

Spearman 0.49 (p 0.024), Pearson 0.61 (p 0.003); against the step radius, Spearman 0.39 (p 0.081). Computed on
exact per-seed values (`analysis/step_dose_across_backbones.txt`); rounded values had given 0.53.
This is post hoc; the recipe factorial tests it prospectively (its amendment 1, committed before any
of its jobs started). Across the v3-recipe studies the pattern is mixed: MobileNetV3 has the largest
excess (median 26) and the only positive there (+1.47), but RegNetY (19, -0.09) is out of order, and no
within-study correlation there is significant (-0.15, +0.17, +0.25).

**What the step's swaps are** (top-76 sets by p3, development labels offline; `analysis/yuval_r18_swaps.txt`).
On B5 the direction does not replicate: share 0.477, net -9 slots (B5 block below).
- TraLO's step swaps 62 slots over 24 seeds (0-7 per seed).
  - The items it brings in are 71.0% true grade 3; the items it pushes out are 45.2%.
  - That is a correct-direction share of 0.629, against 0.50 for CUTPAIR's hinge, and a net +16 slots.
- The same-radius sham swaps 1 slot in total (seed 4016), which leaves cc-F1 unchanged.
- Treating swaps as independent gives a binomial p of 0.005. They are not independent within a seed,
  so the honest test is the per-seed P2 above (p 0.029, Holm 0.088).
- A deeper push is not the way to use this: in our recipe, every depth past the cap cost slots (LEDGER #10).

### Reading, against the readings fixed before the data

- **P1 is null.** His reported PAO > PTO does not reproduce under paired, label-clean conditions. With n = 24 the interval excludes gains above 2.7 points. His paper's advantage would rest on his evaluation artefacts or on his backbone:
  - a single seed;
  - a cap and a stopping rule that read the test set;
  - no reseed in `full_experiment.py`;
  - outcome filtering.

  The B5 block tests the backbone. P1 is null there too (below).
- **P2 is not Holm-positive.** TraLO's direction has no attributable effect that survives correction in his pipeline either. It is directionally positive: +0.73 points, about 0.7 correct slots out of 76. In our recipe the same contrast was +0.14 [-1.21, +1.48] on ResNet18 and +1.47 [+0.43, +2.50] on MobileNetV3, the one attributable positive so far (LEDGER #9). On B5 in his pipeline it is -0.50 (below).
- **The recipe effect is positive.** That is a practical gain for every method, including the post-hoc clipper. It is not evidence for any constraint loss.

**Why PAO cannot help under the cap (post hoc, `analysis/yuval_ranking.py`, output
`analysis/yuval_r18_ranking.txt`).** capped_first fills the 76 slots with the top 76 items by p3, so
only the order of p3 can matter. PAO's loop steers the argmax count, which the cut already fixes.

| pao - pto, n = 24 | Mean [95% CI] | p |
|---|---|---|
| AUC of p3, grade 3 vs rest | -0.006 [-0.013, +0.001] | 0.11 |
| Precision of the 76 capped slots | -0.1 [-3.4, +3.2] points | 0.95 |
| Raw argmax grade-3 count, no cap | -29.3 [-38.7, -20.0] | < 0.001 |
| Raw argmax grade-3 F1, no cap | -6.6 [-9.9, -3.4] points | < 0.001 |
| Raw argmax accuracy, no cap | -1.8 [-3.2, -0.4] points | 0.017 |

- **PAO moves the count, not the order.** Under the cap, its count change is neutralised.
- **Without a cap it is worse.** Its overshoot (65.8 predictions against 106 true grade-3 knees in the pool) costs grade-3 recall.
- **PAO does its job on the training set, and it does not transfer.** Retrain 2 (the first with C > 1) against PTO at the same epoch, 21 seeds (`analysis/yuval_r18_retrain_speed.txt`): 28-82 fewer live training false positives per epoch over epochs 1-9 (epoch 1: -82 [-141, -22]). Yet development accuracy is 0.1-2.0 points lower at each of epochs 1-9, significantly at epochs 1, 2 and 4 (epoch 1: -1.8 [-3.2, -0.5]). Fitting the training false positives harder does not reorder unseen images.
- **TraLO's step is the opposite.** It raises the precision of the capped slots by +0.9 [+0.1, +1.7] points (p 0.029), at a slightly lower AUC (-0.0014, p 0.028). It reorders items near the cut, not the whole ranking.

**Mechanism, for context.**
- In the lab, the step's direction cannot separate right from wrong items even with oracle pool labels (`analysis/lab_pao/fp_lab_out.txt`).
- On these real models, TraLO's count gradient has cosine 0.95 with the gradient over true grade-3 knees and 0.99 with the gradient over the rest (`analysis/grad_alignment_out/`).

## EfficientNet-B5 block (seeds 4100-4123, release 0d70d993), scored 2026-09-27 13:40

Yuval's backbone: timm `efficientnet_b5`, weights `sw_in12k_ft_in1k` (prereg Amendment 1). The
runner, pipeline, arms, cap and endpoints are those of the ResNet18 block. P1-P3 form their own Holm
family.

**Finding: on his own backbone, his loss still does not beat PTO. TraLO's step is null on cc-F1 and
lowers his other metrics. The post-hoc snapshot ensemble beats both.**
- pao - pto is **+0.96 cc-F1 [-0.28, +2.20]** (Holm 0.37). The sign matches his report, but the
  interval covers 0.
- tralo_final - sham_final is **-0.50 [-1.35, +0.34]** (Holm 0.46).
- The snapshot ensemble adds **+4.08 [+3.04, +5.11]** cc-F1 to PTO, which confirms Amendment 2.
- The backbone buys accuracy (+1.58 over ResNet18 in the same pipeline, p 0.0002) but not cc-F1
  (+0.18, p 0.81).

### Integrity

- **Runs.** 24 starts and 24 exit-0 ends across the 8 queue logs (queue tooling from 00fde635).
  Every seed's recorded `tralo/` source hashes equal release 0d70d993's.
- **Built-in checks.** The runner checks common random numbers and the best-weight restore, as in the
  ResNet18 block.
  - All 21 applied steps landed exactly on 76.
  - Three seeds do not bind: 4103, 4117 and 4118, with PTO hard counts 72, 76 and 61.
- **Sham.** Each sham used the target radius. It moved the hard count by -4 to +1, and by 0 in 11 of
  21 seeds. On ResNet18 it moved the count by at most 1.
- **Duplicates.** No probability vector is shared within an arm across the 48 seeds of both blocks
  (144 vectors checked).
- **Determinism.** Not re-checked by re-training on B5. On ResNet18 three seeds reproduced byte for
  byte.

### Primary: cc-F1 of grade 3 under capped_first

n = 24 paired seeds, intent to treat, Holm over P1-P3 within this block.

| Contrast | Mean [95% t CI], points | sd | p | Holm |
|---|---|---|---|---|
| P1 pao - pto | +0.96 [-0.28, +2.20] | 2.94 | 0.123 | 0.368 |
| P2 tralo_final - sham_final | -0.50 [-1.35, +0.34] | 2.00 | 0.229 | 0.458 |
| P3 tralo_final - pto | -0.41 [-1.25, +0.43] | 1.99 | 0.322 | 0.458 |

- **Binding seeds only (n = 21).** P1 is +1.10 [-0.32, +2.52] and P2 is -0.58 [-1.55, +0.39].
- **Power.** With sd 2.94 and n = 24, a two-sided test at 0.05 has 80% power only for an effect of
  about 1.75 points (1.6 slots). P1's +0.96 is below that. So P1 is "not enough measurement for a
  one-slot effect", not "no effect".

### Yuval's own metrics (secondary, no family claim)

| Contrast | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| P1 pao - pto | -0.12 [-1.07, +0.84] | -0.40 [-1.73, +0.92] | -0.81 [-2.07, +0.45] |
| P2 tralo_final - sham_final | -0.94 [-1.77, -0.11] | -1.87 [-3.19, -0.55] | -1.97 [-3.05, -0.88] |
| P3 tralo_final - pto | -0.97 [-1.80, -0.15] | -1.84 [-3.10, -0.57] | -1.99 [-3.08, -0.90] |

TraLO's step costs about 2 points of macro-F1 and weighted-F1 on B5 (unadjusted p 0.007 and 0.001).
On ResNet18 the same contrasts were -0.19 and -0.40, neither significant.

### How PAO behaved

- **Converged 24/24 within 6 retrains.**

  | Retrains | 1 (cap not binding) | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Seeds | 3 | 10 | 5 | 4 | 1 | 1 |

- **It overshoots.** The pool grade-3 count falls from a PTO mean of 99.1 to a final mean of 62.4,
  against the cap of 76. The final C (non-capped classes) averages 2.85.
- **It steers on noise.** Within one retrain, the pool count's epoch-to-epoch sd (epochs 3 onward) has
  a median of 19.5 (range 7.6-56.4). Consecutive retrains differ by a median of 18 (max 65)
  (`analysis/yuval_b5_count_noise.txt`).
- **It fits the training false positives, and that does not transfer.** Retrain 2 against retrain 1 at
  the same epoch, 21 seeds (`analysis/yuval_b5_retrain_speed.txt`):
  - 130 [111, 149] fewer live training false positives at epoch 1, and 20-78 fewer at epochs 2-9;
  - development accuracy is unchanged at every epoch (every interval covers 0).
- **PTO stops early.** Its best epoch averages 5.0 of 10.0 epochs run.

### Ranking (post hoc, `analysis/yuval_b5_ranking.txt`)

| Contrast, n = 24 | Mean [95% CI] | p |
|---|---|---|
| pao - pto: AUC of p3, grade 3 vs rest | -0.0004 [-0.0069, +0.0061] | 0.90 |
| pao - pto: precision of the 76 capped slots | +1.15 [-0.34, +2.64] points | 0.12 |
| pao - pto: raw argmax grade-3 count, no cap | -36.7 [-45.6, -27.8] | < 0.001 |
| pao - pto: raw argmax grade-3 F1, no cap | -8.2 [-10.9, -5.4] points | < 0.001 |
| pao - pto: raw argmax accuracy, no cap | -1.3 [-2.4, -0.2] points | 0.023 |
| tralo_final - sham_final: AUC of p3 | -0.0052 [-0.0098, -0.0006] | 0.029 |
| tralo_final - sham_final: precision of the capped slots | -0.60 [-1.61, +0.41] points | 0.23 |

- **PAO still moves the count, not the order.** Its capped-slot gain (+1.15 points, p 0.12) is the
  whole of P1. The same gain on ResNet18 was -0.1.
- **TraLO's step swaps more, and in the wrong direction** (`analysis/yuval_b5_swaps.txt`).
  - It swaps 199 slots over 24 seeds, against 62 on ResNet18.
  - The items it brings in are 63.8% true grade 3 (127/199); the items it pushes out are 68.3%
    (136/199).
  - That is a correct-direction share of 0.477, net -9 slots (-0.38 per seed). On ResNet18 the share
    was 0.629, net +16.
  - The sham swaps 14 slots, net +2.
- **The dose pattern does not replicate.** P2 against PTO's excess over the cap has Spearman -0.03
  (p 0.914) over the 21 binding seeds, with a median excess of 23. On ResNet18 it was +0.49 (p 0.024).
  Both are post hoc; the recipe factorial tests the pattern prospectively (its amendment 1).

### Recipe and backbone (secondary, unpaired)

| Metric (%) | B5, Yuval's pipeline | ResNet18, Yuval's pipeline | ResNet18, our v3 recipe |
|---|---|---|---|
| cc-F1 | 69.96 (sd 2.17) | 69.78 (sd 3.07) | 64.97 (sd 3.29) |
| Accuracy | 64.03 (sd 1.35) | 62.45 (sd 1.31) | 58.85 (sd 2.59) |
| Macro-F1 | 63.89 (sd 1.68) | 62.83 (sd 2.22) | 59.48 (sd 1.65) |
| Weighted-F1 | 63.22 (sd 1.58) | 61.25 (sd 1.82) | 57.67 (sd 1.75) |

- **B5 against the v3 clipper:** +4.99 cc-F1 and +5.18 accuracy (Welch p < 1e-4 on every metric).
- **B5 against ResNet18 in the same pipeline:** +0.18 cc-F1 (p 0.81), +1.58 accuracy (p 0.0002),
  +1.06 macro-F1 (p 0.069). The backbone buys overall accuracy, not grade-3 F1 under the cap.
- **Against his reported accuracies.** Our B5 PTO reaches 64.4% raw accuracy on the development pool.
  His hard-coded unconstrained accuracies are 66-69%, but on his test split, with a stopping rule that
  reads the test loss. That gap is therefore not attributable.
- **Caveat.** These are three different seed sets, and the v3 clipper uses a different recipe.

### Snapshot ensemble (Amendment 2, confirmatory for this block)

capped_first on the mean of retrain 1's snapshots from epoch max(1, best - 2) to the last epoch
(7.8 snapshots on average), against PTO's restored best epoch (`analysis/yuval_b5_ensemble.txt`):

| Metric | ENS - best [95% t CI] | p |
|---|---|---|
| cc-F1 | +4.08 [+3.04, +5.11] (+3.7 slots of 76) | < 0.001 |
| Accuracy | +2.85 [+2.25, +3.44] | < 0.001 |
| Macro-F1 | +4.08 [+3.39, +4.77] | < 0.001 |

The interval is above 0. By the reading fixed in Amendment 2, the ensemble gain holds on Yuval's
backbone in Yuval's pipeline, so the post-hoc bar there is the ensembled clipper.

**Exploratory: the ensemble against the constraint methods** (post hoc,
`analysis/yuval_ens_vs_pao.py`; outputs `analysis/yuval_{b5,r18}_ens_vs_pao.txt`). Paired by seed,
with counts of seeds where the ensemble is better / worse:

| Block | ENS - pao, cc-F1 | ENS - tralo_final, cc-F1 | ENS - pao, accuracy |
|---|---|---|---|
| B5 | +3.11 [+1.60, +4.62], 20 / 2 | +4.49 [+3.05, +5.93], 20 / 3 | +2.96 [+2.07, +3.85], 23 / 0 |
| ResNet18 | +2.52 [+0.17, +4.86], 17 / 5 | +1.69 [+0.17, +3.22], 13 / 4 | +2.40 [+1.15, +3.64], 20 / 3 |

The script recomputes pao's and tralo_final's capped_first scores and asserts that they match the
stored ones.

**Exploratory: ensembles across seeds, and PAO at equal compute** (post hoc,
`analysis/yuval_deep_ensemble.py`; outputs `analysis/yuval_{b5,r18}_deep_ensemble.txt`).
- The 24 seeds are cut into disjoint groups of k consecutive seeds. Each group's mean probability is
  deployed with capped_first; the table gives the mean over groups.
- PAO costs one training per retrain: 2.71 per seed on B5 and 3.21 on ResNet18. Its equal-compute
  rival is the 3-model ensemble.

| cc-F1 (accuracy), % | 1 model | ENS of 1 run | 3 models (8 groups) | ENS of 3 runs | ENS of all 24 runs | PAO |
|---|---|---|---|---|---|---|
| B5 | 69.96 (64.03) | 74.04 (66.88) | 73.35 (66.51) | 74.45 (68.11) | 75.82 (68.77) | 70.92 (63.92) |
| ResNet18 | 69.78 (62.45) | 72.21 (63.77) | 72.39 (64.92) | 73.49 (64.91) | 74.73 (65.98) | 69.69 (61.38) |

- **At equal compute, three plain models beat PAO:** by 2.4 cc-F1 on B5 and 2.7 on ResNet18.
- **One run's snapshot ensemble already matches three models,** at a third of the training.
- **Past about 3 runs the gain flattens.** The measured ceiling of this pipeline on this pool is
  about 75-76 cc-F1 and 68-69 accuracy (B5).
- These are group means with no paired test; the group counts fall to 1-2 for k >= 12.

### Reading, against the readings fixed before the data

- **P1 is null on both backbones.** By the prereg reading, the advantage in his paper then rests on
  its evaluation artefacts, not on the backbone:
  - a single seed;
  - a cap and a stopping rule that read the test set;
  - no reseed;
  - outcome filtering.

  B5's point estimate is positive, +0.96 points (0.9 slots), and n = 24 cannot exclude a gain of
  that size. Even if real, it is under a quarter of what the post-hoc ensemble adds on the same
  backbone.
- **P2 is null on both backbones, and harmful on B5's secondaries.** TraLO's direction has no
  attributable cc-F1 effect in his pipeline.
  - On B5 it lowers macro-F1 and weighted-F1 by about 2 points, and its swaps are net wrong-way.
  - The ResNet18 block's +0.73, its swap direction and its dose pattern do not replicate. Read them
    as ResNet18-specific or as noise.
- **The recipe effect is positive on both backbones.** It is about +5 cc-F1 for every method.
- **Amendment 2 is confirmed.** The best system measured today is post hoc: B5 in Yuval's pipeline,
  then the snapshot ensemble, then capped_first. That is about 74.0 cc-F1 (69.96 + 4.08) and 66.9
  accuracy (64.03 + 2.85).

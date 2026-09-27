# Result: TraLO and Yuval Kassif's PAO inside Yuval's pipeline (knee, cap 76)

Preregistration: [claude_yuval_pipeline_prereg_20260927.md](claude_yuval_pipeline_prereg_20260927.md).
Scorer: `analysis/score_yuval.py`, run on dsisco01 after all 24 seeds finished. Output:
`analysis/yuval_r18_score.txt`.

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

The sham step never changed the 76-slot grade-3 set: sham equals pto on cc-F1 in 24 of 24 seeds, so
P2 equals P3. On the 21 seeds where the cap binds, P2 is +0.84 [+0.10, +1.58] and P1 is -0.10.

### Yuval's own metrics (secondary)

| Contrast | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| P1 pao - pto | -1.08 [-2.44, +0.28] | -0.19 [-1.87, +1.48] | -0.67 [-2.14, +0.80] |
| P2 tralo_final - sham_final | +0.12 [-0.18, +0.42] | -0.19 [-0.71, +0.33] | -0.40 [-0.85, +0.06] |
| P3 tralo_final - pto | +0.15 [-0.18, +0.48] | -0.17 [-0.71, +0.36] | -0.37 [-0.84, +0.11] |

### How PAO behaved

- **Live, not inert.** At PTO's best epoch there were 97-231 training false positives per epoch (median 162); these are the items PAO up-weights. Training loss was 0.63. The null is not a dead signal.
- **Converged 24/24 within 8 retrains.**

  | Retrains | 1 (cap not binding) | 2 | 3 | 4 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | Seeds | 3 | 9 | 3 | 5 | 2 | 1 | 1 |

- **It overshoots.** The pool grade-3 count falls from a PTO mean of 95.2 to a final mean of 65.8, against the cap of 76.
- **It steers on noise.** Within one retrain, the pool count's epoch-to-epoch sd has a median of 19.3 (range 8.7-32.0); consecutive retrains differ by a median of 12 (max 75). For example, seed 4003 needed 8 retrains, with counts 91, 96, 84, 86, 88, 79, 80, 68 while C rose from 1 to 4.84 (`analysis/yuval_r18_count_noise.txt`).
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
LR decay, normalisation, and a 10% early-stop carve. The factorial that would split them was not run.

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

**TraLO's step gain grows with how far PTO is over the cap.** Per seed, P2 against PTO's excess
(argmax grade-3 count - 76), over the 21 binding seeds:

| Excess | Seeds | Mean P2 | In slots |
|---|---|---|---|
| 1-19 | 10 | +0.11 | +0.1 |
| 20-29 | 4 | +0.27 | +0.25 |
| 30-48 | 7 | +2.20 | +2.0 |

Spearman 0.53 (p 0.013), Pearson 0.61 (p 0.003); against the step radius, Spearman 0.45 (p 0.040).
This is post hoc; the recipe factorial tests it prospectively (its amendment 1, committed before any
of its jobs started). Source: the per-seed lines of `analysis/yuval_r18_score.txt`.

**What the step's swaps are** (top-76 sets by p3, development labels offline):
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

  The B5 block (running) tests the backbone.
- **P2 is not Holm-positive.** TraLO's direction has no attributable effect that survives correction in his pipeline either. It is directionally positive: +0.73 points, about 0.7 correct slots out of 76. In our recipe the same contrast was +0.14 [-1.21, +1.48] on ResNet18 and +1.47 [+0.43, +2.50] on MobileNetV3, the one attributable positive so far (LEDGER #9).
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
- **PAO does its job on the training set, and it does not transfer.** Retrain 2 (the first with C > 1) against PTO at the same epoch, 21 seeds (`analysis/yuval_r18_retrain_speed.txt`): 28-82 fewer live training false positives per epoch (epoch 1: -82 [-141, -22]). Yet development accuracy is 0.5-2 points lower at matched epochs (epoch 1: -1.8 [-3.2, -0.5]; most later CIs cover 0). Fitting the training false positives harder does not reorder unseen images.
- **TraLO's step is the opposite.** It raises the precision of the capped slots by +0.9 [+0.1, +1.7] points (p 0.029), at a slightly lower AUC (-0.0014, p 0.028). It reorders items near the cut, not the whole ranking.

**Mechanism, for context.**
- In the lab, the step's direction cannot separate right from wrong items even with oracle pool labels (`analysis/lab_pao/fp_lab_out.txt`).
- On these real models, TraLO's count gradient has cosine 0.95 with the gradient over true grade-3 knees and 0.99 with the gradient over the rest (`analysis/grad_alignment_out/`).

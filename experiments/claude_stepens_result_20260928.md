# Result: the step-ensemble studies on ResNet18 and RegNetY (knee, Yuval's pipeline, cap 76)

Scored on dsisco01 with the preregistered scorer of each study:

- **ResNet18:** 2026-09-27 23:26, from release 129134dd. Output: `analysis/stepens_r18_score.txt`.
- **RegNetY:** 2026-09-28 04:22, from release f2580776. Output: `analysis/stepens_rgy_score.txt`.

Preregistrations:

- `claude_stepens_prereg_20260927.md`, committed 18:41:43 on 09-27 (129134dd);
- `claude_stepens_rgy_prereg_20260927.md`, committed 19:06:53 (f2580776), before any ResNet18 study result.

Every number below is copied from the two scorer outputs.

## Design, in one paragraph

Each seed trains PTO once in Yuval's pipeline:

- augmentation and the class-balanced sampler;
- Adam 1e-4 with weight decay 1e-4, LR x0.8 every 5 epochs;
- early stopping with patience 5 on a 10% subject carve of TRAIN, best weights restored.

At every epoch, TraLO's targeted step and its sham act on side copies of that epoch's model:

- the step uses the smallest radius along the soft-count gradient that brings the hard grade-3 count on the
  development images to 76;
- the sham moves the same radius in a seeded random direction;
- the global RNG is restored, so PTO's trajectory is untouched.

Each arm (pto, tralo, sham) is averaged over the snapshot window max(1, best - 2)..last and deployed with
`capped_first` at 76.

The primary is development grade-3 cc-F1, Holm over two:

- E1 = ens_tralo - ens_sham;
- E2 = ens_tralo - ens_pto, the thesis bar.

## Integrity

| | ResNet18 | RegNetY |
|---|---|---|
| Seeds complete | 72 of 72, 0 failures | 72 of 72, 0 failures |
| Pilot gate | byte-identical to stored seed 4000 at all 14 epochs (18:59) | byte-identical to stored seed 4400 at all 18 epochs (20:04) |
| Mean window / stepped | 8.0 / 7.8 (min 6) | 8.0 / 7.9 (min 6) |
| Repeated pto prediction vectors | none | none |

## Primary (cc-F1 points, n = 72 each)

| Contrast | ResNet18 | Slots of 76 | Holm | RegNetY | Slots of 76 | Holm |
|---|---|---|---|---|---|---|
| E1 ens_tralo - ens_sham | +1.10 [+0.77, +1.43] | +1.00 [+0.70, +1.30] | < 0.001 | +0.92 [+0.57, +1.27] | +0.83 [+0.52, +1.15] | < 0.001 |
| E2 ens_tralo - ens_pto | +1.11 [+0.78, +1.45] | +1.01 [+0.71, +1.32] | < 0.001 | +0.93 [+0.57, +1.29] | +0.85 [+0.52, +1.17] | < 0.001 |

The per-seed sd of E1 is 1.42 on ResNet18 and 1.49 on RegNetY.

## Reading, fixed before the data

The RegNetY prereg reads the two studies together. Its first reading applies: E1 and E2 are above 0 in both
studies.

> TraLO plus the snapshot ensemble beats the ensembled clipper on two backbones in Yuval's pipeline. That is the
> thesis claim, reported in slots of 76 per backbone.

That is **+1.01 slots of 76 on ResNet18 and +0.85 on RegNetY.**

Scope:

- the knee task, with grade 3 capped at 76, in Yuval's pipeline;
- the development pool: the test split was never read;
- a transductive step, which reads the development images and never their labels.

## Secondary, Yuval's metrics (no family claim; Holm within each metric)

| Metric | Contrast | ResNet18 | Holm | RegNetY | Holm |
|---|---|---|---|---|---|
| accuracy | E1 | +0.66 [+0.44, +0.88] | < 0.001 | +0.65 [+0.36, +0.93] | < 0.001 |
| accuracy | E2 | +0.66 [+0.43, +0.88] | < 0.001 | +0.64 [+0.35, +0.92] | < 0.001 |
| macro-F1 | E1 | -0.11 [-0.32, +0.10] | 0.551 | -0.05 [-0.34, +0.24] | 1.000 |
| macro-F1 | E2 | -0.12 [-0.33, +0.09] | 0.551 | -0.05 [-0.34, +0.24] | 1.000 |
| weighted-F1 | E1 | -0.25 [-0.47, -0.03] | 0.049 | -0.51 [-0.83, -0.20] | 0.003 |
| weighted-F1 | E2 | -0.25 [-0.48, -0.03] | 0.049 | -0.52 [-0.84, -0.20] | 0.003 |

What this means:

- The step buys grade-3 F1 and overall accuracy.
- It pays a little weighted-F1: the other grades lose some F1. On RegNetY that cost is about twice ResNet18's.
- Macro-F1 does not move measurably. That is "not significant", not "no effect".

## Other secondaries (cc-F1, no family claim)

| Contrast | ResNet18 | RegNetY |
|---|---|---|
| P2 tralo_final - sham_final (single model) | +1.07 [+0.64, +1.50] | +0.73 [+0.39, +1.07] |
| ens_pto - pto (ensemble confirmation; set 8 on ResNet18, set 9 on RegNetY) | +2.24 [+1.56, +2.93] | +1.48 [+0.90, +2.06] |
| ens_sham - ens_pto (a random move of the same size, ensembled) | +0.02 [-0.02, +0.05] | +0.02 [-0.04, +0.07] |
| dose: Spearman of E1 against the window's mean excess over the cap | -0.022 (p 0.85) | -0.055 (p 0.65) |

The RegNetY scorer labels its ensemble confirmation "set 8", because it is the ResNet18 scorer's text. The RegNetY
prereg numbers it set 9.

## What it changes

- **The bar.** The ensembled clipper was the bar the training-time loss never cleared. TraLO's step, ensembled,
  clears it on both preregistered backbones.
- **What adds the gain.** The snapshot ensemble on its own adds +2.24 and +1.48 over single models. The ensembled
  step adds about one slot on top of that, and a random move of the same size adds nothing.
- **Generality.** Three blocks now test it, each preregistered and read on its own:
  - `claude_stepens_d2_prereg_20260928.md`: MobileNetV3 and EfficientNet-B5 on dsisco02;
  - `claude_fmow_stepens_prereg_20260928.md`: fmow2 satellite images on dsisco01.
- **Not done.** No confirmation on held-out data, the knee test split or fmow2's reserved countries, has been run.
  That is a decision for the user.

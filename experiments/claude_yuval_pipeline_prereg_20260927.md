# Preregistration: TraLO and Yuval Kassif's PAO inside Yuval's training pipeline (knee, cap 76)

Written 2026-09-27, before any study or pilot output exists. Runner: `tralo/knee_yuval.py`.
Seeds 4000-4023 (study), 4099 (pilot, non-study). ResNet18, grade-3 cap 76 on the 826-image
development pool (70% of the 108 grade-3 cases expected from training prevalence).

## Why

The user asked whether anything in Yuval Kassif's repository
(github.com/YuvalKassif/ConstrainedClassification @ 413d96c) explains why his loss beats
predict-then-optimize (PTO) while TraLO does not. Three findings frame this study:

1. **His loss uses training labels; ours does not.** `losses.CustomLoss` is cross-entropy in
   which every training item whose argmax is the capped class k is weighted by C of its TRUE
   class, with C_k = 1. So only false positives of k are up-weighted. The outer loop
   retrains from scratch and grows C until the argmax count of k meets the cap. TraLO's count
   direction is label-free and demotes true and false positives alike (LEDGER #10).
   Up-weighting training false positives can only act while training false positives exist.
2. **His recipe does not memorise; ours does.** Our recipe (10 epochs, no augmentation,
   uniform batches, final epoch) memorises by epoch 5. That is the dead CUTPAIR gate
   (training-label information at the cut is exhausted). His recipe has augmentation on every
   epoch, a class-balanced sampler, weight decay, LR x0.8 every 5 epochs, and early stopping
   (patience 5) with the best weights restored.
3. **His reported gain is weakly evidenced** (independent audit, 2026-09-27):
   - one seed;
   - the cap and the stopping rule read the test labels and the test-set count;
   - `full_experiment.py` never reseeds between retrains;
   - `weights_impact.py` keeps only runs where PAO beats PTO;
   - no results are committed.

   Our allocator and his are identical: 0 differing items on 300 synthetic matrices and 72
   stored knee runs.

This study runs his pipeline and his outer loop under our rules: paired seeds, a label-free
cap, and development labels used only offline. It then asks whether PAO beats PTO, and
whether TraLO does better in his pipeline than in ours.

## Design (fixed)

**Pipeline.** Every model in this study is trained this way, identically:

| Setting | Value |
|---|---|
| Transforms | Yuval's RGB transforms: resize 224, horizontal flip 0.5, rotation 3, affine translate 0.1 / scale 0.9-1.1, colour jitter 0.2 |
| Normalisation | mean 0.6613, std 0.2123 |
| Sampler | class-balanced, with replacement, one epoch = n draws |
| Optimiser | Adam lr 1e-4, weight decay 1e-4, batch 32 |
| LR schedule | base LR x0.8 every 5 epochs; his per-batch dynamic LR `(1-t) base/mean(C) + t base` |
| Training length | at most 75 epochs; early stopping patience 5 on the loss of a split carved from TRAIN |
| Checkpoint | best weights restored |

The early-stop split holds the subjects whose sha256(ID) is 0 mod 10, with both knees
together. That is about 10% of training; the remaining 90% trains.

Weight decay is part of Yuval's pipeline and every arm and every retrain receives it. It is
not a TraLO lever.

**Outer loop (PAO).** Retrain 1 uses C = 1 (his CustomLoss, which is then CE). After each
retrain:
- count the argmax grade-3 predictions n on the development IMAGES;
- compute F = d^2 (tanh(100 d) + 1) and dF, with d = n - 76;
- stop if F < 1e-5, i.e. n <= 76;
- otherwise set C += (8/600) dF and C_3 = 1, and retrain from the same initialisation with
  the same sampler order and augmentation draws. These are common random numbers, checked by
  hash.

The loop stops after at most 8 retrains; a loop that does not converge is deployed as its
last retrain and reported.

**Arms.** All are deployed with `capped_first` (exactly 76 grade-3 slots):

| Arm | Definition |
|---|---|
| `pto` | retrain 1 (Yuval's PTO) |
| `tralo_final` | pto + one `targeted_step`: TraLO's count direction, the smallest radius that brings the hard count to 76 |
| `sham_final` | pto + the same radius in a seeded random direction, with per-tensor norms matched |
| `pao` | the last retrain (Yuval's PAO) |

When pto's hard count is <= 76, the step is not applied and PAO equals PTO. Those seeds
enter every contrast as exact zeros (intent to treat). The binding-only subset is also
reported.

**Deviations from Yuval, each forced:**
- the early-stop split comes from train, because development labels enter only the scorer;
- the count is taken on the development pool against a label-free cap, not on the test set
  against a test-label count;
- at most 8 retrains;
- ResNet18, because the server is offline and has no cached EfficientNet-B5 weights.

## Endpoints and analysis (fixed)

All analysis uses capped_first on the development pool, paired by seed, with n = 24
de-duplicated by prediction hash and 95% t intervals.

- **Primary**, cc-F1 of grade 3, Holm over three contrasts:
  - **P1** pao - pto: does Yuval's loss beat PTO?
  - **P2** tralo_final - sham_final: is TraLO's direction attributable in this pipeline?
  - **P3** tralo_final - pto.
- **Secondary**, Yuval's own metrics: accuracy, macro-F1 and weighted-F1, for P1-P3,
  reported with no family claim.
- **Recipe effect**, secondary and unpaired: this study's pto against the v3 ResNet18
  clipper (seeds 1801-1824, same pool and cap), Welch t on cc-F1 and accuracy. Caveat: the
  two recipes also differ in the 10% carve and the normalisation.
- **Diagnostics, logged per epoch:**
  - live training false positives, i.e. the items PAO can weight;
  - pool hard/soft counts, stop loss and LR;
  - best epoch, retrains and convergence;
  - the step radius and whether it landed on the cap.

**Readings (fixed before data):**
- **P1 Holm-positive.** His loss beats PTO under paired, label-clean conditions. Next:
  - attribution controls that separate FP-weighting from its bundled LR reduction and
    C-weighted early stopping;
  - a label-aware TraLO variant that weights the count direction by training false
    positives.
- **P1 null.** His reported PAO > PTO does not reproduce here. The advantage in his paper
  would then rest on the evaluation artefacts above and/or on the backbone.
- **P2 positive.** TraLO's step has an attributable effect in a non-memorising pipeline;
  compare with the v3 ResNet18 null.
- **Recipe effect positive.** His pipeline makes every arm better. That is a practical gain
  for all methods, and not evidence for any constraint loss.

## Pilot gate (integrity, not score)

Seed 4099 runs first. It must:
- complete;
- show identical first-epoch sampler-order and augmentation hashes across retrains;
- reproduce each retrain's best-epoch pool output after restoring the best weights;
- land an applied step with hard count <= 76;
- match the sham radius to the target radius.

The pilot also reports whether the cap binds, the stopping epochs and the wall time, to plan
the queue. No pilot score is read before the study's contrasts are fixed; they are fixed
above.

## Compute

About 10-25 min per early-stopped ResNet18 retrain on a Quadro RTX 6000. Per seed that is
1 retrain plus the PAO retrains, 2-8 in total. The queue runs 2 processes per GPU on
dsisco01's four free GPUs. dsisco02 is fully occupied by another user and is not used.

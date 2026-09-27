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

## Amendment 1 (2026-09-27, 09:25, before any B5 output and before the ResNet18 pilot finished): the EfficientNet-B5 block

Yuval's reported models use timm `efficientnet_b5`, with pretrained weights from timm's default
tag `sw_in12k_ft_in1k`. His hard-coded unconstrained accuracies are 66-69%, against 58.9%
for our ResNet18 clipper. So the backbone is the one pipeline difference the ResNet18 block
cannot answer.

The weights file (122,330,162 bytes, sha256 `0e5c09ad...6088aca7`) was downloaded from
huggingface.co/timm/efficientnet_b5.sw_in12k_ft_in1k and uploaded to
`~/tralo-rebuild/data/weights/`. The runner checks its hash before loading it.

- **Block.** Seeds 4100-4123 (study) and 4199 (pilot) use `backbone: efficientnet_b5`: his
  `get_model` call with a fresh 5-way classifier. The runner, pipeline, arms, cap 76, pilot
  gate, endpoints and readings are exactly those above.
- **Primary family.** P1-P3 are Holm-adjusted within this block, a separate family from the
  ResNet18 block. A claim that holds "in both backbones" needs each block's own Holm-adjusted
  result.
- **Cost.** Benchmarked at 0.27 s per training step (44 s of GPU per epoch, 10.6 GB peak) plus
  the serial augmentation. About 12-15 min per early-stopped retrain.

## Amendment 2 (2026-09-27, committed 11:23 as 78ee2e06, before any B5 seed is scored): the snapshot ensemble, B5 block

The ResNet18 block found, exploratory, that the snapshot ensemble adds +2.43 [+0.82, +4.03] cc-F1 on
top of this pipeline (`analysis/yuval_r18_ensemble.txt`). At 11:12, 5 of 24 B5 seeds had finished and
none had been scored. The B5 block therefore tests it as a fresh confirmation on a fourth backbone:

- **Secondary, confirmatory for the B5 block only:** ENS - best on the pto arm, cc-F1. ENS is
  capped_first on the mean of retrain 1's development snapshots from epoch max(1, best - 2) to the
  last epoch run; best is the restored best epoch. The rule is fixed by `analysis/yuval_ensemble.py`
  @ e45a1bca, unchanged. One contrast, so no family correction. Accuracy and macro-F1 are reported
  with it, with no family claim.
- **Reading.** A CI above 0 means the ensemble gain holds on Yuval's backbone in Yuval's pipeline, so
  the post-hoc bar is the ensembled clipper there too. A CI covering 0 means the ensemble gain does not
  transfer to an early-stopped B5.
- P1-P3 and their readings are unchanged.

## Pilots

**ResNet18 pilot, seed 4099** (release bcf5d010, dsisco01 GPU0): the gate passed on every item.

| Gate item | Result |
|---|---|
| Completion | exit 0 in 706 s |
| Common random numbers | 2 retrains with identical first-epoch sampler order (`0225fa4c...`) and first augmented batch (`cc55a941...`) |
| Best-weight restore | reproduced in both retrains; the runner raises otherwise |
| TraLO step | applied, hard count 106 -> 76 |
| Sham | same radius 0.010483; its hard count stays 106 |
| Cap binding | binds (pto hard count 106) |

- Carve: 5,218 train / 560 early-stop. Train label counts: 2062/950/1364/697/145.
- Retrain 1: best epoch 15 of 20.
- PAO: C = 2.6 after one update, and retrain 2 converged with a hard count of 57. It
  overshoots the cap, as the synthetic lab predicted for this pool size.
- The pool's argmax grade-3 count swings 57-140 from epoch to epoch within one retrain, so the
  outer loop steers on a noisy count.
- No pilot score was read.

The ResNet18 study (seeds 4000-4023) was launched at 09:20 server time on release bcf5d010,
in 9 queues on GPU0/2/3, into `runs/claude-yuval-r18`. The B5 pilot 4199 (release 0d70d993,
GPU1) was still running.

**EfficientNet-B5 pilot, seed 4199** (release 0d70d993, GPU1): the gate passed on every item.

| Gate item | Result |
|---|---|
| Completion | exit 0 in 2,552 s |
| Common random numbers | 4 retrains with identical first-epoch order (`544ac59d...`) and first batch (`38bb479d...`) |
| TraLO step | applied, hard count 101 -> 76 |
| Sham | same radius 0.003051; its count stays 101 |
| Cap binding | binds |

- Every retrain's best epoch is 4 of 9: B5 overfits fast in this recipe.
- PAO's loop needed 4 retrains, with counts 101, 77, 107, 59 and C 1, 2.33, 2.39, 4.04.
  - A count one item over the cap (77) produced almost no C change, and the next retrain landed at 107.
  - Within a retrain the pool count has an epoch-to-epoch sd of about 23 (ResNet18 seeds, `analysis/yuval_count_noise.py`), so the loop steers on noise.
- No pilot score was read.

The B5 study (4100-4123) was launched at 10:01 server time on release 0d70d993, into
`runs/claude-yuval-b5`. It uses `tools/claude_claim_queue.sh` from release 00fde635 (queue
tooling only; `tralo/` is byte-identical to 0d70d993), which gives atomic per-seed claims so
queues join as GPUs free.

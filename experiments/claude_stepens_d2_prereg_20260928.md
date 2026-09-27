# Preregistration: the step-ensemble study on MobileNetV3 and EfficientNet-B5 (dsisco02)

Written 2026-09-28 before any run of these two blocks. The commit that adds this file is its timestamp.

## Why

The ResNet18 step-ensemble study (`claude_stepens_prereg_20260927.md`, seeds 4500-4571) cleared its
preregistered bar, n = 72:

- E1 ens_tralo - ens_sham = +1.10 [+0.77, +1.43];
- E2 ens_tralo - ens_pto = +1.11 [+0.78, +1.45];
- both Holm < 0.001 (`analysis/stepens_r18_score.txt`).

Its replication on RegNetY (`claude_stepens_rgy_prereg_20260927.md`, seeds 4600-4671) is running on dsisco01. It
alone decides, with ResNet18, whether this is a thesis claim.

These two blocks ask whether the result carries to two more backbones:

- **MobileNetV3-Large**, where TraLO's single-model step was +0.37 in Yuval's pipeline (small-backbone study, not
  Holm-positive);
- **EfficientNet-B5**, Yuval Kassif's own backbone (the 4100 block of the Yuval-pipeline study).

## Design

The design is identical to the ResNet18 study except for the backbone, the seeds and the host:

- **Task.** Knee (Chen split), grade-3 cap 76, deployed with `capped_first` on the 826-image development pool. The
  test split is never read.
- **Pipeline.** Yuval's pipeline:
  - augmentation and the class-balanced sampler;
  - Adam at 1e-4 with weight decay 1e-4, and LR x0.8 every 5 epochs;
  - early stopping with patience 5 on the 10% subject carve of TRAIN, best weights restored.
- **Training.** PTO only (`max_retrains` 1).
- **`snapshot_steps`.** At every epoch, TraLO's targeted step and its sham run on side copies of that epoch's model.
  The sham uses the same radius in a random direction seeded by seed + 7 + 1000 e. The global RNG is restored, so
  the PTO trajectory is untouched.
- **Ensembles.** Each arm is averaged over the window max(1, best - 2)..last, as in the ResNet18 study.
- **Blocks.**

  | Block | Backbone | Study seeds | Pilot |
  |---|---|---|---|
  | `mn3` | torchvision `mobilenet_v3_large`, ImageNet weights (as in the 4300 block) | 4700-4771, n = 72 | 4300 |
  | `b5` | timm `efficientnet_b5.sw_in12k_ft_in1k`, the uploaded weights with sha256 0e5c09ad... (as in the 4100 block) | 4800-4847, n = 48 | 4100 |

  B5 gets fewer seeds because each B5 seed costs more compute. n is fixed here and is not extended after a result.
- **Host.** Only dsisco02, on GPUs 2 and 3. Another user holds GPUs 0 and 1, and a GPU with another user's process
  is never used. At most 6 of our processes run per GPU. The release is an immutable clone of this commit.
- **Code.**
  - `tralo/knee_yuval.py` admits the new seed blocks and pilots. A pilot seed may also run with `snapshot_steps`
    false (the reference run).
  - `analysis/score_stepens.py` knows the two blocks and their reference jobs.
  - Tests cover both. 15 of 15 code mutations make them fail.

## Pilot gate (same host)

Seeds 4300 (MobileNetV3) and 4100 (B5) each run twice on one dsisco02 GPU:

- the **pilot** (`<seed>_stepens`), with `snapshot_steps` true;
- the **reference** (`<seed>_ref`), the same config with `snapshot_steps` false.

The gate is `analysis/score_stepens.py --gate`. It checks integrity only and prints no score. It passes only if all
of these hold:

- the pilot's PTO pool probabilities are byte-identical to the reference's at every epoch and at the restored best;
- the best and run epochs match;
- both runs start from the same initial weights;
- every step and sham is in spec.

The reference is rerun on dsisco02 because bit-determinism holds within one host and software stack. dsisco02's
Blackwell GPUs differ from dsisco01's Quadro RTX 6000s, so the stored dsisco01 runs cannot serve as the
reference.

A failed gate stops that block before any study seed starts. A failed job stops its launcher. Jobs already running
continue, and the failure is reported.

## Primary endpoint and family

The endpoint is the development cc-F1 of grade 3 under `capped_first`. There are two contrasts per block:

- **E1** ens_tralo - ens_sham: does TraLO's who-signal survive snapshot ensembling on this backbone?
- **E2** ens_tralo - ens_pto: TraLO against the ensembled clipper. This is the thesis bar.

Each contrast is a paired t over seeds, two-sided, with Holm over the two within each block.

## Readings, fixed now

- **A block clears the bar** if E1 and E2 are both above 0 after Holm (alpha 0.05).
- **E1 above 0 but E2 not:** the step survives ensembling but does not beat the ensembled clipper on that backbone.
- **E1 not above 0:** the step's who-signal does not survive ensembling on that backbone.
- **Across backbones:**
  - there is no pooling;
  - the thesis claim is decided by ResNet18 and the RegNetY replication, as their preregistrations say;
  - these blocks measure how far the result generalises, so the report gives each block's estimates and the number
    of backbones that clear the bar;
  - a block that fails limits the scope of the ResNet18 result and does not retract it.
- **Incomplete blocks:** a block not complete when the morning report is written is reported as incomplete, with
  its n, and no reading is drawn from it.

## Secondary (no family claim)

- accuracy, macro-F1 and weighted-F1 for E1 and E2;
- the single-model P2 tralo_final - sham_final;
- ens_pto - pto (the ensemble confirmation);
- ens_sham - ens_pto;
- the dose Spearman of E1 against the window's mean excess over the cap.

## Power

On ResNet18 the per-seed sd of E1 was about 1.40 points (from its CI at n = 72). Taking alpha 0.025 for the smaller
Holm p and 80% power:

| Block | Minimum detectable effect | Power for +0.55 | Power for +1.10 (the ResNet18 effect) |
|---|---|---|---|
| MobileNetV3, n = 72 | 0.52 | 0.84 | above 0.99 |
| B5, n = 48 | 0.64 | 0.65 | above 0.99 |

A B5 null is therefore weak evidence against an effect of +0.55 or smaller.

## Analysis rules

- **Intent to treat.** Every seed that starts is analysed. A failed seed is reported and never replaced.
- **De-duplication.** The scorer refuses repeated pto prediction vectors.
- **Fixed rules.** The window, the cap and the deployment are not changed after a result.

# Preregistration: the step-ensemble study on fmow2 satellite images (a second dataset and modality)

Written 2026-09-28 before any study run. The commit that adds this file is its timestamp.

## Why

On the knee X-rays, the ResNet18 step-ensemble study cleared its preregistered bar at n = 72
(`analysis/stepens_r18_score.txt`):

- E1 ens_tralo - ens_sham = +1.10 [+0.77, +1.43];
- E2 ens_tralo - ens_pto = +1.11 [+0.78, +1.45];
- both Holm < 0.001.

Replications on RegNetY (dsisco01) and on MobileNetV3 and EfficientNet-B5 (dsisco02) are running. Every one of these
is the same dataset: grayscale medical X-rays, five ordinal grades, one capped grade.

This study asks whether the design carries to a different dataset and modality. The new data are fmow2 satellite
images: RGB, 8 land-use classes, and a geographic shift, because train and test come from disjoint countries.

## Data

- **Source.** fmow2 (`~/optloss-audit/data/fmow2/oodslice/`), the rebuilt slice that LEDGER's Datasets section calls
  "the dataset" of the old corpus.
- **Contents.** 17,670 train items from 139 countries and 3,442 test items from 10 other countries, as 224x224 RGB
  uint8 images in 8 classes.
- **Integrity.** The six files are pinned by sha256 in `tralo/fmow_yuval.py`. A run refuses any other bytes.

## Roles, all fixed by country and never by label

| Role | Rule | Countries | Items | Use |
|---|---|---|---|---|
| Training | the rest of the train split | 125 | 15,841 | training |
| Early stopping | the train countries whose sha256 is 0 mod 10 (like the knee subject carve) | 14: AUS, AUT, BDI, BHR, BIH, CMR, KOR, KWT, LKA, RUS, SVN, TUN, TZA, VNM | 1,829 | early stopping |
| Development pool | the 5 test countries with the smallest sha256 | IRQ, NLD, DZA, PHL, TUR | 1,673 | see below |
| Reserved | the other 5 test countries | EGY, CAN, IND, MEX, JPN | 1,769 | none; kept for a later confirmation |

- **Development pool.** The step reads its images only. Its labels enter only the offline scorer and the per-arm
  report files, as in the knee runner.
- **Reserved countries.** They are never used: not in training, a step, a checkpoint choice or a score. Their files
  are only hashed.

## Constraint and deployment

- **Capped class.** Class 1 (crop_field), the corpus's first capped class.
- **Cap.** Pool size // 10 = **167** items, deployed with `capped_first`. The 1/10 rule is the knee rebuild's
  default cap rule (`knee_caps`: n // 10 for the capped class).
- **The cap binds.** The committed code (0399d9e4) allowed a fallback divisor of 20 in case the cap did not bind. To
  check, a 4-epoch smoke run of seed 5099 ran on dsisco02 GPU 3.
  - Only its predicted counts were read: PTO predicted 269 and 228 class-1 items at epochs 1 and 2, against the cap
    of 167.
  - Its labels and its reports were not read. It is not a study run: it is moved to quarantine and never scored.
  - So the divisor is 10, and the fallback is not used.

## Pipeline

The pipeline is Yuval's, exactly as in the knee studies. It is implemented by `tralo.knee_yuval.train_run`, which now
takes the capped class and the transforms as arguments; the knee defaults are unchanged.

- **Augmentation:** horizontal flip, rotation 3, affine with translate 0.1 and scale 0.9-1.1, and colour jitter 0.2.
- **Normalisation:** ImageNet's, which replaces the knee's mean and std.
- **Sampler:** class-balanced.
- **Optimiser:** Adam at 1e-4 with weight decay 1e-4 and batch 32. The LR falls x0.8 every 5 epochs.
- **Stopping:** at most 75 epochs, with patience 5 on the carve's loss and the best weights restored.
- **Backbone:** torchvision `mobilenet_v3_large` with ImageNet weights and a fresh 8-way head.
- **Training:** PTO once.
- **`snapshot_steps`:** at every epoch, TraLO's targeted step and its sham act on side copies of the model. The sham
  uses the same radius in a random direction seeded by seed + 7 + 1000 e. The global RNG is restored, so PTO's
  trajectory is untouched.

## Arms, endpoint and family

- **Arms.** As in the knee studies: ens_pto, ens_tralo and ens_sham, each the mean of that arm's epoch snapshots over
  the window max(1, best - 2)..last.
- **Endpoint.** Development cc-F1 of class 1 under `capped_first` at 167.
- **Contrasts.**
  - **E1** ens_tralo - ens_sham;
  - **E2** ens_tralo - ens_pto, the thesis bar.
- **Test.** Paired t over seeds, two-sided, with Holm over the two.
- **Slots.** One slot is 2 / (167 + n_true) F1 points, where n_true is the pool's number of class-1 items. It is read
  by the scorer offline, and the scorer prints it.

## Seeds, host and gate

- **Study.** Seeds 5000-5047 (n = 48), on dsisco01 only, with at most 4 of our processes per GPU. The RegNetY study
  had finished (72/72) before this launch.
- **Pilot.** Seed 5099 runs twice on dsisco01:
  - the pilot (`5099_stepens`), with the steps on;
  - the reference (`5099_ref`), with the steps off.
- **Gate.** `analysis/score_fmow_stepens.py --gate` passes only if all of these hold:
  - PTO is byte-identical at every epoch and at the restored best;
  - both runs start from the same initial weights;
  - every step is in spec.
- **Order.** To fit the night, the pilot and its reference start first and the study seeds follow at once. The gate
  runs as soon as the pilot and the reference have both ended.
- **Stops.** A failed gate stops the launcher and voids the study, which is then not scored. The study is scored only
  after the gate has passed. A failed job stops the launcher, and running jobs continue.

## Readings, fixed now

- **E1 and E2 above 0 after Holm (alpha 0.05):** the step-ensemble result carries to fmow2 satellite images on
  MobileNetV3.
- **E1 above 0 but E2 not:** the step survives ensembling here but does not beat the ensembled clipper.
- **E1 not above 0:** the step's who-signal does not survive ensembling on fmow2.
- **Pooling:** none with the knee studies.
- **Incomplete study:** if it is not complete when the morning report is written, it is reported as incomplete, with
  its n, and no reading is drawn from it.

## Secondary (no family claim)

- accuracy, macro-F1 and weighted-F1 over the 8 classes, for E1 and E2;
- the single-model P2 tralo_final - sham_final;
- ens_pto - pto;
- ens_sham - ens_pto;
- the dose Spearman of E1 against the window's mean excess over the cap.

## Power

The per-seed sd on fmow2 is unknown. If it matches the knee's (about 1.4 points), n = 48 gives a minimum detectable
effect of about 0.64 points, at 80% power and alpha 0.025.

## Analysis rules

- **Intent to treat.** Every seed that starts is analysed. A failed seed is reported and never replaced.
- **De-duplication.** The scorer refuses repeated pto prediction vectors.
- **Fixed rules.** The window, the cap and the deployment are not changed after a result.

## What differs from the knee design

- the dataset and its modality;
- the capped class and the cap;
- 8 classes;
- ImageNet normalisation;
- the country roles;
- n = 48;
- the same-host reference for the pilot.

## Code

- **Files.** `tralo/fmow_yuval.py`, `analysis/score_fmow_stepens.py`, `tools/claude_fmow_queue.sh` and
  `tools/claude_fmow_launcher.sh`.
- **Mutation checks.**
  - The fmow2 tests catch 18 of 18 code mutations.
  - `knee_yuval.train_run`'s new arguments are tested, and 2 of 2 mutations are caught.
  - The knee tests still pass.

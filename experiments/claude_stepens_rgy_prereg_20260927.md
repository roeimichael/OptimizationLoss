# Preregistration: the step-ensemble study replicated on RegNetY (knee, cap 76)

Committed 2026-09-27 at 19:06:53 (f2580776), before any run of this study (the header first said "about 19:10"). At that point the ResNet18 study
([prereg](claude_stepens_prereg_20260927.md)) had passed its pilot gate, which is integrity only and prints
no score, and none of its study seeds had finished. Runner: `tralo/knee_yuval.py` with `snapshot_steps`.
Scorer: `analysis/score_stepens.py`. Launcher: `tools/claude_stepens_launcher.sh <sha> rgy <pid>`.

## Why

- **The ResNet18 prereg makes a positive a thesis claim only after a preregistered replication on a
  second backbone.** This is that replication. It is fixed now, before the ResNet18 result exists, so
  neither study's outcome can steer the other's design or reading.
- **RegNetY is the other backbone whose single-model step points up in Yuval's pipeline.** Its P2,
  tralo_final - sham_final, is +0.55 [+0.04, +1.06] (Holm 0.073, n = 24)
  ([result](claude_yuval_smallbb_result_20260927.md)). MobileNetV3 is +0.37 and EfficientNet-B5 is -0.50.
- **The question is the same.** Does TraLO's who-signal survive snapshot ensembling (E1)? And does TraLO
  plus the ensemble beat the ensembled clipper (E2)? The snapshot ensemble added +1.56 [+0.45, +2.66] on
  RegNetY in Yuval's pipeline, about three times its single-model P2.
- No data from these seeds has been seen.

## Design (fixed)

Identical to the ResNet18 study except for the backbone and the seeds:
- **Seeds.** 4600-4671 (n = 72), plus a pilot job `4400_stepens`. The pilot reruns stored seed 4400 of the
  small-backbone block (`runs/claude-yuval-rgy/seed4400`, release 7d8f7dd7).
- **Recipe.** torchvision `regnet_y_400mf` with cached ImageNet weights, built as in the small-backbone
  block. Knee grade 3, label-free cap 76 on the development pool. Yuval's full pipeline as in that block:
  - augmentation and a class-balanced sampler;
  - Adam 1e-4 with weight decay 1e-4, LR x0.8 every 5 epochs;
  - early stopping with patience 5 on the 10% subject-level carve from TRAIN, best weights restored, up
    to 75 epochs.

  PTO only (`max_retrains` 1), and every arm is deployed with capped_first.
- **Snapshot steps.** As in the ResNet18 study, at every epoch e two side copies of the model are made:
  - TraLO's targeted step on one (`epochNN_tralo.pt`);
  - the sham on the other: the same radius in a random direction seeded by seed + 7 + 1000e
    (`epochNN_sham.pt`).

  The training model and the global RNG states are untouched, so PTO's trajectory is byte-identical to a
  run without snapshot steps.
- **Ensembles.** ens_pto, ens_tralo and ens_sham are means over the window max(1, best - 2)..last, each
  followed by capped_first.

## Endpoints (fixed)

The same as the ResNet18 study, computed within this study:
- **Primary.** cc-F1 of grade 3, 95% paired t intervals over n = 72 seeds, Holm over two contrasts:
  - E1, ens_tralo - ens_sham;
  - E2, ens_tralo - ens_pto.
- **Secondary** (no family claim): the same list as the ResNet18 prereg. That is Yuval's metrics for E1
  and E2, the single-model P2, ens_pto - pto (ensemble confirmation set 9), ens_sham - ens_pto, and the
  dose relation of E1.
- **Rules.** Intent to treat, and de-duplication by the pto prediction hash.
- **No pooling.** The two backbones are two units, and no pooled estimate is a primary.

## Readings (fixed before data)

**This study alone** is read with the same four readings as the ResNet18 study: E1 and E2 above 0; E1
above 0 with E2 covering 0; E1 covering 0; E2 below 0.

**The two studies together:**
- **E1 and E2 above 0 in both studies.** TraLO plus the snapshot ensemble beats the ensembled clipper on
  two backbones in Yuval's pipeline. That is the thesis claim, reported in slots of 76 per backbone.
- **Above 0 on ResNet18, not on RegNetY.** Not replicated. The ResNet18 result stays a single-backbone
  result and is not a thesis claim.
- **Above 0 on RegNetY, not on ResNet18.** A single-backbone result on RegNetY. It would need its own
  preregistered replication before any claim.
- **E1 covering 0 in both.** In Yuval's pipeline the snapshot ensemble already carries what the step adds.
  The step direction closes there, and the bar stays the ensembled clipper.
- **E2 below 0 in either study.** TraLO plus the ensemble is worse than the ensembled clipper on that
  backbone, whatever the other study shows.

## Power

RegNetY's single-model P2 had a per-seed sd of 1.21. With a similar sd for E1, n = 72 gives a standard
error of about 0.14 points. The minimum detectable effect is then about 0.44 points (80% power, alpha
0.025 for the smaller Holm p). The block's +0.55 would be detected with about 95% power.

## Pilot gate (integrity, not score)

The pilot job `4400_stepens` must show:
- exit 0;
- the RegNet model class, from the initial weights of the stored run (the same initial hash);
- snapshot steps at every epoch, with the sham on the same radius as each step, and every applied step at
  or under the cap;
- PTO byte-identical to the stored run at every epoch, with the same best and last epochs and the same
  restored best weights' output.

The gate is `analysis/score_stepens.py --gate`, and it prints no score.

## Compute and scheduling

- The pilot starts on the first free slot, so it holds at most one slot of the ResNet18 study.
- The study seeds start only after the ResNet18 launcher has exited, that is, once all its seeds are
  claimed. So the ResNet18 result is not delayed by this study.
- A RegNetY job with snapshot steps should take about 30-40 min at three processes per GPU. So 72 jobs on
  dsisco01's 12 slots take about 3-4 h.
- dsisco02 is not used. Another user holds two of its GPUs, and a pilot there could not be byte-identical
  to runs from dsisco01.

## Code change (fixed)

- `tralo/knee_yuval.py`:
  - seeds 4600-4671 are admitted as regnet_y_400mf;
  - the pilot 4400 is admitted, only with `snapshot_steps` true and `max_retrains` 1;
  - the small-backbone block's keys are unchanged.
- `analysis/score_stepens.py`:
  - it scores one block at a time, chosen by the seeds;
  - it rejects mixed blocks, and any seed trained on another architecture;
  - the gate finds the pilot by its job name and requires the stored run's initial weights.
- `tools/claude_stepens_launcher.sh` takes the block and the pid to wait for.
- The tests cover each of these. 30 of 30 mutations of the new and earlier step-ensemble code
  are caught.

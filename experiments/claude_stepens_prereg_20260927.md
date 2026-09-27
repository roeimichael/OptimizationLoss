# Preregistration: does TraLO's step survive snapshot ensembling, and does TraLO plus the ensemble beat the ensembled clipper? (knee, cap 76)

Written 2026-09-27 at about 19:00, before any run of this study. Runner: `tralo/knee_yuval.py` with
`snapshot_steps`. Scorer: `analysis/score_stepens.py`. Launcher: `tools/claude_stepens_launcher.sh`.

## Why

- **The recipe factorial found the first Holm-significant attributable effect of TraLO's step.** P2-pooled,
  tralo_final - sham_final averaged over its eight recipes on ResNet18, is +0.72 [+0.46, +0.97] cc-F1
  points (Holm 0.000, n = 24), about 0.66 of 76 slots
  ([result](claude_recipe_factorial_result_20260927.md)).
  - In the small-backbone blocks of Yuval's pipeline the same contrast points the same way, but neither
    block is significant alone: MobileNetV3 +0.37, RegNetY +0.55 (Holm 0.27 and 0.073).
  - On EfficientNet-B5 it was -0.50.
- **The thesis bar is the ensembled clipper.** A snapshot ensemble of the last epochs adds +1.0 to +4.1
  cc-F1 in every fresh confirmation set so far, and +0.87 to +3.98 in the factorial's cells. That is
  several times the step's single-model effect.
- **So the step clears the bar only if its who-signal survives ensembling.** A single-model gain that
  averaging already captures does not. The step has so far only been applied to the final model. This
  study applies it at every epoch, so TraLO can be ensembled exactly like the clipper and compared at
  equal ensembling.
- No data from these seeds has been seen.

## Design (fixed)

- **Seeds.** 4500-4571 (n = 72) for the study, plus a pilot job `4000_stepens` that reruns seed 4000.
- **Recipe.** ResNet18 with cached ImageNet weights, knee grade 3, label-free cap 76 on the development
  pool. Yuval's full pipeline exactly as in the ResNet18 block (seeds 4000-4023):
  - augmentation and a class-balanced sampler;
  - Adam 1e-4 with weight decay 1e-4, LR x0.8 every 5 epochs;
  - early stopping with patience 5 on the 10% subject-level carve from TRAIN, best weights restored, up
    to 75 epochs.

  PTO only (`max_retrains` 1). Every arm is deployed with capped_first.
- **Snapshot steps.** At every epoch e, after that epoch's pool probabilities are saved (`epochNN.pt`),
  two side copies of the model are made:
  - TraLO's targeted step on one copy: the smallest radius along the soft-count gradient that brings the
    hard grade-3 count to the cap, and no step when the count is already at or under it
    (`epochNN_tralo.pt`);
  - the sham on the other: the same radius and per-tensor norms in a random direction seeded by
    seed + 7 + 1000e (`epochNN_sham.pt`).

  The training model is never touched, and the global RNG states are restored afterwards. So PTO's
  trajectory is byte-identical to a run without snapshot steps.
- The final-model arms tralo_final and sham_final are kept, as the single-model replicate.
- **Ensembles.** ens_pto, ens_tralo and ens_sham are the means of the matching snapshot probabilities over
  the window max(1, best - 2)..last (the rule of `analysis/yuval_ensemble.py`), each followed by
  capped_first.

## Endpoints (fixed)

- **Primary.** cc-F1 of grade 3, 95% paired t intervals over n = 72 seeds, Holm over two contrasts:
  - **E1, ens_tralo - ens_sham:** does the step's who-signal survive snapshot ensembling?
  - **E2, ens_tralo - ens_pto:** TraLO against the ensembled clipper, the thesis bar.
- **Secondary, with no family claim:**
  - accuracy, macro-F1 and weighted-F1 for E1 and E2;
  - single-model P2, tralo_final - sham_final;
  - ens_pto - pto, ensemble confirmation set 8;
  - ens_sham - ens_pto, the cost of a random move of the same size, ensembled;
  - dose: the Spearman correlation between E1 and the window's mean excess of the hard count over the cap.
- **Rules.** Intent to treat: a seed whose count never exceeds the cap enters as exact zeros. Seeds are
  de-duplicated by the pto prediction hash.

## Readings (fixed before data)

- **E1 and E2 both have their CI above 0.** TraLO's step adds who-information that the snapshot ensemble
  does not already carry, and TraLO plus the ensemble beats the ensembled clipper. This clears the thesis
  bar on ResNet18 in Yuval's pipeline. Report it in slots of 76. It becomes a thesis claim only after a
  preregistered replication on a second backbone.
- **E1 above 0, E2 covering 0.** The signal survives averaging, but a move of this size costs about as
  much as it gains. There is no deployable gain over the ensembled clipper.
- **E1 covering 0.** The ensemble already captures the step's single-model gain. The upper bound of E1
  bounds what the step can add over it, and the thesis bar is not met. If the single-model P2 replicates
  the factorial here, the step's effect is real but redundant with averaging.
- **E2 below 0.** TraLO plus the ensemble is worse than the ensembled clipper.

## Power

The single-model P2 had a per-seed sd of 1.27 (the factorial's all-on cell) to 1.54 (the ResNet18 block).
With a similar sd for E1, n = 72 gives a standard error of about 0.15-0.18 points. The minimum detectable
effect is then about 0.47-0.57 points (80% power, alpha 0.025 for the smaller Holm p). The factorial's
+0.72 would be detected with about 95% power, and half of it with about 40%.

## Pilot gate (integrity, not score)

The pilot job `4000_stepens` reruns seed 4000 of the ResNet18 block with snapshot steps. It must show:
- exit 0;
- snapshot steps at every epoch, with the sham on the same radius as each step, and every applied step
  at or under the cap;
- PTO byte-identical to the stored run (`runs/claude-yuval-r18/seed4000/retrain1`): every epoch's pool
  probabilities, the same best and last epochs, and the restored best weights' output.

The gate is `analysis/score_stepens.py --gate`, and it prints no score.

## Compute

A targeted step or sham costs about 30 inferences over the 826 development images, at every epoch. That is
roughly 15-25 min per job at three processes per GPU, so 72 jobs on 12 slots take about 2 h. dsisco01's four
GPUs are free. dsisco02 is fully used by another user and is not used.

## Code change (fixed)

- `tralo/knee_yuval.py`:
  - `snapshot_steps()`;
  - seeds 4500-4571 and the pilot 4000 are admitted only with `snapshot_steps` true and `max_retrains` 1.
- Tests (`tests/test_knee_yuval.py`, `tests/test_score_stepens.py`):
  - the validation rules;
  - a byte-identical PTO trajectory with and without the side steps;
  - the global RNG restored even when a step draws from it;
  - the sham in a different direction from the step;
  - the scorer's window, a built-in signal it must recover, each out-of-spec step record, and a gate that
    passes only a byte-identical pilot.

  18 of 18 mutations of the new code are caught.

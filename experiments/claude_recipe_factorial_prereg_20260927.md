# Preregistration: which part of Yuval's pipeline carries its gain, and does TraLO's step help in any of them (knee, cap 76)

Written 2026-09-27 before any run of this study, and committed at 11:34 as 67ecde20. Runner: `tralo/knee_yuval.py` recipe factorial.
Scorer: `analysis/score_recipe.py`.

## Why

The ResNet18 block of the Yuval-pipeline study
([result](claude_yuval_pipeline_result_20260927.md)) found two things:
- his pipeline's plain model beats our v3 recipe's clipper by +4.81 cc-F1 and +3.60 accuracy (unpaired);
- TraLO's step beats its same-radius sham by +0.73 [+0.08, +1.38], which misses Holm (0.088).

The pipeline changes several components at once. The user asked which parts of Yuval's pipeline, or of
how he worked, made his model better, and whether our loss can be fitted on top.
- This study switches off the three main components one at a time, in a full 2 x 2 x 2 factorial, on
  fresh seeds.
- Every cell also carries TraLO's step and its sham. The step is then tested across eight recipes at
  once, with far more power than one block had.

Prior evidence (not this study's data): in our own recipe, augmentation alone gave +3.48 [+1.71, +5.25]
cc-F1 (CUTPAIR, LEDGER PART 4).

## Design (fixed)

- **Seeds** 4200-4223 (study) and 4299 (pilot). ResNet18 with cached ImageNet weights; knee grade 3;
  label-free cap 76 on the development pool; every arm deployed with capped_first.
- **Held fixed at Yuval's values in every cell:**
  - Adam, lr 1e-4, weight decay 1e-4, LR x0.8 every 5 epochs, batch 32;
  - his knee normalisation, and CE (CustomLoss with C = 1);
  - the 10% subject-level early-stopping carve from TRAIN, so every cell trains on the same 5,218 images.
- **Three switches, all eight combinations per seed:**

  | Switch | On (Yuval) | Off |
  |---|---|---|
  | `augment` (A) | flip, rotation, affine, colour jitter at every epoch | the evaluation transform |
  | `balanced` (S) | class-balanced sampling with replacement | a uniform permutation per epoch |
  | `early_stop` (E) | up to 75 epochs, patience 5 on the carve's loss, best weights restored | exactly 10 epochs, the last kept (v3's length) |

- **Common random numbers within a seed.** All eight cells share the initialisation. Cells with the
  same S share the sampler order. Cells with the same A and S share the augmentation draws. So E off is
  bit-identical to E on through epoch 10.
- **Arms per run:**
  - `pto`, the trained model;
  - `tralo_final`, pto plus TraLO's targeted step at the radius that meets the cap;
  - `sham_final`, pto plus a seeded random step of the same radius.

  There is no PAO: `max_retrains` is 1.
- **Jobs.** A job is `<seed>_a<A>s<S>e<E>`, with configs in `experiments/configs/claude_yuval_<job>.json`.
  It runs through `tools/claude_claim_queue.sh` into `runs/claude-recipe/seed<job>`.

## Endpoints and analysis (fixed)

All analysis uses capped_first on the development pool, cc-F1 of grade 3, 95% t intervals, n = 24
seeds. A seed is the unit: each contrast is formed within a seed, then tested across seeds. Before
scoring, all 192 pto prediction vectors must be distinct.

- **Primary, Holm over four:**
  - **F-A**, the main effect of augmentation on pto: per seed, the mean of the four A-on cells minus the
    mean of the four A-off cells;
  - **F-S**, the same for the balanced sampler;
  - **F-E**, the same for early stopping;
  - **P2-pooled**: per seed, the mean over the eight cells of tralo_final - sham_final.
- **Secondary, with no family claim:**
  - the two-way interactions on pto;
  - P2 in each cell, and how P2 depends on each switch;
  - accuracy, macro-F1 and weighted-F1 for the primaries;
  - ENS - best on pto in each cell, using `analysis/yuval_ensemble.py`'s window rule;
  - the all-on cell against the ResNet18 block's pto, a replication of the pipeline level (unpaired);
  - the all-off cell against the v3 clipper, seeds 1801-1824 (unpaired Welch). This is the part of the
    +4.81 left to the components held fixed.

**Readings (fixed before data):**
- **A main effect with its CI above 0.** Augmentation carries part of the recipe gain, measured against
  the +4.81 total. Predicted from CUTPAIR to be the largest of the three.
- **S or E main effect with its CI above 0.** That component also carries part of the gain. A CI below
  0 means it costs cc-F1 at this cap. No direction is predicted for either.
- **All three CIs cover 0.** The gain lies in the fixed components (weight decay, LR decay,
  normalisation, carve) or in interactions; the all-off cell against v3 tells which.
- **P2-pooled with its CI above 0.** TraLO's step carries an attributable who-signal across recipes;
  report it in slots of 76. It clears the thesis bar only if it also beats the ensembled clipper, so
  it is set beside ENS - best in the same cells.
- **P2-pooled with its CI covering 0.** The ResNet18 block's +0.73 does not generalise across recipes.
  The upper CI bound is the bound on the step's effect.
- A recipe effect of any size is a gain for the post-hoc clipper too. It is not evidence for a
  constraint loss.

## Pilot gate (integrity, not score)

Seed 4299 runs all eight cells first. It must show:
- exit 0 in every cell, with only `retrain1/` and no `pao` arm;
- one `initial_sha256` across the eight cells;
- a `first_order_sha256` shared exactly by the cells with the same S;
- a `first_batch_sha256` shared exactly by the cells with the same (A, S);
- E-off cells that run exactly 10 epochs and keep epoch 10, and E-on cells that restore their best epoch;
- every applied step at hard count <= 76, with the sham radius equal to the target radius.
- added at 12:05, before any job started: each E-off cell bit-identical to its E-on twin's epoch-10 snapshot,
  wherever the twin reached epoch 10. This checks the common random numbers across the whole trajectory. It is
  applied to every study seed too.

The pilot also reports wall time per cell, to plan the queues. No pilot score is read.

## Compute

About 5-6 min per E-on run and 4-5 min per E-off run, at three ResNet18 processes per Quadro RTX 6000
(the ResNet18 block's single-retrain seeds took 307-361 s). The study is 192 runs plus 8 pilot runs,
about 90 min on 12 slots. The queues join dsisco01's GPUs as the EfficientNet-B5 block frees their
memory. dsisco02 is fully occupied by another user and is not used.

## Amendment 1 (2026-09-27, 11:50, before any job of this study has started): P2 dose-response

Exploratory in the ResNet18 block of the Yuval study: per seed, TraLO's step gain rises with how far pto's argmax
grade-3 count exceeds the cap. Over the 21 binding seeds, Spearman is 0.53 (p 0.013). The mean gain is +2.0 slots
where the excess is 30 or more (n 7), and +0.1 where it is below 20 (n 10).

- **Secondary test here**, over the binding jobs:
  - the Spearman correlation between per-job P2 (tralo_final - sham_final, cc-F1) and the excess (pto
    hard_before - 76), pooled over all jobs;
  - the same per seed, with the 24 coefficients t-tested against 0.

  Code: `analysis/score_recipe.py`, committed with this amendment.
- **Reading.** A positive correlation means the step's who-signal grows with the number of items it must
  evict. The within-seed version holds seed-level model quality fixed, though not recipe-level quality.
  A correlation near 0 means the Yuval-block pattern was noise, or a between-seed quality confound.
- The primaries and their readings are unchanged.

Erratum (11:59, still before any job started): on exact per-seed values the motivating Spearman is 0.49
(p 0.024). The 0.53 above came from per-seed values rounded to 0.01 points, which split tied ranks. The test
and its code are unchanged.

# Preregistration: TraLO's step on the small backbones inside Yuval's pipeline (knee, cap 76)

Written 2026-09-27 at about 14:15 server time. At that time no code change, config or output for this
block exists. Runner: `tralo/knee_yuval.py`, extended by two backbone blocks and nothing else.

## Why

- **The one positive.** TraLO's corrected step has one attributable positive: MobileNetV3 in our
  v3 recipe. There, target - sham is **+1.47 [+0.43, +2.50]** cc-F1 (Holm 0.016). RegNetY was null
  (-0.09). Source: `claude_backbone_replication_result_20260926.md`.
  - Offline, each step on either small backbone carried about 0.5 correct slots, about 4x ResNet18's.
- **The v3 recipe memorises** by epoch 5 (the saturation gate). Yuval's pipeline does not
  memorise, and it lifts every method by about 5 cc-F1. In that pipeline TraLO's step minus its sham
  is +0.73 [+0.08, +1.38] on ResNet18 (Holm 0.088) and -0.50 [-1.35, +0.34] on EfficientNet-B5
  (`claude_yuval_pipeline_result_20260927.md`).
- **The question.** Does the MobileNetV3 signal survive a recipe that does not memorise? This is a
  replication of our own positive under the recipe a reviewer would ask for. It is not a search for
  a positive:
  - the blocks, contrasts and readings are fixed here;
  - every Yuval-pipeline P2 is reported together, in one table.

## Design (fixed)

- **Pipeline:** exactly the Design section of `claude_yuval_pipeline_prereg_20260927.md`:
  - Yuval's transforms and normalisation;
  - the class-balanced sampler;
  - Adam 1e-4 with weight decay 1e-4, LR x0.8 every 5 epochs;
  - at most 75 epochs, with early stopping (patience 5) on the train-carved subject split and the
    best weights restored.

  The configs are the study's, with `max_retrains` 1 and a `backbone` key.
- **PTO only (`max_retrains` 1).** PAO is closed (LEDGER PART 4), so there is no outer loop.
- **Arms:** pto, tralo_final and sham_final, defined as in that study and deployed with capped_first
  at cap 76.
- **Blocks,** 24 study seeds each. Both backbones use torchvision ImageNet weights and the v3
  construction, `tralo.knee_e2e_v3.build_model`:

  | Block | Backbone | Study seeds | Pilot seed |
  |---|---|---|---|
  | `mn3` | `mobilenet_v3_large` | 4300-4323 | 4399 |
  | `rgy` | `regnet_y_400mf` | 4400-4423 | 4499 |

- **Power.** The v3 MobileNetV3 contrast had a paired sd of about 2.45 points. With n = 24 and Holm
  over two (first test at 0.025), that gives about 75% power for the v3 effect (+1.47), in a normal
  approximation. A single step's +0.5 slots (0.55 points) cannot be detected at this n (power about
  13%). The design tests the v3 endpoint effect, not a one-step effect.

## Endpoints and analysis (fixed)

All analysis uses capped_first on the development pool, paired by seed, with n de-duplicated by
prediction hash and 95% t intervals.

- **Primary:** P2 = tralo_final - sham_final on grade-3 cc-F1, intent to treat. Holm over the two
  blocks.
- **Secondary, no family claim:**
  - P3 = tralo_final - pto;
  - accuracy, macro-F1 and weighted-F1 for P2 and P3;
  - the binding-seed subset.
- **Secondary confirmatory: the snapshot ensemble.** ENS - best on pto in each block, by the rule
  of Amendment 2 (`analysis/yuval_ensemble.py` @ e45a1bca, unchanged). These are confirmation sets
  6 and 7; a CI above 0 confirms.
- **Secondary unpaired: the recipe.** pto against the v3 clipper of the same backbone, Welch t:
  `claude-repl-mn3` (seeds 3001-3024) and `claude-repl-rgy` (3101-3124).
- **Diagnostics, post hoc:**
  - the swap direction (`analysis/yuval_swaps.py`);
  - P2 against the excess over the cap;
  - best epoch and epochs run.

## Readings (fixed before data)

- **MobileNetV3 P2 Holm-positive.** The MobileNetV3 who-signal survives a non-memorising recipe. So
  TraLO's step adds attributable slots on MobileNetV3 in both recipes. Whether that clears the
  post-hoc bar is read against ENS - best; in v3 it did not.
- **MobileNetV3 P2 null.** The v3 positive is recipe-bound. The claim "a small attributable
  who-signal on smaller backbones" is then scoped to the memorising v3 recipe. If the upper CI is
  below +1.47, an effect the size of the v3 one is excluded in this pipeline.
- **RegNetY.** A Holm-positive P2 would be new, because it was null in v3. A null is the expected
  replication.
- **Every case.** P2 across all four Yuval-pipeline blocks (ResNet18, B5, MobileNetV3, RegNetY) is
  reported in one table, with no pooled claim.

## Pilot gate (integrity, not score)

Seeds 4399 and 4499 run first. Each must pass:
- it completes with exit 0;
- the runner's best-weight restore check passes (the runner raises otherwise);
- an applied step lands on 76, or the cap does not bind;
- the sham radius equals the target radius.

The pilots also report the stopping epochs and the wall time. No pilot score is read.

## Compute and launch

The recipe factorial's ResNet18 jobs take 4-12 min each, with 12 processes sharing dsisco01's four
GPUs (the machine is CPU-bound). The small backbones cost about the same or less, so the 50 runs take
about 1 h.
- They launch after the factorial's last claim, with the same claim-queue tooling and at most 3
  processes per GPU.
- dsisco02 is fully used by another user and is not used.

## Code change (fixed)

- `backbone_for(seed)` maps 4300-4323 and 4399 to `mobilenet_v3_large`, and 4400-4423 and 4499 to
  `regnet_y_400mf`.
- `validate` admits these seeds only with `max_retrains` 1 and no recipe switches.
- `run` builds these backbones with `build_model(backbone)`.

Nothing else changes. The ResNet18, B5 and factorial paths keep their behaviour; the tests check the
mapping and the validation.

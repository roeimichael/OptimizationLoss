# TraLO local-constraint research: status for 1 October morning

Checked 2026-09-30 after the full fmow2 PHR block finished and was
independently scored. This is the current decision document; detailed runs,
per-seed outcomes and training diagnostics remain in the linked reports.
The [paper-facing numerical results draft](tralo_paper_results_draft_20260930.md)
collects absolute metrics, paired effects, uncertainty and negative controls
in manuscript form without claiming a ViT or held-out result.

## Answer today

**There is a narrow knee lead for TraLO's global, targeted snapshot step,
but no demonstrated local-constraint lead.** Under Yuval Kassif's knee
pipeline at grade-3 cap 76, the registered ResNet18 and RegNetY blocks each
added about one correct capped slot over both a dose-matched sham and the
snapshot-ensemble clipper; MobileNetV3 added about 1.18 cc-F1 percentage
points. EfficientNet-B5 reversed sign. All are development-set findings;
the 1,656-image Chen test is sealed. See the
[additional knee result](claude_stepens_additional_result_20260928.md).

The paper-facing knee numbers below are **percentage-point paired effects**
on development grade-3 cc-F1 at cap 76, each against its own snapshot
ensemble control. Intervals are the studies' fixed 95% seed-paired intervals;
no effects are pooled across backbones. The last three columns are
TraLO-minus-PTO secondary effects in percentage points.

| Knee backbone | Seeds | cc-F1 vs sham [95% CI] | cc-F1 vs PTO/Clipper [95% CI] | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 72 | +1.10 [+0.77, +1.43] | +1.11 [+0.78, +1.45] | +0.66 | −0.12 | −0.25 |
| RegNetY | 72 | +0.92 [+0.57, +1.27] | +0.93 [+0.57, +1.29] | +0.64 | −0.05 | −0.52 |
| MobileNetV3-Large | 72 | +1.18 [+0.85, +1.50] | +1.18 [+0.84, +1.51] | +0.87 | +0.93 | +0.46 |
| EfficientNet-B5 | 48 | −0.96 [−1.63, −0.29] | −0.87 [−1.52, −0.22] | −0.22 | −2.00 | −2.06 |

The weighted-F1 decreases on ResNet18 and RegNetY are real tradeoffs; the
MobileNetV3 result has a more favorable secondary profile. ViT has **no
new, matched local-constraint or knee step-ensemble result** in this rebuild
and therefore has no row to fill from older campaigns.

The fmow2 pooled-plus-country local extension has now failed **two distinct
fixed tests**. With a 0.1 full-model step, joint TraLO lost 0.0866 and
0.0697 class-1 cc-F1 to its matched no-step PTO at pooled caps 167 and 83 in
the first 12-seed block. In a separate 12-seed block, its means were 0.4217
and 0.3344 versus PTO 0.4926 and 0.3997; snapshot PHR-ALM did no better
(0.4147 and 0.3344). Both blocks passed independent integrity checks.
These countries have been repeatedly inspected, so their findings constrain
method design but cannot confirm a new winner. See the
[fixed-dose result](fmow_local_fixed_dose_result_20260930.md) and
[PHR result](fmow_local_alm_direction_result_20260930.md).

| fmow2 local cap | PTO/Clipper cc-F1 | Sham cc-F1 | Pooled-only cc-F1 | Joint TraLO cc-F1 | PHR snapshot cc-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 167 | 0.49260 | 0.49217 | 0.43864 | 0.42167 | 0.41471 |
| 83 | 0.39967 | 0.39911 | 0.33501 | 0.33445 | 0.33445 |

These are absolute allocated class-1 cc-F1 means from the complete 12-seed
PHR block, not percentages or results from the earlier fixed-dose block.

| Comparison | Fair interpretation | Current result |
| --- | --- | --- |
| Local TraLO vs PTO/Clipper analogue | Same PTO snapshots, fixed allocator, zero side step | TraLO loses at both fmow2 caps in both complete blocks. |
| Local TraLO vs sham/null | Same snapshots and TraLO per-tensor displacement norms, unrelated direction | Sham remains near PTO; constraint directions cause the large degradation at this dose. |
| Local TraLO vs pooled-only | Same maximum dose, no country gradient | Both harm; country term does not rescue the G=167 result. |
| Local TraLO vs snapshot PHR-ALM | Same snapshots and maximum dose; native activation rules differ | PHR worse at G=167, tied on cc-F1 at G=83; both worse than PTO. This is not full ALM training. |
| Older knee/CIFAR ALM vs TraLO | Frozen-head training, different objective and update schedule | No local fmow2 numerical conclusion can be imported from it. |

For the snapshot study, an ALM side copy with its step disabled has the same
PTO predictions; keeping a dual vector alone cannot alter that copy. The
separate sham tests whether TraLO's observed movement, rather than merely
its dose, matters. The older end-to-end frozen-head study saved distinct
ALM-null and TraLO-null arms, whose predictions matched within that study.

## Why the current local step misses

The local loss and PHR are active: all 73 epoch opportunities at each cap
applied a full 0.1 step in the latest block. Yet a snapshot with about 233
raw class-1 calls falls to only 15–17 after the joint/PHR step. The final
allocator fills the cap, so the harm appears in **which images win the
slots**, not merely in a failure to satisfy the quota. At G=167 the joint
step made 239 correct entries but expelled 402 correct PTO selections across
the 12 seeds; PHR made 242 entries and 421 exits. Every seed lost selected
class-1 true positives versus PTO at both caps. A same-dose sham left raw
calls and allocated scores near PTO. The available evidence strongly points
to excessive, badly directed intervention at the tested radius; it does not
establish an optimal smaller dose or a universal impossibility result.
There is also a structural limit: a count-only loss sees the number or sum of
class-1 calls, not which examples are actually class 1. Shared model weights
can still change rankings, but the count objective itself does not reward
correct entries over incorrect ones. A calibrated step can solve the
overshoot problem and still fail the classification objective.

The training logs also show training loss falling while stop-country loss
rises. This is a separate generalization diagnostic. We should not change
early stopping, the dataset or the objective based on these viewed results
and then claim confirmation on the same countries.

## Next falsifiable path

1. Build a **label-free boundary-calibrated snapshot step** in a new fixed
   release. Compute positive pooled/country soft residuals and directional
   derivatives at the unmodified PTO snapshot. Use a bounded line search to
   cap the step before raw predictions collapse, while checking all active
   scopes for nonworsening and logging conflicts. Pooled and local upper
   bounds are not independent fill targets: local totals exceed the pooled
   cap, so the rule must never require all local quotas to fill at once.
   Derivative finite differences, hard-count behavior, exact dose, PTO/BN/RNG
   parity, label exclusion and allocator recount are mandatory CPU gates.
2. Compare the same calibrated policy against a zero-step PTO/Clipper arm,
   a matched sham/null, pooled-only direction and a PHR direction under the
   same calibration rule. Separate direction from activation/dose effects.
   Predefine one primary metric, contrasts, secondary domination rule,
   seeds, precision and compute ceiling before any GPU pilot. Do not tune a
   radius or accept/reject the method using the already-viewed fmow2 scores.
   The fixed MobileNetV3 design now names pilot 6400, full seeds 6401–6412,
   both caps, six paired contrasts and an 8 GPU-hour ceiling in the
   [boundary protocol](fmow_boundary_mnv3_protocol_20260930.md). The scalar
   policy, joint and PHR side-copy functions pass CPU tests, but runner,
   independent scorer and real-data gradient gates are still being prepared;
   there is no calibrated GPU result yet.
3. Seek an independently auditable group-aware dataset or genuinely sealed
   cohort before a confirmatory multi-dataset claim. The original fmow2
   reserved countries remain unscored in this rebuild, but older project
   work inspected the original test pool; they are not automatically a clean
   independent confirmation. The Chen test remains sealed. A bounded
   source check found [Camelyon17-WILDS](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py)
   a plausible **metadata-audit candidate**: binary tumor labels and slide IDs
   exist, but per-slide positive support, identity overlap and archive access
   are unverified. [Waterbirds](https://github.com/kohpangwei/group_DRO/blob/master/README.md)
   is smaller and accessible, but its land/water groups are composited rather
   than independent natural sources; [CUB source images overlap ImageNet and
   carry noncommercial research terms](https://www.vision.caltech.edu/datasets/cub_200_2011/).
   No new dataset was downloaded. Any new data access or held-out use requires
   its own fixed decision and integrity audit.
4. For a claim against **full ALM**, run a separate matched end-to-end study:
   same data, backbone, initialization, supervised batches, precision,
   epochs, deployment allocator and equal tuning budget, with an explicit
   no-constraint/null arm for each schedule. The current snapshot PHR block
   answers a narrower question and must not be relabeled full ALM.
5. Add a transformer backbone as a separate atomic cell. The existing local
   runner hard-codes MobileNetV3; a no-download ViT-B/16 eight-class model
   shape check passes and both DSI hosts loaded the same cached V1 weights;
   the new eight-class factory on the servers, memory, full gradients,
   matched scorer and runtime are unverified. The
   [ViT preflight](fmow_boundary_vit_preflight_20260930.md) fixes the V1
   weights and full-frame transform disclosure, pilot 6500 and a measured
   24 GPU-hour ceiling before any full 6501–6512 block. Older ViT fmow2
   results used a different schedule and cannot supply matched controls.

No new GPU campaign is authorized merely by this report. The next pilot
depends on a fixed protocol, an independent data boundary or explicitly
exploratory reading, source/data/preprocessing/gradient/scorer gates and a
measured compute projection. Existing immutable releases and all negative
artifacts must remain intact. At the 2026-09-30 14:45 UTC inventory we used
**0 of 8 GPUs**: the four dsisco02 Blackwell cards had `liverty` compute
processes, while the four dsisco01 Quadros were free. This is a point-in-time
inventory, not a reservation or an excuse to bypass scientific gates.

The practical thesis claim today is conditional: **a small global TraLO
snapshot gain exists for some knee backbones on development data; local
country constraints have not yet improved the model, and the full-dose
version is decisively worse on fmow2.** The next useful experiment tests
whether a predeclared boundary-aware step preserves useful class-1 ranking
without the 233-to-15 overshoot, then seeks independent validation. It should
be treated as a new hypothesis, not a repaired score for the completed
blocks.

# Numerical results draft for the TraLO paper (30 September 2026)

This is a **paper-facing results section**, not a claim of held-out
confirmation. All rows below are fixed-block development evaluations. The
complete per-seed outputs and integrity records remain in the linked study
reports. Values in percentage points and decimal F1 units are deliberately
kept separate. No ViT result is available from the rebuilt, matched pipeline.

## Global snapshot intervention on knee radiographs

Yuval Kassif's knee recipe trains each backbone once per seed and evaluates
side-copy constraint steps over the same snapshot window. The deployment rule
allocates exactly 76 grade-3 slots. The primary outcome is allocated grade-3
cc-F1. The PTO arm is a snapshot-ensemble, zero-step post-hoc clipper; the
sham matches TraLO's step dose. For context, the approximate absolute means
below are recomputed from the displayed per-seed scores rounded to 0.01
percentage point. The paired effects and intervals come from each study's
prespecified scorer, not from those rounded means.

| Backbone | Seeds | PTO cc-F1 (%) | TraLO cc-F1 (%) | TraLO − PTO (pp; 95% paired CI) | TraLO − sham (pp; 95% paired CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 72 | 72.61 | 73.72 | +1.11 [+0.78, +1.45] | +1.10 [+0.77, +1.43] |
| RegNetY-400MF | 72 | 73.28 | 74.21 | +0.93 [+0.57, +1.29] | +0.92 [+0.57, +1.27] |
| MobileNetV3-Large | 72 | 73.28 | 74.45 | +1.18 [+0.84, +1.51] | +1.18 [+0.85, +1.50] |
| EfficientNet-B5 | 48 | 74.31 | 73.44 | −0.87 [−1.52, −0.22] | −0.96 [−1.63, −0.29] |

The corresponding TraLO-minus-PTO accuracy, macro-F1 and weighted-F1 effects
(percentage points) are, respectively, +0.66/−0.12/−0.25 for ResNet18,
+0.64/−0.05/−0.52 for RegNetY, +0.87/+0.93/+0.46 for MobileNetV3 and
−0.22/−2.00/−2.06 for EfficientNet-B5. The weighted-F1 losses on ResNet18
and RegNetY preclude an unqualified all-metric win. The B5 sign reversal
precludes a universal backbone claim. The 1,656-image Chen test has not been
scored. Source: [ResNet18/RegNetY complete report](claude_stepens_result_20260928.md),
[MobileNetV3/B5 complete report](claude_stepens_additional_result_20260928.md).

The same global intervention on fmow2 MobileNetV3 (48 seeds, pooled cap167)
had TraLO-minus-PTO **+0.24 percentage points** in class-1 cc-F1 (95% paired
CI −0.09 to +0.57; Holm p=0.308) and **−1.37** points of accuracy (CI −1.63
to −1.12). Thus the knee signal has not transferred as a robust satellite
result. Source: [additional step-ensemble report](claude_stepens_additional_result_20260928.md).

## Pooled-plus-country local intervention on fmow2

The later MobileNetV3 studies used a label-free pooled resource cap and five
country upper bounds, with one common country-aware allocator at deployment.
The first fixed 12-seed 0.1-dose study found joint TraLO-minus-PTO class-1
cc-F1 differences of **−0.0866** at pooled cap167 and **−0.0697** at cap83.
A second, separately seeded 12-seed block compared joint TraLO with a PHR
augmented-Lagrangian **snapshot direction** and retained pooled-only and
same-dose sham controls:

| Pooled cap | PTO/Clipper cc-F1 | Joint TraLO | Sham/null | Pooled-only | PHR snapshot |
| --- | ---: | ---: | ---: | ---: | ---: |
| 167 | 0.49260 | 0.42167 | 0.49217 | 0.43864 | 0.41471 |
| 83 | 0.39967 | 0.33445 | 0.39911 | 0.33501 | 0.33445 |

PHR-minus-PTO paired effects were −0.07789 [−0.09604, −0.05975] at cap167
and −0.06522 [−0.08113, −0.04930] at cap83; four-test Holm-adjusted p values
were 0.00000519 and 0.00000616. PHR-minus-joint was −0.00696
[−0.01079, −0.00313] at cap167 and 0.00000 [−0.00601, +0.00601] at cap83.
Both directions therefore lost to the zero-step allocator at the fixed dose.
This is **not** full ALM training versus full TraLO training. PTO here is a
local post-hoc Clipper analogue, not the historical `clip`/`focal_clip`
implementation. Source: [fixed-dose result](fmow_local_fixed_dose_result_20260930.md),
[PHR direction result and full JSON](fmow_local_alm_direction_result_20260930.md).

The training logs explain why the local result is especially concerning:
all 73 epoch opportunities at each cap applied the 0.1 step, yet mean raw
class-1 calls fell from **232.96** before intervention to **15.25/17.29**
after joint TraLO (caps167/83), versus **233.45/233.44** after the matched
sham. The allocator still filled the exact caps, so the harm was chiefly a
ranking/selection change, not unused resource. At cap167, the joint arm made
239 correct entries into the selected set but expelled 402 correct PTO
selections across seeds; PHR made 242 correct entries and 421 exits. Every
seed lost selected true positives to PTO at both caps. These logs establish
active, harmful interventions **at this radius**, not that every smaller or
different local step must fail.

## Claim boundary and next registered test

The defensible present claim is a small, backbone-dependent **global** TraLO
development gain on knee and a clear **local fixed-dose failure** on fmow2.
The fmow2 development countries have been inspected repeatedly; no new
result on them is independent confirmation. The Chen test and five reserved
fmow2 countries remain outside this table. A boundary-calibrated local step
has a fixed, label-free pilot/full design but **no GPU result yet**. The
MobileNetV3 protocol compares it with PTO, sham, pooled-only and calibrated
PHR at both caps; the ViT-B/16 extension first requires a matched source,
weight, gradient, memory and cost pilot. Those future cells will be reported
with absolute deployed metrics, every seed, paired intervals, secondary
tradeoffs, activation/dose and slot turnover, whether positive or negative.

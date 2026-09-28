# Fixed-dose local-direction fmow2 study

Status: design frozen before any run of this new study. This is a distinct,
exploratory amendment after the preserved failure in
`fmow_joint_local_pilot_failure_20260928.md`; its runs cannot be pooled with
that failed pilot or described as untouched confirmation.

## Question and design decision

The previous pilot stopped because no *sampled* displacement through radius
0.1 made every **raw** class-1 call count feasible at its first snapshot. The
final country-plus-pooled `local_capped_first` allocator is independently
feasible regardless of raw calls. Moreover, a uniform class-1 logit reduction
made raw calls feasible on the saved first snapshot while leaving all final
selected IDs unchanged. Thus raw feasibility is no longer a success gate.

Test whether a country-aware gradient improves the **identity and quality of
the final allocated class-1 set** at a matched parameter dose over the
pooled-only gradient, untouched PTO snapshot ensemble, and seeded random
direction. The treatment is the negative normalized sum of active pooled and
country soft-count gradients, each divided by its fixed capacity. Active
scopes are determined solely by raw hard calls before the step. If gradients
conflict, take the sum direction anyway, record every exact scope derivative,
and treat worsening scopes as adverse mechanism evidence. Reject zero or
nonfinite total gradient. This is a heuristic ranking intervention, not a
projection or guarantee of raw feasibility.

Use exactly radius **0.1 in full parameter L2 norm** when at least one scope
is active. This is the prior study's frozen ceiling, reused transparently as
a single stress dose, not chosen by development F1. It may be too large and
harm quality; that is a reportable result. No radius search, favorable-only
cap adjustment or training-epoch selection follows pilot scores. The
pooled-only control and sham use the treatment's exact radius and the same
PTO snapshot. The sham preserves each tensor's parameter-displacement norm.
When no scope is active, all side arms take no step.

## Fixed data, arms, outcomes

Reuse the byte-audited fmow2 15,841 training / 1,829 country-disjoint stop /
1,673 development split in `fmow_joint_local_protocol_20260928.md`, including
the five development countries and the same MobileNetV3-Large FP32 Yuval
training recipe, augmentation, class-balanced sampler, stopping rule and
snapshot ensemble window. The original other five test countries remain
unscored. The training runner must not load the development label array or
retain individual development labels in its data structures or artifacts;
the independent offline scorer alone reads them.
Compute quotas
only from unlabeled country IDs: pooled G=167 or 83; local Hamilton
size-share limits total B=209 or 104. Every arm uses the exact same final
`local_capped_first` allocator with sample-ID ties and uncapped-class fallback.

The four arms for both G values are `ens_pto`, `ens_joint_fixed`,
`ens_global_dose`, and `ens_sham`. The primary endpoint is deployed class-1
cc-F1, which for one constrained class equals its F1. Prespecified paired
contrasts are joint minus pooled-only and joint minus PTO at each of the two
caps (four contrasts, one Holm family). A positive exploratory signal
requires both contrasts positive after Holm correction at a cap and no
secondary-metric domination by either comparator at that cap. Define domination
before scoring as a negative paired joint-minus-comparator difference whose
two-sided 95% Student-t interval has an upper bound below zero on accuracy,
macro-F1 or weighted-F1. This guardrail does not establish equivalence when
its interval crosses zero. Sham differences are reported as an attribution
diagnostic. Report seed-level scores, paired
mean/SD/Student-t interval, all secondary metrics, raw calls, exact allocated
TP@K, correct entries/exits, country utilization, and every failure. The
development split has been repeatedly inspected; even a positive result here
is not external confirmation.

## Staging and stops

Pilot seed **6199** has two exclusive runs: side-step on and off, on the same
host/precision. The pilot is an integrity and cost gate, not a source of
settings. The fixed exploratory block is **6200–6211** (12 PTO fits), each
with both cap levels derived from the same fit. The full block may launch
only after exact PTO trajectory equality, source/config/data/split and
artifact hashes, label-free quota recount, finite actual gradients and
parameters, fixed-radius total and per-tensor dose controls, same named
allocator, no reserved-country reads, independent metric recomputation,
and complete pilot logs pass. For every active side step, log before/after
pooled and per-country hard and soft calls plus every scope derivative.
Actual hard cap violation after a side step is a **diagnostic**, not a gate.
CPU fixtures must cover conflicting-scope behavior, exact dose, no-step
parity, allocator/ties, and runner-shaped scoring. Any unexplained failed
integrity gate stops expansion and remains recorded; no blind retry.

The added GPU budget is at most **72 GPU-hours** for this distinct study.
Project it from the completed pilot runtime before expansion. If above the
limit, stop and seek a mathematically equivalent speedup with parity proof;
do not shrink the fixed denominator after reading outcomes. Before GPU use,
inspect physical UUID, compute PID, owner and command on both hosts and use
only truly free authorized cards. Pilot at most two cards; full block at most
three concurrently. Exclusive roots and immutable releases are mandatory.
The launch gate projects total GPU-hours as 13 times the step-on pilot duration
(pilot plus 12 full seeds) plus one step-off reference duration, divided by
3600. This projection applies to the pilot's host and precision; a different
host or precision requires a separate label-blind runtime calibration before
using it in the full block, and the total projected GPU-hours must still fit.

Separate follow-on work: extend a mathematically specified ALM to the same
country constraints and compare it at matched schedule/dose and allocator.
The existing global, frozen-feature ALM result is not a fair local baseline.
Further datasets/backbones require their own fixed identity, quota and
held-out protocol rather than selecting a win from this viewed split.

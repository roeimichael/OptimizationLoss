# Prospective fixed-scale tabular-constraint recovery, 2 October 2026

## Question and evidence boundary

Can the already specified **saturating TraLO pooled-plus-group loss** make an
applied training correction, and improve deployed constrained-class F1, when
its one fixed unnormalized parameter step is placed inside a measured safe
dose range? This is a **new dose condition**, not a continuation or repair of
the completed seed-6800 CelebA or seed-6830 ISIC pilots. Those pilots are
preserved as negative mechanism evidence: CelebA level-1 TraLO applied 0/5
corrections; cached ISIC TraLO and PHR at both levels each applied 0/5.
Their development quality was never scored and will not select a setting.

The classifier sees only images. The sex/site tabular column indexes
predeclared pooled and group upper bounds and the identical deployment
allocator for every arm. The TraLO scalar objective and exact derivative in
`tralo/local_bounded_penalty.py`, the PHR recurrence, model, ImageNet weights,
private-label boundary, six-epoch schedule, training/stop split, augmentation,
caps, checkpoint rule, allocator and 0.10 displacement veto remain fixed.
Only the **four predeclared, arm-specific scalar correction step sizes**
change. No per-step normalization, clipping or line search is introduced:
the gradient's saturation and changing magnitude can still affect later
doses. Each treated arm receives the same *maximum projected parameter dose*
at its previously recorded states. Actual dose histories must be measured;
this is maximum-dose matching, not exact epoch-by-epoch dose equality.

## Label-free scale rule frozen before fresh pilots

The completed inactive-arm pilots logged all five raw parameter-gradient
norms per treated arm on a training/stop-labeled, development-*unlabeled*
trajectory. The fixed rule rounds **down** each arm's scale from
`0.08 / that_arm_max_observed_norm`, leaving a 20% margin below the unchanged
0.10 veto at those recorded states:

| Dataset | Treated arm | Maximum old raw norm | New fixed scale | Projected maximum dose |
|---|---|---:|---:|---:|
| CelebA/MobileNetV3 | level1 TraLO | 17.678 | 0.0045 | 0.07955 |
| CelebA/MobileNetV3 | level1 PHR | 28.908 | 0.0027 | 0.07805 |
| CelebA/MobileNetV3 | level2 TraLO | 10.791 | 0.0074 | 0.07985 |
| CelebA/MobileNetV3 | level2 PHR | 7.132 | 0.0112 | 0.07988 |
| ISIC 2020 cached/MobileNetV3 | level1 TraLO | 222.040 | 0.00036 | 0.07993 |
| ISIC 2020 cached/MobileNetV3 | level1 PHR | 2078.850 | 0.000038 | 0.07900 |
| ISIC 2020 cached/MobileNetV3 | level2 TraLO | 202.808 | 0.00039 | 0.07910 |
| ISIC 2020 cached/MobileNetV3 | level2 PHR | 308.211 | 0.00025 | 0.07705 |

It uses no development labels, cc-F1,
predictions' correctness, or outcome-selected cap. It does use an earlier
training trace, so later inference applies only to this explicitly calibrated
condition. A changed trajectory can still exceed the veto; the gate must
record attempted/applied/skipped corrections and stop expansion if an arm
remains inactive. A passing liveness gate does not establish superiority.

## Frozen new jobs and controls

Use fresh, never-before-claimed IDs: CelebA pilot **6880**, fixed 6881–6884;
cached ISIC 2020 pilot **6890**, fixed 6891–6894. The arms stay PTO,
schedule-matched sham, TraLO levels 1/2, and inexact PHR levels 1/2.
The post-hoc Clipper comparator is the identical frozen allocator applied to
the PTO checkpoint probabilities; it receives no constraint-training step.
The task optimizer is unchanged by every constraint correction.
Pilot development labels remain sealed. Run each pilot on a genuinely free
physical GPU after both-host inventory and that host's release/data/gradient
parity checks. Concurrent pilots are allowed only where the particular
prepared cohort is independently verified on each selected host. Inspect
real events, doses, losses and output completeness before any fixed expansion.

The earlier cached ISIC pilot took 6,382 seconds and the CelebA pilot
12,197 seconds. Five similar jobs project 8.86 and 16.94 aggregate GPU-hours,
respectively, before gate/replay overhead or possible changed trajectories.
Each cell has a finite **24 aggregate GPU-hour** ceiling, each job a 6-hour
timeout, and both cells count against the already approved weekend
96 aggregate GPU-hour planning ceiling. A queue may claim fixed seeds only
after its pilot exits zero, an independent label-blind gate verifies hashes,
actual-image replay, pooled/group quota recount, PTO-sham exact parity,
finite task/constraint gradients, nonzero applied corrections in **every**
treated arm, correct dose and cost projection, and the selected GPU remains
genuinely free. A timeout, foreign compute PID, SSH loss or failed gate is a
forensic stop, never permission to reuse a seed.

Only after all four fixed seeds of a cell pass their own independent gates
may the scorer open that cell's development labels. Report every seed's
allocated cc-F1 (primary), accuracy, macro/weighted F1, per-group support
and confusion, feasibility, paired 95% intervals and actual GPU-hours.
No favorable pilot score chooses a backbone, cap or next condition. These
two development cohorts are exploratory; no sealed Chen test or reserved
fMoW country is touched. The comparison tests one dose-calibrated
training condition, not general optimality or fairness.

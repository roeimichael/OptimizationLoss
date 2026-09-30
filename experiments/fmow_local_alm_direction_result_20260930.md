# fmow2 local TraLO versus snapshot PHR-ALM: completed negative result

The fixed 12-seed block 6301–6312 finished on dsisco02 by 2026-09-30
09:05:09 UTC. Each seed trained one unchanged MobileNetV3-Large FP32 PTO
trajectory, then evaluated independent side copies at each epoch. All 12 jobs
exited 0; the offline scorer completed with source, data, artifact, trajectory,
quota, allocator, gradient and dose checks passing. SSH was unavailable when
the queue ended; the original queue and artifacts were inspected after access
returned. No seed was restarted. The immutable run release is
`2f6a68eb006cf2d9cd535a7dfa51f5c09ce9850e`. The exact full score,
including every seed, selected ID and diagnostic, is in
[`fmow_local_alm_direction_result_20260930.json`](fmow_local_alm_direction_result_20260930.json)
(SHA-256 `4d5514e6138750d1eeb316d7ed71b635af944ef7719f01c855060be8cbcca3e5`).

This is a **snapshot step-policy comparison**, not end-to-end ALM training
versus end-to-end TraLO training. PTO is the zero-side-step, fixed-allocator
post-hoc Clipper analogue here. The sham matches TraLO's per-tensor step dose,
not PHR's. The five development countries were repeatedly viewed in earlier
work; these results are exploratory, not independent confirmation. The five
reserved fmow2 countries and the Chen test were not scored.

## Primary result

Allocated class-1 cc-F1 means across 12 paired seeds:

| Pooled cap | PTO / Clipper analogue | Joint TraLO | Pooled-only dose | Dose-matched TraLO sham | PHR-local |
| --- | ---: | ---: | ---: | ---: | ---: |
| G=167 | 0.49260 | 0.42167 | 0.43864 | 0.49217 | 0.41471 |
| G=83 | 0.39967 | 0.33445 | 0.33501 | 0.39911 | 0.33445 |

The four prespecified paired PHR contrasts use two-sided 95% Student-t
intervals, with Holm adjustment across those four tests. Intervals are
descriptive and unadjusted for multiplicity.

| Cap | Contrast in cc-F1 | Paired mean [95% interval] | Holm p |
| --- | --- | ---: | ---: |
| G=167 | PHR − joint TraLO | −0.00696 [−0.01079, −0.00313] | 0.00417 |
| G=167 | PHR − PTO | −0.07789 [−0.09604, −0.05975] | 0.00000519 |
| G=83 | PHR − joint TraLO | 0.00000 [−0.00601, +0.00601] | 1.00000 |
| G=83 | PHR − PTO | −0.06522 [−0.08113, −0.04930] | 0.00000616 |

Neither cap meets the registered positive-signal rule. Accuracy, macro-F1 and
weighted-F1 are also lower for PHR than both PTO and joint TraLO at both caps;
their paired intervals are wholly negative. At G=167, for example, accuracy
is 0.54144 PTO, 0.41328 joint TraLO, and 0.40078 PHR. At G=83 it is
0.53930, 0.42279 and 0.39938. The scorer's complete per-seed values and
contrasts, including all negative results, are retained in the JSON.

## What the training and selection logs show

The independent read-only log audit covers 73 epoch snapshots at each cap:
seeds 6301–6311 ran six epochs each and seed 6312 ran seven. Every joint,
pooled-only, sham and PHR side step was applied in all 73 opportunities, with
finite nonzero gradients and actual full-model displacement approximately
0.1. Thus the result cannot be explained by an inactive ALM penalty or
skipped constraint updates. PHR's penalty reached zero **after** 64/73 steps
at G=167 and 15/73 at G=83; this is not the same as a zero pre-step gradient.

| Snapshot diagnostic (mean of 73) | G=167 | G=83 |
| --- | ---: | ---: |
| Raw class-1 hard calls before side step | 232.96 | 232.96 |
| After joint TraLO | 15.25 | 17.29 |
| After PHR | 14.75 | 15.26 |
| After pooled-only step | 20.52 | 20.52 |
| After sham | 233.45 | 233.44 |

The allocator subsequently filled exactly 167 or 83 pooled slots. Across
seeds at G=167, the joint step replaced 921 PTO-selected slots: 239 incoming
items were true class 1, versus 402 true class-1 exits. PHR replaced 935:
242 correct entries versus 421 correct exits. At G=83, joint had 235 correct
entries versus 352 exits; PHR had 242 versus 359. Each of the 12 seeds lost
selected class-1 true positives versus PTO for both joint and PHR at both
caps. This is measured adverse ranking at the registered 0.1 dose, not a
claim that all local-constraint directions or smaller doses fail.

Mean training loss fell from 0.92705 at each seed's first epoch to 0.10109
at its last; mean stop-country loss rose from 1.23434 to 2.13779. That
growing gap is a diagnostic, not a causal explanation of the side-step
ranking loss. The score audit independently found raw local infeasibility
after 7/73 joint and PHR steps at G=167, and 46/73 joint versus 40/73 PHR
steps at G=83. These raw violations are not final allocator violations.
The log-audit receipt is
`C:/Users/roeym/.codex/rebuild-audit-20260922/fmow_local_alm_full_log_audit_20260930.json`
(SHA-256 `6b6253a991983099017e1352f95bb8bdabf2027ccc556d05feb58a0a2786286d`).

## Interpretation and boundary

At a shared PTO trajectory and maximum L2 dose, changing the fixed local
TraLO direction to the registered PHR-ALM direction did not recover the
allocated score. At G=167 PHR was worse than TraLO; at G=83 their cc-F1 was
equal on average, and both lost to PTO. The near-null sham and negative
pooled-only arm show that the large raw-call collapse came from these
constraint directions at the full step, not from merely copying a snapshot
or moving parameters by 0.1 in an arbitrary matched direction. They do not
identify a better radius.

The older [knee/CIFAR ALM study](alm_two_dataset_result_20260924.md) used a
frozen ResNet18 head and a different update schedule. Its numeric outcomes
cannot be pooled with this snapshot-policy test as a fair local ALM comparison.
The completed [fixed-dose local study](fmow_local_fixed_dose_result_20260930.md)
used different seeds and also lost to PTO. Neither study licenses choosing
a favorable cap, epoch, country, seed, model or dose from the viewed fmow2
development outcomes.

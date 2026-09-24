# Knee occupancy-ranking result, four end-to-end seeds

**Result:** the fixed wrong-occupant ranking formulation did not demonstrate a
grade-3 slot improvement. Its derivative was zero in nearly all intervention
opportunities, so this is a design diagnosis as well as a negative comparison.
The following hard-pair protocol is separate; it is not retroactively part of
these results.

## Identity and gate

The five-arm protocol is
[`knee_cutoff_ranking_protocol_20260924.md`](knee_cutoff_ranking_protocol_20260924.md).
Immutable release `ff080795ee29571b1f2f10a912705354a458bfb8` was pushed
to GitHub and the DSI mirror; 110 tests passed locally and natively on both
hosts, tracked source hashes matched, and the Quadro smoke passed. Four seed
roots are preserved under
`/home/dsi/michaer8/tralo-rebuild/runs/knee-cutoff-20260924/` as
`pilot1401` and `seed1402`–`seed1404`. The independent auditor checked parent,
model, prediction and snapshot hashes; Chen split identity (5778 training,
826 development, 1656 held-out test, no subject/exact-pixel overlap); all 1810
applied task updates per arm; auxiliary dose and no skips; equal warm-up and
minibatch hashes within seed; and recomputed sklearn accuracy and F1 from saved
predictions. All four audits passed. The test split was not scored.

The local independent evidence is in
`C:/Users/roeym/.codex/rebuild-audit-20260922/knee_cutoff_four_seed_analysis.json`,
`knee_cutoff_1401_audit.json` through `knee_cutoff_1404_audit.json`, and
`knee_cutoff_rank_dose.json`. The two named allocators are reported separately;
no validation outcome chose the cap, epoch or learning rate.

## Endpoint comparison

All figures below are **development grade-3 F1 percentages**. Raw means argmax
without a cap. Capped-first fills exactly 76 grade-3 slots using the saved
scores. The Null is CE-only with the same phase boundary and task-Adam reset
as the auxiliary arms; ordinary Clipper retains task-Adam state. Count-only
uses the prior soft-count TraLO update. Rank-only uses training labels to pair
missed true grade-3 images with wrong training occupants of the top-532.
Rank-plus-count accumulates both gradients in one auxiliary Adam step.

| Arm | Raw mean | Capped-first mean | Capped-first seeds 1401 / 1402 / 1403 / 1404 |
|---|---:|---:|---|
| Clipper | 65.58 | 63.74 | 64.84 / 64.84 / 58.24 / 67.03 |
| Phase Null | **66.73** | **64.29** | 61.54 / 62.64 / 64.84 / 68.13 |
| Count-only | 42.84 | 61.54 | 58.24 / 65.93 / 62.64 / 59.34 |
| Rank-only | 66.03 | 63.46 | 63.74 / 62.64 / 64.84 / 62.64 |
| Rank-plus-count | 48.66 | 62.36 | 58.24 / 65.93 / 65.93 / 59.34 |

The prespecified primary contrast, rank-plus-count minus rank-only capped-first
F1, was **−1.10 points** across matched seeds (sample SD 4.01, exploratory 95%
paired t interval [−7.48,+5.29]); its four deltas were −5.49, +3.30, +1.10,
−3.30 points. Rank-only minus phase Null averaged −0.82 points (interval
[−6.05,+4.40]); count-only minus phase Null averaged −2.75 (interval
[−10.63,+5.14]). Count-only raw F1 was below Null in all four seeds, with a
mean paired difference of −23.89 points (interval [−36.13,−11.65]). The
upper-bound-correction results and every saved seed/transition are in the
audited JSON. Four development seeds and repeatedly inspected development
labels are exploratory evidence, not independent publication confirmation.

## Why this loss was mostly absent

Its pair set requires both a missed true training grade-3 image *and* a wrong
non-grade-3 occupant inside the training top-532. The training cohort had about
757 true grade-3 images, so roughly 225 can remain outside an exact 532 slots
even when every slot is correct. Across the four rank-only arms, wrong occupants
appeared at only **2 of 20** post-warmup opportunities: two in seed 1401 epoch
7 and one in seed 1404 epoch 8. The rank-plus-count arms had an active pair at
only **1 of 20** opportunities (seed 1403 epoch 7). At every other opportunity,
active pairs, rank loss and rank gradient were exactly zero. Rank-only was
byte-identical to Null in seeds 1402 and 1403; rank-plus-count was identical to
count-only in seeds 1401, 1402 and 1404. This is predicted by the specified
loss, not evidence of a missing optimizer step or a GPU execution error.

The grade-3 quota can still generalize imperfectly: the training top-532 may
contain no wrong occupants while development top-76 contains only 53–62 correct
images. Thus pure training occupancy is an insufficient stopping signal for a
loss meant to improve development slot quality. The next fixed hypothesis in
[`knee_hard_pair_protocol_20260924.md`](knee_hard_pair_protocol_20260924.md)
keeps pressure on weak true training positives versus strong training negatives
even when the top-K training slots are pure. It still requires a matched Null
and fresh seeds; no current result establishes its benefit.

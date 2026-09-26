# Targeted step at a deeper cut (grade-3 cap 50) -- RESULT: READING 2 (2026-09-26)

Protocol: `claude_targeted_step_cap50_protocol_20260926.md` (fixed before any seed ran).
- Release cd6a205a, dsisco01 GPUs 0-3.
- Seeds 1901-1924 all complete, all reported.
- Scorer output: `analysis/score_target50_20260926.json`.
- Runs: `~/tralo-rebuild/runs/claude-target50-20260926`.

## Integrity

- The pilot (1901) passed its gate.
- All 24 seeds pass every item: matched hashes, 1810 task updates, every targeted step lands
  at or below 50, the sham triggers on its own violation, and first radii are equal.
- Prediction hashes are 24 distinct per arm.
- `tralo_target` and `sham_target` each applied 120 steps (5 per seed: the cap binds at every check).

## Primary (Holm, n = 24, capped_first grade-3 F1 in points; F1 = 2TP/156)

| contrast | mean [95% CI] | p | Holm p | W/T/L | correct slots of 50 |
|---|---|---|---|---|---|
| **C1** tralo_target - tralo_null | +0.96 [-0.87, +2.79] | 0.29 | 0.58 | 12/5/7 | +0.75 [-0.68, +2.18] |
| **C2** tralo_target - sham_target | -0.75 [-2.45, +0.96] | 0.37 | 0.58 | 9/5/10 | -0.58 [-1.91, +0.75] |

**Preregistered reading 2: the negative result holds at a deep cut as well.** The constraint's
information (C2) is bounded below +1.0 F1 point (~0.75 of 50 slots) at 95%.

## Secondary

| contrast | mean [95% CI] | p |
|---|---|---|
| tralo_adam - tralo_null | +1.44 [-0.56, +3.44] | 0.15 |
| sham_target - tralo_null | +1.71 [-0.32, +3.74] | 0.09 |
| clipper - tralo_null | +0.53 [-1.80, +2.87] | 0.64 |
| tralo_target - tralo_adam | -0.48 [-2.42, +1.46] | 0.61 |

- **Raw F1:** target - null -8.78 and target - sham -10.58. This is not a like-for-like
  comparison: the target arm's raw argmax is cut to 50 calls against 106 true grade 3, while the
  others call ~90. It is reported because the protocol lists it, but it measures call volume,
  not ranking.
- **Slot turnover** against the null (reseed floor 0.340): clipper 0.279, adam 0.284,
  target 0.258, sham 0.272.

## Post-hoc (not preregistered): who each step evicts

| arm | steps | items evicted by the step | not truly grade 3 | overlap with the post-hoc cut | post-hoc cut: not grade 3 | capped correct slots per step |
|---|---|---|---|---|---|---|
| tralo_target | 120 | 5412 | 49.3% | 87.1% | 48.4% | +0.62 (sd 1.58) |
| sham_target | 120 | 29 | -- | -- | 48.7% | +0.00 (sd 0.26) |
| tralo_adam | 119 | 8902 | 36.1% | 57.4% | 48.5% | +0.22 (sd 2.66) |

At the deep cut, the step's evictions are marginally more precise than the post-hoc cut
(+0.9 points) and add +0.62 correct slots per step within an epoch. That is the largest
"who" signal measured in this study, but it does not survive to the endpoint: C2 is -0.75.
The next task epoch re-trains the boundary and washes the step out.

## Both operating points together

| cap | C1 target - null | C2 target - sham | step/post-hoc eviction overlap |
|---|---|---|---|
| 76 | +0.96 [-0.96, +2.88] | +0.14 [-1.21, +1.48] | 83% |
| 50 | +0.96 [-0.87, +2.79] | -0.75 [-2.45, +0.96] | 87% |

Two preregistered studies, 48 seeds, same verdict.
- At the right dose, the constraint direction lands the count exactly and evicts the items the
  post-hoc clipper would evict.
- It does not choose better patients than a random move of the same size.

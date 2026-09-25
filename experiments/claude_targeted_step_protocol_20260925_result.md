# TraLO's direction at the dose that lands on the hard cap -- RESULT: READING 2 (2026-09-26)

Protocol: `claude_targeted_step_protocol_20260925.md` (fixed before any seed ran). Release
7c7cd3b7, dsisco01 GPUs 0-3, OMP_NUM_THREADS=8. Seeds 1801-1824 all complete, all reported.
Scorer: `analysis/score_sham.py`; full output is in `analysis/score_target_20260925.json`.
Runs: `~/tralo-rebuild/runs/claude-target-20260925` on dsisco01.

## Integrity

- The pilot (1801) passed every gate item.
- All 24 seeds: warm-up and batch hashes are identical across arms, and every arm has 1810 task updates.
- Every applied `tralo_target` step went from hard_before > 76 to hard_after <= 76.
- Every applied sham step was triggered by its own hard violation.
- When both targeted arms stepped at the first check, their first radii were equal.
- Prediction hashes: 24 distinct per arm, so n = 24 is real (no copies).

`tralo_target` applied 96 steps (sham 94) over 24 x 5 checks, with radii of 0.0002-0.018. The published
step's displacement is 0.10. Sham steps left the hard count essentially unchanged (e.g. 119 -> 118);
targeted steps moved it to exactly the cap.

## Primary (Holm, n = 24, capped_first grade-3 F1 in points)

| contrast | mean [95% CI] | p | Holm p | W/T/L | correct slots of 76 |
|---|---|---|---|---|---|
| **C1** tralo_target - tralo_null | +0.96 [-0.96, +2.88] | 0.31 | 0.62 | 13/0/11 | +0.88 [-0.87, +2.62] |
| **C2** tralo_target - sham_target | +0.14 [-1.21, +1.48] | 0.83 | 0.83 | 9/3/12 | +0.12 [-1.10, +1.35] |

**Pre-registered reading 2: fixing the dose does not rescue the method.** The intervals are
tighter than the stated MDE (4.5 points), because the paired SDs came in at 4.5 (C1) and 3.2 (C2),
below the 7.87 assumed. So this is a bounded null, not an underpowered one:
- the constraint's INFORMATION (C2) is worth at most +1.5 F1 points (~1.3 correct slots) at 95%;
- the whole constrained schedule (C1) is worth at most +2.9 points.

## Secondary (t intervals, not in the family)

| contrast | mean [95% CI] | p | W/T/L |
|---|---|---|---|
| raw F1, tralo_target - tralo_null | -1.29 [-3.66, +1.08] | 0.27 | |
| raw F1, tralo_target - sham_target | -1.17 [-2.77, +0.43] | 0.14 | |
| tralo_adam - tralo_null (published arm) | -2.01 [-4.38, +0.35] | 0.09 | 8/1/15 |
| sham_target - tralo_null | +0.82 [-1.22, +2.87] | 0.41 | 14/2/8 |
| clipper - tralo_null | -0.23 [-1.93, +1.47] | 0.78 | 7/6/11 |
| tralo_target - tralo_adam | **+2.98 [+1.24, +4.71]** | 0.002 | 18/1/5 |

Arm means (capped_first / raw):

| arm | capped_first | raw |
|---|---|---|
| clipper | 64.97 | 66.29 |
| null | 65.20 | 66.81 |
| adam | 63.19 | 38.58 |
| target | 66.16 | 65.52 |
| sham | 66.03 | 66.69 |

Slot turnover against the null (null reseed floor 0.301):

| arm | turnover | vs floor |
|---|---|---|
| clipper | 0.240 | 0.80x |
| adam | 0.282 | 0.94x |
| target | 0.223 | 0.74x |
| sham | 0.230 | 0.76x |

## What this settles

1. **The published arm's damage is the dose.** At the published step size, raw grade-3 F1 falls
   to 38.6. The same direction at the size the cap asks for recovers +3.0 points over it
   (p = 0.002, 18/5). The v1 pilot's 10x overshoot replicates across 24 seeds.
2. **At the right dose, the direction is no better than noise at choosing WHICH patients fill
   the slots.** A random move of identical radius and per-tensor norms changes the slot set by
   the same amount (0.23 vs 0.22 turnover) and scores the same (+0.14). The constraint direction
   is highly informative about HOW MANY: it lands the count exactly, where the sham leaves it
   unchanged. It carries no measurable information about WHO. On capped_first, the allocator
   already fixes how many, so only "who" can score.
3. This closes the strongest remaining objections to the negative result on this cell:
   - the inert controller (D1);
   - the soft-count trigger (D2);
   - non-binding caps (D3);
   - the frozen head (D4);
   - the test-label path (D7);
   - the 10x overshoot.
   It is a single cell (knee, grade-3 cap 76, ResNet18, 5+5 epochs).

## Post-hoc analysis (not preregistered): WHO each step evicts

`analysis/eviction_precision.py` runs offline with development labels, after training. It
compares each applied step's raw grade-3 evictions with the evictions capped_first makes from
the SAME pre-step probabilities. That post-hoc cut is the baseline any step must beat.

| arm | steps | items evicted by the step | not truly grade 3 | overlap with the post-hoc cut | post-hoc cut: not grade 3 | correct capped slots per step |
|---|---|---|---|---|---|---|
| tralo_target | 96 | 1966 | 57.3% | 83.0% | 58.6% | +0.12 (sd 1.20) |
| sham_target | 94 | 6 | -- | -- | 59.4% | -0.01 (sd 0.18) |
| tralo_adam | 104 | 8017 | 36.3% | 27.3% | 57.7% | -2.94 (sd 4.25) |

Grade 3 is 12.8% of the development items, so any eviction rule scores well above chance.

**The targeted constraint step IS the post-hoc clipper, carried out through the weights.**
- It evicts nearly the same items (83% overlap), with slightly LOWER precision (57.3% vs 58.6%).
- That is why it cannot beat the allocator that already scores the endpoint: the soft-count
  gradient ranks items by the same probabilities the allocator cuts on.
- The published dose evicts ~4x the needed number and, past the uncertain items, reaches into
  the true grade 3 (36% precision): that is its damage.

This is the end-to-end, trainable-backbone counterpart of settled result #3
(eviction given the probabilities is already ~optimal).

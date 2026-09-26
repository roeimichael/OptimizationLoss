# Step probe: result -- 2026-09-26

- **Protocol:** `claude_step_probe_20260926.md` plus amendment 1, which fixed the primary tests before scoring.
- **Scorer:** `analysis/score_step_probe.py` at 4b2 HEAD of the branch. Raw output is in `analysis/step_probe_score_20260926.txt`.
- **Seeds:** 2601-2624, cap 50, release e7e02085.

**Seed 2605 is excluded, per amendment 1.** At f = 1.0 its hard count was 43, already under the cap of 50, so the step was not applied. n = 23.

| f | radius | tralo - S | sham - S | tralo - sham | Holm p | not-grade-3 among evicted |
|---|---|---|---|---|---|---|
| 1.0 | 0.012 | +0.61 [-0.01, +1.23] | +0.04 | +0.57 [-0.04, +1.17] | 0.20 | 46.9% |
| 0.8 | 0.017 | +0.57 | +0.04 | +0.52 [-0.27, +1.31] | 0.37 | 43.9% |
| 0.6 | 0.023 | -0.04 | +0.04 | -0.09 [-1.07, +0.90] | 0.86 | 39.9% |
| 0.4 | 0.032 | -2.35 | +0.04 | **-2.39 [-3.90, -0.88]** | 0.014 | 35.6% |
| 0.2 | 0.041 | -6.04 | +0.04 | **-6.09 [-7.73, -4.45]** | <1e-4 | 32.0% |

Units are correct capped slots, with 95% t CIs over seeds. The shams are genuine: they displace probabilities by 0.01-0.14 and swap 0-3 slots, with no net gain.

**Primary 2, the slope of (tralo - sham) on depth:** -8.11 [-10.32, -5.90] slots per unit depth.

## Reading (fixed in amendment 1): (b') plus (c)

- **(b') No depth where the step helps.** The step lands exactly on the cap (f = 1.0) and gives +0.57 slots, which is not significant after Holm. This is consistent with the epoch-6 +1.12 of the cap-50 target study, at about half its size.
- **(c) Pushing deeper actively damages the ranking.** Every step below the cap costs slots, and the damage is steep.
- **Why:** each deeper push evicts a larger share of correct items (47% -> 32% not grade 3). The added push reaches items with high p3, and those are mostly correct. This matches the mechanism in `analysis/lab_synthesis/BOUNDARY_DIRECTION.md` on the rebuild branch: the count's push carries nothing beyond p3, and on items near p3 = 1 it can only remove correct ones.
- **Consequence:** the published 10x overshoot is not merely a wasted dose. At depth it is the damage (LEDGER #9). No depth of the count direction carries more who-information than the exact-cap step, and that step's information does not reach significance.

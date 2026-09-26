# Snapshot-ensemble window study (EXPLORATORY, 2026-09-26)

Not a confirmatory test: windows, averaging spaces and the near-cut band were chosen after the
preregistered 6-10 result was known. Dev ('val') rows only, 826 items, grade-3 slots scored against
labels offline; knee test split untouched. 4 roots x 24 seeds. Scripts: `ens_window.py` (raw, run on
dsisco01 under ~/tralo-rebuild/lab/ens_window) and `summarise.py`; full tables in `summary.txt`,
numbers in `summary.json`, per-(root, seed, arm) raw in `raw.json`. Sanity: win5 gains reproduce
`score_snapshot_ensemble.py` exactly on target-20260925 and bandcons-a1cap50.

## Q1 Window (arm-averaged per seed, correct slots vs BASE, 95% t-CI)
| root | 9-10 | 8-10 | 7-10 | 6-10 | 6-10 +before (10) |
|---|---|---|---|---|---|
| target (cap 76) | +1.35 [0.87,1.83] | +2.48 [1.92,3.05] | +2.62 [1.98,3.27] | +2.98 [2.30,3.67] | +3.36 [2.63,4.09] |
| target50 (cap 50) | +0.97 [0.58,1.37] | +1.31 [0.87,1.74] | +1.78 [1.28,2.29] | +1.93 [1.34,2.53] | +2.02 [1.42,2.61] |
| bandcons (cap 50) | +0.94 [0.51,1.36] | +1.51 [1.07,1.94] | +1.92 [1.49,2.34] | +2.10 [1.72,2.48] | +2.31 [1.89,2.74] |
| cutpair (cap 76) | +0.98 [0.55,1.42] | +1.71 [1.18,2.24] | +2.04 [1.45,2.63] | +2.38 [1.79,2.97] | +2.81 [2.13,3.49] |

Monotone (with diminishing increments) in all 4 roots; 18/21 arms non-decreasing over 2..5, the 3
dips are <= 0.25 slots. Before==after snapshots are identical for clipper/tralo_null in the two
target roots (so "10" == 6-10 there); they differ for every arm in bandcons/cutpair.

## Q2 Averaging space (6-10, arm-averaged; method minus prob-mean, paired t)
| root | prob | geo | logodds | rank | geo-prob | logodds-prob | rank-prob |
|---|---|---|---|---|---|---|---|
| target | +2.98 | +2.46 | +2.88 | +2.36 | -0.53 p=.002 | -0.11 p=.36 | -0.63 p=.001 |
| target50 | +1.93 | +1.86 | +2.07 | +1.82 | -0.07 p=.56 | +0.14 p=.23 | -0.11 p=.30 |
| bandcons | +2.10 | +2.03 | +2.09 | +1.84 | -0.08 p=.47 | -0.01 p=.88 | -0.26 p=.025 |
| cutpair | +2.38 | +2.28 | +2.52 | +2.27 | -0.10 p=.21 | +0.13 p=.20 | -0.12 p=.30 |

Prob-mean and logodds-mean are indistinguishable; geo and rank are never better and sometimes worse.

## Q3 Mechanism
Spearman(per-seed gain, near-cut between-epoch log-odds std), pooled arms x seeds: target +0.02
(p=.86), target50 -0.10 (p=.28), bandcons +0.18 (p=.028, between-arm: bandcons arms have 2-3x the
std and slightly larger gains), cutpair +0.15 (p=.10); per arm all |rho| <= 0.33, none significant.
Single-snapshot correct slots epochs 6..10 are flat within ~1-2 slots (e.g. target clipper
60.0/60.9/59.9/59.1/59.1, ens 62.5); ep10's mean rank among the 5 epochs is 1.2-2.3 (0 = worst,
2 = median). The ensemble beats the best single-epoch mean by 0.6-2.9 slots in all 21 arms: pure averaging,
not a bad final epoch.

## Q4 Slot turnover (mean 1 - overlap/cap)
ENS vs BASE 0.13-0.26; epoch 9 vs 10 0.20-0.35; BASE seed-vs-seed (same val ids, 276 pairs)
0.24-0.40. ENS moves ~70% as many slots as one epoch step and ~55% of the reseed floor, in every arm.

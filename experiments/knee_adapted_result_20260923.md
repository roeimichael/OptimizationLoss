# Knee adapted-backbone results, 23 September 2026

Four independent ResNet18 training seeds901-904. Each backbone received905 training-only CE updates (five epochs), then each Clipper/null/TraLO head received460 supervised updates. TraLO applied15 separate constraint updates. All saved-artifact hashes, warmup/batch identities and zero-skip dose checks passed. Scores were independently checked against sklearn in the runner.

Data: original Chen v1 OAI split;5778 training and826 validation images. Test1656images were audited for overlap but not scored. Same synthetic caps: grade3<=82,grade4<=16. FP32 Blackwell. All comparisons share their seed-specific backbone. The constraint term trains only the final linear head.

Scores are percentage points. ccF1 is mean F1 for grades3 and4. Intervals are unadjusted paired Student-t95% intervals over four seeds, not multiplicity-corrected confirmation.

| Allocation | Metric | Clipper mean | Matched-null mean | TraLO mean | TraLO minus null [95% CI] |
|---|---|---:|---:|---:|---|
| raw | accuracy | 61.259 | 61.350 | 61.289 | -0.061 [-0.253, +0.132] |
| raw | macro_f1 | 61.367 | 62.017 | 61.843 | -0.175 [-0.610, +0.260] |
| raw | cc_f1 | 74.938 | 77.087 | 76.643 | -0.444 [-1.448, +0.560] |
| upper_bound_correction | accuracy | 60.442 | 60.048 | 60.109 | +0.061 [-0.051, +0.172] |
| upper_bound_correction | macro_f1 | 58.646 | 58.309 | 58.386 | +0.077 [-0.087, +0.240] |
| upper_bound_correction | cc_f1 | 68.561 | 68.561 | 68.694 | +0.133 [-0.290, +0.556] |
| capped_first | accuracy | 60.442 | 60.079 | 60.109 | +0.030 [-0.066, +0.127] |
| capped_first | macro_f1 | 58.647 | 58.362 | 58.417 | +0.055 [-0.116, +0.227] |
| capped_first | cc_f1 | 68.561 | 68.694 | 68.827 | +0.133 [-0.290, +0.556] |

## Every seed: capped-first constrained-class F1

| Seed | Clipper | Null | TraLO | TraLO minus null |
|---|---:|---:|---:|---:|
| 901 | 67.665 | 67.665 | 67.665 | +0.000 |
| 902 | 73.911 | 72.848 | 73.380 | +0.532 |
| 903 | 66.601 | 67.665 | 67.665 | +0.000 |
| 904 | 66.069 | 66.601 | 66.601 | +0.000 |

## Interpretation

The supervised representation improved substantially relative to frozen ImageNet features, but that improvement is shared by all methods. The constraint contribution remains small and uncertain. No reliable TraLO advantage is established. This does not prove impossibility or reproduce Yuval's paper.

Engineering note: parallel adaptations used excessive CPU threads and progressed slowly, without skipped updates or failed numerics. New comparison launches902-904 bounded OpenMP/MKL/OpenBLAS threads to4. Seed901 used the earlier default; GPU precision/source stayed fixed.

Full remote evidence: /home/dsi/michaer8/tralo-rebuild/runs/knee-adapted-20260923. Release e101c5bd0dc837cdf048122d2b81ca80cdf8c9ac. Original48 frozen-feature fits remain in the separate knee-chen-20260923 root; all recipes must remain visible.
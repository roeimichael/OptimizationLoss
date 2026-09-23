# Knee first-stage results, 2026-09-23

Independent Chen v1 OAI comparison, not verified Kassif/Singer replication. 5,778 train,826 validation,1,656 test images. No subject-ID or exact decoded-pixel overlap; test images audited but not scored. Frozen ImageNet ResNet18,20 head epochs,warmup5,FP32 Blackwell. Seeds901-904. Synthetic global caps grade3<=82,grade4<=16. Constraint Adam is separate; both Adam learning rates0.001.

All36 fits complete. Each applied460 supervised updates; TraLO applied15 constraint updates. All warmup and batch hashes match per seed; output hashes verified. Independent sklearn macroF1/ccF1/accuracy checks passed in every run. Four head seeds share a frozen feature extractor; these are exploratory development results, not independent backbone training replications.

All scores below are constrained-class F1 percentage points (mean F1 of grades3 and4). Intervals are unadjusted paired Student-t95% intervals with3 degrees of freedom; not multiplicity-adjusted discoveries.

| Recipe | Allocation | Clipper | Matched null | TraLO | TraLO minus null [95% CI] |
|---|---|---:|---:|---:|---|
| none | upper_bound_correction | 35.83 | 38.00 | 23.63 | -14.36 [-22.82, -5.91] |
| none | capped_first | 39.65 | 41.08 | 40.13 | -0.95 [-5.25, +3.35] |
| margin | upper_bound_correction | 35.83 | 38.25 | 23.91 | -14.34 [-23.94, -4.75] |
| margin | capped_first | 39.65 | 41.48 | 40.98 | -0.50 [-3.47, +2.47] |
| false_positive | upper_bound_correction | 35.83 | 37.13 | 22.80 | -14.33 [-22.36, -6.30] |
| false_positive | capped_first | 39.65 | 41.21 | 40.85 | -0.36 [-2.56, +1.83] |

Upper-bound correction only repairs excess; capped-first reallocates using constrained slots first. They are different policies and cannot be pooled. All raw final outputs already satisfied caps in this stage, so upper-bound correction equals raw.

Finding: no reliable TraLO advantage over matched null. Severe raw suppression remains. Base seed901 final constraint step reduced grade3 hard predictions95->29 against cap82; null finished71 grade3 predictions, although its soft probability sum104.09 exceeded82. This documents a soft-count/argmax mismatch and an excessive final step; it does not establish their relative causal contribution.

Next preregistered diagnostic: base recipe only, reduce separate constraint Adam learning rate tenfold to0.0001; supervised rate remains0.001, all other settings/caps/seeds unchanged. One-seed gate then remaining3. This is a development follow-up chosen after observed overshoot, not confirmatory evidence or a search until significance. Margin/FP recipes remain visible.

Artifacts: /home/dsi/michaer8/tralo-rebuild/runs/knee-chen-20260923; local copy C:/Users/roeym/.codex/rebuild-audit-20260922/knee-chen-20260923. Original release fb688418d7852d0f1d2112981ee508dcb418dc95.
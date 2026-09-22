# Four-seed TraLO-derived budget diagnostic

Frozen ResNet18 / CIFAR-100; 10,000 training and 2,000 development images. Seeds [701, 702, 703, 704]; 10 head epochs, warm-up 5; rho 0.5 to 100.0, shared Adam. The previously diagnosed training failure is retained to isolate allocation. This is not a repaired-TraLO test or a run-until-satisfied experiment.

Accuracy is percent correct. Macro-F1 averages all 100 class F1 scores; constrained F1 averages the same 10 constrained classes, including zero-cap classes. F1 is displayed on a 0–100 scale. Higher is better for these scores. Excess is the sum of predictions above individual caps; 0 means feasible. Changed counts label changes relative to that arm’s raw argmax.

Original class:cap pairs are 0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10. All other classes are uncapped. Derived caps are that seed’s raw TraLO counts, with no label access. Upper correction changes excess assignments only. Capped-first rebuilds assignments using the entire probability matrix. Raw has no allocation.

## Observed result and interpretation

The unchanged TraLO recipe predicted only 14, 6, 9 and 7 samples across the ten constrained classes, whose original total capacity was 100. Using these counts as new budgets therefore transferred severe underprediction to Clipper and null. It did not repair the training failure. Under full reallocation at these derived budgets, mean constrained F1 was 6.10 for TraLO, versus 6.32 for either control. At original budgets, the corresponding means were 47.02, 51.78 (Clipper), and 52.24 (null). This is evidence about this failed training recipe, not a rejection of learned budgets with a healthier model.

Self-count upper correction changed zero labels in every seed, as required mathematically. Full reallocation changed 2, 4, 0 and 4 TraLO labels. Every allocated output satisfied its declared budgets. All derived budgets here were tighter than or equal to the originals; the implementation also tests the case of a derived cap exceeding the original.

Data audit: every one of the 12,000 cached feature rows was regenerated from the original images with maximum absolute error 0. Dataset/weights/artifact hashes, split IDs, label order and feature shapes passed. Exact image duplicates within and across the selected splits: zero. No semantic-duplicate/pretraining-overlap guarantee. All 12 runs completed 400 planned supervised updates with zero skipped updates. The 59-test suite passed locally and on both DSI hosts, plus an on-server GPU smoke.

Training release: `d95fc0b3d4aafa550b5b56d70b817ed02c9f0425`, dsisco01 GPU0, FP32. Reporter additionally received a metadata-only change after training. Raw evidence stays outside the source tree at `C:/Users/roeym/.codex/rebuild-audit-20260922/achieved-counts-run`; the matching remote root is `/home/dsi/michaer8/tralo-rebuild/runs/achieved-counts-d95fc0b3-20260922T152521Z`.

## Budgets actually used

| Seed | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 701 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 10 | 14 |
| 702 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 4 | 0 | 0 | 6 |
| 703 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 8 | 0 | 9 |
| 704 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 1 | 0 | 7 |

## Seed 701

| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 | Changed | Excess: original | Excess: derived |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clipper | original | raw | 61.80 | 61.56 | 59.34 | 0 | 70 | 156 |
| clipper | original | upper_bound_correction | 60.50 | 60.25 | 51.96 | 70 | 0 | 86 |
| clipper | original | capped_first | 60.50 | 60.27 | 52.83 | 78 | 0 | 86 |
| clipper | TraLO-derived | upper_bound_correction | 57.50 | 54.95 | 8.06 | 156 | 0 | 0 |
| clipper | TraLO-derived | capped_first | 57.50 | 54.95 | 8.06 | 156 | 0 | 0 |
| tralo_null | original | raw | 62.15 | 61.71 | 58.31 | 0 | 69 | 154 |
| tralo_null | original | upper_bound_correction | 60.95 | 60.49 | 52.10 | 69 | 0 | 85 |
| tralo_null | original | capped_first | 60.95 | 60.55 | 52.83 | 76 | 0 | 86 |
| tralo_null | TraLO-derived | upper_bound_correction | 58.00 | 55.24 | 8.06 | 154 | 0 | 0 |
| tralo_null | TraLO-derived | capped_first | 58.00 | 55.24 | 8.06 | 154 | 0 | 0 |
| tralo | original | raw | 57.40 | 54.73 | 7.22 | 0 | 0 | 0 |
| tralo | original | upper_bound_correction | 57.40 | 54.73 | 7.22 | 0 | 0 | 0 |
| tralo | original | capped_first | 59.80 | 59.39 | 47.31 | 86 | 0 | 86 |
| tralo | TraLO-derived | upper_bound_correction | 57.40 | 54.73 | 7.22 | 0 | 0 | 0 |
| tralo | TraLO-derived | capped_first | 57.45 | 54.81 | 8.06 | 2 | 0 | 0 |

## Seed 702

| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 | Changed | Excess: original | Excess: derived |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clipper | original | raw | 62.00 | 61.83 | 58.99 | 0 | 88 | 182 |
| clipper | original | upper_bound_correction | 60.55 | 60.39 | 49.66 | 88 | 0 | 94 |
| clipper | original | capped_first | 60.60 | 60.45 | 50.33 | 94 | 0 | 94 |
| clipper | TraLO-derived | upper_bound_correction | 57.35 | 55.04 | 5.17 | 182 | 0 | 0 |
| clipper | TraLO-derived | capped_first | 57.35 | 55.04 | 5.17 | 182 | 0 | 0 |
| tralo_null | original | raw | 62.20 | 61.99 | 59.84 | 0 | 89 | 183 |
| tralo_null | original | upper_bound_correction | 60.70 | 60.55 | 49.66 | 89 | 0 | 94 |
| tralo_null | original | capped_first | 60.95 | 60.87 | 52.07 | 95 | 0 | 94 |
| tralo_null | TraLO-derived | upper_bound_correction | 57.55 | 55.25 | 5.17 | 183 | 0 | 0 |
| tralo_null | TraLO-derived | capped_first | 57.55 | 55.25 | 5.17 | 183 | 0 | 0 |
| tralo | original | raw | 57.40 | 54.56 | 3.00 | 0 | 0 | 0 |
| tralo | original | upper_bound_correction | 57.40 | 54.56 | 3.00 | 0 | 0 | 0 |
| tralo | original | capped_first | 60.00 | 59.41 | 45.89 | 94 | 0 | 94 |
| tralo | TraLO-derived | upper_bound_correction | 57.40 | 54.56 | 3.00 | 0 | 0 | 0 |
| tralo | TraLO-derived | capped_first | 57.50 | 54.77 | 5.17 | 4 | 0 | 0 |

## Seed 703

| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 | Changed | Excess: original | Excess: derived |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clipper | original | raw | 61.45 | 61.23 | 57.88 | 0 | 81 | 172 |
| clipper | original | upper_bound_correction | 60.35 | 60.13 | 50.94 | 81 | 0 | 91 |
| clipper | original | capped_first | 60.40 | 60.24 | 51.81 | 93 | 0 | 91 |
| clipper | TraLO-derived | upper_bound_correction | 57.35 | 54.90 | 6.11 | 172 | 0 | 0 |
| clipper | TraLO-derived | capped_first | 57.35 | 54.90 | 6.11 | 172 | 0 | 0 |
| tralo_null | original | raw | 61.60 | 61.43 | 57.77 | 0 | 79 | 170 |
| tralo_null | original | upper_bound_correction | 60.35 | 60.22 | 50.38 | 79 | 0 | 91 |
| tralo_null | original | capped_first | 60.40 | 60.33 | 51.25 | 89 | 0 | 91 |
| tralo_null | TraLO-derived | upper_bound_correction | 57.45 | 55.12 | 6.11 | 170 | 0 | 0 |
| tralo_null | TraLO-derived | capped_first | 57.45 | 55.12 | 6.11 | 170 | 0 | 0 |
| tralo | original | raw | 58.30 | 55.59 | 6.11 | 0 | 0 | 0 |
| tralo | original | upper_bound_correction | 58.30 | 55.59 | 6.11 | 0 | 0 | 0 |
| tralo | original | capped_first | 60.50 | 59.95 | 46.36 | 91 | 0 | 91 |
| tralo | TraLO-derived | upper_bound_correction | 58.30 | 55.59 | 6.11 | 0 | 0 | 0 |
| tralo | TraLO-derived | capped_first | 58.30 | 55.59 | 6.11 | 0 | 0 | 0 |

## Seed 704

| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 | Changed | Excess: original | Excess: derived |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clipper | original | raw | 61.20 | 60.90 | 60.16 | 0 | 91 | 184 |
| clipper | original | upper_bound_correction | 59.90 | 59.47 | 51.03 | 91 | 0 | 93 |
| clipper | original | capped_first | 60.00 | 59.62 | 52.14 | 101 | 0 | 93 |
| clipper | TraLO-derived | upper_bound_correction | 56.70 | 54.10 | 5.94 | 184 | 0 | 0 |
| clipper | TraLO-derived | capped_first | 56.70 | 54.10 | 5.94 | 184 | 0 | 0 |
| tralo_null | original | raw | 61.70 | 61.36 | 60.06 | 0 | 85 | 178 |
| tralo_null | original | upper_bound_correction | 60.25 | 59.82 | 51.94 | 85 | 0 | 93 |
| tralo_null | original | capped_first | 60.30 | 59.89 | 52.81 | 91 | 0 | 93 |
| tralo_null | TraLO-derived | upper_bound_correction | 56.90 | 54.22 | 5.94 | 178 | 0 | 0 |
| tralo_null | TraLO-derived | capped_first | 56.90 | 54.22 | 5.94 | 178 | 0 | 0 |
| tralo | original | raw | 56.80 | 54.10 | 5.07 | 0 | 0 | 0 |
| tralo | original | upper_bound_correction | 56.80 | 54.10 | 5.07 | 0 | 0 | 0 |
| tralo | original | capped_first | 59.65 | 59.11 | 48.54 | 93 | 0 | 93 |
| tralo | TraLO-derived | upper_bound_correction | 56.80 | 54.10 | 5.07 | 0 | 0 | 0 |
| tralo | TraLO-derived | capped_first | 56.80 | 54.09 | 5.07 | 4 | 0 | 0 |

## Means ± seed standard deviation (four seeds)

| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 |
|---|---|---|---:|---:|---:|
| clipper | original | raw | 61.61 ± 0.36 | 61.38 ± 0.40 | 59.09 ± 0.95 |
| clipper | original | upper_bound_correction | 60.33 ± 0.30 | 60.06 ± 0.41 | 50.90 ± 0.94 |
| clipper | original | capped_first | 60.38 ± 0.26 | 60.15 ± 0.36 | 51.78 ± 1.05 |
| clipper | TraLO-derived | upper_bound_correction | 57.22 ± 0.36 | 54.75 ± 0.44 | 6.32 ± 1.23 |
| clipper | TraLO-derived | capped_first | 57.22 ± 0.36 | 54.75 ± 0.44 | 6.32 ± 1.23 |
| tralo_null | original | raw | 61.91 ± 0.31 | 61.62 ± 0.29 | 58.99 ± 1.13 |
| tralo_null | original | upper_bound_correction | 60.56 ± 0.32 | 60.27 ± 0.33 | 51.02 ± 1.19 |
| tralo_null | original | capped_first | 60.65 ± 0.35 | 60.41 ± 0.41 | 52.24 ± 0.75 |
| tralo_null | TraLO-derived | upper_bound_correction | 57.48 ± 0.45 | 54.96 ± 0.49 | 6.32 ± 1.23 |
| tralo_null | TraLO-derived | capped_first | 57.48 ± 0.45 | 54.96 ± 0.49 | 6.32 ± 1.23 |
| tralo | original | raw | 57.48 ± 0.62 | 54.74 ± 0.62 | 5.35 ± 1.80 |
| tralo | original | upper_bound_correction | 57.48 ± 0.62 | 54.74 ± 0.62 | 5.35 ± 1.80 |
| tralo | original | capped_first | 59.99 ± 0.37 | 59.47 ± 0.35 | 47.02 ± 1.17 |
| tralo | TraLO-derived | upper_bound_correction | 57.48 ± 0.62 | 54.74 ± 0.62 | 5.35 ± 1.80 |
| tralo | TraLO-derived | capped_first | 57.51 ± 0.61 | 54.82 ± 0.61 | 6.10 ± 1.38 |

## Paired differences: TraLO minus each control

Differences are percentage/F1 points. Positive favors TraLO; negative favors the control. The interval is a two-sided 95% Student-t interval over four paired seeds, conditional on this fixed development split. These exploratory intervals are not corrected for multiple comparisons, and normality at four seeds is unverified.

| Budget | Allocation | Control | Metric | Seed deltas 701 / 702 / 703 / 704 | Mean | 95% interval |
|---|---|---|---|---|---:|---|
| original | upper_bound_correction | clipper | accuracy | -3.10 / -3.15 / -2.05 / -3.10 | -2.85 | [-3.70, -2.00] |
| original | upper_bound_correction | clipper | macro_f1 | -5.51 / -5.83 / -4.55 / -5.37 | -5.31 | [-6.18, -4.45] |
| original | upper_bound_correction | clipper | cc_f1 | -44.73 / -46.66 / -44.82 / -45.96 | -45.54 | [-47.02, -44.06] |
| original | upper_bound_correction | tralo_null | accuracy | -3.55 / -3.30 / -2.05 / -3.45 | -3.09 | [-4.20, -1.97] |
| original | upper_bound_correction | tralo_null | macro_f1 | -5.76 / -6.00 / -4.64 / -5.72 | -5.53 | [-6.49, -4.56] |
| original | upper_bound_correction | tralo_null | cc_f1 | -44.88 / -46.66 / -44.27 / -46.87 | -45.67 | [-47.72, -43.61] |
| original | capped_first | clipper | accuracy | -0.70 / -0.60 / +0.10 / -0.35 | -0.39 | [-0.96, +0.18] |
| original | capped_first | clipper | macro_f1 | -0.88 / -1.04 / -0.29 / -0.51 | -0.68 | [-1.22, -0.14] |
| original | capped_first | clipper | cc_f1 | -5.51 / -4.44 / -5.45 / -3.61 | -4.75 | [-6.19, -3.31] |
| original | capped_first | tralo_null | accuracy | -1.15 / -0.95 / +0.10 / -0.65 | -0.66 | [-1.53, +0.21] |
| original | capped_first | tralo_null | macro_f1 | -1.16 / -1.46 / -0.38 / -0.79 | -0.95 | [-1.69, -0.20] |
| original | capped_first | tralo_null | cc_f1 | -5.51 / -6.18 / -4.89 / -4.27 | -5.21 | [-6.52, -3.91] |
| TraLO-derived | upper_bound_correction | clipper | accuracy | -0.10 / +0.05 / +0.95 / +0.10 | +0.25 | [-0.50, +1.00] |
| TraLO-derived | upper_bound_correction | clipper | macro_f1 | -0.22 / -0.48 / +0.69 / +0.00 | -0.00 | [-0.80, +0.79] |
| TraLO-derived | upper_bound_correction | clipper | cc_f1 | -0.83 / -2.17 / +0.00 / -0.87 | -0.97 | [-2.39, +0.46] |
| TraLO-derived | upper_bound_correction | tralo_null | accuracy | -0.60 / -0.15 / +0.85 / -0.10 | -0.00 | [-0.97, +0.97] |
| TraLO-derived | upper_bound_correction | tralo_null | macro_f1 | -0.51 / -0.69 / +0.47 / -0.12 | -0.21 | [-1.03, +0.60] |
| TraLO-derived | upper_bound_correction | tralo_null | cc_f1 | -0.83 / -2.17 / +0.00 / -0.87 | -0.97 | [-2.39, +0.46] |
| TraLO-derived | capped_first | clipper | accuracy | -0.05 / +0.15 / +0.95 / +0.10 | +0.29 | [-0.43, +1.00] |
| TraLO-derived | capped_first | clipper | macro_f1 | -0.14 / -0.27 / +0.69 / -0.00 | +0.07 | [-0.61, +0.74] |
| TraLO-derived | capped_first | clipper | cc_f1 | +0.00 / +0.00 / +0.00 / -0.87 | -0.22 | [-0.91, +0.47] |
| TraLO-derived | capped_first | tralo_null | accuracy | -0.55 / -0.05 / +0.85 / -0.10 | +0.04 | [-0.90, +0.97] |
| TraLO-derived | capped_first | tralo_null | macro_f1 | -0.42 / -0.48 / +0.47 / -0.13 | -0.14 | [-0.83, +0.55] |
| TraLO-derived | capped_first | tralo_null | cc_f1 | +0.00 / +0.00 / +0.00 / -0.87 | -0.22 | [-0.91, +0.47] |

Source predictions: `C:\Users\roeym\.codex\rebuild-audit-20260922\achieved-counts-run\comparison`. Full counts, per-class metrics and labels assigned by every policy are in results.json. Independent scikit-learn metrics, artifact hashes, warm-up/batch matching, update counts and self-count identity passed. Four seeds on inspected development data do not establish superiority or absence of bugs.

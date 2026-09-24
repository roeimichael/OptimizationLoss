# New ALM / TraLO results: 24 September 2026

48 new classifier-head fits completed: two datasets, four matched seeds, five methods plus the second TraLO step size. Frozen ImageNet ResNet18 features, FP32 Quadro on dsisco01; no backbone retraining. Runtime release `6411f2b2db81cb7b071a09441cd26d7f82347e98`.

Knee uses 5,778 training and 826 development images. CIFAR-100 uses the previously fixed 10,000/2,000 training/development subset, not the full benchmark. Neither test set was scored. Both development sets were inspected previously.

All methods: 20 epochs, 5 warmup, same initialization and supervised batches. ALM and TraLO nulls are exactly equal in saved predictions for all eight dataset/seed pairs. They are the same control here, not independent evidence.

Exact constrained slots for capped_first: knee grade3=82, grade4=16; CIFAR classes0-9=10 each. Other classes remain unrestricted. Metrics below are percentages; differences are percentage points. cc-F1 is the mean F1 over capped classes; macro-F1 averages all classes.

## What this adds

The knee TraLO/control endpoints reproduce the preceding frozen-feature setup;
those values are not a new discovery. ALM and its null, plus the four-seed
CIFAR comparison, are new evidence. TraLO .001 is worse than .0001 on knee but
has the larger mean on CIFAR. Neither setting has a paired interval wholly
above its matched null on either dataset. Smaller steps are not a universal fix.

ALM lowers knee allocated cc-F1 by 8.06 points versus its own null in this
configuration; the corresponding CIFAR mean is -0.73 points with an interval
crossing zero. This is evidence about this initial fixed-rho, short-budget,
frozen-feature implementation, not a verdict on the ALM literature.

The next informative experiment would separate objective from update schedule:
apply the TraLO and ALM objectives under a shared joint-update schedule, with
matched nulls and a prespecified equal tuning budget. Currently ALM's penalty
shape, projected dual rule and intervention frequency all differ from TraLO,
so these results cannot attribute their difference to one component. This next
experiment is proposed, not run or included in the 48 fits.

## knee

### capped_first

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 49.243 | 37.204 | 39.649 |
| tralo_null | 49.667 | 37.990 | 41.078 |
| alm_null | 49.667 | 37.990 | 41.078 |
| tralo_.0001 | 49.849 | 38.480 | 42.191 |
| tralo_.001 | 49.939 | 37.862 | 40.132 |
| alm | 49.213 | 34.550 | 33.022 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | +1.113 | 1.163 | [-0.737, +2.964] |
| tralo_.0001 minus clipper | +2.542 | 1.511 | [+0.137, +4.947] |
| tralo_.001 minus tralo_null | -0.946 | 2.702 | [-5.246, +3.354] |
| tralo_.001 minus clipper | +0.482 | 3.799 | [-5.563, +6.527] |
| alm minus tralo_null | -8.056 | 2.238 | [-11.618, -4.494] |
| alm minus clipper | -6.627 | 1.686 | [-9.311, -3.944] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 36.294 | 40.413 | 40.413 | 40.945 | 41.477 | 31.111 |
| 902 | 39.151 | 40.945 | 40.945 | 41.477 | 39.683 | 30.381 |
| 903 | 40.945 | 41.477 | 41.477 | 42.009 | 36.826 | 35.564 |
| 904 | 42.207 | 41.477 | 41.477 | 44.334 | 42.541 | 35.032 |

### raw

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 49.425 | 36.292 | 35.830 |
| tralo_null | 49.758 | 37.239 | 37.995 |
| alm_null | 49.758 | 37.239 | 37.995 |
| tralo_.0001 | 49.697 | 36.988 | 37.239 |
| tralo_.001 | 48.335 | 31.785 | 23.630 |
| alm | 48.033 | 29.061 | 17.645 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | -0.756 | 1.201 | [-2.668, +1.156] |
| tralo_.0001 minus clipper | +1.409 | 1.228 | [-0.544, +3.363] |
| tralo_.001 minus tralo_null | -14.365 | 5.316 | [-22.824, -5.905] |
| tralo_.001 minus clipper | -12.200 | 4.304 | [-19.048, -5.351] |
| alm minus tralo_null | -20.350 | 2.599 | [-24.485, -16.215] |
| alm minus clipper | -18.185 | 3.779 | [-24.198, -12.172] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 34.570 | 36.620 | 36.620 | 35.961 | 26.977 | 18.435 |
| 902 | 35.877 | 37.422 | 37.422 | 36.028 | 26.312 | 18.748 |
| 903 | 34.130 | 39.065 | 39.065 | 37.208 | 17.542 | 18.435 |
| 904 | 38.743 | 38.873 | 38.873 | 39.760 | 23.690 | 14.961 |

### upper_bound_correction

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 49.425 | 36.292 | 35.830 |
| tralo_null | 49.758 | 37.239 | 37.995 |
| alm_null | 49.758 | 37.239 | 37.995 |
| tralo_.0001 | 49.697 | 36.988 | 37.239 |
| tralo_.001 | 48.335 | 31.785 | 23.630 |
| alm | 48.033 | 29.061 | 17.645 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | -0.756 | 1.201 | [-2.668, +1.156] |
| tralo_.0001 minus clipper | +1.409 | 1.228 | [-0.544, +3.363] |
| tralo_.001 minus tralo_null | -14.365 | 5.316 | [-22.824, -5.905] |
| tralo_.001 minus clipper | -12.200 | 4.304 | [-19.048, -5.351] |
| alm minus tralo_null | -20.350 | 2.599 | [-24.485, -16.215] |
| alm minus clipper | -18.185 | 3.779 | [-24.198, -12.172] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 34.570 | 36.620 | 36.620 | 35.961 | 26.977 | 18.435 |
| 902 | 35.877 | 37.422 | 37.422 | 36.028 | 26.312 | 18.748 |
| 903 | 34.130 | 39.065 | 39.065 | 37.208 | 17.542 | 18.435 |
| 904 | 38.743 | 38.873 | 38.873 | 39.760 | 23.690 | 14.961 |

## cifar

### capped_first

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 61.975 | 61.592 | 52.057 |
| tralo_null | 62.100 | 61.683 | 51.390 |
| alm_null | 62.100 | 61.683 | 51.390 |
| tralo_.0001 | 62.150 | 61.741 | 51.764 |
| tralo_.001 | 62.200 | 61.822 | 52.245 |
| alm | 61.600 | 61.120 | 50.661 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | +0.374 | 0.443 | [-0.331, +1.078] |
| tralo_.0001 minus clipper | -0.293 | 0.612 | [-1.266, +0.681] |
| tralo_.001 minus tralo_null | +0.855 | 0.592 | [-0.088, +1.797] |
| tralo_.001 minus clipper | +0.188 | 0.713 | [-0.946, +1.322] |
| alm minus tralo_null | -0.729 | 1.478 | [-3.081, +1.622] |
| alm minus clipper | -1.396 | 1.161 | [-3.243, +0.452] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 52.292 | 52.848 | 52.848 | 52.848 | 53.033 | 50.825 |
| 902 | 51.108 | 50.368 | 50.368 | 50.368 | 51.108 | 51.088 |
| 903 | 52.848 | 51.978 | 51.978 | 52.603 | 53.601 | 49.994 |
| 904 | 51.978 | 50.368 | 50.368 | 51.237 | 51.237 | 50.736 |

### raw

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 63.150 | 62.787 | 58.583 |
| tralo_null | 63.287 | 62.917 | 58.534 |
| alm_null | 63.287 | 62.917 | 58.534 |
| tralo_.0001 | 63.263 | 62.888 | 58.350 |
| tralo_.001 | 62.837 | 62.289 | 54.054 |
| alm | 60.575 | 59.527 | 36.670 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | -0.184 | 0.551 | [-1.060, +0.692] |
| tralo_.0001 minus clipper | -0.233 | 0.679 | [-1.313, +0.847] |
| tralo_.001 minus tralo_null | -4.480 | 2.032 | [-7.713, -1.247] |
| tralo_.001 minus clipper | -4.529 | 1.958 | [-7.644, -1.414] |
| alm minus tralo_null | -21.864 | 1.026 | [-23.498, -20.231] |
| alm minus clipper | -21.913 | 1.025 | [-23.544, -20.282] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 57.136 | 56.977 | 56.977 | 57.560 | 52.732 | 36.548 |
| 902 | 57.539 | 57.527 | 57.527 | 56.846 | 55.722 | 35.708 |
| 903 | 60.686 | 60.210 | 60.210 | 59.755 | 54.957 | 37.622 |
| 904 | 58.972 | 59.424 | 59.424 | 59.241 | 52.804 | 36.802 |

### upper_bound_correction

| Method | Accuracy | Macro-F1 | cc-F1 |
|---|---:|---:|---:|
| clipper | 61.913 | 61.503 | 51.434 |
| tralo_null | 62.062 | 61.628 | 50.985 |
| alm_null | 62.062 | 61.628 | 50.985 |
| tralo_.0001 | 62.100 | 61.665 | 51.141 |
| tralo_.001 | 61.938 | 61.358 | 48.206 |
| alm | 60.575 | 59.527 | 36.670 |

| Paired contrast in cc-F1 | Mean difference | Seed SD | 95% t interval |
|---|---:|---:|---:|
| tralo_.0001 minus tralo_null | +0.156 | 0.776 | [-1.078, +1.391] |
| tralo_.0001 minus clipper | -0.293 | 1.080 | [-2.011, +1.426] |
| tralo_.001 minus tralo_null | -2.779 | 2.264 | [-6.382, +0.825] |
| tralo_.001 minus clipper | -3.228 | 2.004 | [-6.417, -0.038] |
| alm minus tralo_null | -14.315 | 1.429 | [-16.590, -12.041] |
| alm minus clipper | -14.764 | 0.869 | [-16.146, -13.382] |

| Seed | clipper | tralo_null | alm_null | tralo_.0001 | tralo_.001 | alm |
|---|---:|---:|---:|---:|---:|---:|
| 901 | 52.292 | 52.848 | 52.848 | 52.848 | 47.851 | 36.548 |
| 902 | 49.488 | 48.747 | 48.747 | 47.878 | 49.130 | 35.708 |
| 903 | 51.978 | 51.978 | 51.978 | 52.603 | 48.620 | 37.622 |
| 904 | 51.978 | 50.368 | 50.368 | 51.237 | 47.226 | 36.802 |

## Validation and limits

All run/artifact hashes, planned/applied updates, matched warmup/batch hashes and exact capped counts passed. Confusion-based F1 was independently recomputed from saved predictions; the pilot/runtime also checked metrics against sklearn. No skipped updates.

Knee: 460 supervised updates per fit; CIFAR: 800. ALM includes the constraint in 345/600 of those updates and performs 15 dual updates. TraLO has up to 15 separate constraint updates. Equal epochs are not equal computation or equal numbers of constraint gradients.

ALM is the documented fixed-rho (.5), fixed-budget inexact inequality PHR baseline. This does not replicate a specific published ALM experiment or establish that ALM was fully optimized. TraLO rates .0001 and .001 were registered before these runs; both are retained.

Paired intervals describe four training seeds on a fixed development split. They do not capture dataset sampling uncertainty and are not corrected for the multiple methods, metrics and datasets. Do not promote a favorable exploratory interval into a general superiority claim.

Evidence: `/home/dsi/michaer8/tralo-rebuild/runs/alm-two-dataset-20260924`; local audit receipts under `C:/Users/roeym/.codex/rebuild-audit-20260922/`. See `alm_two_dataset_20260924.md` for the mathematical definition and registered protocol.

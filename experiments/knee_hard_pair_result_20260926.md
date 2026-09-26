# Knee hard-pair ranking: four end-to-end seeds

**Result:** the hard-pair gradient was active at every planned ranking opportunity, but this fixed recipe did not establish better grade-3 slot selection than its phase-matched Null or ordinary Clipper. Adding the development soft-count term again suppressed raw grade-3 calls and did not improve exact-cap performance over ranking alone.

## Identity and integrity

The design was fixed before these results in [knee_hard_pair_protocol_20260924.md](knee_hard_pair_protocol_20260924.md). Immutable release `eac8db3c069217121a1ff3bba41cdd055bad9ea8` was pushed to GitHub and the DSI mirror and verified with 113 local/native tests, both-host source-byte parity, and Quadro smoke. The first release `3ac900f4a3b188d2ee798fb1614e22a51448927a` failed a native parameter-oracle tolerance; that failure and its measured numerical cause remain in the protocol. No GPU fit used the failed release.

All 20 fits completed on dsisco01 Quadro RTX 6000 FP32, five arms for each of seeds 1501–1504. Seed roots are `/home/dsi/michaer8/tralo-rebuild/runs/knee-hard-pair-20260924/pilot1501` and `seed1502`–`seed1504`. Independent per-seed audits and aggregate analysis are at `C:/Users/roeym/.codex/rebuild-audit-20260922/knee_hard_pair_1501_audit.json` through `knee_hard_pair_1504_audit.json` and `knee_hard_pair_four_seed_analysis.json`. Every audit passed release/data identity, artifact and snapshot hashes, independent sklearn metric recomputation, matched warm-up and batch hashes, 1810 applied task updates per arm, at most five auxiliary updates, no skipped/nonfinite steps, and correct/wrong quota-slot transition checks. Both hard-ranking arms had 532 weakest true grade-3 training images, 532 strongest non-grade-3 training images, and 532² active pairs in all five post-warmup epochs per seed: 40 active ranking opportunities across four seeds. This addresses the zero-gradient failure of the prior wrong-occupant loss. It does not establish that the gradient improves deployment decisions.

The audited Chen split had 5778 training, 826 development, and 1656 held-out test images. The test split was not scored. Each arm used the same ImageNet-initialized ResNet18 architecture, five CE warm-up plus five post-warm-up epochs, grade-3 cap 76, and fixed optimizer settings. The phase Null shares the auxiliary arms' task-optimizer reset; ordinary Clipper retains its task-optimizer state. Training labels selected hard pairs; development labels were used only by the offline auditor. The development split has been inspected repeatedly across studies, so all intervals below are exploratory.

## Development endpoints

Figures are grade-3 F1 percentages. Raw is argmax without allocation. Upper-bound correction only removes calls beyond cap 76. Capped-first assigns exactly 76 grade-3 slots from saved scores; that is the main slot-selection comparison.

| Arm | Raw mean | Raw grade-3 calls, mean | Upper-bound mean | Capped-first mean | Capped-first seeds 1501 / 1502 / 1503 / 1504 |
|---|---:|---:|---:|---:|---|
| Clipper | 66.96 | 89.25 | 65.84 | 65.93 | 69.23 / 63.74 / 63.74 / 67.03 |
| Phase Null | 64.70 | 98.50 | 64.16 | 65.38 | 60.44 / 69.23 / 65.93 / 65.93 |
| Count-only | 33.78 | 22.50 | 33.78 | 61.81 | 63.74 / 56.04 / 63.74 / 63.74 |
| Hard-rank-only | 64.46 | 79.50 | 63.47 | 65.66 | 63.74 / 61.54 / 70.33 / 67.03 |
| Hard-rank-plus-count | 46.77 | 34.50 | 46.77 | 64.56 | 63.74 / 65.93 / 67.03 / 61.54 |

The prespecified primary capped-first contrast, rank-plus-count minus rank-only, was **−1.10 percentage points**, seed deltas 0.00, +4.40, −3.30, −5.49; sample SD 4.30 and exploratory paired 95% t interval [−7.95, +5.75]. Rank-only minus phase Null was **+0.27 points**, seed deltas +3.30, −7.69, +4.40, +1.10; interval [−8.45, +9.00]. Count-only minus Null was −3.57 points, interval [−14.57, +7.43]. Combined minus Null was −0.82, interval [−6.60, +4.95]. None establishes an exact-slot gain. Ordinary Clipper's mean was 65.93 versus rank-only 65.66, so this ranking recipe also did not demonstrate an advantage over the simpler output baseline.

The count component reduced raw grade-3 F1 in all four matched comparisons: combined minus rank-only averaged −17.69 points, interval [−34.08, −1.31], while count-only minus Null averaged −30.92, interval [−44.07, −17.77]. Raw grade-3 calls averaged 34.5 for combined and 22.5 for count-only, versus 79.5 for rank-only and 98.5 for Null. Exact filling recovers many slots, so raw suppression must not be mistaken for an equal capped-first penalty. The full JSON retains both allocation policies and every seed's before/after correct and wrong entries/exits; those transitions are repeated image-update events, not independent patients.

## Interpretation and next decision

The previous occupancy-ranking term often had no gradient; this hard-pair term did. Its activation did **not** translate into a reliable improvement of correct development images in the 76 deployed slots. That narrows the failure mode: inactivity alone was not the obstacle. Training/development transfer, the chosen hardest-pair weighting, and the fixed training schedule remain plausible explanations that these data do not separate. A broader hyperparameter search on this repeatedly viewed development set would not establish a trustworthy win.

The next useful step is to choose and freeze an independent confirmation design with an untouched cohort/split and a deployment utility that explicitly values correct constrained-class membership. Before another full campaign, compare the logged rank-loss and gradient dose with the saved development cutoff scores and correct/wrong slot transitions. Full per-image training margins were not saved, so these artifacts cannot directly establish whether training ranking improved but failed to transfer. This is offline mechanism analysis, with no favorable-only setting selection. A proposed new term should make a falsifiable cutoff-level prediction and face the same matched Null and Clipper on independently held-out data. Do not score the existing 1656-image test split merely to choose the next term.

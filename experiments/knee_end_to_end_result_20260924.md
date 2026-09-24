# End-to-end knee count-loss result, 24 September 2026

The fixed four-seed mechanism test is complete. It **does not establish an advantage for the current TraLO count loss**. With the ResNet18 backbone trainable, the loss strongly reduced raw grade-3 predictions, but did not consistently put more *correct* patients in the 76 deployed grade-3 slots. This addresses a limitation of the earlier frozen-head study: the count gradient reached the image representation here.

## Fixed comparison and provenance

Chen OAI v1: 5,778 training images, 826 development images and a separately retained 1,656-image test split. The test split was audited for overlap but not trained on, scored or used to choose settings. The only constrained class was knee grade 3, with a 76-image development cap derived from training prevalence before this run. Four fresh seeds 1301–1304 each trained the same ImageNet-initialized ResNet18 for five cross-entropy warm-up and five post-warm-up epochs, batch 32 and task Adam rate 0.0001. Clipper continued task Adam; the phase-matched Null and TraLO reset it at the boundary. TraLO alone received one full-development-cohort bounded soft-count update after each post-warm-up epoch, using a separate Adam rate 0.00003. Its two-pass streamed gradient used no development labels; BatchNorm was in evaluation mode during that update. All arms applied 1,810 task updates. TraLO applied all five count updates per seed, with no skipped or nonfinite updates.

Immutable runtime/source release: `484f5a59cba978e61ac1f54e4a9f727a41cd21b3`; all runs were FP32 on dsisco01 Quadro RTX 6000 GPUs. The source passed 100 local tests and byte/native checks on both DSI hosts. Complete run artifacts and exclusive launch receipts are in `/home/dsi/michaer8/tralo-rebuild/runs/knee-e2e-20260924/`. Independent per-seed hash, split, dose, snapshot and scikit-learn metric audits are retained at `C:/Users/roeym/.codex/rebuild-audit-20260922/knee_e2e_130{1,2,3,4}_audit.json`; paired analysis is `knee_e2e_four_seed_analysis.json` (SHA-256 `971d47880183c1868990e45e5146691c3e8186013b29d03f309d422ffb7254a4`). The four audits passed, including matching warm-up state and batch-order hashes within every seed.

## Endpoint results

Grade-3 F1 is `2 TP / (actual grade-3 count + predicted grade-3 count)`, reported as percentage points. `capped_first` assigns exactly 76 grade-3 slots before the remaining classes; `upper_bound_correction` enforces only an upper bound and may leave slots unused. Neither allocator sees labels.

| Seed | Clipper capped-first grade-3 F1 | Phase-matched Null | TraLO | TraLO − Null |
|---|---:|---:|---:|---:|
| 1301 | 68.13 | 63.74 | 59.34 | −4.40 |
| 1302 | 67.03 | 68.13 | 64.84 | −3.30 |
| 1303 | 65.93 | 70.33 | 60.44 | −9.89 |
| 1304 | 70.33 | 59.34 | 68.13 | +8.79 |
| Four-seed mean | 67.86 | 65.38 | 63.19 | **−2.20** |

The paired TraLO-minus-Null mean for capped-first grade-3 F1 was **−2.20 points**, sample SD 7.87, exploratory 95% t interval **[−14.73, +10.33]**. The four seeds do not establish a reliable capped benefit or a precise capped harm. Capped-first accuracy averaged 58.02% for Null versus 59.59% for TraLO; the paired accuracy difference was +1.57 points, interval [−1.71, +4.85]. Macro-F1 averaged 59.28% for Null versus 56.10% for TraLO under capped-first.

The raw output tells a sharper mechanism story. Before allocation, Null predicted grade 3 for **95.75 images on average**, finding 66.25 true grade-3 images; TraLO predicted grade 3 for **26.00 images**, finding 25.00 true grade-3 images. Its raw grade-3 F1 was 37.61% versus Null's 65.71%. The paired raw F1 difference was −28.10 points, exploratory interval [−41.38, −14.81]. The raw penalty mostly removed grade-3 calls, including many correct ones. It did not merely shift confidence while retaining those calls. Under upper-bound correction, TraLO's mean grade-3 count remained 26 and F1 remained 37.61%; capped-first filled the vacant slots and raised TraLO's grade-3 F1 to 63.19%, still below Null's 65.38% mean. Thus **allocation policy materially changes the apparent size of the failure**, and the two policies must not be pooled.

The saved before/after snapshots identify actual slot movement. Across four seeds and five count updates each, capped-first had 262 entering-slot events; 99 entrants were true grade 3, while 131 true grade-3 images exited, a net −32 correct-slot *events*. These events repeat images across updates and are **not** 32 distinct patients or an endpoint effect. Per-seed cumulative net correct-slot events were −18, +2, 0 and −16. In seed 1301's first update, 17 entered (one correct) while 11 correct selections exited. The final seed 1304 still beat its Null, so a harmful immediate slot change does not determine the entire later trajectory.

## Interpretation and next test

The current count objective receives a global soft-count excess but no indication of which patient deserves a capped slot. End-to-end gradients are active and the arithmetic/optimizer checks passed; the model responds by suppressing grade-3 calls broadly. Training labels continue to shape the task through cross-entropy, yet the separate count step can undo useful grade-3 selections. This is an evidence-backed mechanism for these runs, not proof that every global count objective or TraLO variant must fail. The development split has been repeatedly inspected, so these intervals are diagnostic and do not substitute for untouched confirmation.

The next objective should distinguish *wrong occupants from missed true grade-3 cases at the quota cutoff*, using training labels only. It needs its own derivative check, a rank-only matched control, the existing count-only reference, actual update-dose accounting and prespecified seeds before GPU use. See `knee_cutoff_ranking_protocol_20260924.md`.

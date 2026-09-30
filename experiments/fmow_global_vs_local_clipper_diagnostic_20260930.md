# fMoW2 global versus country-aware Clipper: label-free allocation diagnostic

## Question and boundary

Does the historical pooled `capped_first` Clipper allocate the same class-1 slots as the country-aware Clipper required by the fMoW2 dual-constraint question? This is an allocation diagnostic, **not an accuracy comparison**. It uses only the completed PHR-ALM direction study's PTO snapshot probabilities and image IDs/countries. It does not read development labels, reserved countries, or model outcomes, and it launches no GPU work.

The input is the immutable `2f6a68eb006cf2d9cd535a7dfa51f5c09ce9850e` release at `/home/dsi/michaer8/tralo-rebuild/releases/2f6a68eb006cf2d9cd535a7dfa51f5c09ce9850e` and its completed 12-seed run root `/home/dsi/michaer8/tralo-rebuild/runs/fmow-local-alm-direction-20260930-2f6a68eb/full6301_6312`. New diagnostic code was streamed to the host through Python's standard input; neither release nor run artifacts were modified. The full, per-seed provenance and counts are in [the JSON report](fmow_global_vs_local_clipper_diagnostic_20260930.json) (SHA-256 `ee8c50bdd7efadebe0c1be45901cef6c16bf0768f7d683ccd30f5fc2dee378db`). Diagnostic source SHA-256 is `7b3ca8cf05e9e2ce3bbe66c84c5a5a665003714d4b9fc87e8bda45ca7ad1239a`; pool identity SHA-256 is `ab89c2c8e942fc97a14dbfcc2a1f7e9da744fd0869ac1d18a3f8f48f32723daf`.

## Fixed method

For each seed 6301–6312, [the diagnostic](../analysis/clipper_allocator_diagnostic.py) verifies the existing source, configuration, data, launch and completion receipts and every PTO snapshot's creation-time hash. It reconstructs the registered ensemble window from `max(1, best_epoch - 2)` through the final retrain epoch, averaging the same class probabilities for both allocators. It checks the label-free pool identity and derives the already registered global and Hamilton country caps from the same image IDs/countries. It then applies the actual [`capped_first`](../tralo/global_clipper.py) pooled allocator and `allocate_local_capped_first` country-aware allocator to identical probabilities at the 167-slot and 83-slot caps. An independent selected-set oracle checks that the output counts and overlaps reflect one capped class. No label loading or scorer metric path is invoked.

The global policy fills the pooled cap from the highest class-1 probabilities. The local policy first enforces each country's class-1 cap and fills remaining pooled slots from eligible images. This is precisely the policy distinction in `tralo/global_clipper.py:82–138`; the ensemble and provenance rules are in `analysis/score_fmow_local_alm.py:65–137,431–453` and the label-free pool/quota rules in `analysis/score_fmow_local.py:72–109`.

## Result

| Pooled class-1 cap | Seeds with at least one country-cap violation under global Clipper | Mean country-cap excess per seed | Mean selected overlap (global/local) | Overlap range |
| --- | ---: | ---: | ---: | ---: |
| 167 | 12/12 | 79.42 | 87.58/167 | 83–92 |
| 83 | 12/12 | 47.17 | 35.83/83 | 29–42 |

At both caps, global Clipper over-allocated **NLD and PHL in every seed**. For the 167-slot cap, their respective caps were 19 and 23, while global Clipper assigned means of 55.75 and 65.67; country-aware Clipper assigned exactly 19 and 23. For the 83-slot cap, caps were 10 and 11, global means 42.50 and 25.67, and country-aware assignments exactly 10 and 11. The per-seed excess totals were 953 and 566 across 12 seeds; these count excess country assignments, not extra unique images or independent experiments. At each seed/cap, the number of globally selected images absent from the local selection equaled the global country-cap excess. Both policies filled the same pooled cap.

The per-seed excess/overlap pairs (167-slot; 83-slot) are: 6301 `80/87; 44/39`, 6302 `82/85; 54/29`, 6303 `77/90; 44/39`, 6304 `78/89; 46/37`, 6305 `77/90; 50/33`, 6306 `78/89; 47/36`, 6307 `80/87; 46/37`, 6308 `81/86; 46/37`, 6309 `83/84; 45/38`, 6310 `78/89; 51/32`, 6311 `84/83; 52/31`, 6312 `75/92; 41/42`. The JSON preserves each seed's snapshot hashes, ensemble hash, country counts, selected-set hashes and Jaccard overlap.

## Interpretation and checks

The historical pooled Clipper is a valid **global-only** deployment policy, but it is infeasible for these local caps on every audited seed. Therefore its accuracy, even if measured elsewhere, would not be a fair like-for-like comparator for a dual-constraint local loss. The matched null comparator for the existing fMoW2 study is PTO with the **same country-aware Clipper** used for the TraLO and PHR-ALM direction arms; that comparison isolates the training update while holding deployment allocation fixed. These observations do not show that either allocator improves class-1 F1: no labels were read, and no metric was computed.

Focused allocator and diagnostic tests passed: `10 passed, 6 subtests passed` from `tests/test_clipper_allocator_diagnostic.py` and `tests/test_global_clipper.py`. All 12 run receipts, snapshot hashes, source-release checks, pool identities and independent allocation oracles passed. This diagnostic makes no new model-selection decision and does not change the completed negative local-loss or PHR-ALM study results.

# Experiment map

Start with [README.md](README.md) for the current answer and the short PDF. This
map separates the current evidence from historical work. A protocol fixes a
question; a result reports what happened. A pilot or draft is not a result.

## Knee: Kassif comparison and TraLO mechanism

| Read in this order | What it answers |
| --- | --- |
| [Kassif repository audit](claude_yuval_repo_audit_20260927.md) | What the author's code actually trains, changes, and evaluates. |
| [Matched pipeline protocol](claude_yuval_pipeline_prereg_20260927.md) and [result](claude_yuval_pipeline_result_20260927.md) | PAO, PTO, TraLO's one-step direction, and a sham on ResNet18 and EfficientNet-B5. |
| [Prospective persistent knee comparison](knee_persistent_kassif_match_protocol_draft_20261001.md) | New fixed, three-backbone TraLO-versus-PAO/PTO/sham/null design; no training result yet. |
| [Recipe factorial](claude_recipe_factorial_result_20260927.md) | Which parts of the Kassif recipe improve plain training. |
| [Snapshot-step results](claude_stepens_result_20260928.md) and [modern-backbone extension](claude_stepens_additional_result_20260928.md) | A small, backbone-dependent TraLO effect on the unchanged PTO trajectory; this is not persistent constraint training. |
| [Six-cap diagnostic](kassif_single_cap_result_20260924.md) | Frozen-head structural test, not a full-backbone paper comparison. |
| [End-to-end knee result](knee_end_to_end_result_20260924.md) and [ranking result](knee_cutoff_ranking_result_20260924.md) | Earlier full-backbone mechanisms that did not establish a gain. |
| [Hard-pair proposal](knee_hard_pair_protocol_20260924.md) | A specified but unrun training-label rank term. |

Other knee records remain available in this directory: `knee_*`,
`claude_targeted_*`, `claude_backbone_*`, `claude_yuval_smallbb_*`,
`claude_snapshot_*`, `claude_step_*`, `claude_controller_*`, and
`claude_recipe_*`. Their titles are identifiers, not recommendations.

## fMoW2: local-constraint evidence

| Read in this order | What it answers |
| --- | --- |
| [Fixed-dose result](fmow_local_fixed_dose_result_20260930.md) | A negative joint pooled/country step at both fixed caps. |
| [PHR snapshot-direction result](fmow_local_alm_direction_result_20260930.md) | PHR at a snapshot, not full ALM training; negative against unchanged PTO. |
| [MobileNetV3 boundary result](fmow_boundary_mnv3_result_20261001.md) | Audited 12-seed full block, small inconclusive primary gains with secondary harms. |
| [ViT-B/16 boundary result](fmow_boundary_vit_v2_result_20261001.md) | Audited 12-seed full block, no established primary gain. |
| [Prospective full-training comparison](fmow_full_alm_matched_protocol_draft_20261001.md) | Unreleased draft for TraLO, ALM, Clipper, and null; no GPU result. |

Keep the corresponding `fmow_*_protocol*`, `fmow_*_preflight*`, and
`fmow_*_amendment*` beside these results: they establish what was fixed before
scoring and why an integrity check changed. The failed gates are evidence.

## Earlier global and two-dataset work

The [two-dataset ALM comparison](alm_two_dataset_result_20260924.md) and
[constraint-strength sweep](constraint_sweep_result_20260924.md) used earlier
settings and do not stand in for the new full-backbone knee comparison.
[Global Clipper](global_clipper_result_20260922.md),
[global TraLO](global_tralo_result_20260922.md), and
[sample-aware experiments](sample_aware_result_20260924.md) establish earlier
diagnostics. Read their protocols before interpreting their numbers.

## Artifacts and archival

`configs/` contains fixed run inputs; `analysis/` contains independent scorers
and outputs; `output/pdf/` contains presentation copies. Their paths remain
stable for reproducibility. The [provenance archive](archive/provenance_20261001/README.md)
contains 21 unchanged, unreferenced historical notes and a SHA-256 manifest.
Nothing in it was deleted. Older releases retain the original paths.

The knee development pool has been repeatedly inspected. Fresh seeds there are
not an untouched test. The Chen test remains sealed until a fixed method and a
separate evaluation decision are recorded.

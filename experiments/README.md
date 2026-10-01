# TraLO experiments: start here

The top level is a **preserved research record**, not a list of recommended models. Older protocols, failed gates, negative results, and superseded hypotheses remain in place because other reports and immutable releases cite their paths. This index supplies a short reading route; do not infer a winner from a filename or a pilot.

## Current answer (1 October 2026)

Read the [illustrated PDF research report](../output/pdf/tralo_research_status_20261001.pdf) first. It explains the method, Kassif comparison, backbone results, limitations, and open questions. The [numerical paper draft](tralo_paper_results_draft_20260930.md) is a source document, not the final presentation.

| Question | Best entry point | Evidence status |
| --- | --- | --- |
| Does the global TraLO snapshot step help on modern image backbones? | [MobileNetV3 / EfficientNet-B5 result](claude_stepens_additional_result_20260928.md); [ResNet18 / RegNetY result](claude_stepens_result_20260928.md) | Complete development blocks; positive on MobileNetV3, negative on B5; sealed knee test untouched. |
| What did adopting Kassif's pipeline change? | [Independent repository audit](claude_yuval_repo_audit_20260927.md), [fixed protocol](claude_yuval_pipeline_prereg_20260927.md), [paired result](claude_yuval_pipeline_result_20260927.md), [recipe factorial](claude_recipe_factorial_result_20260927.md) | Complete knee development studies; augmentation explains most recipe gain, separate from the constraint step. |
| Does pooled-plus-country local TraLO help on fmow2? | [Fixed-dose result](fmow_local_fixed_dose_result_20260930.md), [PHR direction result](fmow_local_alm_direction_result_20260930.md), [boundary-calibrated MobileNetV3 result](fmow_boundary_mnv3_result_20261001.md) | Fixed 0.1 and PHR snapshot directions failed; calibrated MobileNetV3 full block has an audited, inconclusive primary effect. |
| What is the ViT-B/16 evidence? | [Preflight and protocol](fmow_boundary_vit_preflight_20260930.md), [v2 replay correction](fmow_boundary_vit_v2_replay_protocol_20261001.md) | Twelve full seeds completed; independent scoring remains gated by a numerical replay discrepancy. Pilot is exploratory. |
| Are Clipper, null, and full ALM fairly compared? | [Historical frozen-feature comparison](alm_two_dataset_result_20260924.md), [local Clipper feasibility audit](fmow_global_vs_local_clipper_diagnostic_20260930.md), [prospective full-training design](fmow_full_alm_matched_protocol_draft_20261001.md) | Historical numbers are not a matched full-backbone local comparison. The prospective full-training study has no GPU result yet. |

## How to read this folder

`*_protocol*` and `*_prereg*` fix the question and analysis before scoring. `*_result*` holds outcomes, including negative ones. `*_amendment*` records a correction without rewriting an immutable runner. JSON/CSV files are machine-readable evidence; the matching prose report explains the scope. `configs/` holds fixed run configurations and must retain its paths. Study names beginning `claude_` or `kassif_` are historical experiment identifiers, not evidence grades.

The research progression is: [initial allocation/metric checks](global_clipper_result_20260922.md) -> [knee end-to-end feasibility](knee_end_to_end_result_20260924.md) -> [Kassif pipeline audit and factorial](claude_yuval_repo_audit_20260927.md) -> [global snapshot/backbone study](claude_stepens_result_20260928.md) -> [fmow2 pooled-plus-country local study](fmow_local_fixed_dose_result_20260930.md) -> [smaller boundary-calibrated and ViT work](fmow_boundary_mnv3_protocol_20260930.md). The early phase is retained for provenance, not promoted as the current model.

**Backbone priority:** interpret MobileNetV3-Large, EfficientNet-B5, and ViT-B/16 first. ResNet18 and RegNetY-400MF are useful diagnostic replications, not the main modern-backbone claim. Architecture alone does not turn a development result into held-out confirmation.

The 826-image knee development pool and the five-country, 1,673-image fmow2 development pool have been inspected. The 1,656-image Chen knee test and five reserved fmow2 countries remain sealed. Never select a method, cap, seed, or backbone from those development results and then call another look at the same cohort a confirmation.

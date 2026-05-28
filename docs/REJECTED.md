# Rejected experiments — do not re-introduce without reading this

This file documents backbones and datasets that were tried for the TraLO thesis (transductive prediction-count constraints vs Fioretto LDF / Hounie RCL baselines) and **failed or did not produce a clean win**. Their wrappers/entries have been removed from the active pipeline (`src/models/imagery/`, `src/models/model_factory.py`, `src/utils/data_loader.py`, `src/config_generators/gen_model_search.py`) on 2026-05-28.

Re-adding any of these requires a concrete reason that addresses the failure mode below — otherwise you'll burn GPU time reproducing a known dead end.

The "headroom hypothesis" is the explanatory lens: TraLO's macro-F1 edge appears only when warmup train-acc on derm lands in roughly **[0.70, 0.82]**. Outside that band the warmup is either saturated (argmax locked, no slack for TraLO to redistribute) or degenerate (majority-class collapse, no signal).

---

## Rejected backbones

| Backbone | Mode of failure | Evidence |
|---|---|---|
| **ResNet18** | Saturates derm at warmup epoch 1 (train-acc > 0.84). No headroom for TraLO to exploit. | Memory `project_warmup_sweetspot_2026-05-26`: "ResNet saturates at 1ep". |
| **EfficientNetB0** | Saturates derm warmup; 2-seed apparent win on smoke was noise. | Memory `project_efficientnet_dropped_2026-05-26`. |
| **DenseNet121** | Saturates: ep1 train-acc 0.877 (≥ 0.84). | Probe run in `project_model_search_2026-05-27`. |
| **MNASNet10** | Degenerate: train-acc stuck flat ~0.67 across all 8 warmup epochs (majority-class collapse on derm NV ≈ 67%). Macro-F1 0.27. | Probe run in `project_model_search_2026-05-27`. (`mnasnet1_3` not tested; allowed as a separate candidate.) |
| **RegNetY16GF** (actually `regnet_y_1_6gf`, ~11M params despite the name) | Saturates: ep1 train-acc 0.8439. | Probe run in `project_model_search_2026-05-27`. |
| **SqueezeNet11** | Failed smoke despite ideal warmup band (0.78). On aider **loses both baselines** (Δ Fior −0.0129, Δ Hounie −0.0131). On derm only Hounie-only/Fior-tie. Fire-module architecture does not host the TraLO edge — kills the simple "any mid-band warmup wins" hypothesis. | `project_model_search_2026-05-27` "Architecture-diversity follow-up". |
| **ViTTiny** (`timm.vit_tiny_patch16_224`) | Pure transformer memorizes derm in 1 epoch (ep1 train-acc 0.8279). Smoke = **Fioretto-only / loses Hounie on both** (derm Δh −0.0107, aider Δh −0.0041). Mirror image of ShuffleNet's Hounie-only — interesting datapoint but not a winner. | `project_model_search_2026-05-27`. |

### What this means for new backbone candidates
- Pretrained ImageNet weights + small fine-tuning dataset (derm has 9.6k train) = most large/modern backbones saturate ep1. Mobile-family inverted-residual + depthwise-separable convs are the only known clean winners.
- Mid-band warmup (~0.75) is **necessary but not sufficient** (SqueezeNet had it and still failed both-baselines on aider).
- Reasonable future candidates: MobileViT (under test), EfficientFormer-L1, MNASNet1_3, ConvNeXt-Pico, MobileViT variants.

---

## Rejected datasets

| Dataset | Mode of failure | Evidence |
|---|---|---|
| **TissueMNIST** | Tie everywhere on derm-style winners. Hard / near-random (macro-F1 ~ 0.28). | `project_model_search_2026-05-27`. |
| **PathMNIST** (colon histology, TUM-cap) | Too easy → saturates. MobileNetV2 = Hounie-win / Fior-tie (Δf −0.0013, Δh +0.0089). Doesn't qualify as both-baselines. | `project_model_search_2026-05-27` "Dataset-search". |
| **ISIC2019** (dermoscopy, 8 classes) | MobileNetV2 = loss-Fior / tie-Hounie. | `project_model_search_2026-05-27`. |
| **EuroSAT** (satellite, 10 classes) | Dropped per `docs/PAPER_PLAN.md` v2 (2026-05-24). Failed to give a clean TraLO story. | Memory `project_paper_plan_v2_2026-05-24`. |
| **So2Sat** (real-city groups) | Previously a TraLO win story (Tables 10-11 in old paper plan), dropped 2026-05-24 along with EuroSAT when paper plan v2 narrowed to derm/tissue/aider. **Not failed, but explicitly removed from active scope.** | Memory `project_so2sat_pivot.md`, `project_paper_plan_v2_2026-05-24`. |
| **OctMNIST** | Never properly tested; flagged for deletion to free /home NFS space. | Memory `reference_shared_nfs.md`. |
| **CIFAR-100** | Failed in past sessions; explicitly noted "do NOT re-propose" in the model-search memory. | `project_model_search_2026-05-27` dataset-search decision. |

### What this means for new dataset candidates
- Need: moderate difficulty (not saturable like PathMNIST, not degenerate like ISIC2019/TissueMNIST), an imbalanced constrained class with meaningful real-world stakes, and ideally a group axis for local caps.
- MedMNIST 2D family remains a good source (same data pipeline). Untried as of 2026-05-28: **BloodMNIST** (under test), **RetinaMNIST**, **OrganAMNIST**, **BreastMNIST**.
- Non-MedMNIST options are mostly exhausted (eurosat / so2sat dropped, cifar100 failed, aider already a winner).

---

## How to add a new candidate cleanly

If a future backbone or dataset is to be tried:
1. Read this file first. If the candidate appears above, the burden of proof for retrying it rests on you.
2. Add a wrapper to `src/models/imagery/` (backbone) or a prep script + entry to `IMAGERY_DATASETS` and `DATASETS` (dataset).
3. Probe → cap → smoke per the workflow in `src/config_generators/gen_model_search.py`.
4. If it fails, **append the verdict to this file** with the same row format. Don't silently drop it.

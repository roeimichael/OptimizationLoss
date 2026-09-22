# Yuval Kassif / Gonen Singer: Clipper replication setup

## Status: dataset staged; exact training protocol unresolved

Target: **Adaptive resource-constrained neural networks for multi-class medical
image classification**, DOI [10.1016/j.engappai.2026.115989](https://doi.org/10.1016/j.engappai.2026.115989).
Replicate its predict-then-optimize baseline before adding TraLO. This does not
implement the authors' adaptive cost-sensitive loss.

The supplied `D:/Downloads/S0952197626022736.htm` contains an abstract and metadata,
not the methods, experimental tables or full article body. Its embedded body is
empty. The [publisher preview](https://www.sciencedirect.com/science/article/pii/S0952197626022736)
identifies knee osteoarthritis and DermaMNIST, but does not expose enough detail
for an exact baseline replication. No backbone, hyperparameters, original result
numbers or knee dataset identity have been invented from similar studies.

| Required replication item | Verified / unresolved | Next evidence |
|---|---|---|
| Data domains | Knee osteoarthritis X-rays; DermaMNIST skin lesions | Publisher experimental-study snippet |
| Knee dataset version, download, split and IDs | Unresolved | Full experimental section / author files |
| DermaMNIST version and native resolution | Dataset family confirmed; paper version unresolved | Full paper / dataset-loading code |
| Backbone, classifier head, frozen versus fine-tuned layers, pretrained weights | Unresolved | Architecture and initialization configuration |
| Resize, normalization, augmentation | Unresolved | Exact preprocessing; native 224 differs from 28 resized to 224 |
| Split and patient/lesion handling | Official candidate splits staged; paper split unresolved | Author split indices and lesion IDs |
| Optimizer, schedule, epochs, batch size, seeds, checkpoint selection | Unresolved | Full protocol and configuration |
| Baseline training loss and class weighting | Unresolved | CE versus weighted CE or another objective |
| Constrained classes, capacity construction and rounding | Unresolved | Paper constraint specification and per-run budgets |
| Post-hoc allocation | Unresolved; do NOT assume our greedy policies match the paper | Objective, upper bounds versus equality, solver/greedy ordering, tie handling |
| Metrics, aggregation, reference numbers and uncertainty | Unresolved | Full result tables and exact metric definitions |

## What is ready now

Downloaded the official **28x28 candidate** from [MedMNIST Zenodo record10519652](https://zenodo.org/records/10519652/files/dermamnist.npz?download=1).
This is staging, not a decision that the paper used 28x28. The official
[metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) supplies
the seven-class mapping and archive checksum. The tracked manifest is
`examples/dermamnist_candidate_manifest.json`; the loader preserves the original
split and order. No model has trained on or selected settings using these data.

| Split | Images | Shape | Within-split identical-image extra rows |
|---|---:|---|---:|
| Training | 7,007 | 28x28 RGB uint8 | 1 |
| Validation | 1,003 | 28x28 RGB uint8 | 1 |
| Test | 2,005 | 28x28 RGB uint8 | 0 |

Official MD5 verified: `0744692d530f8e62ec473284d019b0c7`.
Downloaded SHA-256: `1a309fec2e33bb6aba88e7d078e5ccbb9736c84a0b415ac197eb3c8fa331e050`.
No identical image-byte hashes across these three splits. This does **not** mean
different images are from independent lesions or patients. Duplicate rows were
recorded and preserved, not silently removed from the replication candidate.

Local data: `C:/Users/roeym/.codex/rebuild-audit-20260922/yuval-replication-data/`.
Server staging target: `/home/dsi/michaer8/tralo-rebuild/data/yuval-replication/dermamnist-10519652-28/`.
The current candidate is CC BY-NC 4.0; its manifest records the source.

## Data distinction that matters

[MedMNIST+ documentation](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md)
distinguishes higher-resolution images generated from original images from
upsampling the small 28x28 images. Matching the network's input dimensions alone
does not establish equivalent inputs.

The independent [2025 Scientific Data audit](https://www.nature.com/articles/s41597-025-04382-5)
documents lesion overlap in original DermaMNIST partitions. Its corrected
DermaMNIST-C uses different partitions. Replication must preserve and disclose
the paper's actual choice. A separate independence study can later use corrected
splits; it must not be presented as the same experiment.

## Execution sequence once the missing specification arrives

1. Fill `examples/yuval_replication_pending.json` with source-attributed settings
   and exact baseline result targets. The template is deliberately non-runnable.
2. Reconcile the dataset checksum, sample order, class map and resolution with
   the author loader. Audit patient/lesion IDs separately from byte duplicates.
3. Match the raw, unconstrained baseline first: data, architecture, supervised
   objective, training dose and checkpoint selection. Save every raw probability.
4. Implement or verify the paper's actual allocator on hand-computed small cases.
   Our existing upper-correction and capped-first are greedy diagnostics, not
   guaranteed implementations of the paper's predict-then-optimize baseline.
5. Apply paper caps to saved probabilities, independently recount predictions and
   recompute paper metrics. Compare per seed and aggregate against the paper.
6. When author code arrives, compare both allocators on one identical probability
   matrix and compare logits from an identical checkpoint/input batch. Agreement
   in final accuracy alone is weaker evidence than agreement in these components.
7. Only then add TraLO and a matched null under the same fixed deployment policy.

Matching published scores supports replication, not proof that all code is right;
disagreement can arise from undocumented preprocessing or seeds, not only bugs.
Full paper PDF (especially experimental setup, baseline definition and tables)
is needed before a faithful training run. The repository will resolve finer
implementation ambiguities when Yuval supplies it.

## Commands

From the committed immutable release, run without GPU:

```text
python tools/audit_medmnist.py DATA/dermamnist.npz examples/dermamnist_candidate_manifest.json NEW_AUDIT_DIRECTORY
python -m unittest discover -s tests -v
```

No guessed-paper GPU training is dispatched by this setup.

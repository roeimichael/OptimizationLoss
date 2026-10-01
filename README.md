# TraLO research workspace

This repository tests whether a **training-time, label-free prediction-capacity constraint** improves who receives a limited allocation. A cap being met is necessary but is not evidence that the predictions are better. Negative runs, failed integrity gates and old protocols remain available for audit.

Start with the [experiment guide](experiments/README.md) for the current results and the [experiment map](experiments/INDEX.md) for the reading order. The [short meeting PDF](output/pdf/tralo_meeting_brief_20261001_v2.pdf) explains the present conclusion. The [longer numerical PDF](output/pdf/tralo_research_status_20261001_v2.pdf) has full context. [DESIGN.md](DESIGN.md) and [NEXT_STEPS.md](NEXT_STEPS.md) record the rebuilt system and operational history; the main checkout's `docs/FRAMEWORK.md`, `RULESET.md`, `docs/MISSION.md` and `docs/LEDGER.md` govern new studies.

## Current knee comparison

The [prospective knee protocol](experiments/knee_persistent_kassif_match_protocol_draft_20261001.md) fixes a matched comparison of sustained TraLO, Kassif's PAO retraining, plain training plus the same Clipper, a dose-matched random direction, and a no-op control. It uses the Chen knee images and EfficientNet-B5, MobileNetV3-Large and ViT-B/16. The runner, independent scorer and configurations are frozen at release `7a5f68b021d0f0a70185309c4a21be3a27522e5c`. Both-host source/tests and label-free data/weight preflights passed. **No knee training run in this new block has started.** The proposed pilot GPU-hour ceiling is awaiting approval; the sealed Chen test is untouched.

This is a new scientific test because previous encouraging knee results changed side copies of a trained model, leaving the actual training path unchanged. The new runner changes the model after each supervised epoch and continues training. Historical wins and losses are described in the experiment guide and are not pooled with this new study.

## Reading and preservation

`tralo/` contains methods; `analysis/` contains independent scorers; `tests/` holds checks; `experiments/configs/` contains immutable run inputs; `tools/` holds bounded queues. Never run a historical result file as if its title implied a winner. The [recoverable provenance archive](experiments/archive/provenance_20261001/README.md) holds older notes and the original rebuild README by original-path and SHA-256 manifest. Data, predictions, checkpoints and Git objects were not deleted during cleanup.

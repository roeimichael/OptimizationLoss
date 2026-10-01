# TraLO: capacity-aware image classification

TraLO asks whether changing a classifier **during training** can improve which people or items receive a limited number of positive decisions. Meeting a capacity limit is not itself a win: the trained model must make better decisions than an equally constrained ordinary classifier.

The current knee comparison uses the same images, ImageNet backbones, augmentation, sampler, optimizer, stopping data, and deployment allocator for TraLO and Yuval Kassif and Gonen Singer's PAO/PTO baselines. Kassif's PAO changes a supervised cost matrix and retrains. The tested TraLO variant instead takes a gradient of the predicted count on **unlabeled development images** after each supervised epoch, moves just far enough to meet the hard cap, and continues training. A random direction with the same displacement and a no-change control help isolate what that direction contributes. TraLO is a training method, **not a new image backbone**. The exact mathematics and the limits of the novelty claim are in [the method guide](docs/THESIS_METHOD.md).

The Chen knee development cohort has been inspected repeatedly. New seeds on it are useful for a matched comparison, but they are **not held-out confirmation**. Its sealed test cohort remains untouched. The current fixed knee protocol is [here](experiments/knee_persistent_kassif_match_protocol_draft_20261001.md); the [experiment guide](experiments/README.md) separates completed evidence from pilots and work still running. Live run state belongs in immutable run receipts, not in this README.

| Directory | Purpose |
| --- | --- |
| `tralo/` | Methods, shared data/metric primitives, and preserved earlier study runners. The current knee runner is `knee_persistent_match.py`. |
| `analysis/` | Independent scorers and preserved diagnostics. The current knee scorer is `score_knee_persistent_match.py`; historical unreferenced outputs are under `analysis/archive/`. |
| `tests/` | Executable checks of methods, data boundaries, runners, and scorers. |
| `experiments/` | Fixed protocols, immutable run configs, result reports, and recoverable provenance. Start with its README. |
| `tools/` | Bounded, exclusive server queues and preflight tools. |
| `output/pdf/` | Presentation copies, including the [short brief](output/pdf/tralo_meeting_brief_20261001_v2.pdf) and [numerical report](output/pdf/tralo_research_status_20261001_v2.pdf). |

Older scripts and negative results remain recoverable. A historical filename is not a recommendation. The main checkout's `docs/FRAMEWORK.md`, `RULESET.md`, `docs/MISSION.md`, and `docs/LEDGER.md` govern new research. Do not use development labels to tune a constraint, evaluate the sealed test by accident, or change a live release.

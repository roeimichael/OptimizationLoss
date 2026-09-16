# OptimizationLoss (TraLO)

Thesis project. Train neural networks to satisfy **transductive prediction-count
constraints** -- "within group *G*, predict class *C* at most *K* times" -- via soft
constraint optimization during training, and test whether that beats simply clipping the
predictions after the fact.

Active research repository, not a library. The maintained training pipeline and
evidence audit live here; historical experiments are not a second active architecture.

---

## Read this first

> **[`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) is the protocol, and it wins every
> conflict.** It defines the evidence boundary, metrics, validation gates and
> comparison protocol.

Three files carry the rest, and between them they are the whole live record:

| File | Answers |
|---|---|
| [`RULESET.md`](RULESET.md) | How to work. Rules, gates, re-entry checklist. |
| [`docs/MISSION.md`](docs/MISSION.md) | Where the project stands, what is running, what is next. |
| [`docs/LEDGER.md`](docs/LEDGER.md) | What is proved, what is measured, what is closed. |

The user requested an evidence reset on 2026-09-14: old tables and folder names
are not validation. Superseded documents are preserved in `docs/archive/` and
recoverable in full from git history. If other material disagrees with the
framework, the framework wins.

Git tracks source, tests, split metadata and curated notes. Results, checkpoints,
compiled papers and local audit dumps stay outside GitHub. See
[`docs/GIT_TRACKING.md`](docs/GIT_TRACKING.md) for tracking and evidence recovery.

## Method

```
Warm-up (CE, or focal for the focal baseline)
  ──► alternating task and global/local constraint updates (trained arms)
      ──► post-hoc deployment under the same caps
          ──► evaluation / scoring
```

Compared against post-hoc clipping baselines and rival dual methods (Fioretto-LDF,
Hounie-RCL, ALM), plus a matched zero-constraint control. The shared-allocator
correction is still in progress; see MISSION before generating a new comparison.

## Layout

| Path | Purpose |
|---|---|
| `docs/FRAMEWORK.md` | current protocol, evidence boundary and validation gates |
| `configs/protocol.yml` | the fixed experimental protocol |
| `configs/gen_campaign.py` | paired seven-arm campaign generation |
| `src/losses/`, `src/methodologies/` | the constraint losses and the methods being compared |
| `src/models/`, `src/training/`, `src/pipeline/` | models and the run pipeline |
| `src/experiments/` | experiment drivers |
| `scripts/` | dataset preparation, operational validation and `deployed_h2h.py` reporting |
| `evidence/` | historical provenance/prediction tarballs; not the fresh corpus |
| `results/` | fresh run outputs only after reset validation |
| `docs/archive/` | **quarantined.** Superseded documents, the paper tree, audit receipts. Preserve as evidence, never as instructions -- see `docs/archive/README.md` |
| `.github/workflows/preflight.yml` | CI preflight |

## Verify before running

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m scripts.preflight --before-launch
python -m scripts.audit_config
```

Campaign generation and launch must follow the framework and the
`scripts.run_campaign` stage/verify/launch/firstrun/score gates. Explicitly scope
the dispatcher with `EXPERIMENT_DIR` and the approved GPU with
`CUDA_VISIBLE_DEVICES`; a bare dispatcher invocation is not a campaign definition.

GPU runs use the university `dsisco01` and `dsisco02` hosts through `dsihead`.
They share storage but have different GPU generations. VPN access and a working
SSH key are required. The image arrays are server-side and gitignored: a fresh
checkout is not automatically runnable, even when metadata and tests are present.

## Status

Cleanup and validation in progress (September 2026). Empirical superiority is an
open question, not a guarantee. Primary quality metric: deployed cc-F1, alongside
macro-F1, precision/recall and feasibility. A best mean is not automatically a
statistically established win. Historical results are archived recoverably;
fresh executions will remain distinct from genuinely untouched evaluation data.

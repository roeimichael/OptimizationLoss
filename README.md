# OptimizationLoss (TraLO)

Thesis project. Train neural networks to satisfy **transductive prediction-count
constraints** — "within group *G*, predict class *C* at most *K* times" — via soft
constraint optimization during training, and test whether that beats simply clipping the
predictions after the fact.

Active research repo, not a library. The active line of work is here; the clean rewrite
lives in [`OptimizationLossV2`](../research/OptimizationLossV2).

---

## Read this first

> **[`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) is the only operational document.**

It holds the fixed experimental protocol, every idea that has already failed and why, the
current honest status of each claim, and the one open question. Everything else in `docs/`
is history. **If any other file disagrees with `FRAMEWORK.md`, `FRAMEWORK.md` wins** — this
README included. Do not propose, run, or score anything before reading it.

[`CLAUDE.md`](CLAUDE.md) is the short version: the five protocol rules that get broken most
(warm-up/constraint epoch splits, scoring at equal dose, etc.).

## Method

```
Warm-up (CE only)
  ──► constraint optimization  (CE + global + local + KL, lambda ratchet)
      ──► post-hoc adjustment  (global → local → reverify global)
          ──► evaluation / scoring
```

Compared against post-hoc clipping baselines and rival dual methods (Fioretto-LDF,
Hounie-RCL, ALM), plus null and reseed controls.

## Layout

| Path | Purpose |
|---|---|
| `docs/FRAMEWORK.md` | **the protocol + the ledger of what is and is not established** |
| `configs/protocol.yml` | the fixed experimental protocol |
| `configs/gen_campaign.py`, `task_cells.py`, `task_windows.yml` | campaign / cell definitions |
| `src/losses/`, `src/methodologies/` | the constraint losses and the methods being compared |
| `src/models/`, `src/training/`, `src/pipeline/` | models and the run pipeline |
| `src/experiments/` | experiment drivers |
| `scripts/` | probes, audits and scorers (`paper_rows.py`, `deployed_h2h.py`, `cell_table.py`, …) |
| `evidence/` | archived prediction tarballs — the reproducibility anchor |
| `results/` | run outputs |
| `archive/` | superseded material |
| `.github/workflows/preflight.yml` | CI preflight |

## Run

```bash
pip install -r requirements.txt
python main.py            # see configs/protocol.yml for the campaign definition
pytest                    # pytest.ini at repo root
```

GPU runs are executed on the university `dsisco02` server.

## Status

Active (Sept 2026). Large on disk (~9 GB) — `evidence/`, `results/` and caches dominate.

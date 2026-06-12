# Paper-track results archive

Built by `scripts/build_archive.py`. Lossless manifest over paper-track sweeps only.
Probe / dropped-dataset / failed sweeps are intentionally excluded.

## Coverage

- **Cells indexed**: 4385
- **Cells with missing evaluation_metrics.csv (skipped)**: 30
- **Datasets**: `dermmnist` (1702), `tissuemnist` (1363), `aider` (1188), `octmnist` (72), `cifar100` (48), `retinamnist` (12)
- **Models**: `MobileNetV3` (3416), `MobileNetV2` (331), `ShuffleNetV2` (331), `RegNetY400MF` (307)
- **Methods**: `tralo` (997), `fioretto_ldf` (898), `hounie_rcl` (850), `danits_lp` (820), `heuristic` (820)
- **Sweeps**: 37

## Files

- `MASTER_INDEX.csv` — one row per cell, all fields. Source of truth.
- `by_axis/per_dataset.md` — breakdown by dataset
- `by_axis/per_model.md` — breakdown by backbone
- `by_axis/per_method.md` — breakdown by methodology + mean macro-F1
- `by_axis/per_tightness.md` — breakdown by constraint tightness
- `by_axis/per_sweep.md` — what each sweep was for, cell counts
- `tables/pivot_ds_model_method.csv` — cell counts (dataset, model, method)
- `tables/pivot_ds_tight_method.csv` — cell counts (dataset, tightness, method)
- `tables/methodology_means.csv` — mean macro_f1 / sat / flips / ECE per (ds, model, method)
- `tables/paired_tralo_vs_<baseline>.csv` — per-cell (TraLO − baseline) deltas, same-seed paired
- `tables/paired_summary.csv` — W/L/T + mean delta per (baseline, dataset, sym/asym) — paper-table-ready

## How to use

Filter the master CSV. E.g. with pandas:
```python
import pandas as pd
df = pd.read_csv('archive/MASTER_INDEX.csv')
# Headline: tissue MobileNetV3 sym tightness, all methods
df[(df.dataset=='tissuemnist') & (df.model=='MobileNetV3') &
   (df.is_asymmetric==0)].groupby('method').macro_f1.agg(['mean','std','count'])
```

Paired comparison (TraLO vs each baseline, same seed):
```python
key = ['sweep','dataset','model','constraint_tag','seed','data_dir']
wide = df.pivot_table(index=key, columns='method', values='macro_f1')
wide['d_vs_hounie']   = wide['tralo'] - wide['hounie_rcl']
wide['d_vs_fioretto'] = wide['tralo'] - wide['fioretto_ldf']
wide['d_vs_danits']   = wide['tralo'] - wide['danits_lp']
```

## Rebuilding

```bash
cd ~/OptimizationLoss && python scripts/build_archive.py
```

Idempotent — overwrites this archive on each run. Cell paths in MASTER_INDEX
are relative to repo root; raw artifacts (config.json, evaluation_metrics.csv,
final_predictions.csv, training_log.csv) remain in `results/pending_runs/...`.
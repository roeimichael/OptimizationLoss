# `final_AAAI_PAPER/data/` — self-contained paper data

Everything the paper reads lives here. The figure scripts in `../scripts/` and the
committed table `.tex` files draw **only** from this folder. The paper no longer reaches
into the sprawling raw run tree (which is now archived, see bottom), so it cannot
accidentally pull in a non-paper dataset or backbone.

## `corpus/` — aggregated metrics (one row per finished run)

| File | What it is | Consumed by |
|---|---|---|
| `corpus_final.csv` | The canonical corpus. `sweep=='paper_final'` is the locked headline grid: **3 backbones {MobileNetV3, ViTB16, RegNetY400MF} × 3 datasets {tissuemnist, dermmnist, octmnist} × 9 caps × 6 methods × 4 seeds = 1944 rows.** (Other `sweep` values are robustness/exploration rows kept for audit; not in the paper.) | `make_octmnist_fig.py`, `make_deployment_fig.py`, and (by hand) all 4 `tables/*.tex` |
| `extra_robustness_corpus.csv` | Aggregated extra-robustness + component-ablation runs. | ablation table (future work) |
| `ablation_complete.csv` | Digest built from the two CSVs above. | `tables/tab_ablation_complete.tex` |
| `convergence_census.csv` | Per-run epochs-to-feasibility for the §5.4 convergence census: `sat_epoch` and `cep` (= `sat_epoch` minus TraLO's 50-epoch warmup) for DermMNIST+TissueMNIST × {MobileNetV3, RegNetY400MF} × symmetric caps × {tralo, fioretto_ldf, hounie_rcl} × 4 seeds (warmup 50; 216 rows, 18 all-three cells). The §5.4 summary ("median 17/20/44, Hounie slowest in every cell, TraLO first-or-tied in 13 of 17") is over the **17 cells where the cap binds**; it excludes the one degenerate loose-cap cell (`dermmnist / RegNetY400MF / L80`) where all three methods are feasible within ~2 epochs, so there is no convergence race to compare. Seed aggregation is median over the 4 seeds within each cell; the cross-cell summary is the median of those per-cell medians (never a pool of raw runs across cells). | §5.4 convergence claim |
| `kl_sweep.csv` | TissueMNIST `L30_G30`, MobileNetV3, TraLO: constrained-class F1 (class 4) per `alpha_kl ∈ {0,0.1,0.3,1.0}` × seeds. Backs the Supp B KL sweep ("nonzero anchor lowers cc-F1 by 0.01–0.03"). | Supp B KL claim |
| `CAMPAIGN_RESULTS.md` | Notes on the extra-robustness campaign. | reference |

> Tables are **hand-maintained** `.tex` (their generators in `../../scripts/` were
> superseded by manual edits — re-running them would revert those edits). Treat
> `corpus_final.csv` as the source of truth and edit the `.tex` by hand.

## `dynamics/` — per-epoch training logs (the 2 dynamics figures only)

`fig_convergence` and `fig_mechanism` need per-epoch `training_log.csv`, which the corpus
CSV does not carry. Only the exact seed dirs used are copied here (30 log files, class 4;
the w1-probe dirs also carry their `config.json` so the recipe is auditable):

| Figure | Cells copied | Original sweep |
|---|---|---|
| `fig_convergence` | `tissuemnist/tralo/.../L30_G30/seed_{1..4}` | `paper400_tralofix` |
| | `tissuemnist/{fioretto_ldf,hounie_rcl}/.../L30_G30/seed_{1..4}` | `paper400_baselines` |
| | `dermmnist/{tralo,fioretto_ldf,hounie_rcl}/.../L30_G30/seed_{1..4}` | `paperv2_phase1` |
| `fig_mechanism` | `dermmnist/{fioretto_ldf,tralo}/w1_probe/.../L50_G50/seed_{1..3}` | `server_only_sweeps/pushpull_derm_w1` (RegNetY400MF, **warmup=1**) |

The `fig_mechanism` probe is deliberately off the paper grid: warmup=1 keeps the
classifier learning through the whole constraint phase, so the CE-vs-constraint
tug-of-war is visible (on the warmup-50 grid, the CE-saturation gate stops CE updates
within ~2 constraint epochs in every method and the interaction is invisible). Full
paper recipe (`alpha_kl=0`, as in every canonical run — the KL anchor is off throughout
the corpus). The figure uses seed 1; seeds 2–3 replicate (Fioretto max λ 57.5/60.4 vs
TraLO 0.18/0.19).

## Provenance / archive

- `corpus_final.csv` was copied from `paper/aaai_tables/_corpus_with_final.csv`.
- `convergence_census.csv` was rebuilt from `../../archive_experiments/raw_runs_2026-07/
  cells_index.csv` (filter `warmup_epochs==50`, symmetric caps, dedup by newest `mtime`;
  `cep = sat_epoch - 50` for TraLO to net out its warmup offset).
- `kl_sweep.csv` was extracted from the `kl_ablation` runs under
  `../../archive_experiments/raw_runs_2026-07/pending_runs/tissuemnist/tralo/.../kl_ablation/`
  (cc-F1 = `F1_Class4` from each run's `evaluation_metrics.csv`).
- The `dynamics/` L30_G30 logs were copied from `results/pending_runs/` (paths above);
  the w1-probe logs from `../../archive_experiments/raw_runs_2026-07/server_only_sweeps/
  pushpull_derm_w1/RegNetY400MF/dermmnist/L50_G50/`.
- The full raw run tree (2.3 GB: all datasets/backbones/sweeps incl. eurosat and the
  non-paper backbones) and the `paper/aaai_tables/` scratch files were moved to
  `../../archive_experiments/raw_runs_2026-07/` — nothing was deleted; it stays on disk
  and is recoverable. The raw **ablation** sweeps (`g5_component_ablation`,
  `g5_short_warmup`, `kl_ablation`, `tablef_shortwarm`, `server_only_sweeps/*`) live there
  too, should the ablation be extended.

# Live status — pending Blackwell sweeps

**Last updated:** 2026-06-01 (orchestration session, ~18:00 IDT)
**Source of fresh status:** `docs/MISSING_EXPERIMENTS.md` + the per-sweep
`results/pending_runs/<sweep>/` directory (count `evaluation_metrics.csv`).

## Completed Blackwell tables (paper-eligible) ✅

| Table | Cells done | Used by §                       |
|-------|------------|---------------------------------|
| A     | 360        | §5.1 Headline                   |
| B     | 480        | §5.2 Asymmetric (derm)          |
| C     | 480        | §5.3 Backbone-saturated         |
| D     | 360        | §5.4 Multi-class (derm)         |
| E     | 120        | §5.5 Group-column (derm)        |
| F     | 28         | (already in main.tex §3.4)      |
| G     | 16         | (already in main.tex §3.5)      |

## Currently running on Turing (dsisco01, screening only — NOT paper-eligible)

| Sweep              | Backbones        | Datasets         | Cells   | Status                |
|--------------------|------------------|------------------|---------|------------------------|
| `paper_backbones`  | MobileNetV2, RegNetY400MF, ShuffleNetV2 | tissue, derm, aider | 909     | ~97/909 last seen (16:55 launch, ETA ~02:00 IDT) |

Purpose: tells us **which slice of MobileNetV2 actually wins** so we can
issue a targeted Blackwell rerun for G1 instead of running all 360 cells
blindly. RegNet+Shuffle are insurance corroboration we'll only need if
MobileNetV2 fails.

## Queued for Blackwell (dsisco02) — generators ready, launch blocked by SSH gateway

The campus SSH jump host `dsihead.lnx.biu.ac.il` is unreachable as of
2026-06-01 ~18:00. Configs are pre-written; launch commands in
`launch_commands/`. The four sweeps can be fired in any order once SSH
returns.

| Gap | Cells | Wall-clock (2× Blackwell) | Generator                                                    | Launch script                            |
|-----|-------|---------------------------|--------------------------------------------------------------|------------------------------------------|
| G4  |   12  | ~5 min                    | `src/config_generators/gen_g4_table_b_missing_seeds.py`     | `launch_commands/launch_g4.sh`           |
| G1  |  240  | ~3 h                      | `src/config_generators/gen_g1_mobilenetv2.py`               | `launch_commands/launch_g1.sh`           |
| G3  |  360  | ~5 h                      | `src/config_generators/gen_g3_multiclass_tissue.py`         | `launch_commands/launch_g3.sh`           |
| G2  |  960  | ~13 h                     | `src/config_generators/gen_g2_asym_tissue_aider.py`         | `launch_commands/launch_g2.sh`           |

Wall-clock is 2 Blackwell GPUs at the observed Turing throughput (~37
cells/hr/GPU) scaled by Blackwell's measured ~2.5× speedup. Real Blackwell
throughput should beat these.

**Recommended order**: G4 → G1 → G3 → G2 (smallest+highest-impact first).
G2 alone fills a 13-hour overnight window; the rest can sit on a 2nd GPU.

## Recommended sequence when SSH returns

1. Sync the four new generator files (already exist locally, not yet on
   server) and the patched `src/models/model_factory.py` +
   `src/models/imagery/__init__.py`:

   ```bash
   scp src/config_generators/gen_g{1,2,3,4}_*.py dsisco01:~/OptimizationLoss/src/config_generators/
   scp src/models/model_factory.py dsisco01:~/OptimizationLoss/src/models/
   scp src/models/imagery/__init__.py dsisco01:~/OptimizationLoss/src/models/imagery/
   ```

2. Generate configs (on dsisco01; NFS-shared with dsisco02):

   ```bash
   ssh dsisco01 'cd ~/OptimizationLoss && for g in 4 1 3 2; do ~/anaconda3/envs/optloss/bin/python -m src.config_generators.gen_g${g}_*; done'
   ```

3. Launch each sweep — see `launch_commands/launch_g<n>.sh`. Each script
   pins to two dsisco02 GPUs (defaults 2 + 3 to dodge the davidlevin
   GPU0+1 routine and the nirgal GPU3 routine; verify with
   `nvidia-smi` first).

4. Aggregate when done — see `aggregators/agg_g<n>.py`.

## Health rules (DO NOT VIOLATE)

- Max 2 GPUs per dispatcher.
- Never share a physical GPU on dsisco02 — driver crash risk.
- Always check `nvidia-smi` before launching; abort if a GPU shows two distinct users.
- `conda activate optloss` (the base env has CPU-only torch).
- `Hounie CE=nan` warnings are benign — keep going.
- Real errors look like `Traceback` or `[!!]` in the log.

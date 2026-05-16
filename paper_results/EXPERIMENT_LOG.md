# Paper rerun experiment log

**Started**: 2026-05-15 17:37 dsisco02 GPU 0 (PID 2128682)
**Sweep root**: `results/pending_runs/paper_rerun/`
**Log file**: `logs/paper_rerun.log`

## Why a rerun was needed

Two regressions invalidated the May-8 `PAPER_TABLES.md` numbers:

1. **TraLO CE-saturation-skip removed in commit `4c1b1d0` (2026-05-03)**
   - The CE batch loop ran indefinitely even after `train_acc≥0.995`.
   - Bounded penalty `[E/(E+K) + ρ·(E/K)²/(1+(E/K)²)]` saturates as `E→∞`,
     so it cannot overpower a persistent CE force.
   - TraLO no longer satisfied during training. Posthoc patched the gap,
     so the OLD tables read `sat=Y, flips=0` falsely.
   - **Fix**: restored CE-skip (this code change, 2026-05-14).

2. **BF16→FP32 argmax illusion across all three constraint-aware methods**
   - Pre-fix count-forward used autocast → BF16 argmax flipped borderline
     predictions vs FP32 eval. The "raw satisfied" flag inside training
     used BF16; the final evaluation used FP32 → mismatch.
   - Fioretto, Hounie, and TraLO all looked satisfied during training but
     genuinely violated at FP32 eval time.
   - **Fix**: FP32 count-forward (commits `eb24f96`, `04bb7d3`,
     `6f25bd0`) — TraLO + Hounie + Fioretto now consistent with eval.

3. **Hounie primal/dual scale mismatch (commit `2d2dca0`, 2026-05-09)**
   - Hounie's λ and u updates were on incompatible scales → constraint
     enforcement was effectively dampened.
   - **Fix**: `hounie_rerun` sweep ran all prior Hounie cells with the
     fixed code. That data is part of the "valid" set used here.

## Validity of existing data

| Method | Trusted sweeps (post-fix) |
|---|---|
| TraLO | `fix_ce_skip`, `fix1_validation`, `kl_sweep`, `overnight_2026_05_14` |
| Fioretto | `convergence_validation_300`, `fix1_validation`, `overnight_2026_05_14` |
| Hounie | `hounie_rerun`, `convergence_validation_300`, `fix1_validation`, `overnight_2026_05_14` |

Everything older was generated under one or both of the above bugs and is NOT comparable.

## Tier 1 scope (currently running)

**Subset**: TissueMNIST on MobileNetV3 only. 91 missing cells:
- 29 TraLO (mostly tightness extremes L20, L40, L60, L80 — not in overnight grid)
- 37 Fioretto (cells outside the (1,4,7) overnight slice)
- 25 Hounie (same)

Multi-class extensions: `(1,4)` cell × 3 seeds — 3 missing per method.

**Estimated wall time**: ~22 h on 1 GPU at ~15 min/run average (Hounie/Fioretto take ~22 min each, TraLO ~3-5 min with CE-skip).

## TraLO config used (canonical fix1 settings)

```python
{
  "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
  "warmup_epochs": 50, "constraint_epochs": 300,
  "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
  "initial_rho": 5.0, "rho_target": 100.0,
  "alpha_kl": 0.0,              # KL anchor disabled (proven not to help)
  "linear_sat_tail": 0.0,        # β=0 (proven not to help; CE-skip alone suffices)
  "penalty_mode": "both",        # R + Q penalty
  "enable_ce_skip": True,        # THE FIX — March behavior restored
}
```

## Next tiers (queued if Tier 1 succeeds)

- **Tier 2**: extend to ResNet18 + EfficientNetB0 on TissueMNIST (~50 more runs).
- **Tier 3**: DermMNIST, EuroSAT, So2Sat on MobileNetV3 (~70 more runs).
- **Tier all**: full grid (245 runs).

## How tables get regenerated

When runs finish, run:

```bash
python paper_results/regenerate_tables.py
```

This reads `results/pending_runs/paper_rerun/**/evaluation_metrics.csv`,
merges with valid post-fix data from prior sweeps, and writes
`paper_results/PAPER_TABLES_v2.md` with the same structure as the original
`PAPER_TABLES.md` but with fresh, honest numbers.

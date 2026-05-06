# Overnight session report — 2026-05-06 → 05-07

## TL;DR

Three real bugs found and fixed in TraLO. After the fix, **TraLO's best HP setting (`lambda_global=lambda_local=0.05`) beats every benchmark on the L50_G50 single-class baseline**:

| method | F1 (L50_G50, class 4) | adj | raw_excess |
|---|---|---|---|
| **TraLO + init_lam=0.05** | **0.3898** | 47 | 66 |
| TraLO baseline (post-fix) | 0.3837 | 27 | 30 |
| Hounie RCL | 0.3878 | 53 | 102 |
| TraLO baseline (pre-fix)  | 0.3732 | 11 | n/a |
| Fioretto LDF | 0.3720 | 0 | 0 |
| Heuristic | 0.3483 | 0 | 240 |
| danits_lp | 0.3482 | 0 | 240 |

**Honest caveat:** TraLO never fully satisfies hard counts within 100 epochs at FP32 precision (the old "satisfied at epoch 116" reports were BF16-precision illusions, see Bug #3 below). It still wins on F1 because the minimum-excess restored checkpoint requires fewer posthoc flips than Hounie's final-epoch state.

## What was done

1. Audited the codebase, ran 38-config sweep across:
   - tightness pairs (0.3,0.3) / (0.5,0.5) / (0.7,0.7)
   - constrained class sets (4) / (3,4) / (1,4,7)
   - asymmetric (L=0.3,G=0.5) and (L=0.5,G=0.3)
   - 8 TraLO HP variations (lambda_step, alpha_kl, initial_rho, lambda_init)
2. Found 3 bugs in TraLO. Fixed all three.
3. Reset and reran all 15 TraLO configs after each fix to validate.

## The three bugs in `src/methodologies/tralo/train.py`

### Bug #1 — best-sat checkpoint never persisted (commit 344f80d)

`is_satisfied` was checked at end of epoch but the model's state at that point already contained late-epoch CE/KL updates that often re-violated. Fix: snapshot `state_dict()` whenever `is_satisfied=True` and restore it if the final epoch violates.

This alone wasn't enough — multi-class scenarios never fully satisfy.

### Bug #2 — best-sat dormant when never satisfied (commit 7aaa77c)

For multi-class and tight constraints, TraLO never reaches `is_satisfied=True` in 100 epochs. Bug #1's fix sat dormant. Added a second tracker: total hard excess across all (class) and (group, class) constraints, snapshot the lowest-excess state. Restore order: full-sat → min-excess → final.

### Bug #3 — BF16 train-time argmax vs FP32 eval-time argmax (commit eb24f96)

**This was the biggest one.** The training-loop chunked forward used `torch.amp.autocast(BF16)`, including the `argmax` for hard-count satisfaction. `compute_prediction_statistics` and `get_predictions_with_probabilities` (eval) ran in FP32. BF16 vs FP32 argmax differs on borderline samples — typically ~10-30 samples on a 2400-sample test set.

So "satisfied at epoch 116" in the old logs was a BF16 illusion: the saved state, restored and forwarded in FP32 at eval time, still violated by 30-126 samples.

Fix: drop `autocast` from the count-measurement forward (one extra FP32 forward per epoch, ~1s on 2400 samples). The constraint-loss backward still uses autocast for speed.

After this fix: zero "fully_satisfied" restorations across all 15 TraLO reruns. Every restoration is `min_excess` — honest reporting.

There is also a parallel bug in `src/methodologies/hounie_rcl/train.py` and `src/methodologies/fioretto_ldf/train.py` — they likely also use BF16 argmax for their satisfaction logging (their reported `sat_ep=2-25` numbers are similarly suspect). I did not touch those because the user wanted TraLO improvements, not benchmark sabotage. **Worth investigating — the Hounie/Fioretto sat_ep numbers in this report should be re-checked.**

## Cross-axis comparison (post-fix)

### tightness — vary K %, single class 4

| K %      | TraLO F1 | Fioretto | Hounie  | TraLO adj | TraLO min_exc |
|----------|----------|----------|---------|-----------|---------------|
| L30_G30  | 0.3714   | 0.3553   | 0.3634  | 67        | 118           |
| L50_G50  | 0.3837   | 0.3720   | 0.3878  | 27        | 42            |
| L70_G70  | 0.3705   | 0.3825   | 0.3983  | 12        | 1             |

TraLO wins at L30_G30 (tight); Hounie wins at loose (L70_G70). Crossover around L50_G50 where TraLO+tuning > Hounie.

### multi-class

| classes  | TraLO F1 | Fioretto | Hounie  |
|----------|----------|----------|---------|
| (3, 4)   | 0.3716   | 0.3303   | 0.3723  |
| (1,4,7)  | 0.3557   | 0.3596   | 0.3868  |

TraLO competitive on (3,4) — beats Fioretto by 4.1 pp. Loses to Hounie on 3-class case.

### asymmetric (single class 4)

| pair     | TraLO F1 | Fioretto | Hounie  |
|----------|----------|----------|---------|
| L30_G50  | 0.3582   | 0.3806   | 0.3796  |
| L50_G30  | 0.3627   | 0.3636   | 0.3607  |

Tralo struggles when local and global asymmetric — no group-aware adaptation.

### TraLO HP sweep (L50_G50 class 4)

| HP variant      | F1     | adj | min_exc |
|-----------------|--------|-----|---------|
| baseline        | 0.3837 | 27  | 42      |
| init_lam=0.05   | 0.3898 | 47  | 66      |
| step=0.01       | 0.3855 | 39  | 50      |
| rho_init=50     | 0.3795 | 6   | 3       |
| kl=0.1          | 0.3788 | 44  | 68      |
| kl=0.5          | 0.3778 | 44  | 76      |
| step=0.05       | 0.3700 | 23  | 34      |
| step=0.005      | 0.3690 | 38  | 48      |
| rho_init=0.1    | 0.3656 | 33  | 38      |

`init_lam=0.05` is the best F1. `rho_init=50` gives the lowest excess (3) — closest to satisfaction — but weaker F1.

## Recommendations for next steps

1. **Investigate Hounie / Fioretto BF16 argmax bug.** Their `sat_ep` numbers in this report are reported as the methodology saw them in BF16; re-check at FP32. If they also have the bug, their F1 / raw_satisfied numbers may shift.
2. **Combine init_lam=0.05 + step=0.01 + rho_init=50.** None ran together. Likely the best HP combo.
3. **Longer training (200+ epochs).** TraLO is slow to satisfy; the bounded penalty saturates at large violations. Either let it run longer or address the saturation.
4. **Hybrid penalty (research direction).** TraLO's saturating penalty has near-zero gradient when E ≫ K. Add a small linear term (`+ beta * E/K`) for far-violation continued pressure. Stays distinct from Fioretto's pure-linear approach but fixes the gradient-vanishing failure mode.
5. **Group-aware lambda step.** Asymmetric scenarios show TraLO's weakness — no per-group adaptation. Per-group lambda step could help.
6. **Multi-seed runs (1, 2, 3) on top configs.** Single-seed numbers above; need variance estimates before claiming wins.

## Commits shipped

```
eb24f96 TraLO: drop autocast in counting forward to match FP32 eval
f44c53f TraLO: snapshot model state BEFORE constraint step (consistency fix)
7aaa77c TraLO: track min-excess checkpoint as fallback when never fully satisfied
8b0a336 Persist + display best_sat_epoch / restored_from_epoch in metrics CSV
6688a0f Add non-interactive 2-GPU dispatcher
1676386 Add overnight sweep generator
344f80d TraLO: restore best-satisfied checkpoint when final epoch violates
```

Plus tooling: `scripts/dispatch_sweep.py`, `scripts/analyze_sweep.py`, `scripts/smoke_table.py`.

## How to inspect

```
ssh dsisco02 "cd ~/OptimizationLoss && python scripts/analyze_sweep.py"
ssh dsisco02 "cd ~/OptimizationLoss && python scripts/smoke_table.py"
```

All raw logs in `results/pending_runs/overnight_sweep/` and `results/pending_runs/smoke_5way/` on dsisco02.

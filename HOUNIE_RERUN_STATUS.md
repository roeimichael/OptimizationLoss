# Hounie post-bugfix rerun — PAUSED

Stopped 2026-05-10 to run TraLO 1000-epoch convergence test first.

## Progress at pause

| sweep | done | total |
|---|---|---|
| thesis (TissueMNIST headline + tightness) | 21 | 25 |
| thesis_ext (TissueMNIST extended phases) | 0 | 27 |
| thesis_dermmnist | 0 | 25 |
| thesis_eurosat | 0 | 9 |
| thesis_so2sat | 25 | 25 |
| **TOTAL** | **47** | **111** |

## Resume command

```
ssh dsisco02
cd ~/OptimizationLoss
source ~/anaconda3/etc/profile.d/conda.sh
conda activate optloss
nohup python scripts/dispatch_sweep.py --root results/pending_runs/hounie_rerun --gpus 1 > /tmp/hounie_resume.log 2>&1 &
disown
```

64 pending runs remaining at ~10 min/run on single GPU = ~10.7h.

## Fixed Hounie findings so far (47 done)

So2Sat headline (5 seeds, all 3 backbones): fixed Hounie ties or beats TraLO on F1m + ECE + Brier on every backbone. ResNet18 standout: +0.58pp F1m vs TraLO.

TissueMNIST L50 MobileNet (5 seeds): fixed Hounie F1m 0.3856, F1c 0.3299 — beats TraLO's 0.3737, 0.3048 by +1.19pp F1m and +2.51pp F1c.

TissueMNIST tightness (L30, L70): fixed Hounie wins both F1m by ~1.2-1.3pp.

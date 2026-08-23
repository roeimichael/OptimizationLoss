# Paired-significance sweep (TraLO vs baselines)

Seeds paired by (ds, model, cls, grp, tight). Bootstrap p over matched per-seed differences. **WIN** = mean favors TraLO AND p<0.05.

F1-macro: higher is better. Flips: lower is better (diff = baseline - TraLO).


## F1-macro by dataset

### aider (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | +0.0022 | 15/20 | 0.000 | **WIN** |
| hounie_rcl | 20 | +0.0176 | 20/20 | 0.000 | **WIN** |
| tralo_bounded | 20 | +0.0019 | 15/20 | 0.009 | **WIN** |
| danits_lp | 20 | -0.0051 | 3/20 | 0.000 | loss |
| heuristic | 20 | -0.0050 | 3/20 | 0.000 | loss |

### dermmnist (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 220 | +0.0001 | 108/220 | 0.868 | tie |
| hounie_rcl | 220 | +0.0042 | 145/220 | 0.000 | **WIN** |
| tralo_bounded | 220 | +0.0004 | 117/220 | 0.451 | tie |
| danits_lp | 220 | +0.0074 | 139/220 | 0.000 | **WIN** |
| heuristic | 220 | +0.0071 | 139/220 | 0.000 | **WIN** |

### tissuemnist (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 60 | +0.0010 | 33/60 | 0.287 | tie |
| hounie_rcl | 60 | +0.0042 | 36/60 | 0.005 | **WIN** |
| tralo_bounded | 60 | +0.0019 | 35/60 | 0.097 | tie |
| danits_lp | 60 | +0.0161 | 46/60 | 0.000 | **WIN** |
| heuristic | 60 | +0.0151 | 43/60 | 0.000 | **WIN** |


## Post-hoc flips by dataset (diff = baseline - TraLO; + = TraLO needs fewer)

### aider flips

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | +5.3000 | 20/20 | 0.000 | **WIN** |
| hounie_rcl | 20 | +24.8000 | 20/20 | 0.000 | **WIN** |
| tralo_bounded | 20 | +5.4500 | 19/20 | 0.000 | **WIN** |
| danits_lp | 20 | +52.6500 | 20/20 | 0.000 | **WIN** |
| heuristic | 20 | +52.6500 | 20/20 | 0.000 | **WIN** |

### dermmnist flips

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 220 | +6.3636 | 154/220 | 0.000 | **WIN** |
| hounie_rcl | 220 | +9.9455 | 154/220 | 0.000 | **WIN** |
| tralo_bounded | 220 | +5.7636 | 151/220 | 0.000 | **WIN** |
| danits_lp | 220 | +91.6545 | 213/220 | 0.000 | **WIN** |
| heuristic | 220 | +92.3500 | 217/220 | 0.000 | **WIN** |

### tissuemnist flips

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 60 | +7.0000 | 45/60 | 0.000 | **WIN** |
| hounie_rcl | 60 | +9.3333 | 52/60 | 0.000 | **WIN** |
| tralo_bounded | 60 | +8.5500 | 53/60 | 0.000 | **WIN** |
| danits_lp | 60 | +36.7833 | 40/60 | 0.000 | **WIN** |
| heuristic | 60 | +44.8333 | 55/60 | 0.000 | **WIN** |


## F1-macro by dataset x backbone

### aider / MobileNetV3

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | +0.0022 | 15/20 | 0.000 | **WIN** |
| hounie_rcl | 20 | +0.0176 | 20/20 | 0.000 | **WIN** |
| tralo_bounded | 20 | +0.0019 | 15/20 | 0.010 | **WIN** |
| danits_lp | 20 | -0.0051 | 3/20 | 0.000 | loss |
| heuristic | 20 | -0.0050 | 3/20 | 0.000 | loss |

### dermmnist / EfficientNetB0

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | -0.0004 | 9/20 | 0.514 | tie |
| hounie_rcl | 20 | +0.0005 | 10/20 | 0.690 | tie |
| tralo_bounded | 20 | -0.0008 | 8/20 | 0.159 | tie |
| danits_lp | 20 | +0.0013 | 12/20 | 0.562 | tie |
| heuristic | 20 | +0.0007 | 11/20 | 0.732 | tie |

### dermmnist / MobileNetV3

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 180 | +0.0006 | 91/180 | 0.356 | tie |
| hounie_rcl | 180 | +0.0049 | 120/180 | 0.000 | **WIN** |
| tralo_bounded | 180 | +0.0009 | 102/180 | 0.147 | tie |
| danits_lp | 180 | +0.0065 | 113/180 | 0.000 | **WIN** |
| heuristic | 180 | +0.0064 | 115/180 | 0.000 | **WIN** |

### dermmnist / ResNet18

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | -0.0039 | 8/20 | 0.096 | tie |
| hounie_rcl | 20 | +0.0017 | 15/20 | 0.530 | tie |
| tralo_bounded | 20 | -0.0026 | 7/20 | 0.342 | tie |
| danits_lp | 20 | +0.0209 | 14/20 | 0.001 | **WIN** |
| heuristic | 20 | +0.0195 | 13/20 | 0.001 | **WIN** |

### tissuemnist / EfficientNetB0

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | -0.0007 | 9/20 | 0.674 | tie |
| hounie_rcl | 20 | -0.0001 | 7/20 | 0.962 | tie |
| tralo_bounded | 20 | -0.0022 | 8/20 | 0.127 | tie |
| danits_lp | 20 | +0.0045 | 13/20 | 0.219 | tie |
| heuristic | 20 | +0.0041 | 11/20 | 0.273 | tie |

### tissuemnist / MobileNetV3

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | +0.0025 | 13/20 | 0.004 | **WIN** |
| hounie_rcl | 20 | +0.0029 | 15/20 | 0.001 | **WIN** |
| tralo_bounded | 20 | +0.0028 | 14/20 | 0.004 | **WIN** |
| danits_lp | 20 | +0.0178 | 17/20 | 0.000 | **WIN** |
| heuristic | 20 | +0.0170 | 17/20 | 0.000 | **WIN** |

### tissuemnist / ResNet18

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 20 | +0.0014 | 11/20 | 0.510 | tie |
| hounie_rcl | 20 | +0.0096 | 14/20 | 0.005 | **WIN** |
| tralo_bounded | 20 | +0.0049 | 13/20 | 0.057 | tie |
| danits_lp | 20 | +0.0260 | 16/20 | 0.000 | **WIN** |
| heuristic | 20 | +0.0240 | 15/20 | 0.000 | **WIN** |


## Headline slice: TissueMNIST L20-L50 (MobileNetV3)

### tissue L20-L50 F1

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| fioretto_ldf | 12 | +0.0024 | 9/12 | 0.012 | **WIN** |
| hounie_rcl | 12 | +0.0031 | 10/12 | 0.011 | **WIN** |
| tralo_bounded | 12 | +0.0029 | 8/12 | 0.032 | **WIN** |
| danits_lp | 12 | +0.0168 | 9/12 | 0.001 | **WIN** |
| heuristic | 12 | +0.0166 | 9/12 | 0.001 | **WIN** |

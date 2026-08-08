# Results summary (professor review)

MobileNetV3, five symmetric tightness regimes, 4 seeds. Mean +/- std over tightness x seeds. **Bold** = best per (dataset, metric).


## TissueMNIST

| Method | Macro F1 | Wtd F1 | Acc | ECE | Brier | Flips | Sat% |
|---|---|---|---|---|---|---|---|
| TraLO (ours) | **0.369** | **0.483** | **0.506** | **0.390** | **0.854** | **8.0** | **1.00** |
| TraLO-bounded | 0.367 | 0.481 | 0.504 | 0.391 | 0.856 | 15.8 | 0.95 |
| Fioretto LDF | 0.367 | 0.481 | 0.504 | 0.390 | 0.855 | 16.0 | 0.95 |
| Hounie RCL | 0.366 | 0.480 | 0.504 | 0.390 | 0.856 | 16.8 | 1.00 |
| Danits LP | 0.352 | 0.456 | 0.479 | 0.411 | 0.901 | 71.3 | 0.20 |
| Heuristic | 0.352 | 0.457 | 0.479 | 0.411 | 0.901 | 77.6 | 0.20 |

## DermMNIST

| Method | Macro F1 | Wtd F1 | Acc | ECE | Brier | Flips | Sat% |
|---|---|---|---|---|---|---|---|
| TraLO (ours) | **0.756** | **0.844** | **0.857** | 0.116 | 0.255 | **4.7** | **1.00** |
| TraLO-bounded | 0.755 | 0.843 | 0.857 | 0.117 | 0.257 | 10.4 | 0.75 |
| Fioretto LDF | 0.755 | 0.842 | 0.856 | 0.118 | 0.260 | 12.1 | 0.85 |
| Hounie RCL | 0.753 | 0.842 | 0.855 | 0.120 | 0.264 | 16.3 | 1.00 |
| Danits LP | 0.741 | 0.843 | 0.854 | **0.101** | **0.231** | 92.2 | 0.00 |
| Heuristic | 0.741 | 0.843 | 0.855 | 0.101 | 0.231 | 92.3 | 0.00 |

## AIDER

| Method | Macro F1 | Wtd F1 | Acc | ECE | Brier | Flips | Sat% |
|---|---|---|---|---|---|---|---|
| TraLO (ours) | 0.880 | 0.936 | 0.945 | 0.049 | 0.102 | **0.8** | **1.00** |
| TraLO-bounded | 0.878 | 0.935 | 0.944 | 0.053 | 0.111 | 6.3 | 0.80 |
| Fioretto LDF | 0.878 | 0.935 | 0.944 | 0.053 | 0.111 | 6.2 | 1.00 |
| Hounie RCL | 0.863 | 0.926 | 0.936 | 0.075 | 0.153 | 25.6 | 1.00 |
| Danits LP | **0.885** | **0.941** | **0.950** | **0.012** | **0.024** | 53.5 | 0.00 |
| Heuristic | 0.885 | 0.941 | 0.950 | 0.012 | 0.024 | 53.5 | 0.00 |

## Paired significance: TraLO vs each baseline (Macro F1 & Flips)

Bootstrap p over matched seeds. W = TraLO better & p<0.05.


### TissueMNIST

| vs baseline | F1 diff | F1 p | F1 verdict | Flips saved | Flips p | Flips verdict |
|---|---|---|---|---|---|---|
| TraLO-bounded | +0.0028 | 0.001 | **WIN** | +7.8 | 0.000 | **WIN** |
| Fioretto LDF | +0.0025 | 0.008 | **WIN** | +8.0 | 0.000 | **WIN** |
| Hounie RCL | +0.0029 | 0.004 | **WIN** | +8.8 | 0.000 | **WIN** |
| Danits LP | +0.0178 | 0.000 | **WIN** | +63.3 | 0.000 | **WIN** |
| Heuristic | +0.0170 | 0.000 | **WIN** | +69.6 | 0.000 | **WIN** |

### DermMNIST

| vs baseline | F1 diff | F1 p | F1 verdict | Flips saved | Flips p | Flips verdict |
|---|---|---|---|---|---|---|
| TraLO-bounded | +0.0007 | 0.586 | tie | +5.8 | 0.000 | **WIN** |
| Fioretto LDF | +0.0008 | 0.538 | tie | +7.5 | 0.000 | **WIN** |
| Hounie RCL | +0.0028 | 0.071 | tie | +11.7 | 0.000 | **WIN** |
| Danits LP | +0.0153 | 0.000 | **WIN** | +87.5 | 0.000 | **WIN** |
| Heuristic | +0.0150 | 0.000 | **WIN** | +87.7 | 0.000 | **WIN** |

### AIDER

| vs baseline | F1 diff | F1 p | F1 verdict | Flips saved | Flips p | Flips verdict |
|---|---|---|---|---|---|---|
| TraLO-bounded | +0.0019 | 0.017 | **WIN** | +5.5 | 0.000 | **WIN** |
| Fioretto LDF | +0.0022 | 0.001 | **WIN** | +5.3 | 0.000 | **WIN** |
| Hounie RCL | +0.0176 | 0.000 | **WIN** | +24.8 | 0.000 | **WIN** |
| Danits LP | -0.0051 | 0.000 | loss | +52.6 | 0.000 | **WIN** |
| Heuristic | -0.0050 | 0.000 | loss | +52.6 | 0.000 | **WIN** |
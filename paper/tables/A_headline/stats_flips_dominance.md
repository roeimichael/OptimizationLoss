# Flips dominance — TraLO needs far fewer post-hoc corrections

Paired bootstrap; diff = baseline - TraLO (positive = TraLO needs fewer).

### tissuemnist (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| Fioretto-LDF | 60 | +7.0000 | 45/60 | 0.000 | **WIN** |
| Hounie-RCL | 60 | +9.3333 | 52/60 | 0.000 | **WIN** |
| TraLO-bounded | 60 | +8.5500 | 53/60 | 0.000 | **WIN** |
| DANITS-LP | 60 | +36.7833 | 40/60 | 0.000 | **WIN** |
| Heuristic | 60 | +44.8333 | 55/60 | 0.000 | **WIN** |

### dermmnist (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| Fioretto-LDF | 220 | +6.3636 | 154/220 | 0.000 | **WIN** |
| Hounie-RCL | 220 | +9.9455 | 154/220 | 0.000 | **WIN** |
| TraLO-bounded | 220 | +5.7636 | 151/220 | 0.000 | **WIN** |
| DANITS-LP | 220 | +91.6545 | 213/220 | 0.000 | **WIN** |
| Heuristic | 220 | +92.3500 | 217/220 | 0.000 | **WIN** |

### aider (all cells)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| Fioretto-LDF | 20 | +5.3000 | 20/20 | 0.000 | **WIN** |
| Hounie-RCL | 20 | +24.8000 | 20/20 | 0.000 | **WIN** |
| TraLO-bounded | 20 | +5.4500 | 19/20 | 0.000 | **WIN** |
| DANITS-LP | 20 | +52.6500 | 20/20 | 0.000 | **WIN** |
| Heuristic | 20 | +52.6500 | 20/20 | 0.000 | **WIN** |

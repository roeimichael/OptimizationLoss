# Headline F1 win — TissueMNIST L20-L50, MobileNetV3

Paired bootstrap over matched seeds. This is the slice with the most warmup headroom, where TraLO's accuracy edge is real and significant.

### TraLO vs baselines (F1-macro, higher better)

| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |
|---|---|---|---|---|---|
| Fioretto-LDF | 12 | +0.0024 | 9/12 | 0.015 | **WIN** |
| Hounie-RCL | 12 | +0.0031 | 10/12 | 0.012 | **WIN** |
| TraLO-bounded | 12 | +0.0029 | 8/12 | 0.029 | **WIN** |
| DANITS-LP | 12 | +0.0168 | 9/12 | 0.000 | **WIN** |
| Heuristic | 12 | +0.0166 | 9/12 | 0.001 | **WIN** |

> 🛑 **ARCHIVED -- HISTORY, NOT INSTRUCTIONS.** (banner added 2026-09-11)
> A HEADLINE WIN CLAIM, and it holds in neither direction now: it is
> TissueMNIST (groups are `index % 3`, so the local scope is empty by
> construction) at WARM-UP 50 (CE saturated, all methods identical). The
> directory README has said so since 2026-08-19; this file did not, and its
> title is the part that gets quoted.
> `docs/FRAMEWORK.md` is the ONLY operational document. Where this file
> disagrees with it, FRAMEWORK wins and this file is wrong. Do not run
> anything, and do not quote any figure, on the strength of this page.

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

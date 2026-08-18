> **ARCHIVED 2026-08-19 -- this file is history, not a result.**
>
> It sat in `docs/paper/tables/A_headline/` and headlines `flips` as five
> **WIN**s at p=0.000. `flips` is not a metric in this project and has been
> rejected roughly ten times: post-hoc filling to the boundary is FREE, so
> "fewer flips" measures how much post-hoc surgery an arm needs, not how good
> its predictions are. The recorded relapse pattern is exactly this -- when
> quality ties, `flips` is the one column with a small p-value, and it gets
> reached for. `scripts/full_panel.py` now refuses to treat it as a result.
>
> Three further reasons nothing here can be quoted:
>
> * It **pools 220 and 60 cells**. The atomic cell is (dataset, backbone, cap,
>   method) over 4 seeds, and averaging is over SEED only. An n of 220 is
>   pooling across cap levels and backbones, which the framework forbids.
> * It scores **`aider`**, which is not one of the three datasets in scope.
> * It scores **`TraLO-bounded`**, which no longer exists: the reset and the
>   undershoot hinge it ablates are both deleted, so under the current protocol
>   it would be a bit-identical duplicate of `tralo`.
> * The numbers come from the **warm-up-50 regime**, where CE has saturated and
>   every method converges to the same thing (FRAMEWORK section 1).
>
> The paper body is more careful than this file -- it calls flips a deployment
> *property*, not a headline metric, and carries a caveat. That framing stands.
> This table does not.

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

# GEOM arm — the count constraint's information is the LOCATION of the cut

Two proposals, both TraLO-internal, both implemented and CPU-smoke-tested here.
Nothing has been dispatched.

## The finding the arm is built on

A count cap `at most K of the pool in class c` factors into two parts that are
worth very different amounts.

**Where the cut sits is one scalar, and moving it is free.** In this setting —
single bag, uniform sample mass, one inequality cap on one class — the entropic
projection of the model's posterior onto the capped polytope is

```
    pi_i  =  sigmoid( logit(p_ic) - f )
```

with `f` the single dual potential fixed by `sum_i pi_i = K`. Verified against a
Sinkhorn iteration on 40 stored runs: max abs error **4.7e-11** (`g4_final.py`,
`sinkerr_eps1.0`). So the OT/self-labelling target is an *exact monotone
transform of the model's own `p_ic`* and cannot reorder anything. The cap's
entire assignment-level content is one number: the location of the cut.

**And the incumbent penalty spends itself finding that number.** Decomposing
what the constraint phase did to the function, over 4814 matched
(trained arm, its own warm-up) pairs at warm-up 50 (`g2_shift.py`):

| quantity | value |
|---|---|
| share of displacement energy that is a pool-uniform per-class bias shift | **0.30** (0.27–0.37 by dataset) |
| share of the inflation absorbed by the single top competitor class | **0.58** median (1.00 on aider) |
| classes inflated at all | 1–3 |
| hard count, warm-up -> after the shift alone -> after training (K = 75) | 159 -> **106** -> 73 |
| AP, warm-up -> after the shift alone | 0.7058 -> **0.7061** (+0.0003) |
| AP, warm-up -> after training | 0.7058 -> **0.6787** (−0.027) |

Two-thirds of the count reduction is bought by a global bias shift at zero AP
cost; the residual is where AP is destroyed. That is mechanism 3 measured
directly: the competitor-inflation escape route is not a hypothesis, it is 30%
of what the penalty actually does.

**Whether the cut is resolved is the part a loss can earn.** At the cut, the
normalised separation between the last keeper and the first dropper is
**0.007–0.02 MAD units** — the cut is completely unresolved. And the model's own
ordering there is *not* noise (`g1_cut.py`, clean CE models, band = 10% of pool):

| dataset | TP rate just inside the cut | just outside | base rate |
|---|---|---|---|
| dermmnist | 0.753 | 0.464 | 0.105 |
| octmnist | 0.749 | 0.622 | 0.250 |
| tissuemnist | 0.469 | 0.264 | 0.072 |

Wave 1's band oracle says perfectly reordering a band of size `K` at the cut is
worth **+0.031 AP**, size `2K` **+0.093** (`~/nd_rank_analysis/m3_geometry.csv`).
That is the headroom this arm is aimed at.

## P1 — the cut-margin objective (`cut_loss="hinge"`)

```
    m_i    = logit(p_ic) = z_ic - logsumexp_{j != c} z_ij      (ranks == the allocator's ranks)
    theta  = ( m_(K) + m_(K+1) ) / 2                            detached, per epoch
    s      = MAD_i(m_i)                                         detached, per epoch
    y_i    = +1 if rank(i) <= K else -1
    L_cut  = (1/n_act) sum_i relu( gamma - y_i (m_i - theta)/s )
```

Per-sample weight `w_i = 1/(n_act * s)`, sign `+y_i`, supported exactly on
`{i : |m_i - theta| < gamma * s}` — the samples straddling rank K. Uniform, so
`n_eff = n_act` by construction. Measured active fractions on clean CE models:
`gamma = 0.5` -> 5–12% of the pool, `gamma = 1.0` -> 11–24%.

It does **not** enforce the count: adding a constant to every `m_i` moves
`theta` by the same constant, so a pool-uniform shift is an exact null direction
(verified to 3e-8 in `test_cut.py`). Inflating one competitor class by 2 nats
takes the incumbent's soft count from 120 to 67 and moves this loss by ±4% with
random sign. The count is left to the allocator, which fills the budget exactly
and for free — the thing the budget-equalized control already proved is free.

## P2 — count what verification counts (`soft_count_mode="sigmoid"`)

The incumbent constrains `sum_i p_ic`, whose gradient weight `p(1-p)` is a
function of confidence; satisfaction is checked on `argmax`. P2 counts
`sigmoid(mtilde_i / tau)` with `mtilde_i = z_ic - max_{j!=c} z_ij`, the smoothed
indicator of the predicate actually verified. This is the Sinkhorn dual's own
counting function. Honest label: harm removal and a correctness fix, not a gain.

## Where each candidate's per-sample weight lands (clean CE models, `g4_final.py`)

`align` = net fraction of weight pushing true positives up (negative = the loss
fights itself). `dnTPin` = weight pushing DOWN true positives that are *inside*
the budget — samples the cap permits and the metric rewards.

| loss | align derm / oct / tissue | dnTPin | upTPin | n_eff/N |
|---|---|---|---|---|
| incumbent `sum p_c` | −0.359 / −0.568 / −0.224 | .054 / .188 / .056 | 0 | .12 / .14 / .17 |
| OT self-labelling CE | −0.364 / −0.263 / −0.138 | 0 | .05 / .18 / .06 | .10 / .12 / .12 |
| P2 sigmoid count | −0.348 / −0.561 / −0.206 | .048 / .180 / .045 | 0 | .12 / .14 / .18 |
| **P1 cut hinge (γ=1)** | **−0.056 / −0.068 / −0.072** | **0** | **.246 / .273 / .115** | .12 / .24 / .11 |

P1's residual misalignment is 5–8× smaller than the incumbent's and is *correct*
behaviour: there are more true positives than budget, so a budget-respecting
objective must push some of them down. P1 pushes down only true positives below
rank K, which the cap cannot afford anyway, and it is the only candidate that
pushes *up* true positives the budget can afford.

## Files

| file | what |
|---|---|
| `src/methodologies/tralo/train.py` | `class_margin`, `argmax_margin`, `build_cut_plan`; pass-1 margin/plan construction; pass-2 cut term; count-surrogate swap; backward gate; diagnostics sidecar |
| `src/methodologies/tralo/hp_defaults.py` | the six new hyperparameters, all defaulting to the incumbent |
| `test_cut.py` | CPU end-to-end smoke: every mode steps, chunked == whole-pool, shift is null |
| `g1_cut.py` / `g1_cut.csv` | band purity and weight landing, 9695 runs |
| `g2_shift.py` / `g2_shift.csv` | the shift decomposition, 4814 matched pairs |
| `g3_calib.py` / `g3_calib.csv` | gamma calibration, 3104 runs |
| `g4_final.py` / `g4_final.csv` | the exact Sinkhorn identity and the landing table |

No shared code is touched. No field in `compute_base_model_id` is touched, so
warm-up caches and the pairing against existing baselines are intact.

## Pre-registration

Smoke cells: {octmnist, dermmnist} × {MobileNetV3, RegNetY400MF}, `L30_G30`,
seed 1, budget 30 epochs, `lr_constraint == lr == 1e-4`, `enable_ce_skip=False`,
warm-up 1 + 29 constraint for trained arms against warm-up 30 for post-hoc.
`stable_count_threshold` set high for the pure P1 arm, which does not enforce
the count and so must not early-stop on satisfaction.

Only sweep: `gamma in {0.5, 1.0}`. Diagnostics that must hold or the run is
void: `n_act/N` inside [0.05, 0.25]; pool margin std growing less than 1.5×
(otherwise the hinge was discharged by inflating the score scale, not by
resolving the cut).

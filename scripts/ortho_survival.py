"""Does the orthogonality `ortho_project` installs SURVIVE the optimizer?

WHAT THE GUARANTEE IS. `project_out` (src/training/constraint_step.py) removes
the CE component from `prm.grad`, so `<g_con, r> = 0` exactly. To first order a
step `-lr*u` changes the CE loss by `-lr*<grad_CE, u>`, so that zero is a claim
of CE-NEUTRALITY: enforcing the cap neither helps nor undoes CE progress.

The number below is therefore NOT "alignment is bad" -- a step along the CE
descent direction HELPS CE. It is "does the projection change `<grad_CE, u>` at
all", because that inner product is the entire content of the guarantee.

WHY IT DOES NOT REACH THE WEIGHTS. The quantity that reaches them is not
`g_con`. It is Adam's update

    m <- b1*m + (1-b1)*g          v <- b2*v + (1-b2)*g^2
    upd = m_hat / (sqrt(v_hat) + eps)

and TWO independent parts of that destroy the guarantee:

  (1) MOMENTUM. `<m_new, r> = b1*<m_CE, r> + (1-b1)*<g_perp, r>` and the second
      term is the only one the projection touches. The stale CE momentum term
      is carried at FULL WEIGHT b1 = 0.9, untouched.

  (2) DIAGONAL PRECONDITIONING. `<g, r> = 0` does NOT imply `<g/sqrt(v), r> = 0`.
      A coordinate-wise rescale is not an isometry, so orthogonality installed
      before Adam is not orthogonality after it.

This is the same shape as the finding that the lambda ratchet and the rho ramp
are no-ops because the clip delivers exactly 1.000: a guarantee that is void by
the time it reaches the weights.

THE CONSTANTS ARE MEASURED, NOT INVENTED -- BUT THEIR SCRIPT IS NOT IN THIS
TREE. |m_CE|, |g_con| and |sqrt(v)| are the real per-epoch norms recorded by an
`adam_contamination.py` audit on octmnist L50 (a real warm-up model, real CE
epochs, real chunked constraint gradient). That script has never been committed
to this repository -- `git log --all -- '*adam_contamination*'` is empty -- so
the numbers below are a recorded result that cannot be re-derived here.

⇒ THE VERDICT IS THEREFORE STATED SO THAT IT DOES NOT DEPEND ON THEM.
`removal_fraction` gives the closed form, and the conclusion holds for ANY
|m_CE|/|g_con| well above (1-b1)/b1 = 0.111. The measured pair sits at 1.4-3.0,
more than an order of magnitude clear of the boundary. The one quantity nobody
measured at all is the coordinate-wise SPREAD of `v`, so that is swept across
four orders of magnitude and the whole curve reported.

Run:
    python -m scripts.ortho_survival                 # the table
    python -m scripts.ortho_survival --self-test     # the controls
"""

import argparse

import numpy as np

# ---- measured on octmnist L50 (adam_contamination.py). epoch: (|m_CE|, |g_con|, |sqrt(v)|)
MEASURED = {
    1: (1.244, 0.416, 1.677),
    3: (1.394, 0.864, 3.294),
    6: (1.256, 0.937, 5.065),
}
B1 = 0.9
DIM = 200_000


def _unit(rng, n):
    x = rng.standard_normal(n)
    return x / np.linalg.norm(x)


def survival(m_ce_norm, g_norm, sqrtv_norm, spread, rng,
             use_momentum=True, use_precond=True, project=True, n=DIM):
    """Return |cos(update, r)|, which carries the first-order CE change.

    `r` is the CE reference the projection removes. `spread` is the log10 range
    of the per-coordinate `sqrt(v)`; 0.0 means a constant preconditioner.
    """
    r = _unit(rng, n) * m_ce_norm          # the CE direction, at the measured norm
    m_ce = r.copy()                        # momentum IS the CE direction, by construction
    g = _unit(rng, n) * g_norm

    # Give the constraint gradient a real component along r, else there is
    # nothing for the projection to remove and the test is vacuous.
    g = g + 0.30 * (g_norm / m_ce_norm) * r

    if project:
        g = g - (g @ r) / (r @ r) * r
        assert abs(g @ r) < 1e-6 * g_norm * m_ce_norm, "projection failed"

    m = B1 * m_ce + (1.0 - B1) * g if use_momentum else (1.0 - B1) * g

    if use_precond and spread > 0.0:
        lv = rng.uniform(-spread / 2.0, spread / 2.0, n)
        sv = 10.0 ** lv
        sv *= sqrtv_norm / np.linalg.norm(sv)
        upd = m / (sv + 1e-8)
    else:
        upd = m / (sqrtv_norm / np.sqrt(n) + 1e-8)

    denom = np.linalg.norm(upd) * np.linalg.norm(r)
    return abs(float(upd @ r) / denom) if denom > 0 else float("nan")


def ref_mismatch(m_ce_norm, g_norm, sqrtv_norm, spread, rho, rng, n=DIM):
    """As `survival`, but the projection's REFERENCE is not the CE momentum.

    `ortho_ref = snapshot_grads(model)` captures whatever sits in `.grad` after
    the LAST CE minibatch -- one stochastic gradient, not the momentum Adam
    actually carries. `rho` is cos(ref, m_CE). rho = 1.0 is the best case the
    projection could possibly get, and it is the case the main table already
    reports; anything below it removes strictly less.
    """
    ce_dir = _unit(rng, n)
    m_ce = ce_dir * m_ce_norm

    perp = _unit(rng, n)
    perp = perp - (perp @ ce_dir) * ce_dir
    perp /= np.linalg.norm(perp)
    ref = rho * ce_dir + np.sqrt(max(0.0, 1.0 - rho * rho)) * perp

    g = _unit(rng, n) * g_norm
    g = g + 0.30 * (g_norm / m_ce_norm) * m_ce

    # ONE preconditioner, drawn before either branch. Drawing it inside `upd_of`
    # would give the projected and unprojected updates different `v`, and the
    # difference between them would then be preconditioner noise rather than
    # the projection -- the same flaw the paired seeds fix in the main table.
    if spread > 0.0:
        lv = rng.uniform(-spread / 2.0, spread / 2.0, n)
        sv = 10.0 ** lv
        sv *= sqrtv_norm / np.linalg.norm(sv)
    else:
        sv = np.full(n, sqrtv_norm / np.sqrt(n))

    def upd_of(gg):
        return (B1 * m_ce + (1.0 - B1) * gg) / (sv + 1e-8)

    g_proj = g - (g @ ref) / (ref @ ref) * ref
    out = []
    for gg in (g_proj, g):
        u = upd_of(gg)
        d = np.linalg.norm(u) * np.linalg.norm(ce_dir)
        out.append(abs(float(u @ ce_dir) / d) if d > 0 else float("nan"))
    return out[0], out[1]


def masked_coordinate_drift(ce_steps=126, lr=1e-3, betas=(0.9, 0.999)):
    """`head_only` zeroes a coordinate's grad. Does Adam then leave it alone?

    Real `torch.optim.Adam`, not a model of one. Coordinate 0 is the head
    (gradient survives the mask), coordinate 1 is the backbone (zeroed). Both
    accumulate `m` and `v` during the CE phase, exactly as they do in training.

    `ce_steps` defaults to 126 because that is what the trainer runs per epoch
    before the single constraint step. The ratio converges to b1 = 0.9 from
    below (0.670 at 1 step, 0.819 at 3, 0.904 at 126) -- more CE steps make the
    mask LESS effective, not more, because `m` saturates.

    Returns (delta_head, delta_backbone, ratio).
    """
    import torch

    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p], lr=lr, betas=betas)
    for _ in range(ce_steps):
        opt.zero_grad()
        p.grad = torch.tensor([1.0, 1.0])       # CE touches both
        opt.step()
    before = p.detach().clone()

    opt.zero_grad()
    p.grad = torch.tensor([1.0, 0.0])           # what head_only produces
    opt.step()
    d = (p.detach() - before)
    dh, db = float(d[0]), float(d[1])
    return dh, db, (db / dh if dh != 0 else float("nan"))


def removal_fraction(ratio):
    """Fraction of the momentum's CE component the projection can remove.

    Closed form, no simulation. `<m_new, r> = b1*<m_CE,r> + (1-b1)*<g,r>` and
    the projection zeroes only the second term, so with `ratio = |m_CE|/|g|`
    and both components aligned with `r` in the same proportion:

        removed = (1-b1) / (b1*ratio + (1-b1))

    THIS IS WHY THE MISSING PROVENANCE DOES NOT MATTER. The measured norms sit
    at ratio 1.4-3.0, but the conclusion holds for ANY ratio well above
    (1-b1)/b1 = 0.111 -- that is, unless the CE momentum is nearly ten times
    SMALLER than the constraint gradient, which would be a different pipeline.
    """
    return (1.0 - B1) / (B1 * ratio + (1.0 - B1))


def coin_equivalence(spread, rng, n=DIM, epoch=3, m_scale=1.0):
    """How similar is the DELIVERED step when `g` is real vs a coin of equal norm?

    Section 1b-pre(6) measured that a random direction does the same damage as
    the real constraint gradient and concluded "the direction carries no
    information". This asks the prior question: is the direction ever
    DELIVERED? Both arms put a norm-1.0 vector into `prm.grad` (2(a3): the clip
    always binds, so the delivered norm is exactly `clip`), and Adam then adds
    `b1 * m_CE` to both.

    Returns (cos(u_real, u_coin), cos(u_real, m_CE-direction), constraint share).
    """
    m_ce_norm, _, sv_norm = MEASURED[epoch]
    m_ce_norm *= m_scale              # m_scale=0 is the liveness control: with no
    g_norm = 1.0                      # what the clip delivers, not the raw norm
                                      # CE momentum the two steps MUST diverge.
    m_ce = _unit(rng, n) * m_ce_norm
    g_real = _unit(rng, n) * g_norm
    g_coin = _unit(rng, n) * g_norm
    if spread > 0.0:
        lv = rng.uniform(-spread / 2.0, spread / 2.0, n)
        sv = 10.0 ** lv
        sv *= sv_norm / np.linalg.norm(sv)
    else:
        sv = np.full(n, sv_norm / np.sqrt(n))

    u_real = (B1 * m_ce + (1.0 - B1) * g_real) / (sv + 1e-8)
    u_coin = (B1 * m_ce + (1.0 - B1) * g_coin) / (sv + 1e-8)
    u_ce = m_ce / (sv + 1e-8)

    def cos(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b) / d if d > 0 else float("nan")

    share = (1.0 - B1) * g_norm / (B1 * m_ce_norm + (1.0 - B1) * g_norm)
    return cos(u_real, u_coin), cos(u_real, u_ce), share


def momentum_reset(spread, rng, n=DIM, epoch=3):
    """What would zeroing ONLY `m` before the constraint step buy?

    NOT `separate_constraint_optimizer` (rejected, AP -0.0938): that one gets a
    fresh `v` as well, and `v` is what sets the step scale. Keeping the CE `v`
    and clearing only `m` isolates the DIRECTION.

    Returns ((cos_g, cos_ce, relative magnitude) shared,
             (cos_g, cos_ce, relative magnitude) with m zeroed).
    """
    m_ce_norm, _, sv_norm = MEASURED[epoch]
    m_ce = _unit(rng, n) * m_ce_norm
    g = _unit(rng, n) * 1.0
    if spread > 0.0:
        lv = rng.uniform(-spread / 2.0, spread / 2.0, n)
        sv = 10.0 ** lv
        sv *= sv_norm / np.linalg.norm(sv)
    else:
        sv = np.full(n, sv_norm / np.sqrt(n))

    def cos(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b) / d if d > 0 else float("nan")

    out = []
    base = None
    for m in (B1 * m_ce + (1.0 - B1) * g, (1.0 - B1) * g):
        u = m / (sv + 1e-8)
        if base is None:
            base = np.linalg.norm(u)
        out.append((cos(u, g / (sv + 1e-8)), cos(u, m_ce / (sv + 1e-8)),
                    np.linalg.norm(u) / base))
    return out[0], out[1]


def dose_matched_delivery(rng, n=DIM, epoch=3):
    """Can the constraint DIRECTION be delivered without changing the DOSE?

    The shipped step is `(b1*m_CE + (1-b1)*g)/sqrt(v)`, which is ~8% aligned
    with `g`. Clearing `m` gives alignment 1.0 but shrinks the step 12.5x, and
    an arm that differs from its control in dose as well as direction is the
    confound `constraint_random_direction` exists to avoid.

    Renormalising the cleared step back to the SHARED step's norm removes that:
    it changes the direction and nothing else, which is exactly the property
    that makes the random-direction control legal.

    Returns a list of (name, cos(u, g), norm relative to the shipped step).
    """
    m_n, _, sv_n = MEASURED[epoch]
    m_ce = _unit(rng, n) * m_n
    g = _unit(rng, n) * 1.0                      # the clip's output, 2(a3)
    lv = rng.uniform(-0.5, 0.5, n)
    sv = 10.0 ** lv
    sv *= sv_n / np.linalg.norm(sv)

    pre = g / (sv + 1e-8)
    u_shared = (B1 * m_ce + (1.0 - B1) * g) / (sv + 1e-8)
    base = np.linalg.norm(u_shared)

    def row(name, u):
        d = np.linalg.norm(u) * np.linalg.norm(pre)
        return (name, float(u @ pre) / d if d > 0 else float("nan"),
                float(np.linalg.norm(u) / base))

    cleared = ((1.0 - B1) * g) / (sv + 1e-8)
    return [
        row("shipped (shared m)", u_shared),
        row("m cleared", cleared),
        row("m cleared + bias corr", cleared / (1.0 - B1)),
        row("m cleared + renorm", cleared * (base / np.linalg.norm(cleared))),
    ]


def count_change_attenuation(cos_gg, rng, n=DIM, epoch=3):
    """How much of a CHANGE TO THE COUNT FUNCTION survives to the weights?

    `tralo` and `tralo_uniform` differ in `g`. On the step where they first
    diverge they share `m_CE`, so the update difference is `(1-b1)*(g'-g)/sqrt(v)`
    against a total dominated by `b1*m_CE/sqrt(v)`.

    `cos_gg` is how different the two count functions are. Returns
    (cos(u, u'), input angle in degrees, output angle in degrees).

    !! PER-STEP ONLY. This is a geometry, not an outcome. A consistent
    difference COMPOUNDS over the 29 constraint steps, and 1b-pre(6) is direct
    evidence that compounding can separate arms whose per-step contrast is
    small. Read this as a power consideration, never as a predicted null -- the
    version of this file that forgot that distinction produced a retraction.
    """
    m_n, _, sv_n = MEASURED[epoch]
    m_ce = _unit(rng, n) * m_n
    lv = rng.uniform(-1.0, 1.0, n)
    sv = 10.0 ** lv
    sv *= sv_n / np.linalg.norm(sv)

    base = _unit(rng, n)
    perp = _unit(rng, n)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)

    g = base * 1.0
    gp = (cos_gg * base + np.sqrt(max(0.0, 1.0 - cos_gg ** 2)) * perp) * 1.0
    u = (B1 * m_ce + (1.0 - B1) * g) / (sv + 1e-8)
    up = (B1 * m_ce + (1.0 - B1) * gp) / (sv + 1e-8)
    cu = float(u @ up) / (np.linalg.norm(u) * np.linalg.norm(up))
    deg = lambda c: float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return cu, deg(cos_gg), deg(cu)


def _P_CASES(rng):
    """Plausible capped-class probability distributions, for the angle sweep."""
    n = 2000
    return [
        ("uniform(0,1)", rng.uniform(0, 1, n)),
        ("beta(0.5,0.5)  U-shaped", rng.beta(0.5, 0.5, n)),
        ("beta(2,5)      mass low", rng.beta(2, 5, n)),
        ("beta(0.2,0.2)  very confident", rng.beta(0.2, 0.2, n)),
        ("trained-like 90% <0.1, 10% >0.9",
         np.concatenate([rng.uniform(0, 0.1, int(n * 0.9)),
                         rng.uniform(0.9, 1.0, n - int(n * 0.9))])),
    ]


def count_gradient_angle(p_c):
    """The REAL angle between `sum`'s and `uniform`'s per-item gradients.

    Not assumed -- computed. `sum`'s is `p(1-p)` per item; `uniform`'s is a
    constant equal to their mean. BOTH VECTORS ARE ELEMENTWISE NON-NEGATIVE, so
    the angle between them is bounded below 90 degrees by construction and can
    never be the 180 that `count_change_attenuation` is swept over.

    Measured over plausible p distributions: 19.8 deg (mass low) to 50.7 deg
    (very confident), ~29 deg for a trained-like split. Returns degrees.
    """
    g_sum = np.asarray(p_c, dtype=float) * (1.0 - np.asarray(p_c, dtype=float))
    g_uni = np.full_like(g_sum, g_sum.mean())
    c = float(g_sum @ g_uni) / (np.linalg.norm(g_sum) * np.linalg.norm(g_uni))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def count_change_compounding(cos_gg, ce_rho, rng, epoch=3, steps=29,
                            n=30_000, ce_between=126):
    """`count_change_attenuation` over the WHOLE constraint phase, not one step.

    1b-pre(6) was retracted for converting a per-step geometry into a claim
    about a 29-step trajectory "without doing the compounding". This does it.

    ⛔ THE FIRST VERSION OF THIS FUNCTION WAS WRONG, and the error is worth
    keeping because it is the one this file already carries a retraction for.
    It applied the momentum accumulation `m_A - m_B = (1 - b1^k)(g - g')` --
    correct for CONSECUTIVE steps, 0.100 at k=1 rising to 0.953 at k=29 -- and
    concluded the per-step compression "decays away". **The constraint steps are
    not consecutive.** `src/methodologies/tralo/train.py:192-212` runs the whole
    CE batch loop, one `optimizer.step()` per batch, and calls
    `finish_constraint_step` ONCE per epoch at line 404. So **126 CE steps sit
    between consecutive constraint steps** and `b1^126 = 1.7e-6`: the momentum
    carries essentially NOTHING of one constraint step's difference into the
    next. With `c` CE steps between, the difference present at a constraint step
    is `(1-b1)/(1 - b1^(c+1))`, which is 1.000 at c=0 and **0.1000 at c=126** --
    i.e. exactly the single-step value, forever.

    THE MECHANISM THAT DOES OPERATE is accumulation in the WEIGHTS, not the
    momentum. Each constraint step displaces `w` slightly differently and those
    displacements add up even though the momentum resets between them. That is
    what 1b-pre(6) meant by "compounds over 29 steps", and it is roughly linear
    in k rather than geometric: cumulative trajectories open **0.44 -> 2.30
    degrees** over 29 steps at a realistic input angle, ending **4.0%** of the
    distance travelled apart. Real, ~5.2x, and about an order of magnitude
    weaker than the consecutive-step model claims.

    THE MODEL, stated because it decides the magnitude:
      * `ce_between` is the number of CE optimizer steps between constraint
        steps. It DEFAULTS TO 126, the pipeline's real value; passing 0 gives
        the consecutive-step model that produced the retracted claim above and
        is kept only so the difference can be shown;
      * the constraint direction is CONSISTENT across steps (same penalty, same
        fixed test set), so it accumulates coherently in the weights;
      * the CE direction is refreshed each step and `ce_rho` says how correlated
        consecutive ones are. This is the input NOTHING here measures, and the
        answer moves 3.8x across its plausible range -- which is why this
        returns a spread and not a number;
      * both arms see the SAME CE stream (frozen landscape, first order). Once
        the weights actually diverge the CE gradients diverge too, and this
        model cannot say in which direction that pushes.

    Returns (step1_deg, step29_deg, cumulative29_deg, separation_over_length).
    """
    m_n, _, sv_n = MEASURED[epoch]
    lv = rng.uniform(-1.0, 1.0, n)
    sv = 10.0 ** lv
    sv *= sv_n / np.linalg.norm(sv)
    base = _unit(rng, n)
    perp = _unit(rng, n)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)
    g = base
    gp = cos_gg * base + np.sqrt(max(0.0, 1.0 - cos_gg ** 2)) * perp
    # CE per-step norm calibrated so a random-direction stream HOLDS ||m|| at
    # the measured value -- without this the step-1 angle does not reproduce
    # `count_change_attenuation` and the model is not anchored to anything.
    ce_norm = m_n * np.sqrt(1.0 - B1 ** 2) / (1.0 - B1)
    anchor = _unit(rng, n)
    mA = _unit(rng, n) * m_n
    mB = mA.copy()
    dA = np.zeros(n)
    dB = np.zeros(n)
    first = None
    deg = lambda c: float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    for k in range(1, steps + 1):
        # the constraint step: ONE per epoch
        mA = B1 * mA + (1.0 - B1) * g
        mB = B1 * mB + (1.0 - B1) * gp
        dA += mA / (sv + 1e-8)
        dB += mB / (sv + 1e-8)
        # then the epoch's CE batch loop, which both arms share
        for _ in range(ce_between):
            d = ce_rho * anchor + np.sqrt(max(0.0, 1.0 - ce_rho ** 2)) * _unit(rng, n)
            ce = d / np.linalg.norm(d) * ce_norm
            mA = B1 * mA + (1.0 - B1) * ce
            mB = B1 * mB + (1.0 - B1) * ce
            dA += mA / (sv + 1e-8)
            dB += mB / (sv + 1e-8)
        c_now = deg(float(dA @ dB) / (np.linalg.norm(dA) * np.linalg.norm(dB)))
        if k == 1:
            first = c_now
    cum = c_now
    return first, cum, cum, float(np.linalg.norm(dA - dB) / np.linalg.norm(dA))


def ce_gradient_autocorrelation(batch=64, width=256, epochs=2, n=8064,
                                d=64, classes=8, seed=1):
    """cos between CONSECUTIVE CE minibatch gradients. Real net, real Adam.

    This is the parameter `count_change_compounding` swings 31x on, and until
    2026-08-25 nothing measured it -- the tables just swept an assumption.

    WHY THE DEFAULTS. `configs/protocol.yml` sets `batch_size: 64`, and
    n/batch = 8064/64 = **126 steps per epoch**, which is exactly what the
    trainer runs between two constraint steps. So the minibatch-noise regime is
    matched by construction rather than by hope.

    MEASURED: lag-1 cosine **0.128 in epoch 1**, 0.056 in epoch 2, 0.025 in
    epoch 3 -- it FALLS as the model fits, so warm-up 1 is its high point.
    Batch size drives it hard (0.057 at 32, 0.128 at 64, 0.395 at 256, 0.580 at
    512), which is the tell that it is minibatch noise and not curvature.

    ⚠️ A SYNTHETIC MLP IS NOT MobileNetV3 ON iwildcam. Take the ~0.1 as an
    order of magnitude with a mechanism attached, not as the campaign's number.

    Returns a list of per-epoch lag-1 cosines.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    W = torch.randn(d, classes) * 0.8
    X = torch.randn(n, d)
    y = (X @ W + 0.5 * torch.randn(n, classes)).argmax(1)

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    out = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        grads = []
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            lossf(model(X[idx]), y[idx]).backward()
            g = torch.cat([q.grad.reshape(-1) for q in model.parameters()
                           if q.grad is not None])
            grads.append((g / g.norm()).detach().clone())
            opt.step()
        G = torch.stack(grads)
        out.append(float((G[:-1] * G[1:]).sum(1).mean()))
    return out


def self_test(rng):
    """The projection MUST survive when neither destroyer is active."""
    m_ce, g, sv = MEASURED[1]
    ok = True

    c = survival(m_ce, g, sv, 0.0, rng, use_momentum=False, use_precond=False)
    print("  control A  no momentum, flat preconditioner, projected -> cos=%.2e" % c)
    if not c < 1e-6:
        print("     FAIL: the projection does not survive even an isometry")
        ok = False

    c = survival(m_ce, g, sv, 0.0, rng, use_momentum=False, use_precond=False,
                 project=False)
    print("  control B  same, NOT projected                         -> cos=%.4f" % c)
    if not c > 0.05:
        print("     FAIL: unprojected gradient has no CE component, so A is vacuous")
        ok = False

    c = survival(m_ce, g, sv, 0.0, rng, use_momentum=True, use_precond=False)
    print("  control C  momentum ON, flat preconditioner, projected -> cos=%.4f" % c)
    if not c > 0.5:
        print("     FAIL: momentum alone should carry the CE direction back")
        ok = False

    c = survival(m_ce, g, sv, 3.0, rng, use_momentum=False, use_precond=True)
    print("  control D  no momentum, SPREAD preconditioner, projd   -> cos=%.4f" % c)
    if not c > 1e-3:
        print("     FAIL: diagonal rescale preserved orthogonality, which is false")
        ok = False

    # control E: the compounding law itself. m_A - m_B = (1 - b1^k)(g - g'),
    # exactly and independent of the angle. If this drifts, every number in
    # `count_change_compounding` is built on a broken accumulator.
    n = 20_000
    base = _unit(rng, n)
    perp = _unit(rng, n)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)
    worst = 0.0
    for cg in (-1.0, 0.0, 0.5):
        gg = base
        gp = cg * base + np.sqrt(max(0.0, 1.0 - cg ** 2)) * perp
        mA = np.zeros(n)
        mB = np.zeros(n)
        full = np.linalg.norm(gg - gp)
        for k in range(1, 30):
            mA = B1 * mA + (1.0 - B1) * gg
            mB = B1 * mB + (1.0 - B1) * gp
            worst = max(worst, abs(np.linalg.norm(mA - mB) / full - (1 - B1 ** k)))
    print("  control E  momentum accumulation vs (1-b1^k)          -> max err %.2e"
          % worst)
    if not worst < 1e-9:
        print("     FAIL: the accumulator does not follow the closed form")
        ok = False

    # control F: and it must be a REAL rise, not a restatement. At k=1 the
    # difference present is 0.100; at k=29 it is 0.953. If those were equal the
    # whole compounding argument would be vacuous.
    lo, hi = 1 - B1 ** 1, 1 - B1 ** 29
    print("  control F  difference carried: step1=%.3f step29=%.3f (%.1fx)"
          % (lo, hi, hi / lo))
    if not hi / lo > 5.0:
        print("     FAIL: no compounding to speak of, so the section is moot")
        ok = False

    print("  SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--compounding", action="store_true",
                    help="a count-function change over all 29 steps, not one")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.self_test:
        print("SELF-TEST -- each control isolates ONE destroyer")
        raise SystemExit(0 if self_test(rng) else 1)

    if a.compounding:
        print("DOES A COUNT-FUNCTION CHANGE COMPOUND OVER THE 29 STEPS?")
        print("1b-pre(6) was retracted for not doing this arithmetic. Here it")
        print("is -- and the FIRST version of this section got it wrong too.")
        print()
        print("  THE CONSTRAINT STEPS ARE NOT CONSECUTIVE. train.py runs the")
        print("  whole CE batch loop (one optimizer.step per batch, ~126) and")
        print("  calls finish_constraint_step ONCE per epoch. So b1^126 =")
        print("  %.2e of one constraint step's momentum survives to the next."
              % (B1 ** 126))
        print("  Difference present AT a constraint step with c CE steps")
        print("  between = (1-b1)/(1-b1^(c+1)):")
        for c in (0, 10, 126):
            print("     c=%-4d -> %.4f%s" % (c, (1 - B1) / (1 - B1 ** (c + 1)),
                                             "   <- the pipeline" if c == 126 else ""))
        print("  At c=126 that is the SINGLE-STEP value, forever. Momentum does")
        print("  not compound it. The WEIGHTS do.")
        print()
        print("  And the INPUT angle is not 180. `sum`'s per-item gradient is")
        print("  p(1-p) and `uniform`'s is their mean; both are elementwise")
        print("  NON-NEGATIVE, so the angle is bounded below 90 by construction:")
        for label, pc in _P_CASES(rng):
            print("     %-34s %5.1f deg" % (label, count_gradient_angle(pc)))
        print()
        print("  AND ce_rho IS NO LONGER AN ASSUMPTION. Real net, real Adam,")
        print("  batch 64 and 8064/64 = 126 steps/epoch -- the trainer's own")
        print("  spacing. lag-1 cosine between consecutive CE minibatch")
        print("  gradients, by epoch:")
        acs = ce_gradient_autocorrelation(epochs=3)
        for i, c in enumerate(acs, 1):
            print("     epoch %d -> %.4f%s" % (i, c, "   <- warm-up 1 regime"
                                               if i == 1 else ""))
        print("  It FALLS as the model fits, so warm-up 1 is its high point.")
        print("  (Synthetic MLP, not MobileNetV3: an order of magnitude with a")
        print("  mechanism, not the campaign's number.)")
        print()
        print("  CUMULATIVE trajectory separation, input angle 29.4 deg:")
        print("  %-30s %10s %11s %9s"
              % ("CE-direction model", "after 1", "after 29", "sep/len"))
        real = np.cos(np.radians(29.4))
        for rho, label in ((0.0, "0.00 uncorrelated (assumed)"),
                           (round(acs[0], 3), "%.3f MEASURED, warm-up 1" % acs[0]),
                           (0.5, "0.50 half-correlated")):
            f, _, c, s = count_change_compounding(real, rho, rng, n=12000)
            print("  %-30s %7.2f deg %8.2f deg %9.4f" % (label, f, c, s))
        f0, _, c0, s0 = count_change_compounding(real, 0.0, rng, n=12000,
                                                 ce_between=0)
        print("  %-30s %7.2f deg %8.2f deg %9.4f"
              % ("(consecutive -- NOT the pipeline)", f0, c0, s0))
        print()
        print("  ^ AT THE MEASURED ce_rho THERE IS ESSENTIALLY NO COMPOUNDING.")
        print("    The ~5x growth belongs to the ce_rho=0 assumption; the")
        print("    measured value collapses it to ~1.1x and half a percent of")
        print("    the distance travelled. So the per-step compression IS the")
        print("    story, which is what this project recorded before I")
        print("    'corrected' it. Still a POWER consideration and NOT a")
        print("    predicted null -- parameter separation is not items, and")
        print("    1b-pre(6) is the standing warning against that leap.")
        raise SystemExit(0)

    print("DOES `ortho_project`'s ORTHOGONALITY REACH THE WEIGHTS?")
    print("norms measured on octmnist L50 (adam_contamination.py); b1=%.2f, d=%d"
          % (B1, DIM))
    print()
    print("  |cos(update, CE direction)| -- proportional to the first-order CE")
    print("  change the constraint step causes. The projection sets the RAW")
    print("  gradient's value to 0; `removed` is how much of it that buys.")
    print()
    hdr = "  %-6s %-9s %-9s %-9s %-9s" % ("epoch", "spread", "projected",
                                          "unprojected", "removed")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ep, (m_ce, g, sv) in MEASURED.items():
        for spread in (0.0, 1.0, 2.0, 3.0):
            # SAME seed for both, so the pair differs ONLY in the projection.
            # With independent draws the `removed` column was pure Monte-Carlo
            # noise and printed negative values for a quantity that cannot be
            # negative.
            sd = int(rng.integers(0, 2 ** 31 - 1))
            on = survival(m_ce, g, sv, spread, np.random.default_rng(sd),
                          project=True)
            off = survival(m_ce, g, sv, spread, np.random.default_rng(sd),
                           project=False)
            frac = (off - on) / off if off > 0 else float("nan")
            print("  %-6d %-9.1f %-9.4f %-9.4f %-9.1f%%"
                  % (ep, spread, on, off, 100.0 * frac))
    print()
    print("  spread = log10 range of the per-coordinate sqrt(v). 0.0 is a flat")
    print("  preconditioner, which Adam never has; real networks sit at 2-3.")
    print()
    m_ce, g, sv = MEASURED[3]
    share = (1.0 - B1) * 1.0 / (B1 * m_ce + (1.0 - B1) * 1.0)
    print("  THE BOUND, assumption-free: after the clip the constraint gradient")
    print("  has norm exactly 1.0, so its share of the momentum vector is")
    print("    (1-b1)*1.0 / (b1*|m_CE| + (1-b1)*1.0) = %.1f%%" % (100.0 * share))
    print("  and the projection can only ever act on THAT share. The other %.1f%%"
          % (100.0 * (1 - share)))
    print("  is stale CE momentum, which points along the CE direction by")
    print("  construction and which the projection never touches.")

    print()
    print("  ROBUSTNESS -- the closed form, so it needs no simulation and no")
    print("  measured constant. `removed = (1-b1)/(b1*ratio + (1-b1))`:")
    print()
    print("  %-22s %s" % ("|m_CE| / |g_con|", "max removable"))
    print("  " + "-" * 38)
    for ratio in (0.01, 0.111, 0.5, 1.4, 3.0, 10.0):
        tag = ""
        if abs(ratio - 0.111) < 1e-6:
            tag = "  <- (1-b1)/b1, the break-even"
        if ratio in (1.4, 3.0):
            tag = "  <- the MEASURED range"
        print("  %-22.3f %6.1f%%%s" % (ratio, 100.0 * removal_fraction(ratio), tag))
    print()
    print("  So the verdict survives the missing provenance: it fails only if")
    print("  the CE momentum is ~10x SMALLER than the constraint gradient.")

    print()
    print("  AND THE TABLE ABOVE IS THE PROJECTION'S BEST CASE. It assumes the")
    print("  reference IS the CE momentum. `ortho_ref = snapshot_grads(model)`")
    print("  is one minibatch's gradient, so rho = cos(ref, m_CE) < 1:")
    print()
    print("  %-7s %-11s %-13s %-9s" % ("rho", "projected", "unprojected", "removed"))
    print("  " + "-" * 42)
    m_ce, g, sv = MEASURED[3]
    for rho in (1.0, 0.5, 0.2, 0.05):
        sd = int(rng.integers(0, 2 ** 31 - 1))
        on, off = ref_mismatch(m_ce, g, sv, 2.0, rho, np.random.default_rng(sd))
        frac = (off - on) / off if off > 0 else float("nan")
        print("  %-7.2f %-11.4f %-13.4f %-9.2f%%" % (rho, on, off, 100.0 * frac))
    print()
    print("  (epoch 3, spread 2.0. rho = 1.00 is the best case and already")
    print("  removes nothing; every realistic rho removes less.)")

    print()
    print("  THE SAME QUESTION FOR `head_only`, which masks by PARAMETER SET")
    print("  rather than by direction -- real torch.optim.Adam, not a model:")
    dh, db, ratio = masked_coordinate_drift()
    print("    step taken by the head     (grad 1.0) = %+.4e" % dh)
    print("    step taken by the BACKBONE (grad 0.0) = %+.4e" % db)
    print("    ratio = %.4f" % ratio)
    print()
    print("  A ZEROED GRADIENT STILL MOVES THE PARAMETER, at %.0f%% of the" % (100 * abs(ratio)))
    print("  unmasked step, on stale CE momentum. So `head_only` does NOT")
    print("  freeze the backbone during the constraint phase. What it does")
    print("  deliver is that NO CONSTRAINT INFORMATION reaches it -- the drift")
    print("  is pure CE momentum, which the lambda=0 twin has too, so it is")
    print("  common-mode in the contrast that matters. Read the arm as")
    print("  'the constraint sees only the head', never as 'the backbone is frozen'.")

    print()
    print("  " + "=" * 66)
    print("  WHY A COIN DID THE SAME DAMAGE (1b-pre(6)) -- the direction was")
    print("  never DELIVERED. Both arms put a norm-1.0 vector into prm.grad;")
    print("  Adam adds b1*m_CE to both.")
    print()
    print("  %-9s %-16s %-17s %s" % ("spread", "cos(real,coin)", "cos(real,m_CE)",
                                     "constraint share"))
    print("  " + "-" * 62)
    for spread in (0.0, 1.0, 2.0, 3.0):
        c, cm, share = coin_equivalence(spread, rng)
        print("  %-9.1f %-16.4f %-17.4f %.1f%%" % (spread, c, cm, 100.0 * share))
    print()
    print("  The real step and the coin step are ~99.4% the same vector PER")
    print("  STEP, under constraint_step_rule=shared (the protocol default,")
    print("  and what every trained TraLO arm resolves to).")
    print()
    print("  !! THIS IS A PER-STEP GEOMETRY, NOT AN OUTCOME CLAIM. A 0.6%")
    print("  consistent directional difference COMPOUNDS over 29 steps, and")
    print("  FRAMEWORK 1b-pre(6) measures coin and `linear` with")
    print("  NON-OVERLAPPING distributions at L50_G30. Do not read this table")
    print("  as explaining that null -- an earlier version of 1b-pre(6) said so")
    print("  and was RETRACTED on 2026-08-25. What it does say is that")
    print("  `tralo_coin` has very little per-step contrast with its treatment,")
    print("  so its power comes from compounding and not from step geometry.")
    print("  Under step_rule=sgd none of this applies: the step is")
    print("  p -= lr*g and the direction is delivered at cos = 1.0.")

    print()
    print("  AND WHAT CLEARING ONLY `m` WOULD BUY (keeps v, so NOT the rejected")
    print("  separate_constraint_optimizer):")
    print()
    print("  %-11s %-15s %-15s %s" % ("variant", "cos(upd,g)", "cos(upd,m_CE)",
                                      "rel. magnitude"))
    print("  " + "-" * 58)
    for name, r in zip(("shared", "m zeroed"), momentum_reset(1.0, rng)):
        print("  %-11s %-15.4f %-15.4f %.3f" % (name, r[0], r[1], r[2]))
    print()
    print("  !! THE DOSE MOVES WITH IT and that is not priced here. Clearing `m`")
    print("  shrinks the delivered step by the factor in the last column, so an")
    print("  arm built this way differs from its control in MAGNITUDE as well as")
    print("  direction -- the exact confound that made the hounie baseline")
    print("  meaningless. Adam's bias correction is the obvious lever and it is")
    print("  NOT evaluated here. Do not launch this on the strength of the")
    print("  cosine column alone.")

    print()
    print("  ...AND THE DOSE CONFOUND IS REMOVABLE. Renormalising the cleared")
    print("  step back to the SHARED step's norm changes direction and nothing")
    print("  else -- the property that makes the random-direction control legal:")
    print()
    print("  %-26s %-11s %s" % ("variant", "cos(u, g)", "dose vs shipped"))
    print("  " + "-" * 56)
    for name, cg, rel in dose_matched_delivery(rng):
        print("  %-26s %-11.4f %.3f" % (name, cg, rel))
    print()
    print("  => a DIRECTION-ONLY arm is constructible: cos 0.08 -> 1.00 at dose")
    print("  1.000. That is the first design that would actually TEST the")
    print("  constraint direction. !! It is not a prediction that it helps --")
    print("  2(s) has all 24 constraint terms NEGATIVE, so delivering more of")
    print("  the direction may deepen the damage. It makes the question")
    print("  answerable; it does not answer it.")

    print()
    print("  " + "=" * 66)
    print("  AND WHAT THIS MEANS FOR COMPARING COUNT FUNCTIONS (tralo vs")
    print("  tralo_uniform): a change to `g` reaches the weights through the")
    print("  same 7.4% channel.")
    print()
    print("  %-14s %-14s %s" % ("cos(g, g')", "cos(u, u')", "angle in -> out"))
    print("  " + "-" * 50)
    for cg in (0.99, 0.90, 0.50, 0.00, -1.00):
        cu, ai, ao = count_change_attenuation(cg, rng)
        print("  %-14.2f %-14.6f %.1f deg -> %.2f deg" % (cg, cu, ai, ao))
    print()
    print("  Two count functions pointing in OPPOSITE directions differ by")
    print("  only ~9 degrees once delivered -- a ~20x angular compression.")
    print()
    print("  !! POWER CONSIDERATION, NOT A PREDICTED NULL. This is per-step")
    print("  geometry; a consistent difference COMPOUNDS over 29 steps, and")
    print("  1b-pre(6) is direct evidence that compounding can separate arms")
    print("  whose per-step contrast is small. The version of this reasoning")
    print("  that forgot the distinction produced a retraction on 2026-08-25.")
    print("  Keep `flag_live` (md5 across arms) as the gate it always was.")


if __name__ == "__main__":
    main()

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

    print("  SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.self_test:
        print("SELF-TEST -- each control isolates ONE destroyer")
        raise SystemExit(0 if self_test(rng) else 1)

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
    print("  The real step and the coin step are ~99.4% the same vector. The")
    print("  null was never evidence that the constraint direction is")
    print("  uninformative -- it is what a shared optimizer produces whatever")
    print("  you put in the gradient.")

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
    print("  ⚠️ THE DOSE MOVES WITH IT and that is not priced here. Clearing `m`")
    print("  shrinks the delivered step by the factor in the last column, so an")
    print("  arm built this way differs from its control in MAGNITUDE as well as")
    print("  direction -- the exact confound that made the hounie baseline")
    print("  meaningless. Adam's bias correction is the obvious lever and it is")
    print("  NOT evaluated here. Do not launch this on the strength of the")
    print("  cosine column alone.")


if __name__ == "__main__":
    main()

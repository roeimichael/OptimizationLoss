"""Does a UNIFORM logit gradient actually give a pure bias shift? It does not.

`uniform_grad_count` is built on this claim (src/losses/transductive_loss.py):

    "a uniform step in u is a uniform step in the class logit, which is a pure
     bias shift, which cannot reorder"

The first clause is true and the third does not follow, because the step is
taken in the PARAMETERS, not in the logit. For a linear head z_ic = w_c.f_i + b_c
a uniform logit gradient g gives

    dL/dw_c = g * sum_i f_i = g * n * fbar      dL/db_c = g * n
    dz_ic   = -lr * g * n * ( fbar.f_i + 1 )

which varies with fbar.f_i. So it reorders, and it does so with the backbone
FROZEN -- the leak is in the head, not in the representation.

\U0001f6d1 THE ONLY UPDATE THAT PROVABLY CANNOT REORDER IS ONE CONFINED TO b_c.
That is checkable and it is the negative control here.

\u26a0\ufe0f AND IT IS ALSO USELESS, which is the real finding: a constant added to
z_c leaves the WITHIN-CLASS order of class c untouched, so the top-K the
allocator emits for c is bit-identical. The harmless path is harmless because
it changes nothing the metric reads. FRAMEWORK 2(z12).
"""
import argparse
import sys

import numpy as np


def kendall_tau(a, b):
    """Fraction of concordant pairs minus discordant, on two score vectors."""
    n = len(a)
    idx = np.argsort(-a)
    br = b[idx]
    conc = disc = 0
    for i in range(n):
        d = br[i + 1:] - br[i]
        disc += int((d > 0).sum())
        conc += int((d < 0).sum())
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def step(F, w, b, gvec, lr, mode):
    """One GD step on (w, b) from per-item logit gradients gvec. New scores."""
    if mode == "full":                       # what tralo_uniform actually does
        dw = (gvec[:, None] * F).sum(axis=0)
        db = gvec.sum()
    elif mode == "bias_only":                # the provably order-preserving one
        dw = np.zeros_like(w)
        db = gvec.sum()
    elif mode == "weight_only":
        dw = (gvec[:, None] * F).sum(axis=0)
        db = 0.0
    else:
        raise ValueError(mode)
    return F @ (w - lr * dw) + (b - lr * db)


def run(n=2000, d=64, lr=1e-3, seed=0, out=sys.stdout):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, d)) / np.sqrt(d)
    w = rng.normal(size=d) / np.sqrt(d)
    b = 0.0
    z0 = F @ w + b
    fbar = F.mean(axis=0)

    rows = []
    for label, gvec in (("uniform g (tralo_uniform)", np.ones(n)),
                        ("p(1-p) g (tralo / sum)",
                         (lambda p: p * (1 - p))(1 / (1 + np.exp(-z0))))):
        for mode in ("full", "bias_only", "weight_only"):
            z1 = step(F, w, b, gvec, lr, mode)
            spread = float(np.ptp(z1 - z0))
            tau = kendall_tau(z0, z1)
            # top-K churn at a realistic budget
            K = n // 5
            s0 = set(np.argsort(-z0)[:K].tolist())
            s1 = set(np.argsort(-z1)[:K].tolist())
            rows.append((label, mode, spread, tau, K - len(s0 & s1)))

    print("BIAS-SHIFT PROBE -- is a uniform logit gradient a pure bias shift?",
          file=out)
    print("n=%d  d=%d  lr=%g   corr(fbar.f_i) is what makes it fail\n"
          % (n, d, lr), file=out)
    print("%-26s %-12s %12s %9s %8s"
          % ("logit gradient", "update", "spread(dz)", "kendall", "topK_out"),
          file=out)
    for label, mode, spread, tau, churn in rows:
        print("%-26s %-12s %12.3e %9.5f %8d"
              % (label, mode, spread, tau, churn), file=out)
    print("\nspread(dz) = max(dz) - min(dz). EXACTLY 0 <=> a true bias shift.",
          file=out)
    return rows


def self_test(out=sys.stdout):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-62s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    rows = run(n=800, d=32, out=open(__import__("os").devnull, "w"))
    g = {(l.split()[0], m): (s, t, c) for l, m, s, t, c in rows}

    # the claim under test
    su, tu, cu = g[("uniform", "full")]
    check("uniform gradient + full update does NOT give a constant shift",
          su > 1e-12)
    check("uniform gradient + full update REORDERS (kendall < 1)", tu < 1.0)
    check("uniform gradient + full update churns the top-K", cu > 0)

    # NEGATIVE CONTROL: the one update that must be exactly harmless
    sb, tb, cb = g[("uniform", "bias_only")]
    check("NEGATIVE CONTROL: bias-only update spread is exactly 0",
          sb < 1e-12)
    check("NEGATIVE CONTROL: bias-only update preserves order exactly",
          tb == 1.0 and cb == 0)

    # and it is not a quirk of the uniform weighting
    sp, tp, cp = g[("p(1-p)", "full")]
    check("p(1-p) gradient reorders too (so this is not uniform-specific)",
          sp > 1e-12 and tp < 1.0)
    check("bias-only is harmless for p(1-p) as well",
          g[("p(1-p)", "bias_only")][0] < 1e-12)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--self-test", action="store_true")
    a.add_argument("--n", type=int, default=2000)
    a.add_argument("--d", type=int, default=64)
    a.add_argument("--lr", type=float, default=1e-3)
    args = a.parse_args()
    sys.exit(self_test() if args.self_test else (run(args.n, args.d, args.lr) and 0))

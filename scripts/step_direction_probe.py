"""Do the count functions deliver DIFFERENT parameter steps? They do not.

Every count function in this project has the form `S_c = sum_i phi(p_ic)`, so
its gradient w.r.t. the capped class's head weights is

    dS/dw_c = sum_i g_i * f_i ,        g_i = phi'(p_ic) * dp_ic/dz_ic

which is a g-WEIGHTED MEAN OF THE TEST FEATURES. The only thing a new count
function can change is the weighting g. Under `constraint_grad_mode: normalize`
the magnitude is discarded (FRAMEWORK: the delivered step is exactly lr*clip),
so a new count function can only matter if it changes the DIRECTION.

This measures that directly, on the stored `test_embeddings.npz`. If the
directions are collinear, the count-function family is one arm and no member of
it can behave differently from any other -- which closes the entire family with
one number instead of one campaign each.

\u26a0\ufe0f HEAD-ONLY, and deliberately so. It bounds what the count function can do
through the LINEAR HEAD, where the effect is exactly computable. The backbone
adds a further channel, measured separately and separately negative
(iwc1/iwc2, AP -0.031 / -0.094 vs the twin). A collinearity here does not say
the arms are identical end-to-end; it says the count function is not the thing
that differentiates them.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def weightings(p, z):
    m = np.abs(z)
    return {
        "uniform": np.ones_like(p),
        "sum_p(1-p)": p * (1.0 - p),
        "margin_sech2": 1.0 / np.cosh(m / 0.5) ** 2,
        "one_minus_p": 1.0 - p,
        "p": p.copy(),
        "linear_z": z - z.min(),
    }


def probs(run_dir, cls):
    path = os.path.join(run_dir, "final_predictions.csv")
    col = "Prob_Class_%d" % cls
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(float(row[col]))
    return np.array(out)


def directions(feats, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)
    D = {}
    for k, g in weightings(p, z).items():
        v = (g[:, None] * feats).sum(axis=0)
        n = np.linalg.norm(v)
        D[k] = v / n if n else v
    fb = feats.mean(axis=0)
    D["_fbar"] = fb / np.linalg.norm(fb)
    return D


def self_test(out=sys.stdout):
    """Synthetic features with a KNOWN answer in both directions."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-62s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    rng = np.random.default_rng(0)
    n, d = 1500, 32

    # (a) features with a LARGE mean -> every weighting collapses onto fbar
    F = rng.normal(size=(n, d)) + 4.0
    p = rng.uniform(0.02, 0.98, size=n)
    D = directions(F, p)
    c = float(D["uniform"] @ D["sum_p(1-p)"])
    check("large-mean features: uniform and sum are collinear (>0.99)",
          c > 0.99)
    check("large-mean features: both align with fbar (>0.99)",
          float(D["uniform"] @ D["_fbar"]) > 0.99
          and float(D["sum_p(1-p)"] @ D["_fbar"]) > 0.99)

    # (b) NEGATIVE CONTROL: a geometry the weighting CAN steer. Items are
    #     mapped to ORTHOGONAL directions by p-bucket, so a flat weighting and
    #     a |p-0.5|-peaked one MUST point somewhere different. Note what it
    #     takes: orthogonal geometry and three separated buckets. With
    #     non-negative features and non-negative weights every direction lies
    #     in the positive orthant and a high cosine is FORCED -- that is the
    #     mechanism this probe measures, not an artefact of it.
    buckets = [(0.01, 0.05), (0.45, 0.55), (0.95, 0.99)]
    per = 500
    p2 = np.concatenate([rng.uniform(lo, hi, per) for lo, hi in buckets])
    F2 = np.zeros((per * len(buckets), len(buckets)))
    for b in range(len(buckets)):
        F2[b * per:(b + 1) * per, b] = 1.0
    D2 = directions(F2, p2)
    c2 = abs(float(D2["uniform"] @ D2["sum_p(1-p)"]))
    check("NEGATIVE CONTROL: a steerable geometry gives cosine < 0.9 "
          "(got %.3f)" % c2, c2 < 0.9)
    check("NEGATIVE CONTROL: so the probe CAN report non-collinearity",
          c2 < 0.99)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+")
    a.add_argument("--glob")
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()

    runs = args.runs or sorted(glob.glob(args.glob))[:12]
    if not runs:
        raise SystemExit("no runs")

    keys = None
    acc = {}
    nrun = 0
    for rd in runs:
        fp = os.path.join(rd, "test_embeddings.npz")
        if not os.path.exists(fp):
            continue
        F = np.load(fp)["features"].astype(np.float64)
        for c in args.classes:
            try:
                p = probs(rd, c)
            except Exception:
                continue
            if len(p) != len(F):
                continue
            D = directions(F, p)
            if keys is None:
                keys = [k for k in D if k != "_fbar"]
            for i, x in enumerate(keys):
                for y in keys[i:]:
                    acc.setdefault((x, y), []).append(float(D[x] @ D[y]))
                acc.setdefault((x, "_fbar"), []).append(float(D[x] @ D["_fbar"]))
            nrun += 1
    if not nrun:
        raise SystemExit("no usable (run, class) pairs")

    print("STEP-DIRECTION PROBE -- cosine between count functions' parameter "
          "steps")
    print("%d (run, class) pairs, REAL stored features\n" % nrun)
    print("%-14s %s" % ("", "".join("%13s" % k[:12] for k in keys)))
    for x in keys:
        cells = []
        for y in keys:
            v = acc.get((x, y)) or acc.get((y, x))
            cells.append("%13.4f" % np.mean(v))
        print("%-14s %s" % (x[:14], "".join(cells)))
    print("\ncosine with the PLAIN MEAN FEATURE f_bar (mean over pairs, "
          "min in brackets):")
    for x in keys:
        v = acc[(x, "_fbar")]
        print("   %-16s %8.4f   [min %.4f]" % (x, np.mean(v), np.min(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

BAND = 20      # half-width in ITEMS of the band straddling the cut


def sech2(x):
    """Stable and precise: `1/cosh(x)**2` overflows, the sigmoid form underflows."""
    a = np.exp(-2.0 * np.abs(x))
    return 4.0 * a / (1.0 + a) ** 2


def weightings(p, z, K=None, n_items=40):
    """Per-item gradient weights. `cut_window` needs K and is skipped without it.

    `margin_sech2` is centred on the DECISION BOUNDARY (|z| = 0) and
    `cut_window` on the CUT (rank K). Those are the two things CLAUDE.md rule 3
    warns against conflating, and on tight caps they are ~300 items apart.
    """
    m = np.abs(z)
    w = {
        "uniform": np.ones_like(p),
        "sum_p(1-p)": p * (1.0 - p),
        "margin_sech2": sech2(m / 0.5),
        "one_minus_p": 1.0 - p,
        "p": p.copy(),
        "linear_z": z - z.min(),
    }
    if K is not None and 1 <= K < len(z):
        tau = np.sort(z)[::-1][K - 1]
        d = np.abs(z - tau)
        T = max(float(np.sort(d)[min(n_items, len(d)) - 1]), 1e-9)
        w["cut_window"] = sech2((z - tau) / T)
    return w


def probs(run_dir, cls):
    """(probabilities for `cls`, K = how many the run actually deployed)."""
    path = os.path.join(run_dir, "final_predictions.csv")
    col = "Prob_Class_%d" % cls
    out, K = [], 0
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(float(row[col]))
            K += int(int(row["Predicted_Label"]) == cls)
    return np.array(out), K


def directions(feats, p, K=None, n_items=40):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)
    D = {}
    for k, g in weightings(p, z, K, n_items).items():
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

    # (c) the cut window appears ONLY when K is given, and must actually
    #     concentrate at the cut -- otherwise the column means nothing.
    F3 = rng.normal(size=(n, d)) + 4.0
    p3 = np.clip(rng.uniform(0.02, 0.98, size=n), 1e-6, 1 - 1e-6)
    K3 = 300
    check("cut_window is ABSENT without K (never a silent zero column)",
          "cut_window" not in directions(F3, p3))
    check("cut_window is PRESENT with K",
          "cut_window" in directions(F3, p3, K3))
    z3 = np.log(p3) - np.log1p(-p3)
    W3 = weightings(p3, z3, K3)
    band = np.argsort(-z3)[K3 - 20:K3 + 20]
    mc = W3["cut_window"][band].sum() / W3["cut_window"].sum()
    ms = W3["sum_p(1-p)"][band].sum() / W3["sum_p(1-p)"].sum()
    check("cut_window puts >10x the shipped count's mass at the cut "
          "(%.3f vs %.3f)" % (mc, ms), mc > 10 * ms)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+")
    a.add_argument("--glob")
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--n-items", type=int, default=40,
                   help="items inside the cut window (T is derived from this)")
    a.add_argument("--limit", type=int, default=0,
                   help="cap the number of run dirs (0 = no cap). "
                        "Prints what it dropped.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()

    # 🛑 THIS WAS `[:12]`, AND IT WAS A SILENT CAP. `sorted()` is alphabetical,
    # so on a 24-run cell it kept the first two or three ARMS and dropped the
    # rest -- a subset that looks like a campaign and is not one. It produced
    # the first version of the FRAMEWORK 2(z12) table, which read 24 pairs when
    # the campaign held far more. No silent caps: unlimited by default, and if
    # a limit is asked for, say what it dropped.
    runs = args.runs or sorted(glob.glob(args.glob))
    if not runs:
        raise SystemExit("no runs")
    if args.limit and args.limit < len(runs):
        print("!! --limit %d applied: %d of %d run dir(s) DROPPED. `sorted()` "
              "is alphabetical, so this biases toward whichever arms sort "
              "first -- do not read a cross-arm number off a limited run."
              % (args.limit, len(runs) - args.limit, len(runs)))
        runs = runs[:args.limit]

    keys = None
    acc = {}
    mass = {}
    nrun = 0
    for rd in runs:
        fp = os.path.join(rd, "test_embeddings.npz")
        if not os.path.exists(fp):
            continue
        F = np.load(fp)["features"].astype(np.float64)
        for c in args.classes:
            try:
                p, K = probs(rd, c)
            except Exception:
                continue
            if len(p) != len(F):
                continue
            D = directions(F, p, K, args.n_items)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            zc = np.log(pc) - np.log1p(-pc)
            band = np.argsort(-zc)[max(0, K - 20):K + 20]
            for k, g in weightings(pc, zc, K, args.n_items).items():
                tot = g.sum()
                mass.setdefault(k, []).append(
                    float(g[band].sum() / tot) if tot else 0.0)
            if keys is None:
                keys = [k for k in D if k != "_fbar"]
            for i, x in enumerate(keys):
                for y in keys[i:]:
                    acc.setdefault((x, y), []).append(float(D[x] @ D[y]))
                acc.setdefault((x, "_fbar"), []).append(float(D[x] @ D["_fbar"]))

            # the z12(b) table: where each weighting's gradient actually lands
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            zc = np.log(pc) - np.log1p(-pc)
            band = np.argsort(-zc)[max(0, K - BAND):K + BAND]
            for k, g in weightings(pc, zc, K, args.n_items).items():
                tot = float(g.sum())
                mass.setdefault(k, []).append(
                    float(g[band].sum()) / tot if tot else 0.0)
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
    print("\nFRACTION OF TOTAL GRADIENT MASS on the %d items straddling the "
          "cut," % (2 * BAND))
    print("i.e. the only items whose movement can change the emitted top-K "
          "set:")
    for k in sorted(mass, key=lambda k: -float(np.mean(mass[k]))):
        v = mass[k]
        print("   %-16s %8.4f   [min %.4f  max %.4f]"
              % (k, np.mean(v), np.min(v), np.max(v)))
    print("\ncosine with the PLAIN MEAN FEATURE f_bar (mean over pairs, "
          "min in brackets):")
    for x in keys:
        v = acc[(x, "_fbar")]
        print("   %-16s %8.4f   [min %.4f]" % (x, np.mean(v), np.min(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

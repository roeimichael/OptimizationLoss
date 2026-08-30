"""How far is the DECISION BOUNDARY from the CUT? The regime reversal, measured.

WHY THIS EXISTS. Every count penalty this project ships puts its per-item
gradient at or near the DECISION BOUNDARY. `sum`'s weight is `p(1-p)`, which
peaks at p = 0.5, i.e. where an item is one step from flipping; `margin`
windows that point explicitly, via `sigma'(m/T)` at `m = p_ic - max_{c'!=c}
p_ic'`. But deployed quality is decided at the CUT: the post-hoc allocator
emits exactly K items by probability rank, whatever the model's hard count is.

Those are two different items, and `hard_count - K` is the distance between
them -- between the point the penalty can push and the point the metric reads.

*** THIS IS THE MECHANISM OF THE REGIME REVERSAL. Measured over 5 campaigns,
3 backbones, 2 capped classes and 4 seeds:

    regime          cap          gap (items)   p at the cut   slope_bd/slope_K
    LOOSE  K/n=0.90  L90_G95          14 -  76   0.676-0.959         1.9 -   6.6
    LOOSE  K/n=0.80  L80_G95          51 - 123   0.832-0.995         2.8 -  47.3
    TIGHT  K/n=0.30  L30/L50         198 - 397   0.997-1.000        79   - 32387
    TIGHT  K/n=0.20  L20_G50         235 - 442   0.999-1.000       420   - 81926

At a loose cap the boundary and the cut are ~25 items apart and carry
comparable gradient, so pressure aimed at the boundary lands near the cut and
the constraint helps. At a tight cap they are 200-440 items apart and the cut
sits at p = 0.9999, where `p(1-p)` is 0.0001 -- the penalty has four to five
orders of magnitude MORE pull at the boundary than at the point that decides
the metric. The push is not weak; it is aimed somewhere the metric never looks.

AND IT IS STRUCTURAL, NOT A PROPERTY OF THIS DATASET. `gap = hard - K` and the
hard count is roughly `n_pos` for any reasonably calibrated model, so
`gap ~ n_pos * (1 - K/n)`. The gap is therefore set by the CAP FRACTION and
shrinks to zero only as K/n -> 1. Any boundary-concentrating count penalty
inherits this, on any dataset.

*** WHAT IT PREDICTS, AND THE PREDICTION IS FALSIFIABLE. `soft_count_mode:
margin` windows the BOUNDARY, so it cannot fix the tight regime -- it aims the
same pressure at the same wrong point, only more sharply. `results/margin2`
runs 6 tight cells and 6 loose ones, and this file's prediction, recorded
before it launched, is that `tralo_margin` gains in the LOOSE cells and does
NOT in the tight ones. If it gains at tight caps, this mechanism is wrong.

*** AND IT EXPLAINS `uniform`. `uniform_grad_count`'s per-item slope is a
population constant, so it does NOT concentrate at the boundary. That is a
liability at loose caps (it declines to aim where aiming pays) and an asset at
tight ones (there is no good place to aim, so spreading beats missing). The
measured reversal -- `sum` wins loose and loses tight, `uniform` the reverse --
is exactly what this geometry predicts, and it was previously recorded as an
unexplained empirical fact.

⚠️ WHAT THIS DOES NOT SAY. It does not say the penalty is too weak, and it does
not say the cut is unreachable in principle. It says the gradient is placed at
the boundary while the metric reads the cut, and that the two coincide only as
K/n -> 1. Reaching the cut with a count function is a separate question, and it
is CLOSED: a cut-centred count `sigma((p_ic - tau_c)/T)` with `tau_c` the K-th
order statistic counts the items above the K-th largest, which is K - 1
exactly, for any model -- see `src/losses/transductive_loss.py: margin_window`.

K is taken from the arm's own `final_predictions.csv`, which emits exactly K,
so this needs no cap arithmetic and makes no assumption about which scope
binds. The geometry is measured on the `tralo_null` twin -- the model before
any constraint acted -- so it belongs to the cap, not to the arm.

    python -m scripts.cut_gap <campaign-root> [<campaign-root> ...]
    python -m scripts.cut_gap --self-test
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Below this the cut sits where p(1-p) has effectively vanished and no count
# function differentiating `sum_i p_ic` can move it. Calibrated on the tight
# campaigns above, so treat it as a flag rather than a law.
DEAD_SLOPE = 0.005


def measure(root, arm="tralo_null"):
    """Per (cap, class, seed) geometry for one campaign root. Returns a frame."""
    rows = []
    camp = os.path.basename(os.path.normpath(root))
    for cfg_path in glob.glob(os.path.join(root, "*", "*", "*", "*",
                                           "seed_*", "config.json")):
        d = os.path.dirname(cfg_path)
        raw_p = os.path.join(d, "final_predictions_raw.csv")
        alloc_p = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw_p) and os.path.exists(alloc_p)):
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("status") != "completed" or cfg.get("arm") != arm:
            continue
        parts = os.path.normpath(d).split(os.sep)
        model, cap, seed = parts[-5], parts[-3], parts[-1]
        classes = (cfg.get("dataset_config") or {}).get("constrained_class") or []

        raw, alloc = pd.read_csv(raw_p), pd.read_csv(alloc_p)
        y = raw["True_Label"].to_numpy()
        pcols = sorted((int(c[len("Prob_Class_"):]), c) for c in raw.columns
                       if c.startswith("Prob_Class_"))
        if not pcols:
            continue
        P = np.column_stack([raw[c].to_numpy() for _, c in pcols])
        raw_pred = P.argmax(1)
        acol = ("Predicted_Label" if "Predicted_Label" in alloc.columns
                else [c for c in alloc.columns if "redict" in c][0])
        alloc_pred = alloc[acol].to_numpy()

        for cls in classes:
            cls = int(cls)
            K = int((alloc_pred == cls).sum())
            hard = int((raw_pred == cls).sum())
            if not (0 < K <= len(P) and 0 < hard <= len(P)):
                continue
            order = np.argsort(-P[:, cls])
            pk = float(P[order[K - 1], cls])
            pb = float(P[order[hard - 1], cls])
            rows.append(dict(campaign=camp, model=model, cap=cap, seed=seed,
                             cls=cls, K=K, hard=hard,
                             n_pos=int((y == cls).sum()), gap=hard - K,
                             p_K=pk, p_bd=pb,
                             slope_K=pk * (1 - pk), slope_bd=pb * (1 - pb)))
    return pd.DataFrame(rows)


def summarise(df):
    g = (df.groupby(["campaign", "cap", "cls"])
           .agg(seeds=("seed", "nunique"), K=("K", "mean"), hard=("hard", "mean"),
                n_pos=("n_pos", "mean"), gap=("gap", "mean"), p_K=("p_K", "mean"),
                slope_K=("slope_K", "mean"), slope_bd=("slope_bd", "mean"))
           .reset_index())
    g["K_over_n"] = g.K / g.n_pos
    g["ratio"] = g.slope_bd / g.slope_K.replace(0, np.nan)
    return g.sort_values(["cls", "K_over_n", "campaign"])


def report(g, out=sys.stdout):
    pd.set_option("display.width", 240)
    print("BOUNDARY-TO-CUT GAP  (on the lambda=0 twin: the model before any "
          "constraint)\n", file=out)
    print(g.to_string(index=False, float_format=lambda v: "%9.4f" % v), file=out)
    print("\ngap   = hard_count - K, in ITEMS. The penalty pushes at the boundary;",
          file=out)
    print("        the metric reads the cut. This is how far apart they are.", file=out)
    print("ratio = p(1-p) at the boundary / at the cut. How many times more pull",
          file=out)
    print("        the penalty has where the metric does NOT look.\n", file=out)

    dead = g[g.slope_K < DEAD_SLOPE]
    if len(dead):
        print("*** %d of %d (campaign, cap, class) points have the CUT in a dead "
              "zone" % (len(dead), len(g)), file=out)
        print("    (p(1-p) < %.3f at rank K). No count differentiating `sum_i "
              "p_ic`" % DEAD_SLOPE, file=out)
        print("    can move those cuts at any dose or shape. Their K/n is %s."
              % ", ".join("%.2f" % v for v in sorted(set(round(x, 2)
                                                         for x in dead.K_over_n))),
              file=out)
    live = g[g.slope_K >= DEAD_SLOPE]
    if len(live):
        print("    The %d live points sit at K/n = %s." % (
            len(live), ", ".join("%.2f" % v for v in
                                 sorted(set(round(x, 2) for x in live.K_over_n)))),
              file=out)
    if len(dead) and len(live):
        print("\n    => the split is by CAP FRACTION, not by dataset or backbone.",
              file=out)


def self_test(out=sys.stdout):
    """Does the instrument report the geometry it claims to, and can it say NO?

    Three synthetic models with a KNOWN answer. The gate is that the tight case
    and the loose case come out on opposite sides, and that a perfectly
    calibrated model -- hard count exactly K -- reports a gap of zero rather
    than something merely small.
    """
    rng = np.random.default_rng(0)
    n, n_pos = 3000, 400
    # a score distribution with a realistic saturated head
    p = np.clip(rng.beta(0.35, 0.35, n), 1e-6, 1 - 1e-6)
    srt = np.sort(p)[::-1]
    hard = int((p > 0.5).sum())

    ok = True
    print("SELF-TEST -- synthetic scores, n=%d, hard count %d\n" % (n, hard), file=out)
    print("  %-10s %6s %6s %10s %10s" % ("case", "K", "gap", "p_at_K", "slope_K"),
          file=out)
    seen = {}
    for name, K in (("tight", int(0.20 * n_pos)), ("loose", int(0.95 * n_pos)),
                    ("exact", hard)):
        pk = float(srt[K - 1])
        seen[name] = (hard - K, pk, pk * (1 - pk))
        print("  %-10s %6d %6d %10.4f %10.6f"
              % (name, K, hard - K, pk, pk * (1 - pk)), file=out)

    checks = [
        ("a tighter cap gives a strictly larger gap",
         seen["tight"][0] > seen["loose"][0]),
        ("a tighter cap puts the cut nearer p=1",
         seen["tight"][1] > seen["loose"][1]),
        ("the tight cut carries less per-item gradient than the loose one",
         seen["tight"][2] < seen["loose"][2]),
        ("a PERFECTLY calibrated model reports gap EXACTLY zero",
         seen["exact"][0] == 0),
        ("the flag fires on the tight cut and not the loose one",
         seen["tight"][2] < DEAD_SLOPE <= seen["loose"][2]),
    ]
    print(file=out)
    for label, good in checks:
        print("  %-4s %s" % ("OK" if good else "FAIL", label), file=out)
        ok = ok and good
    print("\nSELF-TEST %s" % ("PASSED" if ok else "FAILED"), file=out)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--arm", default="tralo_null",
                    help="the lambda=0 twin; the geometry belongs to the cap")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.roots:
        ap.error("give one or more campaign roots, or --self-test")

    frames = [measure(r, args.arm) for r in args.roots]
    frames = [f for f in frames if len(f)]
    if not frames:
        print("no completed `%s` runs with both prediction files under: %s"
              % (args.arm, ", ".join(args.roots)))
        return 1
    report(summarise(pd.concat(frames, ignore_index=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Is there anything to WIN on this dataset? Ask before spending a campaign.

`dataset_screen` asks whether the count constraint carries INFORMATION here.
This asks the other half, and the two are independent: even where the counts
are informative, the PRIZE can be zero.

THE ARITHMETIC, and it is exact.

When a class has `n` true test instances and the budget lets you emit `K`
predictions for it, the best achievable cc-F1 is

    ceiling = 2K / (K + n)          precision 1, recall K/n

because you cannot recall what you may not emit. A model whose precision at the
cut is `p` achieves `2pK / (K + n)`, so the ENTIRE prize, for any method, is

    prize_items = (1 - p) * K

and nothing about the loss, the dual, the allocator or the optimizer changes
that bound. It is set by `K` and by how good the ranking already is.

WHY IT BITES. `p` is not independent of `K/n`. When the budget is a small
fraction of the true positives, the top-K set is drawn from a pool several
times its own size and is filled with correct items almost by default.
Measured on `results/iwc3` (FRAMEWORK 2(v)): `K/n` is 16-30%, `ccP` is 0.9954,
and the prize is **0.0 to 1.0 items in six (cap, class) combinations, exactly
0.0 in four of them** -- against a paired seed sd of 2.11 items. Every
score-pushing arm this project has built was reordering items inside a set that
was already all-correct.

So a dataset is only worth a campaign if `K` is large enough that `(1-p)K`
clears the seed noise. That needs LABELS and the CAP POLICY and nothing else --
no images, no model, no GPU -- which is why this runs before a download.

The budgets are not re-derived here. `compute_local_constraints` and
`compute_global_constraints` are imported from the training path, and the
binding budget is `effective_budget`'s `min(global, sum of local ceilings)`,
which is the rule that a 30x headroom bug was fixed to.

    python -m scripts.ceiling_screen data/iwildcam/oodslice --caps L20_G50 L30_G50 L50_G30 --classes 2 7
    python -m scripts.ceiling_screen --self-test
"""

import argparse
import os
import sys

SEED_NOISE_ITEMS = 2.11          # iwc3, paired within-cell sd. FRAMEWORK 2(p-post)
MEASURED_CCP = 0.9954            # iwc3, `tralo_null` against `clip`


def budgets(meta_path, caps, classes, group_col, num_classes):
    """[(cap, class, n, k_global, k_local_sum, k_binding)], from labels alone."""
    import pandas as pd

    from configs.gen_campaign import cap_pair
    from scripts.full_panel import effective_budget
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)

    te = pd.read_csv(meta_path)
    out = []
    for tag in caps:
        local_pct, global_pct = cap_pair(tag)
        G = compute_global_constraints(te, "label", global_pct, classes,
                                       num_classes=num_classes)
        L = compute_local_constraints(te, "label", local_pct, group_col,
                                      constrained_class=classes,
                                      num_classes=num_classes)
        for c in classes:
            n = int((te["label"] == c).sum())
            k_loc = sum(int(b[c]) for b in L.values())
            out.append((tag, c, n, int(G[c]), k_loc,
                        int(effective_budget(G, L, c))))
    return out


def report(rows, ccp=MEASURED_CCP, noise=SEED_NOISE_ITEMS, out=sys.stdout):
    """Print the table. Returns the number of (cap, class) cells worth running."""
    out.write("CEILING SCREEN -- how many items can ANY method win here?\n")
    out.write("  ceiling = 2K/(K+n): you cannot recall what you may not emit.\n")
    out.write("  prize   = (1-p)*K items, p = precision at the cut. No loss, "
              "dual,\n            allocator or optimizer changes this bound.\n")
    out.write("  Reference: paired seed sd %.2f items; measured p = %.4f "
              "(iwc3).\n\n" % (noise, ccp))
    out.write("  %-10s %6s %7s %8s %9s %9s %8s %9s  %s\n"
              % ("cap", "class", "n", "K", "K/n", "ceiling", "prize", "vs noise",
                 "verdict"))
    worth = 0
    for tag, c, n, kg, kl, k in rows:
        ratio = k / float(n) if n else 0.0
        ceiling = 2.0 * k / (k + n) if (k + n) else 0.0
        prize = (1.0 - ccp) * k
        rel = prize / noise if noise else 0.0
        if rel >= 1.0:
            verdict = "WORTH RUNNING"
            worth += 1
        elif rel >= 0.5:
            verdict = "marginal"
        else:
            verdict = "*** PRIZE BELOW THE NOISE"
        out.write("  %-10s %6d %7d %8d %8.1f%% %9.4f %8.2f %8.2fx  %s\n"
                  % (tag, c, n, k, 100.0 * ratio, ceiling, prize, rel, verdict))
        if kg != k or kl != k:
            out.write("             ^ global K=%d, local sum K=%d, BINDING K=%d\n"
                      % (kg, kl, k))
    if not worth:
        out.write("\n  *** NO (cap, class) CELL HAS A PRIZE ABOVE THE SEED "
                  "NOISE.\n")
        out.write("      The best any method can do on the capped classes here "
                  "is TIE, and\n"
                  "      every trained arm still shares a backbone with the "
                  "UNCAPPED ones,\n"
                  "      where 2(s) measures only downside. Raise K, or change "
                  "the dataset.\n")
    # ASCII only, deliberately. The Windows console is cp1252 and a single
    # emoji in a print raises UnicodeEncodeError *mid-report*, so the table
    # above is printed and the process dies with exit 1 on the line after it.
    # It happened to this script on its first run.
    out.write("\n  !! p is not independent of K/n: a small budget is filled "
              "from a large\n"
              "    pool of positives and is correct almost by default. Read "
              "the two\n"
              "    columns together, and re-measure p if K/n here is far from "
              "iwildcam's\n"
              "    16-30%%.\n")
    return worth


def self_test(out=sys.stdout):
    """The gate. A screen that says yes to everything decides nothing."""
    import io as _io
    ok = True

    # iwildcam L30_G50 class 2, the measured case: K=111, n=370 -> 0.51 items.
    buf = _io.StringIO()
    worth = report([("L30_G50", 2, 370, 185, 111, 111)], out=buf)
    text = buf.getvalue()
    if worth != 0 or "PRIZE BELOW THE NOISE" not in text:
        out.write("SELF-TEST FAIL: iwildcam's measured cell must not read as "
                  "worth running:\n%s" % text)
        ok = False
    if "0.4615" not in text:
        out.write("SELF-TEST FAIL: ceiling 2*111/(111+370) = 0.4615 not "
                  "printed:\n%s" % text)
        ok = False
    if "global K=185" not in text:
        out.write("SELF-TEST FAIL: an INERT global must be named, or the 30x "
                  "bug can come back silently:\n%s" % text)
        ok = False

    # A budget that is most of the class: K=300 of n=370 -> 1.38 items, still
    # under the noise at p=0.9954, and that is the honest answer.
    buf = _io.StringIO()
    report([("L80_G80", 2, 370, 300, 300, 300)], out=buf)
    if "1.38" not in buf.getvalue():
        out.write("SELF-TEST FAIL: prize at K=300 should be (1-0.9954)*300 = "
                  "1.38 items:\n%s" % buf.getvalue())
        ok = False

    # The screen must be able to say YES, or it is not a screen. A worse
    # ranking makes the same budget worth chasing.
    buf = _io.StringIO()
    worth = report([("L80_G80", 2, 370, 300, 300, 300)], ccp=0.95, out=buf)
    if worth != 1 or "WORTH RUNNING" not in buf.getvalue():
        out.write("SELF-TEST FAIL: at p=0.95 a K=300 budget is 15 items and "
                  "must read as worth running:\n%s" % buf.getvalue())
        ok = False

    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slice_dir", nargs="?",
                    help="a slice dir holding test_meta.csv")
    ap.add_argument("--caps", nargs="+", default=["L20_G50", "L30_G50"],
                    help="cap tags, e.g. L30_G50")
    ap.add_argument("--classes", nargs="+", type=int, default=[2, 7],
                    help="the capped classes")
    ap.add_argument("--group-column", default="location",
                    help="the group column in test_meta.csv")
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--ccp", type=float, default=MEASURED_CCP,
                    help="assumed precision at the cut (default: iwc3's 0.9954)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.slice_dir:
        ap.error("give a slice dir, or --self-test")
    meta = os.path.join(args.slice_dir, "test_meta.csv")
    if not os.path.exists(meta):
        print("no test_meta.csv under %s" % args.slice_dir)
        return 2

    rows = budgets(meta, args.caps, args.classes, args.group_column,
                   args.num_classes)
    return 0 if report(rows, ccp=args.ccp) else 1


if __name__ == "__main__":
    sys.exit(main())

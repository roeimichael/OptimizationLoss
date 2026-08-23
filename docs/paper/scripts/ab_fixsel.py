"""Paired A/B for the checkpoint-degeneracy fix, on the DermMNIST cells.

The patched tree re-ran exactly the 48 Derm configs of headroom_b30_lrc0.0001
with the same seeds and the same cached warm-ups, so every run has a twin that
differs in one thing: whether a feasible-but-degenerate final model is replaced
by the best-filling satisfied checkpoint.

Reported per method, because the fix was applied to all three trained arms and
a fix that only helps TraLO is a different result from one that lifts the field.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

BASE = "results/headroom/headroom_b30_lrc0.0001"
FIX = "newdirections/arm_fixsel/results/fixsel/headroom_b30_lrc0.0001"
KEY = ["dataset", "cap", "model", "seed", "method"]
COLS = ["ccF1eq", "AP", "macroEq", "count_raw"]


def main():
    b = A.rows_for(BASE)
    f = A.rows_for(FIX)
    b = b[b.dataset == "dermmnist"]
    f = f[f.dataset == "dermmnist"]
    print("base derm runs %d   fixed derm runs %d" % (len(b), len(f)))

    d = b.merge(f, on=KEY, suffixes=("_base", "_fix"))
    print("paired %d" % len(d))
    if d.empty:
        return 1
    for c in COLS:
        d["d_" + c] = d[c + "_fix"] - d[c + "_base"]

    d["collapsed_base"] = d.count_raw_base < (d.K_base / 3.0)
    d["collapsed_fix"] = d.count_raw_fix < (d.K_fix / 3.0)
    print("\ncollapsed runs:  before %d   after %d"
          % (int(d.collapsed_base.sum()), int(d.collapsed_fix.sum())))

    print("\n" + "=" * 72)
    print("PAIRED DELTA (fixed - unpatched), DermMNIST")
    print("=" * 72)
    for m, g in d.groupby("method"):
        print("\n%s  (n=%d, %d collapsed before -> %d after)"
              % (m, len(g), int(g.collapsed_base.sum()), int(g.collapsed_fix.sum())))
        for c in COLS:
            v = g["d_" + c]
            print("   %-10s %+9.4f    improved %d/%d    (before %.4f -> after %.4f)"
                  % (c, v.mean(), int((v > 0).sum()), len(v),
                     g[c + "_base"].mean(), g[c + "_fix"].mean()))

    print("\n" + "=" * 72)
    print("ONLY THE RUNS THAT WERE COLLAPSED BEFORE THE FIX")
    print("=" * 72)
    s = d[d.collapsed_base]
    if s.empty:
        print("  none")
    else:
        for m, g in s.groupby("method"):
            print("\n%s  (n=%d)" % (m, len(g)))
            for c in COLS:
                print("   %-10s %+9.4f   (%.4f -> %.4f)"
                      % (c, g["d_" + c].mean(), g[c + "_base"].mean(),
                         g[c + "_fix"].mean()))

    print("\n" + "=" * 72)
    print("RANKING ON DERM AFTER THE FIX (trained arms only, ccF1eq)")
    print("=" * 72)
    t = f.groupby("method")[COLS].mean().round(4)
    t["rank"] = t["ccF1eq"].rank(ascending=False)
    print(t.to_string())
    piv = f.pivot_table(index=["cap", "model", "seed"], columns="method",
                        values="ccF1eq")
    if "tralo" in piv.columns:
        for ref in [c for c in piv.columns if c != "tralo"]:
            g = (piv["tralo"] - piv[ref]).dropna()
            print("  tralo vs %-14s %+0.4f   %d/%d seeds"
                  % (ref, g.mean(), int((g > 0).sum()), len(g)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

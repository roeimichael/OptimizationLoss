"""Stack every scored campaign and read the dual-vs-clipper ordering as a
function of REGIME, on the cells the campaigns share.

The design axes that move between the old corpus and the new one are:
  warmup epochs given to the CLIPPER   (50 / 30 / 10 / 1)
  warmup epochs given to the TRAINED arms (50 / 30 / 10 / 1)
  lr_constraint                        (5e-6 / 5e-5 / 1e-4)
  CE-saturation gate                   (on / off)
and the metric (shipped cc-F1 vs budget-equalized cc-F1).

Shared cells: dataset x model x cap present in every file being stacked.

    python paper/scripts/regime_ladder.py
"""
import os
import sys

import pandas as pd

FILES = [
    ("paper_final  w50 both, lrc5e-6, skipON", "out_paperfinal.csv", None),
    ("warmup_abl   w10 both, lrc5e-6, skipON", "out_warmup_ablation.csv", None),
    ("w1probe      w1  both, lrc5e-6, skipON", "out_warmup1_probe.csv",
     "out_warmup1_probe_s34.csv"),
    ("headroomB30  clip w30 / trained w1, lrc5e-6, skipON",
     "out_headroom_b30.csv", None),
    ("hr lrc5e-5   clip w30 / trained w1, lrc5e-5, skipON",
     "out_headroom_b30_lrc5e-05.csv", "out_headroom_b30.csv"),
    ("hr lrc1e-4   clip w30 / trained w1, lrc1e-4, skipON",
     "out_headroom_b30_lrc0.0001.csv", "out_headroom_b30.csv"),
    ("hr noceskip  clip w30 / trained w1, lrc1e-4, skipOFF",
     "out_headroom_b30_lrc0.0001_noceskip.csv", "out_headroom_b30.csv"),
]
D = "paper/scripts/"
DUAL = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]
TOL = 0.005
CAPS = ["L30_G30", "L50_G50"]
MODELS = ["MobileNetV3", "RegNetY400MF"]


def load(f2, f3):
    d = pd.read_csv(D + f2)
    if f3:
        c = pd.read_csv(D + f3)
        d = pd.concat([d, c[c.method.isin(CLIP)]], ignore_index=True)
    d = d[d.cap.isin(CAPS) & d.model.isin(MODELS)]
    return d


def cellstats(d, metrics):
    rows = []
    for (ds, mo, cap), g in d.groupby(CELL):
        r = {"dataset": ds, "model": mo, "cap": cap}
        for M in metrics:
            piv = g.pivot_table(index="seed", columns="method", values=M)
            hd = [m for m in DUAL if m in piv.columns]
            hc = [m for m in CLIP if m in piv.columns]
            if not hd or not hc:
                continue
            s = piv.dropna(subset=hd + hc)
            if s.empty:
                continue
            r[M] = (s[hd].max(axis=1) - s[hc].max(axis=1)).mean()
            r[M + "_dual"] = s[hd].max(axis=1).mean()
            r[M + "_clip"] = s[hc].max(axis=1).mean()
        # also raw per-method means, so the ladder shows the LEVELS not just gaps
        for m in DUAL + CLIP + ["tralo"]:
            sub = g[g.method == m]
            if len(sub):
                r["AP_" + m] = sub.AP.mean()
                r["cc_" + m] = sub.ccF1eq.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    metrics = ["ccF1adj", "ccF1eq", "AP", "macroEq"]
    print("=" * 118)
    print("DUAL minus CLIP, cells COUNTED, restricted to caps %s and models %s"
          % (CAPS, MODELS))
    print("=" * 118)
    print("%-52s %-6s %-24s %-24s %-24s %-24s"
          % ("regime", "cells", "ccF1adj (shipped)", "ccF1eq (equalized)",
             "AP (allocation-free)", "macroEq"))
    keep = {}
    for lbl, f2, f3 in FILES:
        if not os.path.exists(D + f2):
            print("%-52s MISSING %s" % (lbl, f2))
            continue
        d = load(f2, f3)
        t = cellstats(d, metrics)
        keep[lbl] = t
        line = "%-52s %-6d" % (lbl, len(t))
        for M in metrics:
            if M not in t.columns:
                line += " %-24s" % "-"
                continue
            v = t[M].dropna()
            line += " %+0.4f  D%d/C%d/T%-6d" % (
                v.mean(), int((v > TOL).sum()), int((v < -TOL).sum()),
                len(v) - int((v > TOL).sum()) - int((v < -TOL).sum()))
        print(line)

    print()
    print("=" * 118)
    print("LEVELS, not gaps: mean AP by method within each regime "
          "(AP needs no budget, no threshold)")
    print("=" * 118)
    print("%-52s %8s %8s %8s %8s %8s" %
          ("regime", "heurist", "danits", "fioretto", "hounie", "tralo"))
    for lbl, t in keep.items():
        cols = ["AP_heuristic", "AP_danits_lp", "AP_fioretto_ldf",
                "AP_hounie_rcl", "AP_tralo"]
        vals = [("%8.4f" % t[c].mean()) if c in t.columns else "       -"
                for c in cols]
        print("%-52s %s" % (lbl, " ".join(vals)))

    print()
    print("=" * 118)
    print("SAME, on dermmnist/MobileNetV3 alone (one cell family, no pooling "
          "across datasets)")
    print("=" * 118)
    print("%-52s %8s %8s %8s %8s %8s %8s" %
          ("regime", "AP_heur", "AP_fior", "AP_houn", "AP_tralo",
           "cc_heur", "cc_fior"))
    for lbl, t in keep.items():
        s = t[(t.dataset == "dermmnist") & (t.model == "MobileNetV3")]
        if s.empty:
            print("%-52s  (no dermmnist/MobileNetV3 cells)" % lbl)
            continue
        def g(c):
            return ("%8.4f" % s[c].mean()) if c in s.columns else "       -"
        print("%-52s %s %s %s %s %s %s" %
              (lbl, g("AP_heuristic"), g("AP_fioretto_ldf"), g("AP_hounie_rcl"),
               g("AP_tralo"), g("cc_heuristic"), g("cc_fioretto_ldf")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

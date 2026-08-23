"""Final discrimination: is the dual-vs-clipper flip a metric change, or a
change in which arm got the CE epochs?

Three checks:
  1. does the ordering flip when the METRIC changes but the corpus is held fixed?
  2. does the ordering flip when the WARM-UP PARITY changes but the metric is
     held fixed?
  3. does the per-cell ccF1eq gap track the per-cell AP gap? AP is allocation
     free, so if it does, the difference is a difference in the underlying
     score ranking, not in how the K budget is spent.

    python paper/scripts/final_decomp.py
"""
import os
import sys

import numpy as np
import pandas as pd

D = "paper/scripts/"
DUAL = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]


def cells(files, caps=None, models=None):
    d = pd.concat([pd.read_csv(D + f) for f in files], ignore_index=True)
    if caps:
        d = d[d.cap.isin(caps)]
    if models:
        d = d[d.model.isin(models)]
    rows = []
    for (ds, mo, cap), g in d.groupby(CELL):
        r = {"dataset": ds, "model": mo, "cap": cap}
        for M in ["ccF1adj", "ccF1eq", "AP", "macroEq"]:
            piv = g.pivot_table(index="seed", columns="method", values=M)
            hd = [m for m in DUAL if m in piv.columns]
            hc = [m for m in CLIP if m in piv.columns]
            if not hd or not hc:
                continue
            s = piv.dropna(subset=hd + hc)
            if s.empty:
                continue
            r["d" + M] = (s[hd].max(axis=1) - s[hc].max(axis=1)).mean()
        for m in DUAL + CLIP + ["tralo"]:
            sub = g[g.method == m]
            if len(sub):
                r["AP_" + m] = sub.AP.mean()
        # tralo vs clipper on AP, paired -- equal CE epochs in the noceskip arm
        piv = g.pivot_table(index="seed", columns="method", values="AP")
        hc = [m for m in CLIP if m in piv.columns]
        if "tralo" in piv.columns and hc:
            s = piv.dropna(subset=["tralo"] + hc)
            if not s.empty:
                r["dAP_tralo"] = (s["tralo"] - s[hc].max(axis=1)).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def line(lbl, t):
    o = "%-46s cells=%2d" % (lbl, len(t))
    for M in ["ccF1adj", "ccF1eq", "AP"]:
        c = "d" + M
        if c not in t.columns:
            o += "   %-22s" % "-"
            continue
        v = t[c].dropna()
        o += "   %s %+0.4f [D%d/C%d]" % (M, v.mean(), int((v > 0.005).sum()),
                                         int((v < -0.005).sum()))
    return o


def main():
    CAPS = ["L30_G30", "L50_G50"]
    MODELS = ["MobileNetV3", "RegNetY400MF"]

    print("=" * 118)
    print("CHECK 1+2: dual minus clipper.  D = dual-win cells, C = clipper-win "
          "cells (|delta| > 0.005)")
    print("=" * 118)
    old = [
        ("OLD paper_final       warmup 50 BOTH arms", ["out_paperfinal.csv"]),
        ("OLD paper_backbones   warmup 50 BOTH arms", ["out_paper_backbones.csv"]),
        ("OLD extra_robustness  warmup 50 BOTH arms", ["out_extra_robustness.csv"]),
        ("OLD warmup_ablation   warmup 10 BOTH arms", ["out_warmup_ablation.csv"]),
        ("OLD warmup1_probe     warmup  1 BOTH arms",
         ["out_warmup1_probe.csv", "out_warmup1_probe_s34.csv"]),
    ]
    for lbl, fs in old:
        if not all(os.path.exists(D + f) for f in fs):
            print("%-46s MISSING" % lbl)
            continue
        print(line(lbl, cells(fs)))
    print()
    new = [
        ("NEW headroom lrc5e-6  clip w30 / trained w1",
         ["out_headroom_b30.csv"]),
        ("NEW headroom lrc5e-5  clip w30 / trained w1",
         ["out_headroom_b30_lrc5e-05.csv", "out_headroom_b30.csv"]),
        ("NEW headroom lrc1e-4  clip w30 / trained w1",
         ["out_headroom_b30_lrc0.0001.csv", "out_headroom_b30.csv"]),
        ("NEW headroom noceskip clip w30 / trained w1",
         ["out_headroom_b30_lrc0.0001_noceskip.csv", "out_headroom_b30.csv"]),
    ]
    newt = {}
    for lbl, fs in new:
        t = cells(fs, CAPS, MODELS)
        newt[lbl] = t
        print(line(lbl, t))

    print()
    print("  same 12 cells as the NEW campaign, cut out of the OLD corpus:")
    oldm = cells(["out_paperfinal.csv"], CAPS, MODELS)
    print(line("  OLD paper_final, matched cells", oldm))

    print()
    print("=" * 118)
    print("CHECK 3: does the ccF1eq gap track the ALLOCATION-FREE AP gap?")
    print("=" * 118)
    for lbl, t in list(newt.items()) + [("OLD paper_final matched cells", oldm)]:
        s = t.dropna(subset=["dAP", "dccF1eq"])
        if len(s) < 3:
            continue
        r = np.corrcoef(s.dAP, s.dccF1eq)[0, 1]
        b = np.polyfit(s.dAP, s.dccF1eq, 1)[0]
        print("  %-46s  pearson r = %+0.3f   slope = %+0.2f   (n=%d cells)"
              % (lbl, r, b, len(s)))

    print()
    print("=" * 118)
    print("CHECK 4: TraLO vs clipper on AP in the one arm where the trained "
          "model really did keep a live CE loop")
    print("=" * 118)
    for lbl, t in newt.items():
        if "dAP_tralo" not in t.columns:
            continue
        v = t.dAP_tralo.dropna()
        print("  %-46s  tralo AP minus clipper AP = %+0.4f  (%d cells)"
              % (lbl, v.mean(), len(v)))
    v = oldm.dAP_tralo.dropna()
    print("  %-46s  tralo AP minus clipper AP = %+0.4f  (%d cells)"
          % ("OLD paper_final matched cells", v.mean(), len(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

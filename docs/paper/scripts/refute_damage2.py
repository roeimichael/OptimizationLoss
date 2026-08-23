"""Follow-up: is Spearman(fill, AP)=+0.702 anything but 'hounie collapses on derm'?
Plus the cross-dataset damage-vs-win inversion, on the atomic cell."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
pd.set_option("display.width", 260)

t = pd.read_csv("paper/scripts/out_refute_damage.csv")
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]

print("=" * 118)
print("A. LEAVE-ONE-METHOD-OUT on the derm Spearman(fill, AP) = +0.702")
print("=" * 118)
d = t[t.dataset == "dermmnist"]
r = spearmanr(d["fill"], d["AP"])
print("  all 3 methods       n=%d  rho=%+.3f  p=%.3g" % (len(d), r.correlation, r.pvalue))
for drop in TRAINED:
    g = d[d.method != drop]
    r = spearmanr(g["fill"], g["AP"])
    print("  drop %-13s  n=%d  rho=%+.3f  p=%.3g" % (drop, len(g), r.correlation,
                                                     r.pvalue))
print("\n  fill RANGE per method on derm (do the methods even overlap?):")
for m in TRAINED:
    g = d[d.method == m]
    print("    %-13s fill min %.2f  max %.2f   AP min %.3f max %.3f"
          % (m, g["fill"].min(), g["fill"].max(), g["AP"].min(), g["AP"].max()))
lo = d[d.method == "tralo"]["fill"].min()
hi = d[d.method != "tralo"]["fill"].max()
print("    tralo's LOWEST fill %.2f vs the duals' HIGHEST %.2f -> overlap: %s"
      % (lo, hi, "yes" if lo < hi else "NONE (fill IS the method label)"))

print("\n  cluster-aware check: method-mean fill vs method-mean AP, n=3 clusters")
mm = d.groupby("method")[["fill", "AP", "ccF1eq"]].mean().reindex(TRAINED)
print(mm.to_string(float_format=lambda x: "%.4f" % x))
print("  -> the '48 independent runs, p=2.7e-08' is 3 points; Spearman on n=3 has")
print("     a minimum two-sided p of 0.333.")

print()
print("=" * 118)
print("B. FULL bin table, both columns. The claim quoted 4 of 6 bins, AP only.")
print("=" * 118)
b = pd.cut(d["fill"], [0, 0.25, 0.5, 0.75, 1.0, 1.5, 10],
           labels=["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"])
tb = d.assign(bin=b).groupby("bin", observed=False).agg(
    n=("AP", "size"), AP=("AP", "mean"), ccF1eq=("ccF1eq", "mean"),
    pct_hounie=("method", lambda s: 100.0 * (s == "hounie_rcl").mean()),
    pct_tralo=("method", lambda s: 100.0 * (s == "tralo").mean())).reset_index()
print(tb.to_string(index=False, float_format=lambda x: "%.4f" % x))
ap = tb["AP"].to_numpy()
cc = tb["ccF1eq"].to_numpy()
print("  AP monotone increasing across all 6 bins?      %s" % bool(np.all(np.diff(ap) > 0)))
print("  ccF1eq monotone increasing across all 6 bins?  %s" % bool(np.all(np.diff(cc) > 0)))
print("  ccF1eq over the top three bins (0.5-0.75 -> >1.5): %.4f -> %.4f -> %.4f -> %.4f"
      % (cc[2], cc[3], cc[4], cc[5]))
print("  i.e. in the metric the claim is explaining, MORE fill goes with LESS ccF1eq")
print("  once you are past fill=0.75.")

print()
print("=" * 118)
print("C. CROSS-DATASET INVERSION: TraLO's own AP damage vs whether it wins.")
print("   Atomic cell = (dataset, backbone, cap) over 4 seeds. Cells counted.")
print("=" * 118)
cellrows = []
for (ds, mo, cap), g in t.groupby(CELL):
    p = g.pivot_table(index="seed", columns="method", values="ccF1eq")
    pa = g.pivot_table(index="seed", columns="method", values="dAP")
    pf = g.pivot_table(index="seed", columns="method", values="fill")
    dcc = p["tralo"] - p[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    cellrows.append({"dataset": ds, "model": mo, "cap": cap,
                     "tralo_AP_damage": pa["tralo"].mean(),
                     "tralo_fill": pf["tralo"].mean(),
                     "duals_min_fill": pf[["fioretto_ldf", "hounie_rcl"]].min(axis=1).mean(),
                     "dcc_vs_best_dual": dcc.mean(), "win": dcc.mean() > 0})
C = pd.DataFrame(cellrows).sort_values(CELL)
print(C.to_string(index=False, float_format=lambda x: "%.4f" % x))
print()
for ds, g in C.groupby("dataset"):
    print("  %-12s  mean TraLO AP damage %+.4f   cells won %d/%d"
          % (ds, g["tralo_AP_damage"].mean(), int(g["win"].sum()), len(g)))
r = spearmanr(C["tralo_AP_damage"], C["dcc_vs_best_dual"])
print("\n  Spearman(TraLO's own AP damage, TraLO's ccF1eq margin) over the 12 cells")
print("    = %+.3f  p=%.3g   -> %s" % (r.correlation, r.pvalue,
      "MORE damage goes with a BIGGER win" if r.correlation < 0 else "as claimed"))
print("  TraLO damages the CE model %.1fx MORE on derm (where it wins 4/4) than on"
      % (C[C.dataset == "dermmnist"]["tralo_AP_damage"].mean() /
         C[C.dataset == "octmnist"]["tralo_AP_damage"].mean()))
print("  oct (where it loses 4/4).")

print()
print("=" * 118)
print("D. NO COLLAPSE ON OCT/TISSUE at all, yet the win pattern still splits.")
print("=" * 118)
print(t.groupby(["dataset", "method"])[["fill"]].agg(["mean", "min"])
      .to_string(float_format=lambda x: "%.3f" % x))
print("\n  runs with fill < 0.5 ('collapsed'), by dataset x method:")
print(pd.crosstab(t[t.fill < 0.5]["dataset"], t[t.fill < 0.5]["method"]).to_string())
print("  -> every collapsed run in the campaign is a DERM run, and 39 of them are duals.")

print()
print("=" * 118)
print("E. Sanity on the epoch trap (sparse TraLO log). max(Epoch) vs len(df).")
print("=" * 118)
print(t.groupby(["dataset", "method"])[["max_epoch", "log_rows"]].mean()
      .to_string(float_format=lambda x: "%.1f" % x))
print("  TraLO derm: max(Epoch)=30 but only 8.8 logged rows. Reading len(df) as the")
print("  epoch count would have said 'TraLO trained 9 epochs vs the duals' 22-26', i.e.")
print("  it would have MANUFACTURED an early-stopping explanation for the damage gap.")
print("  Checked: TraLO ran the FULL 30 on every derm run (min=max=30), and the DUALS")
print("  are the ones that stop early on derm (fioretto 21.4, hounie 25.4 of 28).")
print("  So more constraint epochs went to the method that damaged LEAST -> the damage")
print("  ordering is not a dose-response in constraint exposure either.")

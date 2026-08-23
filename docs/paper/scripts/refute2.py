"""Does the collapse story actually EXPLAIN the derm-wins / oct-loses split?

Tests, in order of decisiveness:
  T1  Is the headline metric even sensitive to under-fill?  (ccF1eq re-allocates
      to exactly K from the probability ranking, so a model that argmax-predicts
      the class for 9 samples is scored on its top-K ranking, not on its count.)
  T2  Which dual is the comparator?  The headline is tralo - MAX(duals).  If the
      collapsed dual is never the max, its collapse cannot make the win.
  T3  Does the outcome track ranking (AP) rather than feasibility/fill?
  T4  TISSUEMNIST: same feasibility rate as oct, same non-collapse as oct, but
      the outcome SPLITS by backbone.  One axis, two outcomes.
  T5  Who is actually infeasible as scored, per dataset?  (polarity check)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"
DUALS = ["fioretto_ldf", "hounie_rcl"]
TRAINED = ["tralo"] + DUALS
CELL = ["dataset", "model", "cap"]
fl = lambda x: "%.4f" % x  # noqa: E731


def hdr(s):
    print("\n" + "=" * 112)
    print(s)
    print("=" * 112)


d = A.rows_for(ROOT)
d = d[d.method.isin(TRAINED)].copy()
d["fill"] = d["count_raw"] / d["K"]
mine = pd.read_csv("paper/scripts/out_refute_collapse.csv")
key = ["dataset", "model", "cap", "seed", "method"]
d = d.merge(mine[key + ["final_excess_recomputed", "final_feasible"]], on=key)
print("scored %d runs; cross-check count_raw agrees: %s"
      % (len(d), bool((d.fill.notna()).all())))

hdr("T1. IS ccF1eq SENSITIVE TO UNDER-FILL AT ALL?\n"
    "    analyze_headroom.equalize() ranks by P[:,cls] and takes exactly K.\n"
    "    So the count the model argmaxes is DISCARDED before scoring.\n"
    "    Empirical: within each cell, correlate fill with ccF1eq over the 12 runs.")
rows = []
for (ds, mo, cap), g in d.groupby(CELL):
    r = spearmanr(g["fill"], g["ccF1eq"])
    r2 = spearmanr(g["fill"], g["AP"])
    rows.append({"dataset": ds, "model": mo, "cap": cap, "n": len(g),
                 "rho_fill_ccF1eq": r.correlation, "p_cc": r.pvalue,
                 "rho_fill_AP": r2.correlation, "p_AP": r2.pvalue})
t1 = pd.DataFrame(rows)
print(t1.to_string(index=False, float_format=fl))
print("\n  cells where fill significantly predicts ccF1eq (p<0.05): %d / %d"
      % (int((t1.p_cc < 0.05).sum()), len(t1)))
print("  and the constrained-count that IS scored is identical for every method:")
print("   ", d.groupby("method")["count"].describe()[["min", "max"]].to_string()
      .replace("\n", "\n    "))

hdr("T2. WHICH DUAL IS THE COMPARATOR (tralo - MAX(duals))?")
piv = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq").dropna()
fillp = d.pivot_table(index=CELL + ["seed"], columns="method", values="fill")
out = []
for (ds, mo, cap), g in piv.groupby(CELL):
    f = fillp.loc[g.index]
    argmax_is_fior = (g["fioretto_ldf"] >= g["hounie_rcl"])
    mx = g[DUALS].max(axis=1)
    out.append({
        "dataset": ds, "model": mo, "cap": cap,
        "vMax": (g.tralo - mx).mean(),
        "vFior": (g.tralo - g.fioretto_ldf).mean(),
        "vHoun": (g.tralo - g.hounie_rcl).mean(),
        "best_is_fioretto": "%d/4" % int(argmax_is_fior.sum()),
        "fill_of_best": np.where(argmax_is_fior, f["fioretto_ldf"],
                                 f["hounie_rcl"]).mean(),
        "fill_houn": f["hounie_rcl"].mean(),
    })
t2 = pd.DataFrame(out).sort_values(CELL)
print(t2.to_string(index=False, float_format=fl))
print("\n  On dermmnist the collapsed dual (hounie, fill 0.14-0.48) is the max in")
print("  %s of 16 seed-comparisons; the comparator that sets the headline is the"
      % int(sum(int(x.split('/')[0]) for x in t2[t2.dataset == 'dermmnist'].best_is_fioretto)
            and 16 - sum(int(x.split('/')[0]) for x in
                         t2[t2.dataset == 'dermmnist'].best_is_fioretto)))
print("  UNcollapsed one.")

hdr("T3. DOES THE OUTCOME TRACK RANKING (AP) INSTEAD?  per cell, tralo - max(duals)")
apiv = d.pivot_table(index=CELL + ["seed"], columns="method", values="AP").dropna()
rows = []
for (ds, mo, cap), g in apiv.groupby(CELL):
    c = piv.loc[g.index]
    rows.append({"dataset": ds, "model": mo, "cap": cap,
                 "dAP_vs_max": (g.tralo - g[DUALS].max(axis=1)).mean(),
                 "dccF1eq_vs_max": (c.tralo - c[DUALS].max(axis=1)).mean()})
t3 = pd.DataFrame(rows).sort_values(CELL)
t3["same_sign"] = np.sign(t3.dAP_vs_max) == np.sign(t3.dccF1eq_vs_max)
print(t3.to_string(index=False, float_format=fl))
print("\n  sign agreement AP vs ccF1eq: %d / %d cells   Spearman %.3f"
      % (int(t3.same_sign.sum()), len(t3),
         spearmanr(t3.dAP_vs_max, t3.dccF1eq_vs_max).correlation))

hdr("T4. TISSUEMNIST vs OCTMNIST on the claim's own two axes")
agg = d[d.method.isin(DUALS)].groupby("dataset").agg(
    dual_feasible_as_scored=("final_feasible", "sum"),
    n=("final_feasible", "size"), mean_fill=("fill", "mean")).reset_index()
print(agg.to_string(index=False, float_format=fl))
print("\n  outcome (tralo - max duals, ccF1eq), cells counted:")
for ds, g in t2.groupby("dataset"):
    print("    %-12s  WIN %d cells   LOSS %d cells   (deltas %s)"
          % (ds, int((g.vMax > 0.005).sum()), int((g.vMax < -0.005).sum()),
             " ".join("%+.4f" % v for v in g.vMax)))

hdr("T5. POLARITY: who is infeasible AS SCORED (raw count vs K, before post-hoc)?")
p = d.groupby(["dataset", "method"]).agg(
    feasible=("final_feasible", "sum"), n=("final_feasible", "size"),
    mean_fill=("fill", "mean"), mean_excess=("final_excess_recomputed", "mean")
).reset_index()
print(p.to_string(index=False, float_format=fl))

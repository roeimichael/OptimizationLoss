"""Classify + stress-test. K recomputed from true labels, independent of taxonomy.py."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.training.constraints import compute_global_constraints  # noqa: E402
from src.utils.constants import UNLIMITED  # noqa: E402

D = pd.read_csv("paper/scripts/out_redo_tax.csv")
CELL = ["dataset", "model", "cap"]

Ks = []
for _, r in D.iterrows():
    cfg = json.load(open(os.path.join(r.path, "config.json")))
    fr = pd.read_csv(os.path.join(r.path, "final_predictions_raw.csv"))
    lp, gp = cfg["constraint"]
    ncls = len([c for c in fr.columns if c.startswith("Prob_Class_")])
    df = pd.DataFrame({"label": fr["True_Label"].to_numpy(int)})
    G = compute_global_constraints(df, "label", gp, constrained_class=[int(r.cls)],
                                   num_classes=ncls)
    Ks.append(int(G[int(r.cls)]) if G[int(r.cls)] < UNLIMITED else -1)
D["K"] = Ks
assert (D.K > 0).all()

# reference labels
T = pd.read_csv("paper/scripts/out_taxonomy.csv")
T["key"] = T.path
D["key"] = D.path
ref = dict(zip(T.key, T.klass))
refK = dict(zip(T.key, T.K))
refn = dict(zip(T.key, T.n_sat))
refE = dict(zip(T.key, T.epochs_run))

print("K disagreements vs out_taxonomy.csv:",
      int(sum(refK[k] != v for k, v in zip(D.key, D.K))))
print("n_sat disagreements:",
      int(sum(refn[k] != v for k, v in zip(D.key, D.n_sat))))
print("epochs_run disagreements:",
      int(sum(refE[k] != v for k, v in zip(D.key, D.E))))


def classify(s_str, count_raw, K, tail, coll_div):
    s = np.array([int(c) for c in s_str], dtype=int)
    if s.sum() == 0:
        return "NEVER_SAT"
    if coll_div is not None and count_raw < K / float(coll_div):
        return "COLLAPSE"
    held = bool(len(s) >= tail and s[-tail:].all())
    if not held:
        return "SAT_THEN_DRIFT"
    if int(((s[:-1] == 1) & (s[1:] == 0)).sum()) >= 1:
        return "OSC_THEN_LOCK"
    return "LOCK_AND_HOLD"


D["klass"] = [classify(r.s, r.count_raw, r.K, 5, 3.0) for _, r in D.iterrows()]
print("\n=== MY re-derived populations (tail=5, collapse=K/3) ===")
print(D.klass.value_counts().to_string())
print("\nlabel disagreements vs out_taxonomy.csv:",
      int(sum(ref[k] != v for k, v in zip(D.key, D.klass))))
print("\n", pd.crosstab(D.method, D.klass).to_string())

print("\n" + "=" * 90)
print("SENSITIVITY: number of the 144 labels that CHANGE vs the (tail=5, K/3) baseline")
print("=" * 90)
base = D.klass.tolist()
grid = []
for tail in [3, 4, 5, 6, 7]:
    for cd in [2.0, 2.5, 3.0, 4.0, None]:
        lab = [classify(r.s, r.count_raw, r.K, tail, cd) for _, r in D.iterrows()]
        ch = sum(a != b for a, b in zip(base, lab))
        vc = pd.Series(lab).value_counts()
        grid.append({"tail": tail, "collapse": ("none" if cd is None else "K/%g" % cd),
                     "changed": ch, "pct_changed": 100.0 * ch / 144,
                     "NEVER_SAT": vc.get("NEVER_SAT", 0),
                     "SAT_THEN_DRIFT": vc.get("SAT_THEN_DRIFT", 0),
                     "LOCK_AND_HOLD": vc.get("LOCK_AND_HOLD", 0),
                     "COLLAPSE": vc.get("COLLAPSE", 0),
                     "OSC_THEN_LOCK": vc.get("OSC_THEN_LOCK", 0)})
G = pd.DataFrame(grid)
print(G.to_string(index=False))

print("\n" + "=" * 90)
print("THE INFERRED-TAIL KNOB: what happens if TraLO's unlogged break epoch is NOT")
print("reconstructed (i.e. take the log at face value)?")
print("=" * 90)
D2 = D.copy()


def strip_inferred(r):
    return r.s[:-1] if r.inferred else r.s


D2["s2"] = [strip_inferred(r) for _, r in D2.iterrows()]
lab2 = [classify(r.s2, r.count_raw, r.K, 5, 3.0) for _, r in D2.iterrows()]
print("runs with an inferred tail epoch: %d" % int(D.inferred.sum()))
print("labels changed if not reconstructed: %d" % sum(a != b for a, b in zip(base, lab2)))
print(pd.Series(lab2).value_counts().to_string())

print("\n" + "=" * 90)
print("EPOCH BUDGET actually run, by method (the compute confound)")
print("=" * 90)
print(D.groupby("method").agg(E_mean=("E", "mean"), E_min=("E", "min"),
                              E_max=("E", "max"), rows=("nrows", "mean"),
                              inferred=("inferred", "sum")).to_string())

print("\n" + "=" * 90)
print("CELL-COUNTED VIEW: 12 cells x 3 methods. Is the class a property of the RUN")
print("or of the CELL? For each (cell,method) the 4 seeds get how many distinct labels?")
print("=" * 90)
g = D.groupby(CELL + ["method"]).klass.agg(lambda x: "|".join(sorted(set(x))))
n = D.groupby(CELL + ["method"]).klass.nunique()
print("of the 36 (cell,method) groups, %d are label-homogeneous across their 4 seeds"
      % int((n == 1).sum()))
print(g.to_string())

print("\n" + "=" * 90)
print("MODAL CLASS PER CELL (counting cells, not runs): 12 cells")
print("=" * 90)
for (ds, mo, cp), gg in D.groupby(CELL):
    print("  %-12s %-14s %-8s K=%3d  %s" % (ds, mo, cp, gg.K.iloc[0],
                                            dict(gg.klass.value_counts())))
D.to_csv("paper/scripts/out_redo_tax2.csv", index=False)

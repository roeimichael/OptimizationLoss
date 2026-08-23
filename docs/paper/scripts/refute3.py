"""Three probes the claim has to survive.

P1  Decompose the 'infeasibility' on oct/tissue into GLOBAL excess vs LOCAL
    (per-group) excess.  The claim measures collapse on the GLOBAL count but
    measures feasibility on global+local.  If oct's residual excess is local,
    the two halves of the claim are about different constraints.
P2  Remove the between-method confound: within (cell, method), does a seed that
    collapses harder rank worse?  Centered within cell AND method, pooled.
P3  TISSUEMNIST per backbone: same feasibility, same fill, opposite outcome.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
from src.utils.constants import UNLIMITED                                  # noqa: E402
from src.training.constraints import (compute_global_constraints,          # noqa: E402
                                      compute_local_constraints)
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


rows = []
for cfgp in sorted(glob.glob(ROOT + "/**/config.json", recursive=True)):
    cfg = json.load(open(cfgp))
    d0 = os.path.dirname(cfgp)
    raw = os.path.join(d0, "final_predictions_raw.csv")
    if not os.path.exists(raw):
        continue
    t = pd.read_csv(raw)
    dc = cfg.get("dataset_config") or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    ncls = int(dc["num_classes"])
    y = t["True_Label"].to_numpy(int)
    rp = t["Predicted_Label"].to_numpy(int)
    g = (t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns
         else np.zeros(len(t), int))
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g})
    G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                   num_classes=ncls)
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[cls],
                                  num_classes=ncls)
    if G[cls] >= UNLIMITED:
        continue
    K = int(G[cls])
    cnt = int((rp == cls).sum())
    gexc = max(0, cnt - K)
    lexc = 0
    for gid, b in (L or {}).items():
        if b[cls] < UNLIMITED:
            gc = int(((rp == cls) & (g == gid)).sum())
            lexc += max(0, gc - int(b[cls]))
    rows.append({"dataset": cfg["dataset_mode"], "model": cfg["model_name"],
                 "cap": cfg["constraint_tag"], "method": cfg["methodology"],
                 "seed": (cfg.get("hyperparams") or {})["seed"],
                 "K": K, "count_raw": cnt, "fill": cnt / K,
                 "glob_exc": gexc, "loc_exc": lexc, "n_groups": len(L or {}),
                 "glob_feasible": int(gexc == 0),
                 "both_feasible": int(gexc == 0 and lexc == 0)})
E = pd.DataFrame(rows)
E = E[E.method.isin(TRAINED)]

hdr("P1. GLOBAL vs LOCAL: where does the 'infeasibility' live?  (duals only, n=32/ds)")
du = E[E.method.isin(DUALS)]
agg = du.groupby("dataset").agg(
    n=("glob_exc", "size"), n_groups=("n_groups", "first"),
    glob_feasible=("glob_feasible", "sum"), both_feasible=("both_feasible", "sum"),
    mean_glob_exc=("glob_exc", "mean"), mean_loc_exc=("loc_exc", "mean"),
    mean_fill=("fill", "mean")).reset_index()
print(agg.to_string(index=False, float_format=fl))
print("\n  -> On octmnist the GLOBAL cap (the one 'collapse' is measured against)")
print("     is met by %d/32 dual runs; the residual excess is %.0f%% local."
      % (agg[agg.dataset == "octmnist"].glob_feasible.iloc[0],
         100 * agg[agg.dataset == "octmnist"].mean_loc_exc.iloc[0]
         / max(1e-9, agg[agg.dataset == "octmnist"].mean_loc_exc.iloc[0]
               + agg[agg.dataset == "octmnist"].mean_glob_exc.iloc[0])))
print("\n  per method:")
print(du.groupby(["dataset", "method"]).agg(
    glob_feas=("glob_feasible", "sum"), both_feas=("both_feasible", "sum"),
    g_exc=("glob_exc", "mean"), l_exc=("loc_exc", "mean"),
    fill=("fill", "mean")).reset_index().to_string(index=False, float_format=fl))

hdr("P2. WITHIN (cell x method): does a harder-collapsing SEED rank worse?\n"
    "    fill and AP both centered inside (dataset,model,cap,method); n=16 per\n"
    "    dataset-method.  Kills the between-method confound (hounie is both the\n"
    "    lowest-fill and the lowest-quality dual on derm).")
d = A.rows_for(ROOT)
d = d[d.method.isin(TRAINED)].copy()
d["fill"] = d["count_raw"] / d["K"]
key = ["dataset", "model", "cap", "method"]
d["fill_c"] = d["fill"] - d.groupby(key)["fill"].transform("mean")
d["AP_c"] = d["AP"] - d.groupby(key)["AP"].transform("mean")
d["cc_c"] = d["ccF1eq"] - d.groupby(key)["ccF1eq"].transform("mean")
print("  pooled over cells, within method:")
for (ds, m), gg in d.groupby(["dataset", "method"]):
    r1 = spearmanr(gg.fill_c, gg.AP_c)
    r2 = spearmanr(gg.fill_c, gg.cc_c)
    print("    %-12s %-14s n=%2d   rho(fill,AP)=%+.3f p=%.2f   rho(fill,ccF1eq)=%+.3f p=%.2f"
          % (ds, m, len(gg), r1.correlation, r1.pvalue, r2.correlation, r2.pvalue))
print("\n  and with method effects removed but methods pooled (n=48/ds):")
for ds, gg in d.groupby("dataset"):
    r1 = spearmanr(gg.fill_c, gg.AP_c)
    r2 = spearmanr(gg.fill_c, gg.cc_c)
    print("    %-12s n=%d  rho(fill,AP)=%+.3f p=%.3f   rho(fill,ccF1eq)=%+.3f p=%.3f"
          % (ds, len(gg), r1.correlation, r1.pvalue, r2.correlation, r2.pvalue))
print("\n  (compare: WITHOUT centering, i.e. letting method identity in)")
for ds, gg in d.groupby("dataset"):
    r = spearmanr(gg.fill, gg.AP)
    print("    %-12s rho(fill,AP)=%+.3f p=%.2g" % (ds, r.correlation, r.pvalue))

hdr("P3. TISSUEMNIST BY BACKBONE: the claim's two axes vs the outcome")
piv = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq").dropna()
res = []
for (ds, mo, cap), gg in piv.groupby(CELL):
    sub = E[(E.dataset == ds) & (E.model == mo) & (E.cap == cap)
            & (E.method.isin(DUALS))]
    res.append({"dataset": ds, "model": mo, "cap": cap,
                "dual_fill": sub.fill.mean(),
                "dual_feas_glob": "%d/8" % int(sub.glob_feasible.sum()),
                "dual_feas_both": "%d/8" % int(sub.both_feasible.sum()),
                "vMax_ccF1eq": (gg.tralo - gg[DUALS].max(axis=1)).mean()})
R = pd.DataFrame(res).sort_values(CELL)
print(R.to_string(index=False, float_format=fl))

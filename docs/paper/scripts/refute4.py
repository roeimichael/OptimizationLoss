"""Final: is collapse NECESSARY or SUFFICIENT for the outcome?
Plus a schema-trap audit of TraLO's sparse log (never infer epochs from len(df)).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"
DUALS = ["fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
fl = lambda x: "%.4f" % x  # noqa: E731

d = A.rows_for(ROOT)
d = d[d.method.isin(["tralo"] + DUALS)].copy()
d["fill"] = d["count_raw"] / d["K"]
cc = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq").dropna()
fi = d.pivot_table(index=CELL + ["seed"], columns="method", values="fill")

print("=" * 112)
print("PER-SEED: identity and FILL of the dual that actually sets the headline")
print("(headline = tralo - max(fioretto, hounie), so only the max matters)")
print("=" * 112)
rows = []
for (ds, mo, cap), g in cc.groupby(CELL):
    f = fi.loc[g.index]
    isf = g["fioretto_ldf"] >= g["hounie_rcl"]
    bfill = np.where(isf, f["fioretto_ldf"], f["hounie_rcl"])
    rows.append({"dataset": ds, "model": mo, "cap": cap,
                 "comparator": "fior %d/4, houn %d/4" % (int(isf.sum()), int((~isf).sum())),
                 "fill_of_comparator": bfill.mean(),
                 "fill_min_seed": bfill.min(), "fill_max_seed": bfill.max(),
                 "delta_ccF1eq": (g.tralo - g[DUALS].max(axis=1)).mean()})
R = pd.DataFrame(rows).sort_values("fill_of_comparator")
R["tralo"] = np.where(R.delta_ccF1eq > 0, "WIN", "LOSS")
print(R.to_string(index=False, float_format=fl))

print()
print("=" * 112)
print("IS COLLAPSE NECESSARY?  (a TraLO win where the comparator did NOT collapse)")
print("=" * 112)
nec = R[(R.tralo == "WIN") & (R.fill_of_comparator >= 0.9)]
print(nec.to_string(index=False, float_format=fl) if len(nec) else "  none")

print()
print("=" * 112)
print("IS COLLAPSE SUFFICIENT?  (a TraLO loss where the comparator DID under-fill)")
print("=" * 112)
suf = R[(R.tralo == "LOSS") & (R.fill_of_comparator < 0.9)]
print(suf.to_string(index=False, float_format=fl) if len(suf) else "  none")

print()
print("=" * 112)
print("THE INVERSION PAIR (same backbone, same cap, matched):")
print("=" * 112)
pr = R[(R.model == "MobileNetV3") & (R.cap == "L50_G50")]
print(pr.to_string(index=False, float_format=fl))

print()
print("=" * 112)
print("SCHEMA-TRAP AUDIT: TraLO's log is sparse; epochs from Epoch.max(), not len(df)")
print("=" * 112)
tr = []
for cfgp in sorted(glob.glob(ROOT + "/**/config.json", recursive=True)):
    cfg = json.load(open(cfgp))
    if cfg.get("methodology") != "tralo":
        continue
    p = os.path.join(os.path.dirname(cfgp), "training_log.csv")
    if not os.path.exists(p):
        continue
    lg = pd.read_csv(p)
    ec = "Epoch" if "Epoch" in lg.columns else "epoch"
    e = pd.to_numeric(lg[ec], errors="coerce").dropna()
    tr.append({"dataset": cfg["dataset_mode"], "rows_in_file": len(lg),
               "n_numeric_rows": len(e), "epoch_max": float(e.max())})
T = pd.DataFrame(tr)
print(T.groupby("dataset").agg(n=("epoch_max", "size"),
                               mean_rows_in_file=("rows_in_file", "mean"),
                               mean_epoch_max=("epoch_max", "mean"),
                               min_epoch_max=("epoch_max", "min"),
                               max_epoch_max=("epoch_max", "max")
                               ).to_string(float_format=fl))
print("\n  If anyone had used len(df) as the epoch count for TraLO they would have")
print("  read ~%.0f epochs instead of ~%.0f." % (T.rows_in_file.mean(), T.epoch_max.mean()))

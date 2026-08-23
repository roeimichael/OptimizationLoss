"""Per-cell fill = evaluated model's raw constrained-class count / K, per method.
Grounds the 'under-fill is the log signature' lead. Never pools cells.
    python paper/scripts/fill_map.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
TR = ["tralo", "fioretto_ldf", "hounie_rcl"]

for camp in ["results/headroom/headroom_b30_lrc0.0001_noceskip",
             "results/headroom/headroom_b30_lrc0.0001"]:
    d = A.rows_for(camp)
    d = d[d.method.isin(TR)].copy()
    cl = A.rows_for("results/headroom/headroom_b30")
    cl = cl[cl.method == "heuristic"].copy()
    d = pd.concat([d, cl], ignore_index=True)
    d["fill"] = d["count_raw"] / d["K"]
    print("=" * 88)
    print(camp)
    print("=" * 88)
    p = d.pivot_table(index=CELL, columns="method", values="fill")
    print(p.to_string(float_format=lambda x: "%.3f" % x))
    print()

# dose-response: within-cell deviation of ccF1eq / AP vs fill bin, study campaign
d = A.rows_for("results/headroom/headroom_b30_lrc0.0001_noceskip")
d = d[d.method.isin(TR)].copy()
d["fill"] = d["count_raw"] / d["K"]
for m in ["ccF1eq", "AP"]:
    if m in d.columns:
        d["d_" + m] = d[m] - d.groupby(CELL)[m].transform("mean")
bins = [0, 0.33, 0.60, 0.85, 1.10, 1.25, 99]
d["bin"] = pd.cut(d["fill"], bins)
cols = ["d_ccF1eq"] + (["d_AP"] if "d_AP" in d.columns else [])
print("=" * 88)
print("DOSE-RESPONSE  fill bin -> within-cell deviation (study campaign, 144 trained runs)")
print("=" * 88)
g = d.groupby("bin", observed=True).agg(n=("fill", "size"), **{c: (c, "mean") for c in cols})
print(g.to_string(float_format=lambda x: "%+.4f" % x))

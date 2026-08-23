"""Was there a better-filling checkpoint to restore, per method?

The degeneracy fix restores the highest-filling SATISFIED epoch. If a run was
degenerate from its first satisfied epoch onward there is nothing better to
restore, and a zero delta is correct behaviour rather than a fix that quietly
skipped that arm. So for every collapsed run: compare the best count reached at
any satisfied epoch against the count it actually ended on.

NOTE the logs carry repeated header rows (one per phase), so every numeric
column is coerced and the header rows dropped.
"""
import glob, json, os
import pandas as pd

ROOT = "newdirections/arm_fixsel/results/fixsel/headroom_b30_lrc0.0001"
rows = []
for cfg in glob.glob(ROOT + "/**/config.json", recursive=True):
    c = json.load(open(cfg))
    dc = c.get("dataset_config", {}) or {}
    cc = dc.get("constrained_class")
    cls = int(cc[0] if isinstance(cc, (list, tuple)) else cc)
    p = os.path.join(os.path.dirname(cfg), "training_log.csv")
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    hard, lim, sat = "Hard_Class%d" % cls, "Limit_Class%d" % cls, "Global_Satisfied"
    if hard not in d.columns:
        continue
    for col in [hard, lim, sat]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[hard])
    if not len(d):
        continue
    K = float(d[lim].dropna().iloc[0]) if lim in d.columns else float("nan")
    h = d[hard].to_numpy(float)
    s = d[sat].to_numpy() if sat in d.columns else None
    hs = h[s == 1] if s is not None and (s == 1).any() else None
    rows.append({"method": c["methodology"], "epochs": len(d), "K": K,
                 "final": h[-1], "max_any": h.max(),
                 "best_satisfied": (hs.max() if hs is not None else float("nan")),
                 "n_satisfied": int((s == 1).sum()) if s is not None else 0})
d = pd.DataFrame(rows)
d["collapsed"] = d["final"] < d["K"] / 3.0
d["restorable"] = d["best_satisfied"] - d["final"]
print("runs %d" % len(d))
print("\nALL RUNS, per method")
print(d.groupby("method")[["epochs", "K", "final", "max_any", "best_satisfied",
                           "n_satisfied", "restorable"]].mean().round(1).to_string())
print("\nCOLLAPSED RUNS ONLY -- was a better SATISFIED checkpoint available?")
c = d[d.collapsed]
print("collapsed n = %d" % len(c))
if len(c):
    print(c.groupby("method")[["K", "final", "max_any", "best_satisfied",
                               "n_satisfied", "restorable"]]
          .agg(["mean", "max"]).round(1).to_string())

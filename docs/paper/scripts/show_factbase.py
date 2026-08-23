"""Print the fact base + a consistency check against the established stratified
numbers. Read-only; consumes paper/scripts/out_factbase*.csv."""
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
F = pd.read_csv("paper/scripts/out_factbase.csv")
R = pd.read_csv("paper/scripts/out_factbase_perrun.csv")

# Ceiling: with exactly K predicted positives and T true positives, F1 <= 2K/(K+T)
F["ccF1eq_ceiling"] = 2 * F["K"] / (F["K"] + F["n_true_cls"])
F["pct_of_ceiling"] = F["ccF1eq"] / F["ccF1eq_ceiling"]

for camp, g in F.groupby("campaign"):
    print("\n" + "=" * 150)
    print("CAMPAIGN %s" % camp)
    print("=" * 150)
    print(g[["dataset", "model", "cap", "method", "n_seeds", "K", "clip_raw",
             "ccF1eq", "ccF1eq_ceiling", "pct_of_ceiling", "AP", "macroEq",
             "count_raw", "count_adj", "sat", "n_collapsed"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

print("\n" + "=" * 150)
print("CONSISTENCY CHECK vs established stratified result: tralo - best(duals), ccF1eq, per cell")
print("=" * 150)
CELL = ["dataset", "cap", "model", "seed"]
for camp, g in R.groupby("campaign"):
    piv = g.pivot_table(index=CELL, columns="method", values="ccF1eq")
    if "tralo" not in piv.columns:
        continue
    have = [m for m in ["fioretto_ldf", "hounie_rcl"] if m in piv.columns]
    s = piv.dropna(subset=["tralo"]).copy()
    s["ref"] = s[have].max(axis=1)
    s["d"] = s["tralo"] - s["ref"]
    t = s.reset_index().groupby(["dataset", "model", "cap"])["d"].agg(["mean", "size"])
    print("\n--- %s ---" % camp)
    print(t.to_string(float_format=lambda x: "%+.4f" % x))

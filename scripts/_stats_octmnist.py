"""OctMNIST 4-seed verdict: per-tightness seed-avg grid + paired Wilcoxon
(TraLO vs each baseline). Dedup by (tight,method,seed) BEFORE pairing to avoid
the n-inflation bug. Pairs on matched (tight,seed)."""
import glob
import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
    HAVE = True
except Exception:
    HAVE = False

ROOTS = sorted(glob.glob("results/pending_runs/octmnist_s*"))
ORDER = ["heuristic", "danits_lp", "fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]
KEYS = {"acc": "Accuracy", "f1m": "F1 (Macro)", "ccf1": "F1_Class2",
        "ccrec": "Recall_Class2", "flips": "Flips Required", "sat": "Raw All Satisfied"}

rows = []
for root in ROOTS:
    seed = root.split("octmnist_s")[1]
    for f in glob.glob(f"{root}/MobileNetV3/*/*/seed_*/evaluation_metrics.csv"):
        parts = f.split("/")
        rec = {"seed": seed, "tight": parts[-4], "method": parts[-3]}
        m = pd.read_csv(f).set_index("Metric")["Value"]
        for k, mk in KEYS.items():
            rec[k] = float(m[mk]) if mk in m.index else float("nan")
        rows.append(rec)

df = pd.DataFrame(rows)
# dedup: average any accidental duplicate (tight,method,seed)
df = df.groupby(["tight", "method", "seed"], as_index=False)[list(KEYS)].mean()
print(f"cells: {len(df)}  seeds: {sorted(df.seed.unique())}  "
      f"tightness: {sorted(df.tight.unique())}  "
      f"(per cell n_seeds = {df.groupby(['tight','method']).size().unique()})\n")

for tight in sorted(df.tight.unique()):
    sub = df[df.tight == tight]
    agg = sub.groupby("method")[list(KEYS)].mean().reindex(
        [m for m in ORDER if m in sub.method.unique()])
    print(f"===== {tight}  (seed-avg, n={sub.seed.nunique()}) =====")
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    bt = agg.loc[[m for m in ["fioretto_ldf", "hounie_rcl", "tralo_bounded"]
                  if m in agg.index], "ccf1"].max()
    bp = agg.loc[[m for m in ["heuristic", "danits_lp"] if m in agg.index], "ccf1"].max()
    t = agg.loc["tralo", "ccf1"]
    print(f"  --> TraLO cc-F1={t:.3f} | best other-trained={bt:.3f} (Δ{t-bt:+.3f}) | "
          f"best post-hoc={bp:.3f} (Δ{t-bp:+.3f})\n")

# ---- paired Wilcoxon: TraLO vs each baseline, pooled over (tight,seed) ----
print("===== Paired Wilcoxon (TraLO vs baseline), pooled over (tight,seed) =====")
k = ["tight", "seed"]
ta = df[df.method == "tralo"][k + list(KEYS)]
for base in ["heuristic", "danits_lp", "fioretto_ldf", "hounie_rcl", "tralo_bounded"]:
    bb = df[df.method == base][k + list(KEYS)]
    mg = ta.merge(bb, on=k, suffixes=("_t", "_b"))
    print(f"\n  TraLO vs {base}  (n={len(mg)} matched pairs)")
    for col in ["ccf1", "ccrec", "f1m", "acc", "flips"]:
        d = (mg[f"{col}_t"] - mg[f"{col}_b"]).values
        w = "W" if (d > 0).sum() > (d < 0).sum() else ("L" if (d < 0).sum() > (d > 0).sum() else "T")
        p = wilcoxon(d).pvalue if (HAVE and np.any(d != 0)) else float("nan")
        print(f"    {col:6s} Δ={d.mean():+.4f}  win/tie/loss={int((d>0).sum())}/"
              f"{int((d==0).sum())}/{int((d<0).sum())}  p={p:.4f}  [{w}]")

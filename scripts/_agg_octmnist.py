"""Aggregate octmnist probe-smoke: drusen (class 2) cc-F1/recall + macro/acc/flips/sat,
per method x tightness, per seed and averaged. Decides if octmnist is a TraLO-favorable
3rd dataset (TraLO best among trained AND beats post-hoc on cc-F1)."""
import glob
import os
import pandas as pd

ROOTS = ["results/pending_runs/octmnist_s1", "results/pending_runs/octmnist_s2"]
ORDER = ["heuristic", "danits_lp", "fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]
KEYS = {"acc": "Accuracy", "f1m": "F1 (Macro)", "ccf1": "F1_Class2",
        "ccrec": "Recall_Class2", "flips": "Flips Required", "sat": "Raw All Satisfied"}

rows = []
for root in ROOTS:
    seed = root[-1]
    for f in glob.glob(f"{root}/MobileNetV3/*/*/seed_*/evaluation_metrics.csv"):
        parts = f.split("/")
        tight = parts[-4]
        method = parts[-3]
        m = pd.read_csv(f).set_index("Metric")["Value"]
        rec = {"seed": seed, "tight": tight, "method": method}
        for k, mk in KEYS.items():
            rec[k] = float(m[mk]) if mk in m.index else float("nan")
        rows.append(rec)

if not rows:
    print("no completed cells yet")
    raise SystemExit

df = pd.DataFrame(rows)
print(f"cells: {len(df)}  seeds: {sorted(df.seed.unique())}  "
      f"tightness: {sorted(df.tight.unique())}\n")

for tight in sorted(df.tight.unique()):
    print(f"===== {tight}  (seed-avg over {df[df.tight==tight].seed.nunique()} seeds) =====")
    sub = df[df.tight == tight]
    agg = sub.groupby("method")[list(KEYS)].mean()
    agg = agg.reindex([m for m in ORDER if m in agg.index])
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    trained = [m for m in ["fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]
               if m in agg.index]
    posthoc = [m for m in ["heuristic", "danits_lp"] if m in agg.index]
    if "tralo" in agg.index and trained:
        bt = agg.loc[[m for m in trained if m != "tralo"], "ccf1"].max()
        bp = agg.loc[posthoc, "ccf1"].max() if posthoc else float("nan")
        t = agg.loc["tralo", "ccf1"]
        print(f"  --> TraLO cc-F1={t:.3f} | best other-trained={bt:.3f} (Δ{t-bt:+.3f}) | "
              f"best post-hoc={bp:.3f} (Δ{t-bp:+.3f})")
    print()

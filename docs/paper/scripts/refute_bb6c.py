"""Decisive test: is the cep=1 count the WARM-UP state, or is it taken AFTER a
full CE epoch that runs at lr_constraint?

train.py:145-147 sets   for pg in optimizer.param_groups: pg["lr"] = lr_constraint
BEFORE the CE pass of the very first constraint epoch. lr_constraint is NOT in the
warm-up cache key. So:
  * if cep=1 were the warm-up state, hard_cep1 must be IDENTICAL between the
    lrc0.0001 and lrc5e-05 campaigns (same seed -> same cached warm-up).
  * if a CE epoch has already run at lr_constraint, they must DIFFER.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ROOTS = {
    "lrc1e-4_noceskip": "results/headroom/headroom_b30_lrc0.0001_noceskip",
    "lrc1e-4_ceskip":   "results/headroom/headroom_b30_lrc0.0001",
    "lrc5e-5":          "results/headroom/headroom_b30_lrc5e-05",
}


def cls_of(c):
    v = (c.get("dataset_config") or {}).get("constrained_class")
    return int(v[0] if isinstance(v, (list, tuple)) else v)


rows = []
for tag, root in ROOTS.items():
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cp)
        c = json.load(open(cp))
        hp = c.get("hyperparams") or {}
        if c.get("methodology") != "tralo":
            continue
        p = os.path.join(d, "training_log.csv")
        if not os.path.exists(p):
            continue
        lg = pd.read_csv(p)
        hc = "Hard_Class%d" % cls_of(c)
        if hc not in lg.columns:
            continue
        t = pd.DataFrame({"E": pd.to_numeric(lg["Epoch"], errors="coerce"),
                          "hard": pd.to_numeric(lg[hc], errors="coerce")}).dropna()
        t = t[t.E == 2]
        if not len(t):
            continue
        rows.append(dict(camp=tag, dataset=c.get("dataset_mode"),
                         model=c.get("model_name"), cap=c.get("constraint_tag"),
                         seed=hp.get("seed"), warmup=hp.get("warmup_epochs"),
                         lrc=hp.get("lr_constraint"), lr=hp.get("lr"),
                         hard_cep1=float(t.hard.iloc[0])))
R = pd.DataFrame(rows)
print("runs with an Epoch==2 row, by campaign:")
print(R.groupby("camp").agg(n=("seed", "size"), warmup=("warmup", "unique"),
                            lr=("lr", "unique"), lrc=("lrc", "unique")).to_string())

key = ["dataset", "model", "cap", "seed"]
P = R.pivot_table(index=key, columns="camp", values="hard_cep1")
have = [c for c in ["lrc1e-4_noceskip", "lrc1e-4_ceskip", "lrc5e-5"] if c in P.columns]
print("\nper-seed hard count at Epoch==2 (cep=1), by campaign:")
print(P[have].dropna().to_string())

print("\n" + "=" * 100)
if "lrc5e-5" in P.columns and "lrc1e-4_noceskip" in P.columns:
    q = P[["lrc1e-4_noceskip", "lrc5e-5"]].dropna()
    same = int(np.isclose(q["lrc1e-4_noceskip"], q["lrc5e-5"]).sum())
    print("lr_constraint 1e-4 vs 5e-5, SAME seed and SAME cached warm-up:")
    print("  identical hard_cep1 : %d of %d run pairs" % (same, len(q)))
    print("  mean |difference|   : %.1f counts"
          % (q["lrc1e-4_noceskip"] - q["lrc5e-5"]).abs().mean())
    print("  -> if cep=1 were the warm-up state these would be identical.")
if "lrc1e-4_ceskip" in P.columns and "lrc1e-4_noceskip" in P.columns:
    q2 = P[["lrc1e-4_noceskip", "lrc1e-4_ceskip"]].dropna()
    same2 = int(np.isclose(q2["lrc1e-4_noceskip"], q2["lrc1e-4_ceskip"]).sum())
    print("\ncontrol: same lr_constraint, only the CE-saturation gate differs")
    print("  (the gate cannot fire by epoch 1) -> expect IDENTICAL")
    print("  identical hard_cep1 : %d of %d run pairs" % (same2, len(q2)))

print("\n" + "=" * 100)
print("backbone ordering of hard_cep1 under lr_constraint=5e-5 (tissuemnist):")
if "lrc5e-5" in P.columns:
    z = R[(R.camp == "lrc5e-5") & (R.dataset == "tissuemnist")]
    if len(z):
        print(z.groupby("model").hard_cep1.agg(["mean", "min", "max", "size"]).to_string())

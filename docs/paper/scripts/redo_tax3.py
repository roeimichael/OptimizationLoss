"""Structural probes: is the taxonomy measuring dynamics, or the trainer's own
early-stop flag? And is NEVER_SAT a GLOBAL-cap statement or a LOCAL-group one?"""
import glob
import json
import os

import numpy as np
import pandas as pd

D = pd.read_csv("paper/scripts/out_redo_tax2.csv")
NCE = 29


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_log(p):
    t = pd.read_csv(p, dtype=str, low_memory=False)
    k = t.columns[0]
    return t[t[k] != k]


D["held"] = D.klass.isin(["LOCK_AND_HOLD", "OSC_THEN_LOCK"])
D["early_stop"] = D.E < NCE
print("=" * 92)
print("A. IS held_tail JUST THE TRAINER'S EARLY-STOP FLAG?")
print("   Both trainers break at stable_count >= 5. A run that reaches epoch 29")
print("   without breaking CANNOT have 5 satisfied tail epochs -- by construction.")
print("=" * 92)
print(pd.crosstab(D.held, D.early_stop, rownames=["held_tail(=LOCK|OSC)"],
                  colnames=["early_stopped (E<29)"]).to_string())
mism = D[D.held != D.early_stop]
print("\nmismatched runs: %d" % len(mism))
if len(mism):
    print(mism[["dataset", "model", "cap", "method", "seed", "E", "klass", "s"]].to_string(index=False))

print("\n" + "=" * 92)
print("B. NEVER_SAT: global cap vs local groups. TraLO only logs an epoch when the")
print("   JOINT constraint holds, or every 5th epoch. So on the every-5th rows we can")
print("   still read Global_Satisfied. How many NEVER_SAT tralo runs met the GLOBAL cap?")
print("=" * 92)
out = []
for _, r in D[(D.method == "tralo")].iterrows():
    t = read_log(os.path.join(r.path, "training_log.csv"))
    ep = num(t["Epoch"])
    t = t[ep.notna()]
    gs = num(t["Global_Satisfied"]).fillna(0).astype(int)
    ls = num(t["Local_Satisfied"]).fillna(0).astype(int)
    out.append({"path": r.path, "dataset": r.dataset, "model": r.model, "cap": r.cap,
                "seed": r.seed, "klass": r.klass, "n_rows": len(t),
                "n_G": int((gs == 1).sum()), "n_L": int((ls == 1).sum()),
                "n_joint": int(((gs == 1) & (ls == 1)).sum()),
                "n_G_not_L": int(((gs == 1) & (ls == 0)).sum()),
                "n_L_not_G": int(((gs == 0) & (ls == 1)).sum())})
T = pd.DataFrame(out)
ns = T[T.klass == "NEVER_SAT"]
print("tralo NEVER_SAT runs: %d" % len(ns))
print("  of those, runs with >=1 LOGGED epoch where Global_Satisfied==1: %d"
      % int((ns.n_G > 0).sum()))
print("  of those, runs with >=1 LOGGED epoch where Local_Satisfied==1 : %d"
      % int((ns.n_L > 0).sum()))
print("  G-satisfied-but-L-violated logged epochs, summed over NEVER_SAT runs: %d"
      % int(ns.n_G_not_L.sum()))
print("  L-satisfied-but-G-violated logged epochs, summed over NEVER_SAT runs: %d"
      % int(ns.n_L_not_G.sum()))
print("\n  per-dataset breakdown of tralo NEVER_SAT runs that DID meet the global cap:")
print(ns.groupby(["dataset", "model", "cap"]).agg(
    n=("path", "size"), any_global_ok=("n_G", lambda x: int((x > 0).sum())),
    rows_logged=("n_rows", "mean")).to_string())

print("\n  ALL tralo runs: logged epochs meeting G only / L only / joint")
print(T.groupby("klass").agg(n=("path", "size"), rows=("n_rows", "mean"),
                             G_only=("n_G_not_L", "sum"),
                             L_only=("n_L_not_G", "sum"),
                             joint=("n_joint", "sum")).to_string())

print("\n" + "=" * 92)
print("C. THE CENSORING PROBLEM. TraLO logs iff joint-satisfied or (epoch+1)%5==0.")
print("   So the JOINT trace is exact, but a GLOBAL-ONLY trace is observable on at")
print("   most 6 of 29 epochs. Rows per tralo log:")
print("=" * 92)
print(T.n_rows.describe().to_string())
print("\n  tralo runs whose log has ONLY the every-5th rows (i.e. 6 or 7 rows, zero")
print("  extra satisfied epochs): %d of 48" % int(((T.n_joint == 0)).sum()))

print("\n" + "=" * 92)
print("D. DUAL logs are dense (1 row/epoch), TraLO's are sparse. Median rows:")
print("=" * 92)
print(D.groupby("method").agg(rows=("nrows", "median"), E=("E", "median"),
                              n_sat=("n_sat", "median")).to_string())
print("\n  For the duals a global-only trace is ALSO unavailable: they log only")
print("  total_excess (global+local, clipped at 0) and all_satisfied.")

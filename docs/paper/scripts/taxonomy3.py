"""Third pass. Four things pass 2 left open:

  1. do all three methods really start from the identical warm-up state?
     (if yes, epoch-1 excess is a property of the CELL and SEED, not the method,
      and can be used as an exogenous predictor of the eventual class)
  2. does the starting distance predict which class a run lands in?
  3. within a fixed (cell, method) -- 4 seeds -- does a lower raw count go with
     a worse outcome? this is the only collapse control that is not confounded
     by method or cell.
  4. the CE-saturation gate fired on the duals but not on TraLO in this
     campaign. the sibling campaign headroom_b30_lrc0.0001 has the gate on for
     everyone; comparing TraLO across the two isolates it.

    python paper/scripts/taxonomy3.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import taxonomy as T                                                  # noqa: E402

CELL = ["dataset", "model", "cap"]
P = lambda t: print(t.to_string(float_format=lambda x: "%.4f" % x))    # noqa: E731

D = pd.read_csv("paper/scripts/out_taxonomy2.csv")
for c in ["ccF1eq", "AP"]:
    D["d_" + c] = D[c] - D.groupby(CELL)[c].transform("mean")

print("=" * 96)
print("1. DO ALL THREE METHODS START FROM THE SAME WARM-UP STATE?")
print("   epoch-1 excess, per (dataset, model, cap, seed), one column per method")
print("=" * 96)
piv = D.pivot_table(index=CELL + ["seed"], columns="method", values="ex_first")
piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
print("  %d (cell,seed) triples; epoch-1 excess identical across all 3 methods in %d of them"
      % (len(piv), int((piv.spread == 0).sum())))
print("  max disagreement: %.0f" % piv.spread.max())
P(piv.head(8))
cepiv = D.pivot_table(index=CELL + ["seed"], columns="method", values="ex_first")
D["ex1"] = D.groupby(CELL + ["seed"]).ex_first.transform("mean")

print("\n" + "=" * 96)
print("2. STARTING DISTANCE -> CLASS.  x1 = epoch-1 excess / K")
print("=" * 96)
D["x1K"] = D.ex1 / D.K
P(D.groupby("klass").agg(n=("path", "size"), x1_excess=("ex1", "mean"),
                         x1_over_K=("x1K", "mean"), K=("K", "mean")))
D["x1b"] = pd.cut(D.x1K, [0, 1, 2, 3, 5, 100],
                  labels=["<1", "1-2", "2-3", "3-5", ">5"])
print("\n  P(class) by starting-distance bin (rows sum to 1):")
ct = pd.crosstab(D.x1b, D.klass, normalize="index")
P(ct)
print("\n  counts:")
P(pd.crosstab(D.x1b, D.klass))
print("\n  epoch-1 excess/K per cell (a property of the cell+backbone, not the method):")
P(D.groupby(CELL).agg(x1_over_K=("x1K", "mean"), K=("K", "mean"),
                      never_sat_frac=("klass", lambda x: (x == "NEVER_SAT").mean())))
print("\n  correlation of x1/K with P(NEVER_SAT) across the 12 cells: %.3f"
      % D.groupby(CELL).agg(a=("x1K", "mean"),
                            b=("klass", lambda x: (x == "NEVER_SAT").mean())).corr().iloc[0, 1])

print("\n" + "=" * 96)
print("3. WITHIN (cell x method) -- 36 groups of 4 seeds -- does a lower raw")
print("   count go with a worse outcome? Values are centred inside each group,")
print("   so cell difficulty and method identity are both differenced out.")
print("=" * 96)
g = D.groupby(CELL + ["method"])
for c in ["count_raw", "ratio", "ccF1eq", "AP", "n_sat", "first_sat", "epochs_run"]:
    D["w_" + c] = D[c] - g[c].transform("mean")
sub = D.dropna(subset=["w_ratio", "w_ccF1eq"])
print("  n=%d runs in %d groups" % (len(sub), g.ngroups))
for x in ["w_ratio", "w_count_raw", "w_n_sat", "w_first_sat", "w_epochs_run"]:
    s = sub.dropna(subset=[x])
    print("    corr(%-14s, w_ccF1eq) = %+0.3f   corr(.., w_AP) = %+0.3f   n=%d"
          % (x, s[x].corr(s.w_ccF1eq), s[x].corr(s.w_AP), len(s)))
print("\n  same, restricted to hounie_rcl on dermmnist (where every collapse lives):")
s = sub[(sub.method == "hounie_rcl") & (sub.dataset == "dermmnist")]
print("    n=%d  corr(w_ratio, w_ccF1eq) = %+0.3f   corr(w_ratio, w_AP) = %+0.3f"
      % (len(s), s.w_ratio.corr(s.w_ccF1eq), s.w_ratio.corr(s.w_AP)))

print("\n" + "=" * 96)
print("4. PER-DATASET: within-cell correlation of raw/K with the outcome")
print("=" * 96)
for ds, gg in D.groupby("dataset"):
    a = gg.ratio - gg.groupby(CELL).ratio.transform("mean")
    print("  %-12s n=%d   corr(raw/K, ccF1eq)=%+0.3f   corr(raw/K, AP)=%+0.3f"
          % (ds, len(gg), a.corr(gg.d_ccF1eq), a.corr(gg.d_AP)))

print("\n" + "=" * 96)
print("5. THE CE-SATURATION GATE ACROSS SIBLING CAMPAIGNS")
print("=" * 96)
for root in ["results/headroom/headroom_b30_lrc0.0001_noceskip",
             "results/headroom/headroom_b30_lrc0.0001",
             "results/headroom/headroom_b30_lrc5e-05"]:
    n = {}
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cp))
        m = cfg.get("methodology")
        if m not in ("tralo", "fioretto_ldf", "hounie_rcl"):
            continue
        v = cfg["hyperparams"].get("enable_ce_skip", "ABSENT")
        n[(m, str(v))] = n.get((m, str(v)), 0) + 1
    print("  %-52s %s" % (os.path.basename(root), dict(sorted(n.items()))))

print("\n  TraLO on dermmnist, noceskip campaign vs the gate-on sibling:")
rows = []
for root, tag in [("results/headroom/headroom_b30_lrc0.0001_noceskip", "gate OFF for tralo"),
                  ("results/headroom/headroom_b30_lrc0.0001", "gate ON for everyone")]:
    R = T.__dict__  # reuse the scorers
    for cp in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        cfg = json.load(open(cp))
        if cfg.get("methodology") not in ("tralo", "fioretto_ldf", "hounie_rcl"):
            continue
        if cfg.get("dataset_mode") != "dermmnist":
            continue
        d = os.path.dirname(cp)
        sc = T.score_run(d, cfg)
        if sc is None:
            continue
        rows.append({"campaign": tag, "method": cfg["methodology"],
                     "model": cfg["model_name"], "cap": cfg["constraint_tag"],
                     "seed": cfg["hyperparams"]["seed"], "ccF1eq": sc["ccF1eq"],
                     "AP": sc["AP"], "ratio": sc["count_raw"] / sc["K"]})
C = pd.DataFrame(rows)
if not C.empty:
    P(C.pivot_table(index=["model", "cap", "method"], columns="campaign",
                    values=["ccF1eq", "ratio"]))
    print("\n  paired by (model,cap,method,seed): gate OFF minus gate ON")
    pv = C.pivot_table(index=["model", "cap", "method", "seed"],
                       columns="campaign", values="ccF1eq")
    if pv.shape[1] == 2:
        dlt = pv.iloc[:, 0] - pv.iloc[:, 1]
        P(dlt.groupby(level="method").agg(["count", "mean",
                                           lambda x: int((x > 0).sum())]))

print("\n" + "=" * 96)
print("6. THE ONE SAT_THEN_DRIFT RUN WITH count_adj/K = 0.56")
print("=" * 96)
D["adjK"] = D.count_adj / D.K
P(D.nsmallest(4, "adjK")[["dataset", "model", "cap", "method", "seed", "klass",
                          "count_raw", "count_adj", "K", "adjK", "ccF1eq", "flips"]])

print("\n" + "=" * 96)
print("7. DELIVERABLE TABLE: class -> outcome, with the method held fixed")
print("=" * 96)
tab = D.pivot_table(index="klass", columns="method",
                    values="d_ccF1eq", aggfunc=["mean", "size"])
P(tab)
print("\n  and per dataset, which class does each method occupy, and who wins the cell:")
for (ds, mo, cp), gg in D.groupby(CELL):
    best = gg.groupby("method").ccF1eq.mean().idxmax()
    kl = gg.groupby("method").klass.agg(lambda x: x.value_counts().idxmax())
    print("  %-11s %-13s %-8s  winner=%-13s | tralo=%-15s fioretto=%-15s hounie=%-15s"
          % (ds, mo, cp, best, kl.get("tralo", "-"), kl.get("fioretto_ldf", "-"),
             kl.get("hounie_rcl", "-")))

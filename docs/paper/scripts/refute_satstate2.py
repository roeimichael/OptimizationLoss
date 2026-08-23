"""Follow-up to refute_satstate.py.

(A) Freeze audit: prove that a dual's "satisfied state" epochs took zero
    optimizer steps, by showing the dual variable is bit-identical too.
(B) Matched-gate head-to-head: the same crosstab with enable_ce_skip=True for
    ALL THREE arms (campaign headroom_b30_lrc0.0001).
(C) A/B of TraLO alone across the two campaigns (the flag is the only diff).
(D) Cells counted, not pooled, + does a longer hold predict a better outcome?
"""
import glob
import json
import os

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]
NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"
GATE = "results/headroom/headroom_b30_lrc0.0001"
P = lambda x: print(x.to_string())  # noqa: E731
pd.set_option("display.width", 250)

print("=" * 100)
print("A. FREEZE AUDIT -- fioretto_ldf, every run whose max consecutive hold is 5")
print("   On a fully-frozen epoch the CE loop iterates over [], the constraint")
print("   block is gated off by has_work, and the dual update adds step*0.")
print("   If max_lambda_g is ALSO bit-identical, no parameter of any kind moved.")
print("=" * 100)
D = pd.read_csv("paper/scripts/out_refute_satstate.csv")
hits = D[(D.method == "fioretto_ldf") & (D.maxrun_inf >= 5)]
allfrozen = 0
rows = []
for _, r in hits.iterrows():
    t = pd.read_csv(os.path.join(r["path"], "training_log.csv"))
    t = t[t["epoch"].astype(str) != "epoch"]
    for c in ["epoch", "ce_loss", "constraint_loss", "total_excess",
              "all_satisfied", "max_lambda_g"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    tail = t.tail(5)
    frozen = (tail.ce_loss.isna().all() and (tail.constraint_loss == 0).all()
              and tail.max_lambda_g.nunique() == 1 and (tail.all_satisfied == 1).all())
    allfrozen += frozen
    rows.append({"cell": "%s/%s/%s/s%d" % (r["dataset"], r["model"], r["cap"], r["seed"]),
                 "last5_epochs": "%d-%d" % (tail.epoch.min(), tail.epoch.max()),
                 "CE_all_NaN": bool(tail.ce_loss.isna().all()),
                 "cstr_all_0": bool((tail.constraint_loss == 0).all()),
                 "lambda_constant": bool(tail.max_lambda_g.nunique() == 1),
                 "ZERO_STEPS_TAKEN": bool(frozen)})
P(pd.DataFrame(rows))
print("\n  fioretto runs whose entire 5-epoch 'hold' involved ZERO optimizer steps: %d of %d"
      % (allfrozen, len(hits)))

print("\n  same audit for hounie_rcl (CE off, but the constraint step keeps firing):")
h = D[(D.method == "hounie_rcl") & (D.maxrun_inf >= 5)]
n_ceoff = n_cstr = 0
for _, r in h.iterrows():
    t = pd.read_csv(os.path.join(r["path"], "training_log.csv"))
    t = t[t["epoch"].astype(str) != "epoch"]
    for c in ["ce_loss", "constraint_loss"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    tail = t.tail(5)
    n_ceoff += bool(tail.ce_loss.isna().all())
    n_cstr += bool((tail.constraint_loss > 0).all())
print("    runs whose whole 5-epoch hold had CE switched off : %d of %d" % (n_ceoff, len(h)))
print("    ...of which the ONLY live gradient was the constraint (pushes the")
print("       count further DOWN, away from the cap)                 : %d" % n_cstr)

print("\n" + "=" * 100)
print("B/C. THE FLAG, NOT THE METHOD.  --no-ce-skip only reached the tralo arm.")
print("=" * 100)
G = pd.read_csv("paper/scripts/out_refute_headroom_b30_lrc0.0001.csv")
for lbl, X in [("headroom_b30_lrc0.0001_noceskip  (gate OFF for tralo only)", D),
               ("headroom_b30_lrc0.0001           (gate ON for all three)", G)]:
    print("\n  %s" % lbl)
    ct = pd.crosstab(X.method, X.maxrun_inf)
    for c in range(6):
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[sorted(ct.columns)]
    P(ct)
    print("    never satisfies : " +
          "  ".join("%s=%d" % (m, int((g.n_sat_lit == 0).sum()))
                    for m, g in X.groupby("method")))

print("\n  the two campaigns' DUAL arms, digit for digit (they share the same configs):")
for m in ["fioretto_ldf", "hounie_rcl"]:
    a = D[D.method == m].maxrun_inf.value_counts().sort_index().to_dict()
    b = G[G.method == m].maxrun_inf.value_counts().sort_index().to_dict()
    print("    %-13s noceskip=%s   gateon=%s   identical=%s" % (m, a, b, a == b))
a = D[D.method == "tralo"].maxrun_inf.value_counts().sort_index().to_dict()
b = G[G.method == "tralo"].maxrun_inf.value_counts().sort_index().to_dict()
print("    %-13s noceskip=%s   gateon=%s   identical=%s" % ("tralo", a, b, a == b))

print("\n" + "=" * 100)
print("D. COUNT THE CELLS. mean max-consecutive-hold per (dataset,backbone,cap).")
print("=" * 100)
for lbl, X in [("gate OFF for tralo only", D), ("gate ON for all three", G)]:
    mp = X.pivot_table(index=CELL, columns="method", values="maxrun_inf", aggfunc="mean")
    mp["tralo_below_BOTH"] = (mp.tralo < mp.fioretto_ldf) & (mp.tralo < mp.hounie_rcl)
    mp["tralo_>=_one"] = ~mp["tralo_below_BOTH"]
    print("\n  %s" % lbl)
    P(mp)
    print("    cells where tralo holds strictly less than BOTH duals: %d of %d"
          % (int(mp.tralo_below_BOTH.sum()), len(mp)))

print("\n" + "=" * 100)
print("E. DOES A LONGER HOLD PREDICT A BETTER OUTCOME? (within-cell deviations)")
print("=" * 100)
F = pd.read_csv("paper/scripts/out_factbase_perrun.csv")
F = F[F.campaign.astype(str).str.contains("noceskip", na=False)]
key = CELL + ["method", "seed"]
M = D.merge(F[key + ["ccF1eq", "AP", "count_raw", "K"]], on=key, how="inner",
            suffixes=("", "_f"))
print("  merged runs: %d" % len(M))
if len(M):
    for c in ["ccF1eq", "AP"]:
        M["d_" + c] = M[c] - M.groupby(CELL)[c].transform("mean")
    P(M.groupby("method").agg(n=("path", "size"),
                              maxrun=("maxrun_inf", "mean"),
                              ccF1eq=("ccF1eq", "mean"),
                              corr_hold_vs_d_ccF1eq=("maxrun_inf",
                                                     lambda x: x.corr(M.loc[x.index, "d_ccF1eq"])),
                              corr_hold_vs_d_AP=("maxrun_inf",
                                                 lambda x: x.corr(M.loc[x.index, "d_AP"]))))
    print("\n  per-DATASET: tralo's mean hold vs tralo's ccF1eq margin over best dual")
    for ds, g in M.groupby("dataset"):
        piv = g.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
        piv = piv.dropna()
        marg = (piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1))
        hold = g[g.method == "tralo"].maxrun_inf.mean()
        dh = g[g.method != "tralo"].maxrun_inf.mean()
        print("    %-12s tralo hold=%.2f  dual hold=%.2f  tralo ccF1eq margin vs best dual=%+.4f"
              % (ds, hold, dh, marg.mean()))

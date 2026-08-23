"""Second pass on out_taxonomy.csv: robustness of the class boundaries, the
within-cell within-method contrasts that decide whether a class CAUSES a bad
outcome or merely co-occurs with a bad method, and the TraLO-only count trace.

    python paper/scripts/taxonomy2.py
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

D = pd.read_csv("paper/scripts/out_taxonomy.csv")
for c in ["ccF1eq", "AP", "macroEq"]:
    D["d_" + c] = D[c] - D.groupby(CELL)[c].transform("mean")

print("=" * 96)
print("A. MAX CONSECUTIVE SATISFIED EPOCHS -- is satisfaction a state or a transient?")
print("=" * 96)
runs = []
for _, r in D.iterrows():
    d = r["path"]
    cfg = json.load(open(os.path.join(d, "config.json")))
    dc = cfg["dataset_config"]
    cls = dc["constrained_class"]
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    if r["method"] == "tralo":
        tr, _ = T.trace_tralo(os.path.join(d, "training_log.csv"), cls)
    else:
        tr = T.trace_dual(os.path.join(d, "training_log.csv"))
    tr = tr.sort_values("e")
    E = int(r["epochs_run"])
    sm = dict(zip(tr["e"].astype(int), tr["sat"].astype(int)))
    if r["method"] == "tralo" and r["inferred_tail"]:
        sm[E] = 1
    s = np.array([sm.get(e, 0) for e in range(1, E + 1)])
    best = cur = 0
    for v in s:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    cnt = tr["count"].to_numpy(float)
    runs.append({"path": d, "max_run_sat": best,
                 "cnt_min": np.nanmin(cnt) if np.isfinite(cnt).any() else np.nan,
                 "cnt_max": np.nanmax(cnt) if np.isfinite(cnt).any() else np.nan,
                 "cnt_first": cnt[0] if len(cnt) else np.nan,
                 "cnt_last": cnt[-1] if len(cnt) else np.nan})
R = pd.DataFrame(runs)
D = D.merge(R, on="path")
P(D.groupby(["dataset", "method"]).agg(n=("path", "size"),
                                       n_sat=("n_sat", "mean"),
                                       max_run_sat=("max_run_sat", "mean"),
                                       max_run_sat_max=("max_run_sat", "max"),
                                       epochs=("epochs_run", "mean")))
print("\n  distribution of max consecutive satisfied epochs, by method")
P(pd.crosstab(D.method, D.max_run_sat))

print("\n" + "=" * 96)
print("B. CLASS x CAP, and CLASS x BACKBONE")
print("=" * 96)
P(pd.crosstab([D.cap], D.klass))
P(pd.crosstab([D.model], D.klass))

print("\n" + "=" * 96)
print("C. DOES COLLAPSE CAUSE THE BAD OUTCOME, OR IS IT JUST HOUNIE?")
print("   Contrast collapsed vs non-collapsed runs of the SAME method inside the")
print("   SAME cell. Only cells that contain both are usable.")
print("=" * 96)
D["coll"] = D.count_raw < D.K / 3.0
found = 0
for (ds, mo, cp, me), g in D.groupby(CELL + ["method"]):
    if g.coll.nunique() < 2:
        continue
    found += 1
    a, b = g[g.coll], g[~g.coll]
    print("  %-11s %-13s %-8s %-13s  collapsed n=%d ccF1=%.4f AP=%.4f raw=%.0f | "
          "not n=%d ccF1=%.4f AP=%.4f raw=%.0f | delta ccF1=%+.4f AP=%+.4f"
          % (ds, mo, cp, me, len(a), a.ccF1eq.mean(), a.AP.mean(), a.count_raw.mean(),
             len(b), b.ccF1eq.mean(), b.AP.mean(), b.count_raw.mean(),
             a.ccF1eq.mean() - b.ccF1eq.mean(), a.AP.mean() - b.AP.mean()))
print("  usable within-cell within-method contrasts: %d" % found)

print("\n  and the same for the 3 derm cells that contain collapsed runs, per seed:")
sub = D[(D.dataset == "dermmnist")].sort_values(CELL + ["method", "seed"])
P(sub[["model", "cap", "method", "seed", "klass", "count_raw", "K", "count_adj",
       "n_sat", "first_sat", "epochs_run", "ccF1eq", "AP", "flips"]])

print("\n" + "=" * 96)
print("D. IS THE OUTCOME MONOTONE IN raw/K, OR IS ONLY COLLAPSE BAD?")
print("=" * 96)
bins = [0, 1 / 3, 0.6, 0.85, 1.0, 1.25, 10]
lab = ["<0.33", "0.33-0.60", "0.60-0.85", "0.85-1.00", "1.00-1.25", ">1.25"]
D["rb"] = pd.cut(D.ratio, bins=bins, labels=lab)
P(D.groupby("rb", observed=True).agg(n=("path", "size"),
                                     d_ccF1eq=("d_ccF1eq", "mean"),
                                     d_AP=("d_AP", "mean"),
                                     ccF1eq=("ccF1eq", "mean"),
                                     flips=("flips", "mean")))

print("\n" + "=" * 96)
print("E. POST-HOC MASKING: does count_adj hide the collapse?")
print("=" * 96)
P(D.groupby("klass").agg(n=("path", "size"), K=("K", "mean"),
                         count_raw=("count_raw", "mean"),
                         count_adj=("count_adj", "mean"),
                         adj_over_K=("count_adj", "mean"),
                         flips=("flips", "mean")))
D["adjK"] = D.count_adj / D.K
print("\n  count_adj / K by class (post-hoc target utilisation):")
P(D.groupby("klass").adjK.agg(["mean", "min", "max"]))

print("\n" + "=" * 96)
print("F. CLASS-BOUNDARY SENSITIVITY (do the conclusions survive other thresholds?)")
print("=" * 96)
for thr in [2.0, 3.0, 4.0]:
    c = D.count_raw < D.K / thr
    print("  collapse at K/%.0f : n=%d  mean d_ccF1eq=%+.4f  d_AP=%+.4f  (rest %+.4f/%+.4f)"
          % (thr, c.sum(), D.d_ccF1eq[c].mean(), D.d_AP[c].mean(),
             D.d_ccF1eq[~c].mean(), D.d_AP[~c].mean()))
for k in [3, 5, 7]:
    held = D.apply(lambda r: r["n_sat"] > 0, axis=1)
    print("  tail window %d: (recomputation needs the trace; reported in G)" % k
          if False else "", end="")
print("\n  tail-window sensitivity, recomputed from the traces:")
for k in [3, 5, 7]:
    lab2 = []
    for _, r in D.iterrows():
        d = r["path"]
        cfg = json.load(open(os.path.join(d, "config.json")))
        cls = cfg["dataset_config"]["constrained_class"]
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        if r["method"] == "tralo":
            tr, _ = T.trace_tralo(os.path.join(d, "training_log.csv"), cls)
        else:
            tr = T.trace_dual(os.path.join(d, "training_log.csv"))
        E = int(r["epochs_run"])
        sm = dict(zip(tr["e"].astype(int), tr["sat"].astype(int)))
        if r["method"] == "tralo" and r["inferred_tail"]:
            sm[E] = 1
        s = np.array([sm.get(e, 0) for e in range(1, E + 1)])
        if s.sum() == 0:
            lab2.append("NEVER_SAT")
        elif r["count_raw"] < r["K"] / 3.0:
            lab2.append("COLLAPSE")
        elif not (len(s) >= k and s[-k:].all()):
            lab2.append("SAT_THEN_DRIFT")
        elif ((s[:-1] == 1) & (s[1:] == 0)).sum() >= 1:
            lab2.append("OSC_THEN_LOCK")
        else:
            lab2.append("LOCK_AND_HOLD")
    D["k%d" % k] = lab2
    print("   tail=%d  " % k, dict(pd.Series(lab2).value_counts()))
print("\n  agreement between tail=5 and tail=3 / tail=7 labels: %d / %d of 144"
      % ((D.klass == D.k3).sum(), (D.klass == D.k7).sum()))

print("\n" + "=" * 96)
print("G. TraLO ONLY -- the count trace the duals never write")
print("   (5-epoch resolution + every satisfied epoch, so dips between samples")
print("    that never reached satisfaction are invisible)")
print("=" * 96)
t = D[D.method == "tralo"]
P(t.groupby(CELL).agg(K=("K", "mean"), cnt_first=("cnt_first", "mean"),
                      cnt_min=("cnt_min", "mean"), cnt_max=("cnt_max", "mean"),
                      cnt_last=("cnt_last", "mean"),
                      raw_final=("count_raw", "mean"), n_sat=("n_sat", "mean")))
print("\n  TraLO runs whose logged count ever fell below K (i.e. crossed the cap): %d of %d"
      % (int((t.cnt_min < t.K).sum()), len(t)))
print("  TraLO runs whose logged count NEVER fell below K: %d"
      % int((t.cnt_min >= t.K).sum()))

print("\n" + "=" * 96)
print("H. CE-SATURATION GATE: who stopped training CE, and where")
print("=" * 96)
P(D.groupby(["dataset", "method"]).agg(n=("path", "size"),
                                       stopped_CE=("ce_skip_e", lambda x: int(x.notna().sum())),
                                       median_stop=("ce_skip_e", "median")))
D["cestop"] = D.ce_skip_e.notna()
print("\n  outcome of dual runs that stopped CE vs those that did not (within-cell dev):")
P(D[D.method != "tralo"].groupby(["dataset", "cestop"]).agg(
    n=("path", "size"), d_ccF1eq=("d_ccF1eq", "mean"), d_AP=("d_AP", "mean"),
    ratio=("ratio", "mean")))

D.to_csv("paper/scripts/out_taxonomy2.csv", index=False)
print("\nwrote paper/scripts/out_taxonomy2.csv")

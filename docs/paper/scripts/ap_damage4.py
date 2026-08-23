"""Nail down the mechanism, and separate it from two artifacts.

Built on out_ap_damage3.csv (576 runs, 4 campaigns) plus fresh log reads.

Checks, in order:
  A  Is `headroom_b30` an INERT-CONSTRAINT control?  lr_constraint=5e-6 is the
     learning rate of the WHOLE phase-2 optimizer (src/methodologies/tralo/
     train.py builds the optimizer with lr_constraint), not just of the
     constraint step, so that campaign is 1 epoch at 1e-4 + 29 epochs at 5e-6.
     If the three methods produce the SAME predictions there, the constraint
     did nothing and any dAP is an under-training artifact.
  B  Are the dual runs in lrc0.0001 and lrc0.0001_noceskip the same files?
     (`enable_ce_skip` is only set on tralo configs.)
  C  The CE-skip A/B on TraLO, paired by (dataset, backbone, cap, seed):
     same lr, same seeds, one flag.
  D  TraLO count-trajectory: how deep the count dived, and WHEN.
  E  Final existence-proof table with raw satisfaction attached.

    python paper/scripts/ap_damage4.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")

CELL = ["dataset", "model", "cap"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    d = pd.read_csv("paper/scripts/out_ap_damage3.csv")
    print("loaded %d runs, campaigns: %s" % (len(d), sorted(d.campaign.unique())))

    print("\n" + "=" * 96)
    print("A. IS headroom_b30 AN INERT-CONSTRAINT CONTROL?")
    print("   For each (dataset,backbone,cap,seed): spread of AP across the three")
    print("   trained methods.  Near zero => the three algorithms produced the same")
    print("   model => the constraint term had no effect at that lr_constraint.")
    print("=" * 96)
    for camp, g in d.groupby("campaign"):
        piv = g.pivot_table(index=CELL + ["seed"], columns="method", values="AP")
        piv = piv.dropna()
        spread = (piv.max(axis=1) - piv.min(axis=1))
        cnt = g.pivot_table(index=CELL + ["seed"], columns="method",
                            values="count_raw").dropna()
        cspread = (cnt.max(axis=1) - cnt.min(axis=1))
        print("  %-18s  n=%3d  AP spread across methods: mean %.5f  max %.5f"
              "   |  raw-count spread: mean %6.1f  max %5.0f"
              % (camp, len(piv), spread.mean(), spread.max(),
                 cspread.mean(), cspread.max()))
    print("\n  -> compare with the pure-CE reference spread (0.000 by construction)")
    print("     and with the clean campaign, where the methods must differ.")

    print("\n  b30 per-dataset mean dAP by method (should be identical if inert):")
    b = d[d.campaign == "b30_lrc5e-06"]
    print(b.pivot_table(index="dataset", columns="method", values="dAP").round(4).to_string())
    print("\n  b30: mean |count_raw - K|/K and n_sat_total  (did it ever reach the cap?)")
    b2 = b.copy()
    b2["gap"] = (b2.count_raw - b2.K).abs() / b2.K
    print(b2.groupby(["dataset", "method"])[["gap", "n_sat_total", "eval_c"]]
          .mean().round(3).to_string())

    print("\n" + "=" * 96)
    print("B. ARE THE DUAL RUNS SHARED BETWEEN lrc0.0001 AND lrc0.0001_noceskip?")
    print("=" * 96)
    r1 = "results/headroom/headroom_b30_lrc0.0001"
    r2 = "results/headroom/headroom_b30_lrc0.0001_noceskip"
    same, diff, missing = 0, 0, 0
    examples = []
    for p in sorted(glob.glob(r1 + "/**/final_predictions_raw.csv", recursive=True)):
        q = p.replace(r1, r2)
        if not os.path.exists(q):
            missing += 1
            continue
        meth = json.load(open(os.path.join(os.path.dirname(p), "config.json")))["methodology"]
        if md5(p) == md5(q):
            same += 1
        else:
            diff += 1
        if len(examples) < 3:
            examples.append((meth, md5(p) == md5(q)))
    print("  identical predictions: %d   different: %d   missing counterpart: %d"
          % (same, diff, missing))
    for camp in ["lrc1e-04_ceskip", "lrc1e-04_noskip"]:
        g = d[d.campaign == camp]
        print("  %-16s mean dAP by method: %s" % (
            camp, g.groupby("method")["dAP"].mean().round(4).to_dict()))
    byme = []
    for meth in ["fioretto_ldf", "hounie_rcl", "tralo"]:
        a = d[(d.campaign == "lrc1e-04_ceskip") & (d.method == meth)]
        c = d[(d.campaign == "lrc1e-04_noskip") & (d.method == meth)]
        m = a.merge(c, on=CELL + ["seed"], suffixes=("_skip", "_nos"))
        byme.append((meth, len(m), float((m.AP_skip - m.AP_nos).abs().max())))
    print("  max |AP difference| between the two campaigns, per method:")
    for meth, n, mx in byme:
        print("    %-14s n=%2d   max|dAP| = %.6f" % (meth, n, mx))

    print("\n" + "=" * 96)
    print("C. THE CE-SKIP A/B ON TraLO (paired: same dataset, backbone, cap, seed;")
    print("   same lr=lr_constraint=1e-4; the ONLY difference is enable_ce_skip)")
    print("=" * 96)
    a = d[(d.campaign == "lrc1e-04_ceskip") & (d.method == "tralo")]
    c = d[(d.campaign == "lrc1e-04_noskip") & (d.method == "tralo")]
    m = a.merge(c, on=CELL + ["seed"], suffixes=("_skip", "_nos"))
    m["gain"] = m.dAP_nos - m.dAP_skip
    print("  n paired runs = %d" % len(m))
    t = m.groupby(CELL)[["dAP_skip", "dAP_nos", "gain", "unspent_skip", "unspent_nos",
                         "traj_min_over_K_skip", "traj_min_over_K_nos"]].mean()
    print(t.round(4).to_string())
    print("\n  cells where turning the CE gate OFF removes damage (gain>0): %d of %d"
          % (int((t.gain > 0).sum()), len(t)))
    print("  mean gain by dataset: %s"
          % m.groupby("dataset")["gain"].mean().round(4).to_dict())
    print("  Spearman(dAP, traj_min_over_K) over the 32 A/B runs: rho=%.3f p=%.2e"
          % spearmanr(pd.concat([m.dAP_skip, m.dAP_nos]),
                      pd.concat([m.traj_min_over_K_skip, m.traj_min_over_K_nos]))[:2])

    print("\n" + "=" * 96)
    print("D. TraLO COUNT TRAJECTORY -- how deep, and WHEN.")
    print("   Read straight out of Hard_Class{c} in each TraLO training_log.csv.")
    print("   Sparse log: rows exist at every 5th epoch, at the first constraint")
    print("   epoch, and at EVERY satisfied epoch, so the minimum below is a")
    print("   LOWER BOUND on the true dive depth (an unlogged epoch could be worse).")
    print("=" * 96)
    rows = []
    for p in d[d.method == "tralo"].path.unique():
        cfg = json.load(open(os.path.join(p, "config.json")))
        cls = cfg["dataset_config"]["constrained_class"]
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        t = pd.read_csv(os.path.join(p, "training_log.csv"))
        ep, hard = num(t["Epoch"]), num(t["Hard_Class%d" % cls])
        lim = num(t["Limit_Class%d" % cls])
        ok = ep.notna() & hard.notna()
        ep, hard, lim = ep[ok].to_numpy(), hard[ok].to_numpy(), lim[ok].to_numpy()
        K = float(lim[0])
        wu = (cfg.get("hyperparams") or {}).get("warmup_epochs", 1)
        c = ep - wu
        i = int(np.argmin(hard))
        below = c[hard < K]
        rows.append({"path": p, "min_c": float(c[i]), "min_over_K": float(hard[i] / K),
                     "first_below_c": float(below.min()) if len(below) else np.nan,
                     "n_logged": len(c)})
    tj = pd.DataFrame(rows)
    dt = d[d.method == "tralo"].merge(tj, on="path")
    for camps, lab in [(["lrc1e-04_noskip"], "clean campaign only (lr_c==lr, gate off)"),
                       (["lrc1e-04_ceskip", "lrc1e-04_noskip"], "both lr_c=1e-4 campaigns"),
                       (["b30_lrc5e-06"], "INERT control (lr_c=5e-6)")]:
        s = dt[dt.campaign.isin(camps)]
        cell = s.groupby(CELL)[["dAP", "min_over_K", "min_c", "first_below_c",
                                "unspent", "dose_active", "eval_c"]].mean()
        r, p_ = spearmanr(cell.min_over_K, cell.dAP)
        r2, p2 = spearmanr(cell.unspent, cell.dAP)
        r3, p3 = spearmanr(cell.dose_active, cell.dAP)
        print("\n  --- %s  (%d runs, %d cells)" % (lab, len(s), len(cell)))
        print("      rho(dAP, deepest dive min/K)  = %+.3f  p=%.2e" % (r, p_))
        print("      rho(dAP, final unspent)       = %+.3f  p=%.2e" % (r2, p2))
        print("      rho(dAP, dose_active epochs)  = %+.3f  p=%.2e" % (r3, p3))
        print(cell.round(3).to_string())

    print("\n  WHEN does the count first go below the cap? (TraLO, clean campaign)")
    s = dt[dt.campaign == "lrc1e-04_noskip"]
    print(s.groupby(CELL)[["first_below_c", "min_c", "eval_c", "last_c", "dAP"]]
          .mean().round(2).to_string())
    print("\n  NOTE: AP is recorded ONCE, at the end, from final_predictions_raw.csv.")
    print("  No per-epoch probabilities or per-epoch AP exist anywhere in the")
    print("  corpus, so the epoch at which AP actually fell CANNOT be read off the")
    print("  logs.  Everything above is the count trajectory, not the AP trajectory.")

    print("\n" + "=" * 96)
    print("E. EXISTENCE PROOF -- clean campaign, cells with no AP damage, with the")
    print("   raw (pre-post-hoc) satisfaction rate attached so that 'no damage'")
    print("   cannot be bought by simply ignoring the constraint.")
    print("=" * 96)
    n = d[d.campaign == "lrc1e-04_noskip"].copy()
    sat = []
    for p in n.path:
        t = pd.read_csv(os.path.join(p, "evaluation_metrics.csv"))
        mm = dict(zip(t.Metric.astype(str), t.Value.astype(str)))
        sat.append(float(mm.get("Raw All Satisfied", "nan")))
    n["raw_sat"] = sat
    cm = n.groupby(CELL + ["method"])[["dAP", "dPrecK", "raw_sat", "unspent",
                                       "count_raw", "K", "n_sat_total",
                                       "dose_active", "eval_c"]].mean().reset_index()
    cm["budget_used"] = cm.count_raw / cm.K
    keep = cm[(cm.dAP >= -0.005) & (cm.raw_sat >= 0.5)]
    print("  cells with dAP >= -0.005 AND raw satisfaction >= 0.5 (majority of seeds")
    print("  hit the cap natively, so the constraint really was enforced):")
    print(keep.sort_values("dAP", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  all 36 cells for reference:")
    print(cm.sort_values("dAP", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""REFUTATION pass on the 'damage-avoidance' claim.

Claim under test:
  TraLO's derm win is a damage-avoidance win -- it degrades the plain-CE model
  least, not satisfies the cap best.
  Evidence quoted: derm AP cost tralo -0.0594 / fioretto -0.0967 / hounie -0.2034;
  ccF1eq cost tralo -0.0113 / fioretto -0.0379 / hounie -0.0932;
  Spearman(fill, AP) over all 48 derm trained runs = +0.702, p=2.7e-08.

Everything here is re-derived from raw files. Reuses analyze_headroom.equalize
(the metric definition under discussion) but recomputes every aggregate.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402
from src.training.constraints import (compute_global_constraints,  # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED  # noqa: E402

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
TR_ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CL_ROOT = "results/headroom/headroom_b30"

pd.set_option("display.width", 260)
FF = lambda x: "%.4f" % x  # noqa: E731


def max_epoch(d):
    """Trap-safe epoch extraction: sparse TraLO logs, case differs by method,
    headers repeat mid-file."""
    p = os.path.join(d, "training_log.csv")
    if not os.path.exists(p):
        return np.nan, np.nan
    try:
        t = pd.read_csv(p)
    except Exception:
        return np.nan, np.nan
    col = None
    for c in t.columns:
        if c.strip().lower() == "epoch":
            col = c
            break
    if col is None:
        return np.nan, np.nan
    e = pd.to_numeric(t[col], errors="coerce").dropna()
    if e.empty:
        return np.nan, np.nan
    return float(e.max()), int(len(e))


def scan(root, methods):
    out = {}
    dupes = 0
    for cfgp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        if m not in methods:
            continue
        d = os.path.dirname(cfgp)
        if not os.path.exists(os.path.join(d, "final_predictions_raw.csv")):
            continue
        hp = cfg.get("hyperparams") or {}
        key = (cfg.get("dataset_mode"), cfg.get("model_name"),
               cfg.get("constraint_tag"), hp.get("seed"), m)
        if key in out:
            dupes += 1
        out[key] = (d, cfg)
    if dupes:
        print("!! %d DUPLICATE keys in %s (silently overwritten)" % (dupes, root))
    return out


def load(d, cfg):
    t = pd.read_csv(os.path.join(d, "final_predictions_raw.csv"))
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = t[cols].to_numpy(float)
    y = t["True_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
    dc = cfg.get("dataset_config") or {}
    c = dc.get("constrained_class")
    c = int(c[0] if isinstance(c, (list, tuple)) else c)
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = compute_global_constraints(df, "label", gp, constrained_class=[c],
                                   num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[c],
                                  num_classes=P.shape[1])
    return P, y, g, c, G, L, t["Predicted_Label"].to_numpy(int)


def score(P, y, g, c, G, L):
    eq = A.equalize(P, g, G, L, c)
    return dict(
        AP=average_precision_score((y == c).astype(int), P[:, c]),
        ccF1eq=f1_score(y, eq, labels=[c], average="macro", zero_division=0),
        k_used=int((eq == c).sum()),
        TP=int(((eq == c) & (y == c)).sum()),
        n_true=int((y == c).sum()))


def main():
    tr = scan(TR_ROOT, TRAINED)
    cl = scan(CL_ROOT, ["heuristic", "danits_lp"])
    print("trained runs found: %d   clip runs found: %d" % (len(tr), len(cl)))

    rows = []
    ctrl_cache = {}
    for (ds, mo, cap, sd, m), (d, cfg) in tr.items():
        P, y, g, c, G, L, raw = load(d, cfg)
        K = int(G[c])
        if K >= UNLIMITED:
            continue
        s = score(P, y, g, c, G, L)
        me, nrows = max_epoch(d)
        hp = cfg.get("hyperparams") or {}
        r = {"dataset": ds, "model": mo, "cap": cap, "seed": sd, "method": m,
             "K": K, "raw_count": int((raw == c).sum()),
             "max_epoch": me, "log_rows": nrows,
             "warmup": hp.get("warmup_epochs"), "cep": hp.get("constraint_epochs"),
             **s}
        r["fill"] = r["raw_count"] / float(K)
        for cm in ["heuristic", "danits_lp"]:
            ref = cl.get((ds, mo, cap, sd, cm))
            if ref is None:
                continue
            kk = (ds, mo, cap, sd, cm)
            if kk not in ctrl_cache:
                Pc, yc, gc, cc, Gc, Lc, rawc = load(*ref)
                sc = score(Pc, yc, gc, cc, Gc, Lc)
                sc["raw_count"] = int((rawc == cc).sum())
                sc["P_c"] = Pc[:, cc]
                ctrl_cache[kk] = sc
            sc = ctrl_cache[kk]
            tag = "" if cm == "heuristic" else "_lp"
            r["AP_ce" + tag] = sc["AP"]
            r["ccF1eq_ce" + tag] = sc["ccF1eq"]
            r["TP_ce" + tag] = sc["TP"]
            r["raw_ce" + tag] = sc["raw_count"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t["dAP"] = t["AP"] - t["AP_ce"]
    t["dcc"] = t["ccF1eq"] - t["ccF1eq_ce"]
    t.to_csv("paper/scripts/out_refute_damage.csv", index=False)

    # ---------------------------------------------------------------- STEP 0
    print("\n" + "=" * 120)
    print("STEP 0  control integrity")
    print("=" * 120)
    print("  runs with a matched heuristic control: %d / %d"
          % (t["AP_ce"].notna().sum(), len(t)))
    if "AP_ce_lp" in t.columns:
        dd = (t["AP_ce"] - t["AP_ce_lp"]).abs().max()
        print("  max |AP_ce(heuristic) - AP_ce(danits_lp)| = %.3g  "
              "(claim says the two share raw predictions)" % dd)
    # the control has constraint_epochs=0, so L30 and L50 must be the SAME model
    z = t[t.method == "tralo"].pivot_table(index=["dataset", "model", "seed"],
                                           columns="cap", values="AP_ce")
    if z.shape[1] == 2:
        dd = (z.iloc[:, 0] - z.iloc[:, 1]).abs()
        print("  control AP identical across L30/L50 for the same seed?  "
              "max diff %.3g  -> %d truly distinct control models per dataset "
              "(not 16)" % (dd.max(), z.shape[0] // 3 if False else len(z) // 1))
        print("    (derm: 2 backbones x 4 seeds = 8 distinct controls reused "
              "across 2 caps -> the '16 seed-cells' are 8 pairs)")

    # ---------------------------------------------------------------- STEP 1
    print("\n" + "=" * 120)
    print("STEP 1  reproduce the quoted derm numbers (mean over 16 seed-cells)")
    print("=" * 120)
    dm = t[t.dataset == "dermmnist"]
    q = dm.groupby("method").agg(n=("dAP", "size"), AP_cost=("dAP", "mean"),
                                 cc_cost=("dcc", "mean"),
                                 AP=("AP", "mean"), ccF1eq=("ccF1eq", "mean"),
                                 AP_ce=("AP_ce", "mean"),
                                 cc_ce=("ccF1eq_ce", "mean")).reindex(TRAINED)
    print(q.to_string(float_format=lambda x: "%.4f" % x))
    print("\n  quoted: AP cost  tralo -0.0594  fioretto -0.0967  hounie -0.2034")
    print("  quoted: cc cost  tralo -0.0113  fioretto -0.0379  hounie -0.0932")

    # ---------------------------------------------------------------- STEP 2
    print("\n" + "=" * 120)
    print("STEP 2  THE CONTROL CANCELS.  Same control run per (ds,bb,cap,seed) for")
    print("        all three methods => cost differences ARE raw differences.")
    print("=" * 120)
    piv_cost = dm.pivot_table(index=CELL + ["seed"], columns="method", values="dcc")
    piv_raw = dm.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    pa_cost = dm.pivot_table(index=CELL + ["seed"], columns="method", values="dAP")
    pa_raw = dm.pivot_table(index=CELL + ["seed"], columns="method", values="AP")
    for lab, pc, pr in [("ccF1eq", piv_cost, piv_raw), ("AP", pa_cost, pa_raw)]:
        for other in ["fioretto_ldf", "hounie_rcl"]:
            a = (pc["tralo"] - pc[other])
            b = (pr["tralo"] - pr[other])
            print("  %-7s tralo-%-13s :  cost-gap mean %+.4f   raw-gap mean %+.4f"
                  "   max|diff| %.3g" % (lab, other, a.mean(), b.mean(),
                                         (a - b).abs().max()))
    print("  => subtracting the control is an EXACT no-op for method comparisons.")

    # ---------------------------------------------------------------- STEP 3
    print("\n" + "=" * 120)
    print("STEP 3  does 'damages least' PREDICT 'wins'?  All 3 datasets, per cell.")
    print("=" * 120)
    out = []
    for (ds, mo, cap), g in t.groupby(CELL):
        pc = g.pivot_table(index="seed", columns="method", values="dAP")
        pk = g.pivot_table(index="seed", columns="method", values="ccF1eq")
        if "tralo" not in pk.columns:
            continue
        d_cc = pk["tralo"] - pk[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
        rank_ap = pc.mean().rank(ascending=False)  # 1 = least damage
        out.append({"dataset": ds, "model": mo, "cap": cap,
                    "dccF1_vs_best_dual": d_cc.mean(),
                    "tralo_wins": d_cc.mean() > 0,
                    "seeds_won": int((d_cc > 0).sum()),
                    "AP_dmg_tralo": pc["tralo"].mean(),
                    "AP_dmg_fior": pc["fioretto_ldf"].mean(),
                    "AP_dmg_houn": pc["hounie_rcl"].mean(),
                    "tralo_least_AP_damage": rank_ap["tralo"] == 1.0})
    o = pd.DataFrame(out).sort_values(CELL)
    print(o.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  CONTINGENCY (count cells, do not average):")
    print(pd.crosstab(o["tralo_least_AP_damage"], o["tralo_wins"],
                      rownames=["tralo damages LEAST (AP)"],
                      colnames=["tralo WINS ccF1eq"]).to_string())
    n_least = int(o["tralo_least_AP_damage"].sum())
    n_win = int(o["tralo_wins"].sum())
    print("  tralo damages least in %d/%d cells but wins in only %d/%d cells."
          % (n_least, len(o), n_win, len(o)))

    # ---------------------------------------------------------------- STEP 4
    print("\n" + "=" * 120)
    print("STEP 4  EPOCH CONFOUND. How many epochs did each method actually run?")
    print("        (sparse TraLO log -> use max(Epoch), never len(df))")
    print("=" * 120)
    e = t.groupby(["dataset", "method"]).agg(
        max_ep_mean=("max_epoch", "mean"), max_ep_min=("max_epoch", "min"),
        max_ep_max=("max_epoch", "max"), log_rows=("log_rows", "mean"),
        cep=("cep", "mean")).reindex(
        pd.MultiIndex.from_product([sorted(t.dataset.unique()), TRAINED]))
    print(e.to_string(float_format=lambda x: "%.1f" % x))
    print("\n  per-run epochs vs AP damage, derm (does less training = less damage?):")
    dm2 = t[t.dataset == "dermmnist"]
    r = spearmanr(dm2["max_epoch"], dm2["dAP"])
    print("    Spearman(max_epoch, dAP) over 48 derm runs = %+.3f  p=%.2g"
          % (r.correlation, r.pvalue))
    print(dm2.groupby("method")[["max_epoch", "dAP", "dcc", "fill"]].mean()
          .reindex(TRAINED).to_string(float_format=lambda x: "%.4f" % x))

    # ---------------------------------------------------------------- STEP 5
    print("\n" + "=" * 120)
    print("STEP 5  the Spearman(fill, AP)=+0.702 -- pooled across 3 methods.")
    print("=" * 120)
    for ds, g in t.groupby("dataset"):
        r = spearmanr(g["fill"], g["AP"])
        print("\n  %-12s POOLED n=%d  rho=%+.3f p=%.2g" % (ds, len(g), r.correlation,
                                                           r.pvalue))
        for m in TRAINED:
            gm = g[g.method == m]
            rm = spearmanr(gm["fill"], gm["AP"])
            print("      within %-13s n=%2d  rho=%+.3f  p=%.3g   "
                  "(fill %.2f, AP %.3f)"
                  % (m, len(gm), rm.correlation, rm.pvalue, gm["fill"].mean(),
                     gm["AP"].mean()))
        # within CELL x METHOD (the atomic unit), 4 seeds each -> pool the rhos
        rs = []
        for (mo, cap, m), gg in g.groupby(["model", "cap", "method"]):
            if len(gg) >= 3 and gg["fill"].nunique() > 1:
                rr = spearmanr(gg["fill"], gg["AP"]).correlation
                if not np.isnan(rr):
                    rs.append(rr)
        if rs:
            print("      within (cell x method), %d groups of 4 seeds: "
                  "mean rho %+.3f, median %+.3f, %d/%d positive"
                  % (len(rs), np.mean(rs), np.median(rs),
                     int(np.sum(np.array(rs) > 0)), len(rs)))

    print("\n  derm bin table quoted by the claim, with METHOD COMPOSITION:")
    g = t[t.dataset == "dermmnist"].copy()
    b = pd.cut(g["fill"], [0, 0.25, 0.5, 0.75, 1.0, 1.5, 10],
               labels=["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"])
    g["bin"] = b
    tb = g.groupby("bin", observed=False).agg(n=("AP", "size"), AP=("AP", "mean"),
                                              ccF1eq=("ccF1eq", "mean")).reset_index()
    comp = pd.crosstab(g["bin"], g["method"])
    print(tb.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print(comp.to_string())

    # ---------------------------------------------------------------- STEP 6
    print("\n" + "=" * 120)
    print("STEP 6  is TraLO's derm win explained by fill?  Compare cells where")
    print("        TraLO's fill is HIGHER vs LOWER than the duals'.")
    print("=" * 120)
    pf = t.pivot_table(index=CELL + ["seed"], columns="method",
                       values=["fill", "ccF1eq", "AP"])
    rr = []
    for idx, row in pf.iterrows():
        f = row["fill"]
        k = row["ccF1eq"]
        if "tralo" not in k.index:
            continue
        bd = k[["fioretto_ldf", "hounie_rcl"]].idxmax()
        rr.append({"dataset": idx[0], "model": idx[1], "cap": idx[2], "seed": idx[3],
                   "fill_tralo": f["tralo"], "fill_best": f[bd], "best": bd,
                   "dcc": k["tralo"] - k[bd], "dfill": f["tralo"] - f[bd]})
    R = pd.DataFrame(rr)
    for ds, g in R.groupby("dataset"):
        rho = spearmanr(g["dfill"], g["dcc"])
        print("  %-12s n=%d  mean dfill %+.3f  mean dcc %+.4f  "
              "Spearman(dfill,dcc) %+.3f p=%.3g"
              % (ds, len(g), g["dfill"].mean(), g["dcc"].mean(), rho.correlation,
                 rho.pvalue))
    print("\n  derm seed-cells where TraLO's fill is LOWER than the best dual's, "
          "yet TraLO still wins ccF1eq:")
    dsub = R[R.dataset == "dermmnist"]
    bad = dsub[(dsub["dfill"] < 0) & (dsub["dcc"] > 0)]
    print("    %d of %d derm seed-cells" % (len(bad), len(dsub)))
    if len(bad):
        print(bad.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  wrote paper/scripts/out_refute_damage.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

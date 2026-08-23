"""What is ccF1eq actually measuring, and how far does each method move the
constrained-class ORDERING away from the plain-CE model?

ccF1eq re-allocates the budget: it takes the top-K by P[:, c] (respecting local
room) and scores F1 for class c. So it is a pure RANKING metric at fixed budget.
Its numerator is the number of true positives inside that top-K set, so a
ccF1eq gap converts exactly into a number of SAMPLES:

    F1_c = 2*TP / (K_used + n_true)      =>   dF1 = 2*dTP / (K_used + n_true)

Reference model = `heuristic` from results/headroom/headroom_b30 (warmup 30,
constraint_epochs 0): it is the plain-CE model at the same total epoch budget
and the same seed, so it is the "constraint off" control. `danits_lp` shares the
identical raw predictions (same warm-up), so either works.

    python paper/scripts/rank_ds.py --trained results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
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

CELL = ["dataset", "model", "cap"]
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]


def scan(root, methods):
    out = {}
    for cfgp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        if m not in methods:
            continue
        d = os.path.dirname(cfgp)
        f = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(f):
            continue
        hp = cfg.get("hyperparams") or {}
        key = (cfg.get("dataset_mode"), cfg.get("model_name"),
               cfg.get("constraint_tag"), hp.get("seed"), m)
        out[key] = (d, cfg)
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
    raw = t["Predicted_Label"].to_numpy(int)
    return P, y, g, c, G, L, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    args = ap.parse_args()

    tr = scan(args.trained, TRAINED)
    cl = scan(args.clip, ["heuristic"])

    rows = []
    for (ds, mo, cap, sd, m), (d, cfg) in tr.items():
        P, y, g, c, G, L, raw = load(d, cfg)
        K = int(G[c])
        if K >= UNLIMITED:
            continue
        eq = A.equalize(P, g, G, L, c)
        n_true = int((y == c).sum())
        k_used = int((eq == c).sum())
        tp = int(((eq == c) & (y == c)).sum())
        r = {"dataset": ds, "model": mo, "cap": cap, "seed": sd, "method": m,
             "K": K, "n_true": n_true, "k_used": k_used, "TP": tp,
             "ccF1eq": f1_score(y, eq, labels=[c], average="macro", zero_division=0),
             "AP": average_precision_score((y == c).astype(int), P[:, c]),
             "raw_count": int((raw == c).sum())}
        ref = cl.get((ds, mo, cap, sd, "heuristic"))
        if ref is not None:
            Pc, yc, gc, cc, Gc, Lc, rawc = load(*ref)
            r["AP_ce"] = average_precision_score((yc == cc).astype(int), Pc[:, cc])
            r["rho_vs_ce"] = spearmanr(P[:, c], Pc[:, cc]).correlation
            topk = set(np.argsort(-P[:, c])[:K])
            topk_ce = set(np.argsort(-Pc[:, cc])[:K])
            r["topK_overlap"] = len(topk & topk_ce) / float(K)
            eqc = A.equalize(Pc, gc, Gc, Lc, cc)
            r["TP_ce"] = int(((eqc == cc) & (yc == cc)).sum())
            r["ccF1eq_ce"] = f1_score(yc, eqc, labels=[cc], average="macro",
                                      zero_division=0)
            r["raw_count_ce"] = int((rawc == cc).sum())
        rows.append(r)

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    print("=" * 128)
    print("Per cell x method (mean over 4 seeds). AP_ce/ccF1eq_ce/TP_ce = the plain-CE")
    print("30-epoch model at the SAME seed (headroom_b30 heuristic). rho = Spearman of")
    print("P[:,c] vs that model; topK_overlap = |top-K cap top-K_CE| / K.")
    print("=" * 128)
    agg = t.groupby(CELL + ["method"]).agg(
        K=("K", "mean"), n_true=("n_true", "mean"),
        raw=("raw_count", "mean"), raw_ce=("raw_count_ce", "mean"),
        TP=("TP", "mean"), TP_ce=("TP_ce", "mean"),
        ccF1eq=("ccF1eq", "mean"), ccF1eq_ce=("ccF1eq_ce", "mean"),
        AP=("AP", "mean"), AP_ce=("AP_ce", "mean"),
        rho=("rho_vs_ce", "mean"), ovl=("topK_overlap", "mean")).reset_index()
    agg["dAP"] = agg["AP"] - agg["AP_ce"]
    agg["dTP"] = agg["TP"] - agg["TP_ce"]
    print(agg.to_string(index=False, float_format=lambda x: "%.4f" % x))

    print()
    print("=" * 128)
    print("PAIRED per-seed: tralo minus best-of-duals, in ccF1eq and in TRUE POSITIVES")
    print("dF1 = 2*dTP / (k_used + n_true) exactly, so the samples column is the same fact.")
    print("=" * 128)
    piv = t.pivot_table(index=CELL + ["seed"], columns="method",
                        values=["ccF1eq", "TP", "AP"])
    out = []
    for (ds, mo, cap, sd), r in piv.iterrows():
        f = r["ccF1eq"]
        tp = r["TP"]
        if "tralo" not in f.index:
            continue
        best = f[["fioretto_ldf", "hounie_rcl"]].max()
        bm = f[["fioretto_ldf", "hounie_rcl"]].idxmax()
        out.append({"dataset": ds, "model": mo, "cap": cap, "seed": sd,
                    "tralo": f["tralo"], "fior": f["fioretto_ldf"],
                    "houn": f["hounie_rcl"], "best_dual": bm,
                    "dF1": f["tralo"] - best,
                    "dTP": tp["tralo"] - tp[bm],
                    "TP_tralo": tp["tralo"], "TP_best": tp[bm]})
    o = pd.DataFrame(out).sort_values(CELL + ["seed"])
    print(o.to_string(index=False, float_format=lambda x: "%.4f" % x))

    print()
    print("=" * 128)
    print("CELL SUMMARY: mean dF1, mean dTP, and how many of the 4 seeds tralo won")
    print("=" * 128)
    s = o.groupby(CELL).agg(dF1=("dF1", "mean"), dTP=("dTP", "mean"),
                            seeds_won=("dF1", lambda x: int((x > 0).sum())),
                            n=("dF1", "size")).reset_index()
    print(s.to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Is the budget-equalized result an artifact of the re-scoring? Three checks.

The claim under test is that TraLO's tight-cap constrained-class lead is quota
utilization rather than better ranking. Before anyone acts on that, the
re-scoring itself has to be shown correct, and the conclusion has to be shown
independent of the allocation rule the re-scoring invents.

  A. REPRODUCTION. Re-score each run's SHIPPED predictions and compare to the
     cc-F1 the run itself wrote at the time. If these disagree, the scoring
     function is wrong and nothing else here means anything.
  B. CAPS. The post-hoc clippers allocate exactly K by construction, so their
     realized count is a ground-truth readout of K. If the reconstructed K
     matches it, the caps were rebuilt correctly.
  C. ALLOCATION-FREE RANKING. Average precision on the constrained class uses
     the scores directly and never picks a threshold or a budget, so it cannot
     be confounded by how much quota a method spent. If TraLO ranks better, AP
     says so regardless of any allocation rule. Also reported: each trained
     arm truncated to the WEAKEST arm's realized count, which compares them at
     a matched budget without inventing a filling rule at all.

    python paper/scripts/verify_equalization.py --dataset octmnist --cap L30_G30
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.getcwd())
from src.training.constraints import (compute_global_constraints,   # noqa: E402
                                      compute_local_constraints)

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl", "tralo_bounded"]
CLIP = ["heuristic", "danits_lp"]


def stored_cc_f1(run_dir, cls):
    p = os.path.join(run_dir, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return None
    for row in csv.DictReader(open(p)):
        if (row.get("Metric") or "").strip() == "F1_Class%d" % cls:
            try:
                return float(row["Value"])
            except (TypeError, ValueError):
                return None
    return None


def truncate_to(y_proba, gids, loc, cls, n_keep, base):
    """Keep only the n_keep highest-scoring constrained-class predictions.

    No filling and no new allocator: this only removes, so it cannot advantage
    the arm being cut. It answers "at the same number of predictions, whose are
    better?" using each arm's own ordering.
    """
    y = base.copy()
    idx = np.where(y == cls)[0]
    if len(idx) <= n_keep:
        return y
    drop = idx[np.argsort(y_proba[idx, cls])][:len(idx) - n_keep]
    other = y_proba[drop].copy()
    other[:, cls] = -np.inf
    y[drop] = np.argmax(other, axis=1)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pending_runs/paper_final")
    ap.add_argument("--dataset", default="octmnist")
    ap.add_argument("--cap", default="L30_G30")
    args = ap.parse_args()

    rows = []
    for cfg_path in glob.glob(args.root + "/**/config.json", recursive=True):
        cfg = json.load(open(cfg_path))
        if cfg.get("dataset_mode") != args.dataset:
            continue
        if cfg.get("constraint_tag") != args.cap:
            continue
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        fin = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw) and os.path.exists(fin)):
            continue
        r = pd.read_csv(raw)
        cols = sorted((c for c in r.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        P = r[cols].to_numpy(float)
        y = r["True_Label"].to_numpy(int)
        g = r["Group_ID"].to_numpy(int) if "Group_ID" in r.columns else None
        cls = int(cfg["dataset_config"]["constrained_class"])
        lp, gp = cfg["constraint"]
        df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
        G = compute_global_constraints(df, "label", gp, constrained_class=cls,
                                       num_classes=P.shape[1])
        L = compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=cls,
                                      num_classes=P.shape[1])
        shipped = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
        binary = (y == cls).astype(int)
        rows.append({
            "model": cfg["model_name"], "method": cfg["methodology"],
            "seed": cfg["hyperparams"]["seed"], "cls": cls, "K": int(G[cls]),
            "count": int((shipped == cls).sum()),
            "cc_f1_recomputed": f1_score(y, shipped, labels=[cls],
                                         average="macro", zero_division=0),
            "cc_f1_stored": stored_cc_f1(d, cls),
            "ap": average_precision_score(binary, P[:, cls]),
            "auroc": roc_auc_score(binary, P[:, cls]),
            "_P": P, "_y": y, "_g": g, "_L": L, "_shipped": shipped,
        })

    if not rows:
        print("no runs matched")
        return 1
    d = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows])

    print("=" * 72)
    print("A. REPRODUCTION -- recomputed cc-F1 vs what the run itself stored")
    ok = d.dropna(subset=["cc_f1_stored"]).copy()
    ok["diff"] = (ok.cc_f1_recomputed - ok.cc_f1_stored).abs()
    print("   %d runs compared; max |difference| = %.6f; agree to 4dp: %d/%d"
          % (len(ok), ok["diff"].max(),
             int((ok["diff"] < 5e-5).sum()), len(ok)))
    if ok["diff"].max() >= 5e-5:
        print(ok.nlargest(5, "diff")[["model", "method", "seed",
                                      "cc_f1_recomputed", "cc_f1_stored"]]
              .to_string(index=False))

    print("\nB. CAPS -- the clippers allocate exactly K, so their count reads K back")
    c = d[d.method.isin(CLIP)]
    print("   %d clipper runs; count == reconstructed K in %d of them"
          % (len(c), int((c["count"] == c.K).sum())))

    print("\nC. ALLOCATION-FREE RANKING -- average precision and AUROC on the")
    print("   constrained class. These use the scores directly: no budget, no")
    print("   threshold, so quota utilization cannot touch them.")
    t = d[d.method.isin(TRAINED + CLIP)]
    print(t.groupby("method")[["ap", "auroc", "count", "cc_f1_recomputed"]]
          .mean().reindex(TRAINED + CLIP).round(4).to_string())

    piv = d[d.method.isin(TRAINED)].pivot_table(index=["model", "seed"],
                                                columns="method", values="ap")
    piv["gap"] = piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    print("\n   paired AP gap, TraLO - best trained dual: %+.4f (seeds won %d/%d)"
          % (piv.gap.mean(), int((piv.gap > 0).sum()), len(piv)))
    print(piv.groupby(level=0).gap.mean().round(4).to_string())

    print("\n   Truncated to the WEAKEST arm's count per cell (removal only,")
    print("   no filling rule invented):")
    out = []
    for (model, seed), grp in pd.DataFrame(rows).groupby(["model", "seed"]):
        sub = grp[grp.method.isin(TRAINED)]
        if len(sub) < len(TRAINED):
            continue
        n_keep = int(sub["count"].min())
        rec = {"model": model, "seed": seed, "n_keep": n_keep}
        for _, r in sub.iterrows():
            yt = truncate_to(r["_P"], r["_g"], r["_L"], r["cls"], n_keep,
                             r["_shipped"])
            rec[r["method"]] = f1_score(r["_y"], yt, labels=[r["cls"]],
                                        average="macro", zero_division=0)
        out.append(rec)
    o = pd.DataFrame(out)
    o["gap"] = o["tralo"] - o[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    print(o.groupby("model")[TRAINED + ["gap", "n_keep"]].mean().round(4).to_string())
    print("   pooled paired gap at matched count: %+.4f (seeds won %d/%d)"
          % (o.gap.mean(), int((o.gap > 0).sum()), len(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

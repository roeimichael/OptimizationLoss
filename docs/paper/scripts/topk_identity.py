"""Is the whole pipeline just "take the top K by p[.,c]"?

Reading src/utils/posthoc_adjustment.py, the greedy path (which runs in 55 of 59
granularity runs; the LP fires in 4) is:

    phase 1, over limit:  sorted_idx = indices[argsort(  y_proba[indices, c])]
    phase 2, under limit: sorted_idx = indices[argsort(- y_proba[indices, c])]

i.e. it removes the LOWEST-probability members of the capped class and admits the
HIGHEST-probability outsiders, until the count is exactly K. Composed, that is
the definition of thresholding the ranking at the budget.

If that is really what happens, then the set of items finally labelled c is
EXACTLY the top K of the ranking induced by p[., c] -- and the consequence is
severe: the pipeline's output depends on p only through its ORDER. Probability
VALUES are discarded. Two models with the same ranking and completely different
calibration produce byte-identical predictions.

This checks it directly, per run: rebuild top-K from the stored probabilities and
compare the resulting label set against the labels the pipeline actually wrote.
Reported as Jaccard so a near-miss is visible rather than collapsing to "False".

Also reports, per run, whether the ccF1 computed from pure top-K equals the
ccF1 of the shipped predictions -- the same claim read on the metric that
actually decides arms.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def per_run(cfg_path):
    cfg = json.load(open(cfg_path))
    d = os.path.dirname(cfg_path)
    f = os.path.join(d, "final_predictions.csv")
    if not os.path.exists(f):
        return None
    dc = cfg.get("dataset_config") or {}
    cc = dc.get("constrained_class")
    if cc is None:
        return None
    classes = [cc] if isinstance(cc, int) else list(cc)
    df = pd.read_csv(f)
    y = df["True_Label"].to_numpy()
    pred = df["Predicted_Label"].to_numpy()
    P = df[[c for c in df.columns if c.startswith("Prob_Class_")]].to_numpy()

    out = []
    for c in classes:
        K = int((pred == c).sum())          # the budget the pipeline landed on
        if K == 0:
            continue
        s = P[:, c]
        topk = np.zeros(len(s), bool)
        topk[np.argsort(-s)[:K]] = True     # top K by p[.,c]
        got = (pred == c)
        inter = int((topk & got).sum())
        union = int((topk | got).sum())
        jac = inter / union if union else 1.0

        def f1(mask):
            tp = int((mask & (y == c)).sum())
            npos = int((y == c).sum())
            return 2.0 * tp / (mask.sum() + npos) if (mask.sum() + npos) else 0.0

        out.append({"cell": "%s/%s/%s" % (cfg.get("dataset_mode"),
                                          cfg.get("model_name"),
                                          cfg.get("constraint_tag")),
                    "arm": cfg.get("arm"), "cls": c, "K": K,
                    "jaccard": jac, "identical": jac == 1.0,
                    "f1_shipped": f1(got), "f1_topk": f1(topk),
                    "lp": bool((cfg.get("results") or {}).get("lp_fallback_used"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for camp in a.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            r = per_run(p)
            if r:
                rows.extend(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs")
        return
    print("class-level observations:", len(df), " runs:", len(df) // max(1, df.cls.nunique()))
    print("\nIs the shipped label set EXACTLY top-K by p[.,c]?")
    print("  identical: %d / %d  (%.1f%%)"
          % (df.identical.sum(), len(df), 100.0 * df.identical.mean()))
    print("  mean Jaccard: %.6f   min: %.6f" % (df.jaccard.mean(), df.jaccard.min()))
    print("  max |ccF1 shipped - ccF1 topK|: %.6f"
          % (df.f1_shipped - df.f1_topk).abs().max())
    print("\nby arm:")
    g = df.groupby("arm").agg(n=("jaccard", "size"), ident=("identical", "mean"),
                              jac=("jaccard", "mean"),
                              dF1=("f1_shipped", lambda s: 0.0))
    for arm, r in g.iterrows():
        sub = df[df.arm == arm]
        print("  %-16s n=%3d  identical=%5.1f%%  meanJac=%.5f  max|dccF1|=%.6f"
              % (arm, r["n"], 100.0 * r["ident"], r["jac"],
                 (sub.f1_shipped - sub.f1_topk).abs().max()))
    if df.lp.any():
        print("\nruns where the LP fallback fired (greedy could not finish):")
        sub = df[df.lp]
        print("  n=%d  identical=%.1f%%  meanJac=%.5f"
              % (len(sub), 100.0 * sub.identical.mean(), sub.jaccard.mean()))
    bad = df[~df.identical].sort_values("jaccard")
    if len(bad):
        print("\nworst non-identical:")
        print(bad.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

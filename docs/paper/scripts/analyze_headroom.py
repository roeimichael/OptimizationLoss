"""Harvest the compute-matched headroom campaign. RUN ON THE SERVER.

Every arm in `headroom_b30` received exactly the same total optimizer epochs --
post-hoc arms `warmup_epochs=30`, trained arms `warmup_epochs=1` plus 29
constraint epochs -- on the same data from the same warm-up cache. So a
difference between arms is a difference between OBJECTIVES, not between compute
budgets. That matters because the apparent short-warm-up advantage in the old
corpus was a 26-epochs-versus-1-epoch artifact.

The question: in the regime where the CE still has headroom, does training under
the constraint produce a BETTER MODEL than training plainly and clipping?

Decided on average precision, which uses the scores directly and never picks a
threshold or spends a budget, so quota utilization -- the confound that killed
the original headline -- cannot touch it. cc-F1 is reported at equal budget for
the same reason.

Results are stratified by dataset and never pooled across them: OctMNIST reaches
0.997 train accuracy inside 30 epochs while DermMNIST does not, so the two sit in
different regimes and their average would describe neither.

    python paper/scripts/analyze_headroom.py [--root results/headroom]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
from src.utils.constants import UNLIMITED                            # noqa: E402
from src.training.constraints import (compute_global_constraints,    # noqa: E402
                                      compute_local_constraints)

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "cap", "model", "seed"]


def equalize(P, gids, glob_c, loc, cls):
    K = int(glob_c[cls])
    order = np.argsort(-P[:, cls])
    room = {int(g): int(l[cls]) for g, l in loc.items()} if (gids is not None and loc) else {}
    chosen = np.zeros(len(P), dtype=bool)
    taken = 0
    for i in order:
        if taken >= K:
            break
        if room:
            g = int(gids[i])
            if room.get(g, 0) <= 0:
                continue
            room[g] -= 1
        chosen[i] = True
        taken += 1
    other = P.copy()
    other[:, cls] = -np.inf
    y = np.argmax(other, axis=1)
    y[chosen] = cls
    return y


def rows_for(root):
    out = []
    for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        d = os.path.dirname(cfg_path)
        raw, fin = (os.path.join(d, "final_predictions_raw.csv"),
                    os.path.join(d, "final_predictions.csv"))
        if not (os.path.exists(raw) and os.path.exists(fin)):
            continue
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        if not cols:
            continue
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
        G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                       num_classes=P.shape[1])
        L = compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=[cls],
                                      num_classes=P.shape[1])
        if G[cls] >= UNLIMITED:
            continue
        rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
        eq = equalize(P, g, G, L, cls)
        hp = cfg.get("hyperparams") or {}
        out.append({
            # Join key. (dataset, model, cap, seed) is NOT unique once a campaign
            # runs several arms or methods, so anything joining log-derived
            # features onto these rows must key on the directory.
            "path": d,
            "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
            "model": cfg.get("model_name"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"),
            # Smoke campaigns run several variants of ONE methodology, so
            # `method` cannot separate them and (dataset, model, cap, seed)
            # collides across arms. `arm` is the only discriminator.
            "arm": cfg.get("arm"),
            "warmup": hp.get("warmup_epochs"),
            "cepochs": hp.get("constraint_epochs"),
            "K": int(G[cls]), "count": int((rel == cls).sum()),
            # count_raw is the model's OWN count before post-hoc adjustment.
            # `count` is measured after adjustment, which fills up to K, so it
            # reads ~K even when the network has collapsed and predicts the
            # constrained class for almost nobody. Any check for that collapse
            # must use count_raw or it will find nothing.
            "count_raw": int((rawp == cls).sum()),
            "sat": int((rawp != rel).sum() == 0),
            "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
            "ccF1eq": f1_score(y, eq, labels=[cls], average="macro", zero_division=0),
            "macroEq": f1_score(y, eq, average="macro", zero_division=0),
        })
    return pd.DataFrame(out)


def paired(d, metric, refs, label):
    piv = d.pivot_table(index=CELL, columns="method", values=metric)
    have = [m for m in refs if m in piv.columns]
    if "tralo" not in piv.columns or not have:
        return None
    s = piv.dropna(subset=["tralo"]).copy()
    s["ref"] = s[have].max(axis=1)
    s = s.dropna(subset=["ref"])
    if s.empty:
        return None
    gap = s["tralo"] - s["ref"]
    return {"cmp": label, "metric": metric, "n": len(gap), "mean": gap.mean(),
            "won": int((gap > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = rows_for(args.root)
    if d.empty:
        print("no scorable runs")
        return 1
    print("scored %d runs" % len(d))
    print("\nepoch budget actually used (confirms the matching held):")
    print(d.groupby("method")[["warmup", "cepochs"]].mean().round(1).to_string())

    print("\n" + "=" * 78)
    print("MEANS BY DATASET x METHOD  (compute-matched, 30 epochs every arm)")
    print("=" * 78)
    for ds, grp in d.groupby("dataset"):
        print("\n--- %s ---" % ds)
        print(grp.groupby("method")[["AP", "ccF1eq", "macroEq", "count", "K", "sat"]]
              .mean().reindex(TRAINED + CLIP).round(4).to_string())

    print("\n" + "=" * 78)
    print("PAIRED, TraLO minus comparator, within (dataset,cap,model,seed)")
    print("=" * 78)
    rows = []
    for ds, grp in d.groupby("dataset"):
        for metric in ["AP", "ccF1eq", "macroEq"]:
            for refs, lab in [(["fioretto_ldf", "hounie_rcl"], "best dual"),
                              (CLIP, "clipper")]:
                r = paired(grp, metric, refs, lab)
                if r:
                    r["dataset"] = ds
                    rows.append(r)
    o = pd.DataFrame(rows)
    for metric in ["AP", "ccF1eq", "macroEq"]:
        print("\n%s" % metric)
        t = o[o.metric == metric].pivot_table(index="dataset", columns="cmp",
                                              values=["mean", "won", "n"])
        print(t.round(4).to_string())

    if args.out:
        d.to_csv(args.out, index=False)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""INDEPENDENT re-scoring of a campaign from RAW run files.

Deliberately does NOT import analyze_headroom / score_dualsvsclip: K, the
equal-budget allocation and both F1s are recomputed here from scratch so a bug
in the existing helpers cannot propagate into the check.

Also records what the existing scripts do not:
  count_eq   -- how many items the equal-budget allocator actually placed
                (if this is < K the "budget-equalized" metric is not equalized)
  deficit    -- K - count_adj, the unfilled shipped budget
  n_groups, local caps sum

    python paper/scripts/rm_score.py --root <dir> --out <csv>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def K_of(y, pct):
    """round(true_count * pct) -- the pipeline's own definition."""
    return int(np.round((y).sum() * pct))


def equalize(P, gids, K, local_caps, cls):
    """Greedy: take the K highest-scoring items for `cls`, honouring per-group
    caps; everything else gets argmax over the non-constrained classes."""
    order = np.argsort(-P[:, cls], kind="stable")
    room = dict(local_caps) if local_caps else None
    chosen = np.zeros(len(P), dtype=bool)
    taken = 0
    for i in order:
        if taken >= K:
            break
        if room is not None:
            g = int(gids[i])
            if room.get(g, 0) <= 0:
                continue
            room[g] -= 1
        chosen[i] = True
        taken += 1
    other = P.copy()
    other[:, cls] = -np.inf
    yhat = np.argmax(other, axis=1)
    yhat[chosen] = cls
    return yhat, taken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rows, skip = [], {"nopred": 0, "badcfg": 0, "unlimited": 0}
    paths = sorted(glob.glob(a.root + "/**/config.json", recursive=True))
    for n, cp in enumerate(paths):
        if n % 400 == 0:
            print("  %d/%d" % (n, len(paths)), file=sys.stderr)
        try:
            cfg = json.load(open(cp))
        except Exception:
            skip["badcfg"] += 1
            continue
        d = os.path.dirname(cp)
        fraw, ffin = (os.path.join(d, "final_predictions_raw.csv"),
                      os.path.join(d, "final_predictions.csv"))
        if not (os.path.exists(fraw) and os.path.exists(ffin)):
            skip["nopred"] += 1
            continue
        t = pd.read_csv(fraw)
        pc = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                    key=lambda c: int(c.rsplit("_", 1)[1]))
        if not pc or "True_Label" not in t.columns:
            skip["nopred"] += 1
            continue
        P = t[pc].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        gids = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config") or {}
        cls = dc.get("constrained_class")
        if cls is None:
            skip["badcfg"] += 1
            continue
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        if gp is None or gp >= 1e9:
            skip["unlimited"] += 1
            continue
        pos = (y == cls)
        K = K_of(pos, gp)
        local = None
        if gids is not None and lp is not None and lp < 1e9:
            local = {}
            for g in np.unique(gids):
                local[int(g)] = int(np.round((pos & (gids == g)).sum() * lp))
        rel = pd.read_csv(ffin)["Predicted_Label"].to_numpy(int)
        eq, taken = equalize(P, gids, K, local, cls)
        hp = cfg.get("hyperparams") or {}
        rows.append({
            "campaign": a.tag or a.root,
            "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
            "model": cfg.get("model_name"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"), "sweep": cfg.get("sweep"),
            "arm": cfg.get("arm"),
            "warmup": hp.get("warmup_epochs"),
            "cepochs": hp.get("constraint_epochs"),
            "lr": hp.get("lr"), "lr_c": hp.get("lr_constraint"),
            "ce_skip": hp.get("enable_ce_skip"),
            "cls": cls, "n_pool": len(y), "n_true": int(pos.sum()),
            "K": K, "K_local_sum": (sum(local.values()) if local else -1),
            "n_groups": (len(local) if local else 0),
            "count_raw": int((rawp == cls).sum()),
            "count_adj": int((rel == cls).sum()),
            "count_eq": int(taken),
            "deficit": K - int((rel == cls).sum()),
            "AP": average_precision_score(pos.astype(int), P[:, cls]),
            "ccF1adj": f1_score(y, rel, labels=[cls], average="macro",
                                zero_division=0),
            "ccF1eq": f1_score(y, eq, labels=[cls], average="macro",
                               zero_division=0),
            "macroAdj": f1_score(y, rel, average="macro", zero_division=0),
            "macroEq": f1_score(y, eq, average="macro", zero_division=0),
            "path": d,
        })
    o = pd.DataFrame(rows)
    o.to_csv(a.out, index=False)
    print("scored %d runs from %s -> %s  skipped=%s" %
          (len(o), a.root, a.out, skip))
    return 0


if __name__ == "__main__":
    sys.exit(main())

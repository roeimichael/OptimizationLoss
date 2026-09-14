"""Development diagnostics at each actual group/class allocation cut.

Diagnostics use the stored deployed selection, including global competition.
They do not select a campaign or certify learnability.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from src.training.constraints import (
    compute_global_constraints,
    compute_local_constraints,
    normalize_constrained_classes,
)
from src.utils.constants import UNLIMITED


def effective_budget(G, L, c):
    local = sum(bounds[c] for bounds in L.values()) if L else UNLIMITED
    return min(G[c], local)


def run_axes(parts):
    if len(parts) < 5:
        raise SystemExit("REFUSED: run path must include backbone/dataset/cap/arm/seed")
    return parts[-5], parts[-4]


def load(d):
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    t = pd.read_csv(d / "final_predictions_raw.csv")
    cols = sorted((int(c[11:]), c) for c in t if c.startswith("Prob_Class_"))
    P = t[[c for _, c in cols]].to_numpy(float)
    if not len(P) or not np.isfinite(P).all() or (P < 0).any() or (P.sum(1) <= 0).any():
        raise ValueError("invalid probabilities in %s" % d)
    P = P / P.sum(axis=1, keepdims=True)
    y, g = t["True_Label"].to_numpy(int), t["Group_ID"].to_numpy()
    classes = normalize_constrained_classes(cfg["dataset_config"]["constrained_class"])
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g})
    G = compute_global_constraints(
        df, "label", gp, constrained_class=classes, num_classes=P.shape[1]
    )
    L = compute_local_constraints(
        df, "label", lp, "grp", constrained_class=classes, num_classes=P.shape[1]
    )
    deployed = pd.read_csv(d / "final_predictions.csv")
    if (
        len(deployed) != len(t)
        or not np.array_equal(deployed["True_Label"], y)
        or not np.array_equal(deployed["Group_ID"], g)
    ):
        raise ValueError("raw/deployed row identity mismatch")
    return y, g, P, classes, G, L, deployed["Predicted_Label"].to_numpy(int)


def group_headroom(y, groups, probabilities, classes, G, L, deployed):
    rows = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        for c in classes:
            k = min(len(idx), int(L[group][c]), int(G[c]))
            selected = idx[deployed[idx] == c]
            emitted = len(selected)
            tp = int((y[selected] == c).sum())
            n = int((y[idx] == c).sum())
            rows.append(
                dict(
                    group=str(group),
                    class_id=c,
                    K=k,
                    effective_class_K=int(effective_budget(G, L, c)),
                    emitted=emitted,
                    support=n,
                    selected_tp=tp,
                    selected_errors=emitted - tp,
                    outside_tp=n - tp,
                    correctable=min(emitted - tp, n - tp),
                    cut_probability=float(probabilities[selected, c].min())
                    if emitted
                    else None,
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--control", default="clip")
    args = ap.parse_args()
    found = 0
    try:
        for p in sorted(Path(args.root).rglob("config.json")):
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if cfg.get("arm") != args.control or cfg.get("status") != "completed":
                continue
            if run_axes(p.parent.parts) != (cfg["model_name"], cfg["dataset_mode"]):
                raise ValueError("run path axes differ from config")
            y, g, P, classes, G, L, deployed = load(p.parent)
            for row in group_headroom(y, g, P, classes, G, L, deployed):
                row.update(
                    dataset=cfg["dataset_mode"],
                    backbone=cfg["model_name"],
                    cap=cfg["constraint_tag"],
                    seed=cfg["hyperparams"]["seed"],
                )
                print(json.dumps(row, sort_keys=True))
                found += 1
        if not found:
            raise ValueError("no completed control runs with predictions")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print("FAIL headroom: %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

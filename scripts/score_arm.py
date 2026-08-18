"""Score a proposed loss change against the frozen benchmark. RUN ON THE SERVER.

Every research arm in newdirections/ runs the SAME four smoke cells and is
scored here against numbers that already exist in the frozen grid, so a new idea
costs four runs rather than a re-run of every baseline.

The comparison is only trustworthy because the warm-up cache key
(`compute_base_model_id`) covers model, lr, dropout, batch size, warm-up epochs,
pretrained, dataset and seed but NOT the methodology. An arm that clones a
frozen-grid config and changes only the loss therefore starts from bit-identical
warm-up weights as the baseline it is being compared to. Changing any hyper-
parameter in that key silently retrains the warm-up, breaks the pairing, and
reintroduces the cross-campaign drift (0.027 cc-F1) that already corrupted one
ablation row. So: change the loss, never the cache key.

METRICS, in the order they decide an arm's fate:

  AP        average precision on the constrained class. Allocation-free -- no
            threshold, no budget -- so quota filling cannot manufacture it. This
            is the metric the budget-equalized control did NOT kill, and the only
            one on which beating the post-hoc clipper means anything. PRIMARY.
  ccF1eq    constrained-class F1 with every arm filled to exactly K. Budget is
            held fixed, so this measures allocation quality, not utilization.
  macroEq   overall macro-F1 at equal budget. GUARD: an arm that buys AP by
            wrecking the other classes has not won anything.
  count/K   realized constrained-class count before any post-hoc edit.
  sat       whether the run satisfied natively (equivalently, flips == 0).

The bar an arm has to clear is printed with the results: at these cells the
post-hoc clipper has the BEST AP of any method, because at warm-up 50 the CE has
saturated and every trained method can only re-threshold a frozen score vector.
Beating the clipper on AP therefore requires changing what the network learns,
not where the cut is placed. An arm that only moves counts around has not.

    python paper/scripts/score_arm.py --build-reference
    python paper/scripts/score_arm.py --arm results/softtopk --name soft-topk
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

# Tight caps, two datasets, two backbones, one seed: the cheapest cut that still
# spans dataset and architecture. Every baseline already exists here.
CELLS = [("octmnist", "L30_G30", "MobileNetV3", 1),
         ("octmnist", "L30_G30", "RegNetY400MF", 1),
         ("dermmnist", "L30_G30", "MobileNetV3", 1),
         ("dermmnist", "L30_G30", "RegNetY400MF", 1)]
REF_METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
DEFAULT_REF = "newdirections/bench/headroom_reference.csv"

# The reference is the COMPUTE-MATCHED short-warm-up campaign, not the frozen
# warm-up-50 grid. Two reasons, and both are load-bearing:
#
#   Regime. At warm-up 50 the CE-saturation gate has already fired, so nothing
#   is learning during the constraint phase and every method can only
#   re-threshold a frozen score vector. Optimal re-thresholding IS the post-hoc
#   clipper, so that regime is unwinnable by construction and tells us nothing
#   about a new loss.
#
#   Fairness. The post-hoc arms do no constraint-phase training at all -- they
#   train `warmup_epochs` and allocate -- so at short warm-up an unmatched
#   comparison pits a ~26-epoch model against a 1-epoch model. The headroom
#   campaign pins every arm to the same total optimizer epochs (post-hoc arms
#   warmup=B; trained arms warmup=1 + constraint_epochs=B-1), which is the only
#   comparison that isolates the objective from the compute.
DEFAULT_ROOTS = ["results/headroom"]
DEFAULT_WARMUP = 1


def equalize(y_proba, gids, glob_c, loc, cls):
    """Fill to exactly K: the K highest-scoring samples get the constrained
    class subject to each group's cap, everything else takes its best remaining
    class. Same rule the post-hoc clipper follows, applied to every arm, which
    is what makes the budget a constant instead of a free variable."""
    K = int(glob_c[cls])
    order = np.argsort(-y_proba[:, cls])
    room = {int(g): int(l[cls]) for g, l in loc.items()} if (gids is not None and loc) else {}
    chosen = np.zeros(len(y_proba), dtype=bool)
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
    other = y_proba.copy()
    other[:, cls] = -np.inf
    y = np.argmax(other, axis=1)
    y[chosen] = cls
    return y


def score_run(run_dir, cfg):
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    fin = os.path.join(run_dir, "final_predictions.csv")
    if not (os.path.exists(raw) and os.path.exists(fin)):
        return None
    t = pd.read_csv(raw)
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if not cols:
        return None
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
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[cls],
                                  num_classes=P.shape[1])
    if G[cls] >= UNLIMITED:
        return None
    rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
    eq = equalize(P, g, G, L, cls)
    return {
        "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
        "model": cfg.get("model_name"), "seed": (cfg.get("hyperparams") or {}).get("seed"),
        "method": cfg.get("methodology"), "K": int(G[cls]),
        "count": int((rel == cls).sum()),
        "sat": int((rawp != rel).sum() == 0),
        "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
        "ccF1eq": f1_score(y, eq, labels=[cls], average="macro", zero_division=0),
        "macroEq": f1_score(y, eq, average="macro", zero_division=0),
    }


def collect(roots, keep=None):
    rows = []
    for root in roots:
        for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                continue
            hp = cfg.get("hyperparams") or {}
            key = (cfg.get("dataset_mode"), cfg.get("constraint_tag"),
                   cfg.get("model_name"), hp.get("seed"))
            if keep and key not in keep:
                continue
            if keep and hp.get("warmup_epochs") != 50:
                continue
            r = score_run(os.path.dirname(cfg_path), cfg)
            if r:
                r["path"] = cfg_path
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--arm", help="results dir of the proposed change")
    ap.add_argument("--name", default="ARM")
    ap.add_argument("--roots", nargs="*", default=["results/pending_runs"])
    args = ap.parse_args()

    if args.build_reference:
        keep = {(d, c, m, s) for d, c, m, s in CELLS}
        d = collect(args.roots, keep)
        d = d[d.method.isin(REF_METHODS)]
        d = d.sort_values("path").drop_duplicates(
            subset=["dataset", "cap", "model", "seed", "method"], keep="first")
        os.makedirs(os.path.dirname(args.ref), exist_ok=True)
        d.to_csv(args.ref, index=False)
        print("wrote %s (%d rows)\n" % (args.ref, len(d)))
        print(d.groupby("method")[["AP", "ccF1eq", "macroEq", "count", "K", "sat"]]
              .mean().reindex(REF_METHODS).round(4).to_string())
        print("\nper cell:")
        print(d.pivot_table(index=["dataset", "model"], columns="method",
                            values="AP").round(4).to_string())
        return 0

    if not args.arm:
        print("need --arm or --build-reference")
        return 1
    ref = pd.read_csv(args.ref)
    arm = collect([args.arm])
    if arm.empty:
        print("no scorable runs under %s" % args.arm)
        return 1
    arm["method"] = args.name

    both = pd.concat([ref, arm], ignore_index=True)
    print("=" * 78)
    print("ARM: %s   (%d runs)" % (args.name, len(arm)))
    print("=" * 78)
    for metric in ["AP", "ccF1eq", "macroEq"]:
        piv = both.pivot_table(index=["dataset", "model"], columns="method",
                               values=metric)
        order = [m for m in REF_METHODS if m in piv.columns] + [args.name]
        print("\n%s" % metric)
        print(piv[order].round(4).to_string())

    # Paired deltas on the cells the arm actually ran, against the two
    # comparators that matter: the incumbent method and the clipper that the
    # budget control showed to be the real bar.
    j = arm.merge(ref, on=["dataset", "cap", "model", "seed"], suffixes=("_a", "_r"))
    print("\n" + "-" * 78)
    for base in ["tralo", "heuristic"]:
        s = j[j.method_r == base]
        if s.empty:
            continue
        print("vs %-10s  dAP %+0.4f (%d/%d)   dccF1eq %+0.4f   dmacroEq %+0.4f"
              % (base, (s.AP_a - s.AP_r).mean(),
                 int((s.AP_a > s.AP_r).sum()), len(s),
                 (s.ccF1eq_a - s.ccF1eq_r).mean(),
                 (s.macroEq_a - s.macroEq_r).mean()))
    print("\nnative satisfaction: %d/%d   mean count/K: %.2f"
          % (int(arm.sat.sum()), len(arm), (arm["count"] / arm.K).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Every metric that could support (or sink) a claim, on one campaign, seed-paired.

Three families, kept separate on purpose because they can be gamed differently:

ALLOCATION-FREE -- computed from PROBABILITIES only, so neither the post-hoc
    adjustment nor budget filling can touch them. This is the family that
    survived the quota-fill audit, and calibration belongs to it: post-hoc
    adjustment rewrites LABELS, never probabilities.
        AP      average precision, constrained class
        AUROC   ranking quality, prevalence-independent
        ECE     expected calibration error (15 bins), lower is better
        Brier   multiclass Brier score, lower is better
        NLL     negative log-likelihood, lower is better
        ConfGap mean confidence when correct minus when wrong, higher is better

BUDGET-EQUALIZED -- every arm filled to exactly K by the same rule, so the
    budget is a constant and only allocation QUALITY varies.
        ccP / ccR / ccF1     constrained-class precision, recall, F1
        macroP / macroR / macroF1
        acc

AS-RUN -- what the method actually did before any equalization.
        sat      native satisfaction (flips == 0)
        cnt/K    realized constrained count over the cap
        flips    how many predictions post-hoc had to change

Usage:
    python full_panel.py <campaign_dir> --control <arm>
"""
import argparse
import glob
import importlib.util as ilu
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             log_loss)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = ilu.spec_from_file_location("_sa", os.path.join(_HERE, "score_arm.py"))
_sa = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sa)


def equalize_multi(y_proba, gids, glob_c, loc, classes):
    """Feasible max-probability allocation over ALL capped classes.

    The single-class `equalize` in score_arm.py picks top-K for one class and
    then argmaxes the rest with no budget check, so with several capped classes
    it produces an allocation that violates every cap but the first. This walks
    all (item, capped class) pairs in descending probability and assigns while
    the class has global and local room, then gives each leftover item its best
    class that still has room.

    With one capped class this is exactly the old behaviour.
    """
    n, n_cls = y_proba.shape
    room_g = {int(c): int(glob_c[c]) for c in classes}
    room_l = {}
    if gids is not None and loc:
        for g, lim in loc.items():
            for c in classes:
                if c < len(lim) and lim[c] < _sa.UNLIMITED:
                    room_l[(int(g), int(c))] = int(lim[c])
    assigned = np.full(n, -1, dtype=int)
    pairs = [(y_proba[i, c], i, int(c)) for c in classes for i in range(n)]
    pairs.sort(key=lambda t: -t[0])
    for _, i, c in pairs:
        if assigned[i] != -1 or room_g[c] <= 0:
            continue
        key = (int(gids[i]), c) if gids is not None else None
        if key is not None and key in room_l:
            if room_l[key] <= 0:
                continue
            room_l[key] -= 1
        assigned[i] = c
        room_g[c] -= 1
    capped = set(int(c) for c in classes)
    free = np.where(assigned == -1)[0]
    if len(free):
        alt = y_proba[free].copy()
        for c in capped:
            # a capped class is only still available to a leftover item if it
            # has BOTH global and local room; otherwise mask it out entirely.
            blocked = room_g[c] <= 0
            for j, i in enumerate(free):
                key = (int(gids[i]), c) if gids is not None else None
                if blocked or (key is not None and key in room_l and room_l[key] <= 0):
                    alt[j, c] = -np.inf
        assigned[free] = np.argmax(alt, axis=1)
    return assigned


def ece(y, P, bins=15):
    """Expected calibration error on the top-1 prediction."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return e


def brier(y, P):
    oh = np.zeros_like(P)
    oh[np.arange(len(y)), y] = 1.0
    return float(((P - oh) ** 2).sum(axis=1).mean())


def panel(run_dir, cfg):
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
    if not np.isfinite(P).all():
        # A diverged run: probabilities NaN, `status` still "completed" because
        # nothing in the pipeline guards divergence. Dropping it loudly beats
        # crashing the campaign (which hides every healthy run behind it) and
        # beats coercing the NaNs (which lets a diverged run contribute a number).
        print("  DROPPED (NON-FINITE probabilities):", raw)
        return None
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    y = t["True_Label"].to_numpy(int)
    rawp = t["Predicted_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None

    dc = cfg.get("dataset_config", {}) or {}
    cls_raw = dc.get("constrained_class")
    # ALL capped classes, not just the first. Scoring a multi-class campaign on
    # cls[0] reports one class's metric under the campaign's name, and equalizing
    # only cls[0] leaves the other caps free -- so the "budget-equalized" family
    # was neither budget-equalized nor multi-class.
    classes = ([int(c) for c in cls_raw] if isinstance(cls_raw, (list, tuple))
               else [int(cls_raw)])
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = _sa.compute_global_constraints(df, "label", gp, constrained_class=classes,
                                       num_classes=P.shape[1])
    L = _sa.compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=classes, num_classes=P.shape[1])
    classes = [c for c in classes if G[c] < _sa.UNLIMITED]
    if not classes:
        return None
    cls = classes[0]
    rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
    eq = (_sa.equalize(P, g, G, L, cls) if len(classes) == 1
          else equalize_multi(P, g, G, L, classes))
    # Per-class scores averaged over the capped classes; identical to before
    # when only one class is capped.
    ybin = (y == cls).astype(int)
    s = P[:, cls]
    _ap = float(np.mean([average_precision_score((y == c).astype(int), P[:, c])
                         for c in classes]))
    _auc = float(np.mean([roc_auc_score((y == c).astype(int), P[:, c])
                          for c in classes
                          if 0 < (y == c).sum() < len(y)])) if any(
        0 < (y == c).sum() < len(y) for c in classes) else np.nan
    _Ksum = float(sum(G[c] for c in classes))
    _rawcnt = float(sum((rawp == c).sum() for c in classes))
    _relcnt = float(sum((rel == c).sum() for c in classes))
    conf = P.max(axis=1)
    ok = (P.argmax(axis=1) == y)

    return {
        "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
        "cap": cfg.get("constraint_tag"),
        # part of the cell key: a swept dimension that lives only in the config
        # gets pooled, which is how the granularity sweep was first misread
        "capped": "-".join(str(x) for x in (
            cfg.get("dataset_config", {}).get("constrained_class", [])
            if isinstance(cfg.get("dataset_config", {}).get("constrained_class"), list)
            else [cfg.get("dataset_config", {}).get("constrained_class")])),
        "seed": (cfg.get("hyperparams") or {}).get("seed"),
        "arm": cfg.get("arm"),
        # -------- allocation-free
        "AP": _ap,
        "AUROC": _auc,
        "ECE": ece(y, P),
        "Brier": brier(y, P),
        "NLL": log_loss(y, P, labels=list(range(P.shape[1]))),
        "ConfGap": float(conf[ok].mean() - conf[~ok].mean()) if (~ok).any() else np.nan,
        # -------- budget-equalized
        "ccP": precision_score(y, eq, labels=classes, average="macro", zero_division=0),
        "ccR": recall_score(y, eq, labels=classes, average="macro", zero_division=0),
        "ccF1": f1_score(y, eq, labels=classes, average="macro", zero_division=0),
        "macroP": precision_score(y, eq, average="macro", zero_division=0),
        "macroR": recall_score(y, eq, average="macro", zero_division=0),
        "macroF1": f1_score(y, eq, average="macro", zero_division=0),
        "acc": accuracy_score(y, eq),
        # -------- as-run
        "sat": float((rawp != rel).sum() == 0),
        # RAW count, before post-hoc. cnt_over_K below is measured after the
        # adjustment and is therefore ~1.0 for every arm by construction -- it
        # says nothing. This is the one that measures how far the trained model
        # sits from feasibility on its own.
        "raw_over_K": _rawcnt / _Ksum,
        "cnt_over_K": _relcnt / _Ksum,
        "flips": float((rawp != rel).sum()),
        "flips_over_K": float((rawp != rel).sum()) / _Ksum,
    }


# higher-is-better for everything except these.
#
# raw_over_K and flips_over_K need care. raw_over_K's ideal is 1.0, not 0, so
# "lower is better" only holds while an arm sits ABOVE the cap -- undershooting
# would waste budget and would still read as an improvement here. Both arms
# over-predict by a wide margin in every campaign measured so far (clip 2.35x,
# TraLO 1.71x), so the direction is unambiguous, but main() asserts it rather
# than trusting it.
LOWER_BETTER = {"ECE", "Brier", "NLL", "flips", "raw_over_K", "flips_over_K"}
ABOVE_CAP_ONLY = {"raw_over_K"}
GROUPS = [
    ("ALLOCATION-FREE  (probabilities only -- budget filling cannot touch these)",
     ["AP", "AUROC", "ECE", "Brier", "NLL", "ConfGap"]),
    # ccP/ccR/ccF1 are ONE result, not three. With the budget pinned to exactly
    # K, ccP = TP/K, ccR = TP/n_pos and ccF1 = 2TP/(K+n_pos) -- all three are
    # monotone in the same TP count, so they agree to the fourth decimal of the
    # p-value by construction. Quote one of them, never three as if they
    # corroborated each other.
    ("BUDGET-EQUALIZED (filled to exactly K; ccP/ccR/ccF1 are one metric in three costumes)",
     ["ccP", "ccR", "ccF1", "macroP", "macroR", "macroF1", "acc"]),
    # 🛑 NOT A RESULT FAMILY. Post-hoc adjustment fills to the constraint
    # boundary for free at the end of every pipeline, so "how far outside the
    # cap the raw model sat" and "how many flips the free step performed" buy
    # no advancement whatsoever. A method that halves the flip count and ties
    # on quality has produced NOTHING. These rows stay only as debugging
    # telemetry and are printed under a header that says so -- never rank arms
    # on them, never put them in a paper, never call one a WIN.
    ("DIAGNOSTIC ONLY  -- NOT RESULTS. Post-hoc filling is free; flips and raw\n"
     "                   count buy no advancement. Do not rank arms on these.",
     ["sat", "raw_over_K", "flips", "flips_over_K", "cnt_over_K"]),
]
NON_SCORING = {"sat", "raw_over_K", "flips", "flips_over_K", "cnt_over_K"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--campaign", required=True, nargs="+")
    a.add_argument("--control", required=True)
    a.add_argument("--percell", action="store_true")
    args = a.parse_args()

    rows = []
    for camp in args.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(p))
            except Exception:
                continue
            r = panel(os.path.dirname(p), cfg)
            if r:
                rows.append(r)
    if not rows:
        sys.exit("no scorable runs")
    df = pd.DataFrame(rows)
    arms = sorted(df.arm.unique())
    if args.control not in arms:
        sys.exit("control %r not among %s" % (args.control, arms))

    key = ["dataset", "model", "cap", "seed"]
    print("arms:", {a_: int((df.arm == a_).sum()) for a_ in arms})
    print("cells:", df.groupby(["dataset", "model", "cap", "capped"]).ngroups,
          " seeds:", df.seed.nunique())

    for arm in arms:
        if arm == args.control:
            continue
        print()
        print("=" * 100)
        print("%s   vs   %s        (paired on %s)" % (arm, args.control, "+".join(key)))
        print("=" * 100)
        for title, metrics in GROUPS:
            print("\n  " + title)
            print("  %-9s %10s %10s %10s %8s %9s   %s"
                  % ("metric", "control", arm, "delta", "cells", "wilcoxon", "verdict"))
            for m in metrics:
                # Restrict to the PAIR being compared before dropping
                # NaNs. Pivoting over every arm means a third arm that
                # is still running deletes its missing seeds from this
                # comparison too -- measured on results/beta, where an
                # unrelated 2/4 arm cut a complete 4-seed comparison to
                # two pairs and made it read as a tie.
                pairdf = df[df["arm"].isin([args.control, arm])]
                q = pairdf.pivot_table(index=key, columns="arm",
                                       values=m).dropna()
                if args.control not in q or arm not in q:
                    continue
                c, t = q[args.control], q[arm]
                d = t - c
                if m in ABOVE_CAP_ONLY and (c.min() < 1.0 or t.min() < 1.0):
                    # An arm dipped below the cap, so "lower is better" no
                    # longer describes this column. Say so instead of scoring it.
                    print("  %-9s   UNDERSHOOTS the cap in some runs "
                          "(min %.3f/%.3f) -- direction undefined, not scored"
                          % (m, c.min(), t.min()))
                    continue
                better = (d < 0) if m in LOWER_BETTER else (d > 0)
                try:
                    pv = stats.wilcoxon(t, c)[1]
                except Exception:
                    pv = np.nan
                # A claim needs BOTH a majority of cells and significance.
                if m in NON_SCORING:
                    # Never emits WIN/LOSS. These cannot support a claim, and
                    # printing a verdict next to them is how they keep getting
                    # quoted as one.
                    v = "(not a result)"
                elif np.isnan(pv):
                    v = "-"
                elif pv < 0.05 and better.sum() > len(d) / 2:
                    v = "*** WIN"
                elif pv < 0.05:
                    v = "*** LOSS"
                elif better.sum() > len(d) * 0.7:
                    v = "lean win"
                elif better.sum() < len(d) * 0.3:
                    v = "lean loss"
                else:
                    v = "tie"
                print("  %-9s %10.4f %10.4f %+10.4f %5d/%-3d %9.4f   %s"
                      % (m, c.mean(), t.mean(), d.mean(), better.sum(), len(d), pv, v))
                if args.percell:
                    per = d.groupby(level=[0, 1]).mean().round(4)
                    print("            per-cell: %s"
                          % {"/".join(k[:2]): v for k, v in per.items()})


if __name__ == "__main__":
    main()

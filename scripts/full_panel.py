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
import collections
import hashlib
import io
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             log_loss)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.score_arm import equalize                            # noqa: E402
from src.training.constraints import (compute_global_constraints,  # noqa: E402
                                      compute_local_constraints, normalize_constrained_classes)
from src.utils.constants import UNLIMITED                          # noqa: E402


def _one(series):
    """Aggregator for the seed pivot: there must be exactly ONE run per
    (cell, seed, arm). More than one means the pairing key is missing a
    dimension, and silently averaging them is how a swept axis gets pooled."""
    vals = series.dropna()
    if len(vals) > 1:
        raise ValueError(
            "%d runs share one (cell, seed, arm) key -- the pairing key is "
            "missing a dimension that the campaign varies. Averaging them "
            "would pool the swept axis." % len(vals))
    return vals.iloc[0] if len(vals) else float("nan")


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
                if c < len(lim) and lim[c] < UNLIMITED:
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
        # An all -inf row means no class has room left: the instance is
        # infeasible, and np.argmax returns 0, silently assigning class 0 past
        # its own budget. The trainer's allocator logs exactly this; here it was
        # silent. Harmless when a strict subset is capped (0 in 2000 random
        # instances) and real when every class is (53 in 200).
        stuck = ~np.isfinite(alt).any(axis=1)
        if stuck.any():
            print("  WARNING: equalize_multi found no feasible class for %d "
                  "item(s) -- every capped class is full, so the caps cannot "
                  "all be met and these assignments VIOLATE one." % stuck.sum())
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


_degenerate = set()   # capped classes absent from a scope, reported once


def panel(run_dir, cfg):
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    fin = os.path.join(run_dir, "final_predictions.csv")
    if not (os.path.exists(raw) and os.path.exists(fin)):
        return None
    t = pd.read_csv(raw)
    # Hash the file the model actually wrote, before any allocator touches it.
    # Two arms with the same digest did the same thing, whatever their configs
    # claim -- and the ALLOCATED predictions can differ while the raw ones are
    # identical, which is exactly `clip` vs `lp` and is NOT an inert flag.
    raw_md5 = hashlib.md5(io.open(raw, "rb").read()).hexdigest()[:12]
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
    if "Group_ID" not in t.columns:
        # Silently defaulting every row to group 0 collapses all groups into
        # one, so every LOCAL cap silently becomes the global cap and the run
        # scores as a plausible but wrong result. The writer always emits this
        # column today (runner -> eval -> logging, and group_ids is mandatory
        # in the loader), so this is unreachable by construction -- which is
        # exactly why it must raise rather than paper over the day it is not.
        raise SystemExit(
            "%s has no Group_ID column. Every local cap would collapse into "
            "the global one and the run would score as if it had no groups."
            % raw)
    g = t["Group_ID"].to_numpy(int)

    dc = cfg.get("dataset_config", {}) or {}
    cls_raw = dc.get("constrained_class")
    # ALL capped classes, not just the first. Scoring a multi-class campaign on
    # cls[0] reports one class's metric under the campaign's name, and equalizing
    # only cls[0] leaves the other caps free -- so the "budget-equalized" family
    # was neither budget-equalized nor multi-class.
    # THE normalizer. This was a FOURTH inline copy of the scalar-or-list
    # pattern the other three were just unified into -- in the file that scores
    # every result. On `constrained_class: null` it raised a bare
    # "TypeError: int() argument must be ... not 'NoneType'" instead of the
    # informative ValueError every other site gives.
    classes = normalize_constrained_classes(cls_raw)
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g})
    G = compute_global_constraints(df, "label", gp, constrained_class=classes,
                                       num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=classes, num_classes=P.shape[1])
    # Both scopes: eval.py now ENFORCES a local-only cap, so dropping such a
    # class here would make the scorer and the pipeline disagree about what is
    # constrained -- and the framework prescribes sweeping G < L next.
    classes = [c for c in classes
               if G[c] < UNLIMITED
               or any(b[c] < UNLIMITED for b in L.values())]
    if not classes:
        return None
    cls = classes[0]
    rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
    eq = (equalize(P, g, G, L, cls) if len(classes) == 1
          else equalize_multi(P, g, G, L, classes))
    # Per-class scores averaged over the capped classes; identical to before
    # when only one class is capped.
    # ONE class set for both. AUROC has to drop a class with no positive (or
    # no negative) instance -- it is undefined there -- and AP silently
    # contributed 0.0 for that class instead of dropping it, so the two numbers
    # described DIFFERENT populations while sitting in the same table, both
    # finite and both plausible. constraints.py:36-43 deliberately permits K=0
    # for a class absent from a scope, so this is reachable, not hypothetical.
    scorable = [c for c in classes if 0 < (y == c).sum() < len(y)]
    if len(scorable) < len(classes):
        _degenerate.update(set(classes) - set(scorable))
    _ap = float(np.mean([average_precision_score((y == c).astype(int), P[:, c])
                         for c in scorable])) if scorable else np.nan
    _auc = float(np.mean([roc_auc_score((y == c).astype(int), P[:, c])
                          for c in scorable])) if scorable else np.nan
    _Ksum = float(sum(G[c] for c in classes))
    _rawcnt = float(sum((rawp == c).sum() for c in classes))
    _relcnt = float(sum((rel == c).sum() for c in classes))
    conf = P.max(axis=1)
    ok = (P.argmax(axis=1) == y)

    return {
        "raw_md5": raw_md5,
        "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
        "cap": cfg.get("constraint_tag"),
        # part of the cell key: a swept dimension that lives only in the config
        # gets pooled, which is how the granularity sweep was first misread
        # A FIFTH copy lived here and was worse than the fourth: on None it did
        # not raise at all, it produced the cell label "None" -- a
        # plausible-looking wrong row in the cell KEY itself. Unreachable only
        # because the line above happened to raise first, which is a landmine,
        # not a defused case.
        "capped": "-".join(str(x) for x in classes),
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
        # NOT satisfaction. Both allocators fill the budget UPWARD to exactly K
        # (heuristic/train.py pass 1 walks every (item, capped class) pair;
        # posthoc_adjustment phase 2 runs with force_exact=True), so this is 1
        # only when the raw count already equals K in every scope. A model that
        # satisfies the cap with room to spare scores the same as one that
        # grossly violates it. It was called `sat` and read as satisfaction.
        "count_eq_K": float((rawp != rel).sum() == 0),
        # THIS is satisfaction: the raw count is within every limit, global and
        # local, before any post-hoc adjustment. Undershoot counts as satisfied,
        # because the constraint is `count <= K`.
        "raw_feasible": float(
            all(int((rawp == c).sum()) <= G[c] for c in classes)
            and all(int(((rawp == c) & (g == gi)).sum()) <= lim[c]
                    for gi, lim in L.items() for c in classes
                    if c < len(lim))),
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
# TraLO 1.71x), so the direction is unambiguous, but main() branches to UNDERSHOOT on it rather
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
     ["raw_feasible", "count_eq_K", "raw_over_K", "flips", "flips_over_K",
      "cnt_over_K"]),
]
BH_ALIAS_OF = {"ccP": "ccF1", "ccR": "ccF1"}
BH_ALIASES = set(BH_ALIAS_OF)

NON_SCORING = {"raw_feasible", "count_eq_K", "raw_over_K", "flips", "flips_over_K", "cnt_over_K"}


# Arms with constraint_epochs == 0: the warm-up IS the run, so their
# predictions cannot vary with the cap and must not be flagged for it.
POSTHOC_ARMS = {"clip", "focal_clip", "lp", "focal_lp",
                "cb_lp", "la_lp"}

RAW_MD5 = {}          # arm -> {cell: md5 of final_predictions_raw.csv}


def _signflip_p(x):
    """Exact two-sided sign-flip permutation p over D cluster means.

    All 2^D sign assignments are enumerated -- D is the number of datasets, so
    at most 3 here and the enumeration is free. Returns (p, D).
    """
    x = np.asarray([v for v in np.asarray(x, float) if np.isfinite(v)])
    D = len(x)
    if D == 0:
        return float("nan"), 0
    obs = abs(x.mean())
    hits = 0
    for mask in range(1 << D):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(D)])
        if abs(float((x * signs).mean())) >= obs - 1e-15:
            hits += 1
    return hits / float(1 << D), D


def _clustered_readout(results, pvals, control, arm):
    """The honest n for a claim meant to generalize is the DATASET count.

    Averaging seeds within a cell fixed the seed-level dependence, but every
    cell inside one dataset still shares that dataset's fixed test set and the
    K derived from it. The Wilcoxon above treats each cell as an independent
    draw regardless, so adding a backbone or a cap level buys resolution, not
    independence -- only a new dataset buys a genuinely independent test set.

    So this reports the same deltas clustered to one value per dataset and
    tested by exact sign flip, and prints the floor that unit imposes. It is
    printed BESIDE the per-cell table, never instead of it: the per-cell test
    is the right unit for "did this move on the cells we ran", and this one is
    the right unit for "does this generalize".
    """
    rowset = [(m, r) for _t, m, r in results
              if r[0] == "OK" and m in pvals and np.isfinite(pvals[m])]
    if not rowset:
        return
    print("\n  CLUSTERED BY DATASET  (%s vs %s) -- the generalization unit"
          % (arm, control))
    D = None
    printed = False
    for m, r in rowset:
        d = r[3]
        try:
            per_ds = d.groupby(level=0).mean()
        except Exception:
            continue
        p, D = _signflip_p(per_ds.values)
        if not D:
            continue
        printed = True
        detail = "  ".join("%s %+.4f" % (str(k)[:6], v)
                           for k, v in per_ds.items())
        print("  %-9s cell p=%.4f | clustered p=%.4f over %d dataset(s):  %s"
              % (m, pvals[m], p, D, detail))
    if printed and D:
        floor = 2.0 ** (1 - D)
        print("  ^ exact sign-flip floor at %d dataset(s) is p=%.3f%s"
              % (D, floor, "" if floor < 0.05 else
                 "  -- NO all-dataset campaign this project can run reaches "
                 "p<0.05 on this unit"))
        if D < 3:
            print("    (only %d of the 3 datasets are present, so this says "
                  "nothing about generality)" % D)
    print()


def _reordering_check(rows):
    """Did the constraint phase reorder the capped class, or only shift it?

    Nine of the thirteen scored metrics are exactly invariant to a monotone
    transform of the capped class's score column, so an arm can move its soft
    count a long way and change nothing a metric can see. tau near 1.0 with a
    large bias_shift IS that case, and it is the difference between "this arm
    did nothing" and "this arm did something the scorer is blind to".

    Printed, never gated: it describes what happened, and a low tau is not by
    itself good news -- reordering badly is also reordering.
    """
    print("REORDERING (capped-class test ranking, warm-up -> scored model)")
    seen = False
    for arm in sorted({r["arm"] for r in rows}):
        per = [r for r in rows if r["arm"] == arm and r.get("reordering")]
        if not per:
            continue
        seen = True
        taus, shifts, resids = [], [], []
        for r in per:
            for st in r["reordering"].values():
                taus.append(st.get("kendall_tau"))
                shifts.append(st.get("bias_shift"))
                resids.append(st.get("shift_residual_sd"))
        taus = [t for t in taus if t is not None and np.isfinite(t)]
        shifts = [v for v in shifts if v is not None and np.isfinite(v)]
        resids = [v for v in resids if v is not None and np.isfinite(v)]
        if not taus:
            continue
        tau, shift = float(np.mean(taus)), float(np.mean(shifts or [np.nan]))
        resid = float(np.mean(resids or [np.nan]))
        verdict = ""
        if tau > 0.99 and abs(shift) > 0.1:
            verdict = "  <-- BIAS ONLY: the count moved, the ranking did not"
        elif tau > 0.999:
            verdict = "  <-- the ranking is unchanged"
        print("  %-14s n=%2d  tau=%.4f  bias_shift=%+.4f  resid_sd=%.4f%s"
              % (arm, len(taus), tau, shift, resid, verdict))
    if not seen:
        print("  (no run carries the diagnostic -- post-hoc arms never run a")
        print("   constraint phase, and runs made before it was persisted have")
        print("   no `reordering` key in config.json)")
    print()


def _allocator_check(rows):
    """An arm that fell through to the LP is not running the allocator it names.

    `targeted_correction` hands an infeasible greedy allocation to `_fallback_lp`
    without saying so anywhere a reader will look. The flag has been recorded on
    every run since the pipeline was written and read by nothing.
    """
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["arm"]][1] += 1
        if r.get("lp_fallback"):
            per[r["arm"]][0] += 1
    hits = {a: v for a, v in per.items() if v[0]}
    if not hits:
        return
    print("")
    print("ALLOCATOR SWAP -- these arms did NOT run the allocator they are named for")
    for arm, (n, tot) in sorted(hits.items()):
        print("  *** %s: %d of %d runs fell through to the LP fallback (%.0f%%). "
              "The greedy allocation was infeasible, so a DIFFERENT algorithm "
              "produced those predictions." % (arm, n, tot, 100.0 * n / tot))


def _identity_check(rows):
    """House rule 3: md5 the raw predictions across arms, BEFORE any metric.

    Two arms whose raw predictions hash identically on every seed of every cell
    did not do two different things. Five occurrences on record, most recently
    `clip` and `focal_clip` sharing a base_model_id so focal_clip silently
    loaded clip's warm-up. The tell is always available and always ignored,
    because it lives in a file nobody opens once the metrics table exists.
    """
    per_arm = collections.defaultdict(dict)
    for r in rows:
        # `capped` belongs in the key. It is in the PAIRING key below, and
        # --campaign takes several roots, so two roots that differ only in the
        # capped class collide here and half the runs never get hashed.
        cell = (r["dataset"], r["model"], r["cap"], r["capped"], r["seed"])
        per_arm[r["arm"]][cell] = r["raw_md5"]
    arms = sorted(per_arm)
    RAW_MD5.clear()
    RAW_MD5.update(per_arm)
    print("")
    print("RAW-PREDICTION IDENTITY (house rule 3, before any metric)")
    dead = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = set(per_arm[a]) & set(per_arm[b])
            if not shared:
                continue
            same = sum(per_arm[a][c] == per_arm[b][c] for c in shared)
            if same == len(shared):
                dead.append((a, b, len(shared)))
            elif same:
                print("  %s vs %s: %d of %d cells bit-identical"
                      % (a, b, same, len(shared)))
    for a, b, n in dead:
        print("  *** %s and %s emit BIT-IDENTICAL raw predictions on all %d "
              "cell-seeds. Whatever separates them in the config is INERT. "
              "Any delta below is allocator-only." % (a, b, n))
    # A cap level that changes nothing is the same failure wearing a different
    # hat: the baseline runs in the multiclass campaign were bit-identical
    # across caps, so 12 cells rested on 6 models.
    for arm in arms:
        if arm in POSTHOC_ARMS:
            continue
        by_cap = collections.defaultdict(dict)
        for (ds, mdl, cap, capped, seed), h in per_arm[arm].items():
            by_cap[(ds, mdl, capped, seed)][cap] = h
        collapsed = [k for k, v in by_cap.items()
                     if len(v) > 1 and len(set(v.values())) == 1]
        if collapsed:
            print("  *** %s: raw predictions IDENTICAL across cap levels in %d "
                  "of %d (dataset, backbone, seed) groups -- those cells are "
                  "ONE run counted twice, not two levels."
                  % (arm, len(collapsed), len(by_cap)))
    skipped_ph = sorted(set(arms) & POSTHOC_ARMS)
    if skipped_ph:
        print("  (cross-cap check skipped for %s: a post-hoc arm's final model IS"
              % ", ".join(skipped_ph))
        print("   its warm-up model, and base_model_id correctly excludes the cap,")
        print("   so identical raw predictions across caps are EXPECTED there.")
        print("   NOTE: for the six allocation-free metrics this means the control's")
        print("   value is duplicated across cap levels, so the effective n is")
        print("   cells / n_cap_levels, not cells.)")
    if not dead:
        print("  every arm pair differs on at least one cell-seed")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--campaign", required=True, nargs="+")
    a.add_argument("--control", required=True)
    a.add_argument("--percell", action="store_true")
    a.add_argument("--allow-weak-control", action="store_true",
                   help="permit --control focal_clip even though clip is present")
    args = a.parse_args()

    rows = []
    skipped = collections.Counter()
    unscorable = []
    prov = collections.Counter()
    for camp in args.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(p))
            except Exception:
                continue
            # Only completed runs. The scorer ignored `status` entirely, so a
            # campaign of `diverged` and `pending` runs scored normally -- and
            # regenerating a campaign overwrites a non-completed config in
            # place while leaving the OLD final_predictions.csv on disk, so the
            # previous code's predictions get scored as the new code's result.
            if cfg.get("status") != "completed":
                skipped[cfg.get("status", "no status")] += 1
                continue
            r = panel(os.path.dirname(p), cfg)
            if not r:
                # A COMPLETED run that cannot be scored vanished with no
                # message: missing prediction files, no Prob_Class_ columns, or
                # no capped class surviving the filter. Silently dropping a
                # completed run is how a comparison loses pairs -- the failure
                # that made in-flight campaigns read as ties.
                unscorable.append(os.path.dirname(p))
            if r:
                r["lp_fallback"] = bool(
                    cfg.get("results", {}).get("lp_fallback_used", False))
                r["reordering"] = cfg.get("reordering") or {}
                prov[(cfg.get("code_version"),
                      cfg.get("data_fingerprint"))] += 1
                rows.append(r)
    if _degenerate:
        print("NOTE: capped class(es) %s have no positive or no negative instance "
              "in some run and are excluded from AP and AUROC alike. Both metrics "
              "now average over the SAME classes."
              % sorted(_degenerate))
    # Two code versions or two data fingerprints in one comparison means the
    # arms were not produced by the same pipeline against the same data.
    # FRAMEWORK records that results either side of the 2026-08-19 n_chunks
    # removal are not comparable, and gen_campaign re-stamps code_version on
    # PENDING runs while leaving completed ones alone, so one tree legitimately
    # carries two. check_parity catches this, but it runs BEFORE the campaign.
    if len(prov) > 1:
        print("REFUSED: these runs do not share a provenance --")
        for (cv, df), n in sorted(prov.items(), key=lambda kv: -kv[1]):
            print("   %4d run(s)  code_version=%s  data_fingerprint=%s"
                  % (n, cv, df))
        sys.exit("Scoring across them would compare two pipelines, or two "
                 "datasets, as if they were one arm-vs-arm difference.")
    if unscorable:
        print("*** %d run(s) are COMPLETED but produced nothing scorable. They are"
              % len(unscorable))
        print("    missing from every comparison below:")
        for d in unscorable[:10]:
            print("      %s" % d)
        if len(unscorable) > 10:
            print("      ... and %d more" % (len(unscorable) - 10))
    _allocator_check(rows)
    _identity_check(rows)
    _reordering_check(rows)
    if skipped:
        print("skipped %d run(s) that are not completed: %s"
              % (sum(skipped.values()), dict(skipped)))
    if not rows:
        sys.exit("no scorable runs")
    df = pd.DataFrame(rows)
    arms = sorted(df.arm.unique())
    if args.control not in arms:
        sys.exit("control %r not among %s" % (args.control, arms))
    if args.control == "focal_clip" and "clip" in arms and not args.allow_weak_control:
        sys.exit("REFUSED: --control focal_clip while `clip` is in this campaign. "
                 "`clip` is the stronger quality bar -- it beats focal_clip by "
                 "more than TraLO does -- and headlining against focal_clip is "
                 "retraction (d) in FRAMEWORK section 2. Pass "
                 "--allow-weak-control to report it anyway, alongside clip.")

    # `capped` belongs in the PAIRING key, not only in the cell count printed
    # below it. Without it, pivot_table's default aggfunc="mean" averages two
    # capped-class settings into one pair, so a +0.40 cell and a -0.40 cell
    # collapse to an exact tie while the header still reports two cells. That is
    # mistake-pattern 6 (pooling the swept axis) presenting as mistake-pattern 8
    # (a bug that reads as a tie) -- inside the scorer written to prevent both.
    key = ["dataset", "model", "cap", "capped", "seed"]
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
        # Do the arms actually emit the same predictions? Computed once per
        # arm, from the md5s panel() already recorded, so the verdict can say
        # "bit-identical" only when that is literally true.
        shared = set(RAW_MD5.get(args.control, {})) & set(RAW_MD5.get(arm, {}))
        identical = bool(shared) and all(
            RAW_MD5[args.control][k] == RAW_MD5[arm][k] for k in shared)

        results = []          # (title, metric, row-tuple) collected, then BH
        for title, metrics in GROUPS:
            for m in metrics:
                # Restrict to the PAIR being compared before dropping
                # NaNs. Pivoting over every arm means a third arm that
                # is still running deletes its missing seeds from this
                # comparison too -- measured on results/beta, where an
                # unrelated 2/4 arm cut a complete 4-seed comparison to
                # two pairs and made it read as a tie.
                pairdf = df[df["arm"].isin([args.control, arm])]
                q = pairdf.pivot_table(index=key, columns="arm",
                                       values=m, aggfunc=_one).dropna()
                if args.control not in q or arm not in q:
                    continue
                # Average seeds WITHIN a cell before testing. The atomic unit
                # is the cell, not the seed-pair: seeds share the test set, the
                # cap and the cached warm-up, so testing them as independent
                # inflates type-I error to 11-22% under the null.
                cell = q.groupby(level=[0, 1, 2, 3]).mean()
                c, t = cell[args.control], cell[arm]
                d = t - c
                if m in ABOVE_CAP_ONLY and (c.min() < 1.0 or t.min() < 1.0):
                    results.append((title, m, ("UNDERSHOOT", c, t, d, None, None)))
                    continue
                results.append((title, m, ("OK", c, t, d, None, None)))

        # Benjamini-Hochberg across this arm's scoring metrics. 13 metrics gives
        # a measured family-wise error of 28% under a true null.
        pvals = {}
        for title, m, r in results:
            if r[0] != "OK" or m in NON_SCORING:
                continue
            if m in BH_ALIASES:
                # ccP, ccR and ccF1 are one result in three costumes. Counting
                # them three times in the family widens the callable p
                # threshold by 2.54x, so only ccF1 ENTERS THE FAMILY -- but the
                # other two now get their OWN Wilcoxon rather than inheriting
                # ccF1's p. Sharing a SIGN is guaranteed (same K and n_pos per
                # cell make all three positive multiples of the same TP-diff);
                # sharing a RANK of |delta| across cells, which is what the
                # signed-rank test actually consumes, is not. It held on every
                # archive checked, but it was an assumption printed as a result.
                try:
                    pvals[m] = stats.wilcoxon(r[2], r[1], zero_method="zsplit")[1]
                except Exception:
                    pvals[m] = np.nan
                continue
            d = r[3]
            if (d == 0).all():
                # A metric that did not move is a legitimate NON-REJECTION, so
                # it belongs in the family at p=1.0. Dropping it shrank the BH
                # denominator m in q = p*m/i, which makes every OTHER metric's q
                # SMALLER -- more lenient -- than a fixed-family BH gives. The
                # comment above calibrates the family at a fixed metric count;
                # this made the count vary with the data.
                pvals[m] = 1.0
                continue
            try:
                pvals[m] = stats.wilcoxon(r[2], r[1], zero_method="zsplit")[1]
            except Exception:
                pvals[m] = np.nan
        # the aliases carry their own p but must not widen the family
        finite = sorted((v, k) for k, v in pvals.items()
                        if np.isfinite(v) and k not in BH_ALIASES)
        qvals = {}
        for i, (pv, m) in enumerate(finite, 1):
            qvals[m] = min(1.0, pv * len(finite) / i)
        for i in range(len(finite) - 2, -1, -1):      # enforce monotonicity
            qvals[finite[i][1]] = min(qvals[finite[i][1]], qvals[finite[i + 1][1]])
        # the aliases inherit their representative's q, so the table still
        # prints a number for them without inflating the family
        for alias, rep in BH_ALIAS_OF.items():
            # q only. The p is the alias's own, computed above.
            if rep in qvals:
                qvals.setdefault(alias, qvals[rep])
            if (alias in pvals and rep in pvals and np.isfinite(pvals[alias])
                    and np.isfinite(pvals[rep])
                    and abs(pvals[alias] - pvals[rep]) > 1e-9):
                print("  NOTE: %s p=%.4f differs from its BH representative %s "
                      "p=%.4f." % (alias, pvals[alias], rep, pvals[rep]))
                print("        They share a sign by construction but not a rank "
                      "of |delta|, so the")
                print("        q shown for %s is %s's and is only indicative."
                      % (alias, rep))

        shown = None
        for title, m, r in results:
            if title != shown:
                shown = title
                print("\n  " + title)
                print("  %-9s %10s %10s %10s %14s %9s %8s   %s"
                      % ("metric", "control", arm, "delta",
                         "better/worse", "wilcoxon", "BH q", "verdict"))
            kind, c, t, d, _, _ = r
            if kind == "UNDERSHOOT":
                print("  %-9s   UNDERSHOOTS the cap in some cells "
                      "(min %.3f/%.3f) -- direction undefined, not scored"
                      % (m, c.min(), t.min()))
                continue
            gain = -d if m in LOWER_BETTER else d
            better, worse = int((gain > 0).sum()), int((gain < 0).sum())
            tied = len(d) - better - worse
            pv, qv = pvals.get(m, np.nan), qvals.get(m, np.nan)

            if m in NON_SCORING:
                # Checked FIRST. flips / raw-count-over-K / count_eq_K are not
                # results at any value, so they must never render as a
                # movement verdict, not even "no movement".
                v = "(not a result)"
            elif (d == 0).all() and identical:
                # Both conditions. The metric did not move AND the arms emit
                # byte-identical raw predictions -- only then is "the treatment
                # did nothing" a statement about the treatment rather than
                # about one metric's resolution.
                v = "DEAD FLAG (bit-identical)"
            elif (d == 0).all():
                # The arms DO differ; this metric just cannot see it. Common
                # and not a bug: a treatment that shifts calibration without
                # reordering leaves every ranking and threshold metric exactly
                # equal while ECE, Brier and NLL all move.
                v = "no movement in this metric (arms differ)"
            elif not np.isfinite(pv):
                v = "-"
            elif better + worse < 6:
                # The floor is set by the NON-ZERO pairs, which is what the
                # signed-rank test actually consumes -- exactly 2^(1-n) at n of
                # them. This used to gate on len(d), the total cell count, while
                # printing a floor computed from better+worse: 6 cells with one
                # exact tie carry a true floor of 0.0625 and rendered as an
                # ordinary row with no caveat. An integer-valued metric at small
                # K ties often, so that is not a corner case.
                v = "%d non-zero pair(s) of %d cells, min attainable p=%.3f "                    "-- NOT CALLABLE" % (
                        better + worse, len(d), 2.0 ** (1 - max(1, better + worse)))
            elif qv < 0.05 and better > worse:
                v = "*** WIN"
            elif qv < 0.05 and worse > better:
                v = "*** LOSS"
            elif pv < 0.05 and better > worse:
                v = "win (not after BH)"
            elif pv < 0.05 and worse > better:
                v = "loss (not after BH)"
            elif better > len(d) * 0.7:
                v = "lean win"
            elif worse > len(d) * 0.7:
                v = "lean loss"
            else:
                v = "tie"
            print("  %-9s %10.4f %10.4f %+10.4f  %4d/%-4d t=%-3d %9.4f %8s   %s"
                  % (m, c.mean(), t.mean(), d.mean(), better, worse, tied,
                     pv, ("%.4f" % qv) if np.isfinite(qv) else "-", v))
            if args.percell:
                # levels 0..3 = dataset, model, cap, capped. Dropping `cap` here
                # averaged a win at L50_G30 with a loss at L30_G30 into one row
                # and hid the sign flip -- in the only output that claims to
                # show the atomic cell.
                per = d.groupby(level=[0, 1, 2, 3]).mean().round(4)
                print("            per-cell: %s"
                      % {"/".join(str(x) for x in k): v for k, v in per.items()})

        _clustered_readout(results, pvals, args.control, arm)

        # The scope BH actually covers, stated because nothing stated it.
        print("  BH controls the false-discovery rate across the %d metrics IN "
              "THIS TABLE," % len([m for m in pvals if m not in BH_ALIASES]))
        print("  for THIS arm against THIS control. It does NOT correct across "
              "the other arms")
        print("  scored in the same run, nor across the dozens of campaigns this "
              "project has")
        print("  run. A q<0.05 here is one arm's family, not the project's.")
        print()


if __name__ == "__main__":
    main()

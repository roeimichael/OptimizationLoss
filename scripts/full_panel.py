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
import math
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


# Allocation-free metrics that get their own power statement. They cannot be
# moved by post-hoc filling, which is exactly why a verdict may rest on them --
# and why a tie in them must never be reported without its seed cost.
#
# THE SPLIT IS NOT COSMETIC. "Allocation-free" is two different guarantees and
# only one of them answers this project's question:
#
#   RANKING     AP, AUROC. Invariant to ANY strictly monotone rescale of the
#               scores, so they move only if the ORDER changed -- and order is
#               the only thing a top-K allocator reads. A move here is the
#               representation channel of FRAMEWORK 2(p).
#   CALIBRATION ECE, Brier, NLL, ConfGap. Move when the order changes OR when
#               the probabilities are merely rescaled with the order intact.
#               A temperature or prior shift moves these and provably cannot
#               change any top-K set (2(j)), so a calibration-only move is a
#               real effect that changes NO allocation.
#
# Measured on the stored evidence: focal_clip vs clip moves ECE -0.069 on 6 of
# 6 cells while AUROC moves +0.005 and AP -0.0001. Reading that as "the
# probabilities changed, so the representation channel is live" would be the
# error this split exists to block.
FREE_RANKING = ("AP", "AUROC")
FREE_CALIBRATION = ("ECE", "Brier", "NLL", "ConfGap")
FREE_RESOLUTION = FREE_RANKING + FREE_CALIBRATION

# The budget-equalized table gets one too. It has no items scale outside ccF1,
# and printing macro-F1 -- the paper's headline -- with no seed cost is the same
# omission ConfGap had.
EQ_RESOLUTION = ("ccP", "ccR", "ccF1", "macroP", "macroR", "macroF1", "acc")


RUN_DIRS = {}         # df index -> run directory, for the collision message
LEAF_DEPTH = 5        # model/dataset/cap/arm/seed_N -- the per-cell path tail


def effective_budget(G, L, c):
    """The budget that actually BINDS: min(global, sum of the local ceilings).

    🛑 THIS USED TO READ `int(G[c])`, THE GLOBAL ALONE, AND IT INFLATED THE
    PRIZE BY AN ORDER OF MAGNITUDE ON iwildcam. Local caps are per-group
    ceilings, so their SUM already bounds the count; whenever that sum is below
    the global, the global is INERT and cannot be reached. `gen_campaign` says
    so out loud for every cap it emits ("INERT GLOBAL: K=185 is above the local
    sum 111, so it can never bind"), and this tool ignored it.

    The ceiling is `2K/(K+n)`, so an over-large K raises it twice over -- it
    both admits more true positives and enlarges the denominator more slowly.
    On iwildcam L30_G50 class 2 the global is 185 against a local sum of 111,
    which reads a ceiling of 0.667 where the reachable one is 0.462, and prints
    59 items of headroom where the real gap is 4.0. Measured 2026-08-24 against
    the equalized top-K counted directly off the stored predictions.

    The module docstring already warned that "local caps can put it out of
    reach". That was a comment describing a defect instead of a fix, and it
    stood while the number it qualified was quoted as the project's effect
    size.

    UNLIMITED local ceilings are excluded from the sum rather than added, since
    a group with no ceiling places no bound on the total.
    """
    k = int(G[c]) if G[c] < UNLIMITED else UNLIMITED
    parts = [int(b[c]) for b in L.values() if b[c] < UNLIMITED]
    if parts and len(parts) == len(L):
        # Every group is capped, so the sum is a real bound on the total. If
        # even one group is uncapped the local scope bounds nothing globally.
        k = min(k, sum(parts))
    return k

def _collision_msg(idx):
    """Name the RIGHT cause when two runs land on one (cell, seed, arm) key.

    TWO different mistakes produce this collision and they need opposite fixes,
    so a message that names only one sends the reader hunting in the wrong
    place:

      * `--campaign` was pointed at a tree holding MORE THAN ONE campaign, so
        the colliding runs are the same cell under two different roots. The
        stored-evidence tarball is exactly this shape -- `mcbar` and
        `multiclass` side by side -- and the earlier message ("the pairing key
        is missing a dimension") sent the reader into the scorer when the fix
        was to pass a narrower `--campaign`.
      * the campaign really does sweep an axis the pairing key does not name,
        and averaging the runs would pool it.

    The tell is WHERE the paths diverge. Identical cell tail under different
    prefixes is two campaigns; anything else is a missing dimension. Either way
    the paths themselves are printed, because that is what the reader needs.
    """
    paths = [RUN_DIRS[i] for i in idx if i in RUN_DIRS]
    head = "%d runs share one (cell, seed, arm) key." % len(idx)
    if len(paths) < 2:
        return (head + " Averaging them would pool whatever axis separates "
                "them, so the pairing key is missing a dimension the campaign "
                "varies. (No run paths recorded, so which one cannot be said.)")
    parts = [os.path.normpath(q).replace(os.sep, "/").split("/") for q in paths]
    lines = [head]
    if len({"/".join(q[-LEAF_DEPTH:]) for q in parts}) == 1:
        lines.append("  SAME cell path under DIFFERENT roots: `--campaign` "
                     "points at a tree holding more than one campaign. Score "
                     "each campaign root separately -- pooling them is not a "
                     "scorer setting to change.")
    else:
        lines.append("  Same layout, different run paths: the campaign sweeps "
                     "an axis the pairing key does not name, and averaging "
                     "these would pool it.")
    lines += ["    " + q for q in sorted(paths)[:6]]
    if len(paths) > 6:
        lines.append("    ... and %d more" % (len(paths) - 6))
    return chr(10).join(lines)


def _one(series):
    """Aggregator for the seed pivot: there must be exactly ONE run per
    (cell, seed, arm). More than one means either two campaigns got pooled or
    the pairing key is missing a dimension, and silently averaging them is how
    a swept axis gets pooled. `_collision_msg` separates the two."""
    vals = series.dropna()
    if len(vals) > 1:
        raise ValueError(_collision_msg(list(vals.index)))
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


def ece(y, P):
    """Expected calibration error on the top-1 prediction, 15 bins."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
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
    # THE BINDING budget per class, not the global alone. Local caps are
    # per-group ceilings so their SUM already bounds the count, and on iwildcam
    # the global sits ABOVE that sum and can never bind. Reading the global here
    # inflated the denominator 1.67x on L30_G50 -- the same defect that made
    # `headroom.py` print 59 items of prize where the real gap is 2.0
    # (fixed 2026-08-24). These three quantities are NOT metrics under rule 5,
    # but they are printed, and a diagnostic with a wrong denominator still
    # misleads.
    _Ksum = float(sum(effective_budget(G, L, c) for c in classes))
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
        # How many ITEMS one unit of capped-class F1 is worth here. F1 =
        # 2TP/(K+n) is linear in TP, and `eq` holds exactly K predictions per
        # capped class by construction, so both numbers are already in hand.
        #
        # SUM, not mean: ccF1 is MACRO-averaged over the m capped classes, so a
        # delta d means sum_c dF1_c = m*d, and the items it costs are summed
        # across classes. Taking the mean here understated the count by exactly
        # m -- a factor of 3 on dermmnist's three capped classes.
        "items_per_001": float(np.sum(
            [0.01 * (int((eq == c).sum()) + int((y == c).sum())) / 2
             for c in classes])) if len(classes) else np.nan,
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



def _resolution_readout(perseed, df, arm, control):
    """How many seeds this contrast needs, in the campaign's own unit.

    WHY THIS IS PRINTED AND NOT LEFT TO THE READER. The table above reports a
    delta, a p and a BH q, and none of the three says whether the comparison
    could have SEEN the effect it is reporting on. A tie at four seeds means
    "no effect" or "not enough seeds", and those are opposite conclusions.

    The seed sd is measured WITHIN a cell and then pooled across cells, because
    that is the replication the campaign actually buys when it adds a seed.
    Cells with one seed contribute nothing to the sd and are counted out loud.

    ⚠️ DO NOT ASSUME the null contrast is the quieter one. Every TRAINED arm
    shares a `base_model_id` with its zero-dose null -- warm-up 1 for both --
    while `clip` cannot, because the protocol gives post-hoc arms warm-up 30.
    That much is structural and verified from the run configs. It is tempting
    to conclude the shared warm-up cancels as common mode and makes the null
    contrast cheaper in seeds. **It was tested and it does not hold.** On
    `results/dualbar2` at two seeds the isolation swung 0.2 items for `alm` --
    which is where the idea came from -- and 8.8 items for `fioretto`, worse
    than that arm's own 6.5-item swing against `clip`. The warm-up draw is
    shared; the constraint phase's own stochasticity is not, and it can
    dominate. Which contrast is quieter is therefore an empirical question per
    campaign, which is the reason this readout measures it instead of
    reasoning about it.
    """
    if perseed is None or not len(perseed):
        return
    try:
        from scripts.frozen_head_probe import seeds_needed
    except Exception:                                  # noqa: BLE001
        return

    scale = {}
    for _, r in df.iterrows():
        v = r.get("items_per_001")
        if v is None or not np.isfinite(v):
            continue
        scale.setdefault((r["dataset"], r["model"], r["cap"], r["capped"]),
                         []).append(float(v))
    if not scale:
        return

    sds, ns, means = [], [], []
    for key, g in perseed.groupby(level=[0, 1, 2, 3]):
        sc = scale.get(key)
        if not sc:
            continue
        sc = float(np.mean(sc))
        means.append(float(g.mean()) / 0.01 * sc)
        ns.append(len(g))
        if len(g) >= 2:
            sds.append(float(g.std(ddof=1)) / 0.01 * sc)
    if not means:
        return

    print("")
    print("  RESOLUTION of this contrast -- can it see what it is reporting?")
    single = sum(1 for n in ns if n < 2)
    if not sds:
        print("     every cell has ONE seed, so the seed sd is not estimable and")
        print("     nothing in the table above is separable from seed noise.")
        return
    sd = float(np.mean(sds))
    eff = abs(float(np.mean(means)))
    # the FEWEST seeds in any cell, not the median: power is set by the
    # least-replicated cell, and a median of [2, 1] truncating to 1 reported a
    # cell count that matched no cell.
    have = int(min(ns))
    print("     paired seed sd  %6.2f items  (within cell, pooled over %d cell(s))"
          % (sd, len(sds)))
    print("     observed d ccF1 %+6.2f items  at %d seed(s) in the "
          "least-replicated cell" % (float(np.mean(means)), have))
    if eff > 0:
        need = seeds_needed(eff, sd)
        verdict = ("POWERED" if have >= need else
                   "UNDERPOWERED -- a tie here is not evidence of no effect")
        print("     to detect an effect THIS SIZE at 80%% power needs ~%d seeds "
              "per cell: %s" % (need, verdict))
    if single:
        print("     %d cell(s) have a single seed and contribute no sd" % single)


def detectable_at(sd, n_seeds, power=0.80, alpha=0.05):
    """Smallest effect `n_seeds` can see -- the number a NULL has to be stated with.

    `seeds_needed` answers "how many seeds would I need"; this inverts it to
    "what would I have caught with the seeds I have". They are the same
    arithmetic (`n = z^2 sd^2 / d^2` solved the other way, `d = z sd / sqrt n`),
    but only one of them can be written into a conclusion.

    FRAMEWORK 2(p) requires the iwc1 null be stated as an equivalence -- not
    "AUROC was flat" but "any AUROC effect larger than X would have been seen,
    and none was" -- because a flat result with no bound attached is the "tie
    means no effect" conflation the RESOLUTION block exists to stop. The panel
    already held sd and the seed count; it just never printed the third number.
    """
    if not (np.isfinite(sd) and sd > 0 and n_seeds and n_seeds > 0):
        return float("nan")
    z = 1.959963985 + 0.8416212336
    if (alpha, power) != (0.05, 0.80):
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0) + NormalDist().inv_cdf(power)
    return float(z * sd / np.sqrt(float(n_seeds)))


def _perseed_rows(perseed):
    """[(metric, mean delta, within-cell seed sd, min seeds)] -- the three
    numbers every power statement needs, computed once for whichever family
    asks. The sd is pooled WITHIN cells, never across them (house rule 4)."""
    rows = []
    for m, series in sorted(perseed.items()):
        if series is None or not len(series):
            continue
        sds, ns, means = [], [], []
        for _key, g in series.groupby(level=[0, 1, 2, 3]):
            means.append(float(g.mean()))
            ns.append(len(g))
            if len(g) >= 2:
                sds.append(float(g.std(ddof=1)))
        if not means:
            continue
        rows.append((m, np.mean(means), np.mean(sds) if sds else None,
                     int(min(ns))))
    return rows


def _power_row(m, eff, sd, have, seeds_needed):
    """One printed line: delta, sd, seeds, the bound, the verdict."""
    if sd is None:
        return ("     %-8s %+12.4f %12s %6d %12s   every cell has ONE seed "
                "-- not separable from seed noise" % (m, eff, "n/a", have, "n/a"))
    mde = detectable_at(sd, have)
    if abs(eff) <= 0:
        return ("     %-8s %+12.4f %12.4f %6d %12.4f   exactly zero -- "
                "bound the null at this size, do not call it no effect"
                % (m, eff, sd, have, mde))
    need = seeds_needed(abs(eff), sd)
    verdict = ("POWERED" if have >= need else
               "UNDERPOWERED (~%d needed) -- a tie here is NOT evidence "
               "of no effect" % need)
    return ("     %-8s %+12.4f %12.4f %6d %12.4f   %s"
            % (m, eff, sd, have, mde, verdict))


def _resolution_eq_readout(perseed, arm, control):
    """Power for the BUDGET-EQUALIZED metrics, in their OWN units.

    WHY THIS EXISTS. The items block above prices `d ccF1` and nothing else,
    because `items = dF1 * (K+n)/2` is an F1 identity over the CAPPED classes
    and does not extend to macro-F1 or accuracy. So the panel printed
    **macro-F1 -- the metric the paper headlines** -- with a delta, a
    better/worse count, a Wilcoxon p and no seed cost anywhere. That is the
    ConfGap defect a second time, in the family that carries the paper's claim,
    and it is worse here: macro-F1 is known to be carried by the UNCAPPED
    classes, which swing with the seed, so it is the noisiest number on the
    page and the one quoted most often.

    No items conversion is invented. Native units, and `detectable` says what
    the seeds present would have caught.

    ⚠️ ccP / ccR / ccF1 are one metric in three costumes, and so are
    macroP / macroR / macroF1 -- all monotone in the same counts. Three lines
    agreeing is arithmetic, not corroboration.
    """
    rows = _perseed_rows(perseed)
    if not rows:
        return
    try:
        from scripts.frozen_head_probe import seeds_needed
    except Exception:                                  # noqa: BLE001
        return

    print("")
    print("  RESOLUTION of the BUDGET-EQUALIZED metrics -- native units.")
    print("  `d ccF1` also appears in ITEMS above; that conversion is an F1")
    print("  identity over the capped classes and does NOT extend to macroF1")
    print("  or acc, which is why they are priced here instead of converted.")
    print("  NOTE: macroF1 is carried by the UNCAPPED classes, so it is the")
    print("  noisiest line on this page and the one the paper headlines.")
    print("     %-8s %12s %12s %6s %12s   %s"
          % ("metric", "observed d", "seed sd", "seeds", "detectable",
             "verdict"))
    order = {m: i for i, m in enumerate(EQ_RESOLUTION)}
    for m, eff, sd, have in sorted(rows, key=lambda r: order.get(r[0], 99)):
        print(_power_row(m, eff, sd, have, seeds_needed))


def _resolution_free_readout(perseed, arm, control):
    """Power for the ALLOCATION-FREE metrics, in their OWN units.

    WHY THIS EXISTS SEPARATELY. The RESOLUTION block above converts to ITEMS
    via `items_per_001`, which is an F1 identity (`F1 = 2TP/(K+n)`) and does not
    apply to AP or AUROC. So for years it printed a power statement for exactly
    one metric family -- and the family it covered is the one post-hoc filling
    can reach. Whenever a verdict rests on the allocation-free family instead,
    a flat table carried NO power statement at all, which is precisely the
    "no effect" vs "not enough seeds" conflation the items block was built to
    prevent. FRAMEWORK 2(p) pre-registers the iwc1 verdict on d AP and d AUROC,
    so that gap had to close before the campaign lands rather than after.

    No items conversion is invented here. AP and AUROC are reported in native
    units, which is honest and still answers the only question that matters:
    is the observed effect large against the seed noise of THIS campaign.
    """
    if not perseed:
        return
    try:
        from scripts.frozen_head_probe import seeds_needed
    except Exception:                                  # noqa: BLE001
        return

    rows = _perseed_rows(perseed)
    if not rows:
        return

    print("")
    print("  RESOLUTION of the ALLOCATION-FREE metrics -- these cannot be moved")
    print("  by post-hoc filling, so a verdict resting on them needs its own")
    print("  power statement. Native units, NOT items: the items scale is an F1")
    print("  identity and does not apply here.")
    print("  RANKING moves = the order changed = the only channel a top-K")
    print("  allocator can see. CALIBRATION moves alone = a rescale, which")
    print("  provably leaves every top-K set untouched (FRAMEWORK 2(j)).")
    print("  `detectable` is what the seeds present WOULD have caught. State a")
    print("  flat result with it -- \"any effect above this would have shown\"")
    print("  -- never as \"no effect\", which the seeds cannot support.")
    order = {m: i for i, m in enumerate(FREE_RESOLUTION)}
    rows.sort(key=lambda r: order.get(r[0], 99))
    fam = None
    print("     %-8s %12s %12s %6s %12s   %s"
          % ("metric", "observed d", "seed sd", "seeds", "detectable",
             "verdict"))
    for m, eff, sd, have in rows:
        this = "RANKING" if m in FREE_RANKING else "CALIBRATION"
        if this != fam:
            fam = this
            print("    -- %s --" % this)
        print(_power_row(m, eff, sd, have, seeds_needed))


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


COLLAPSE_DROP_AT_GAP_1 = 0.02


def _collapse_threshold(gap):
    """How big a terminal accuracy DROP counts as a collapse, over `gap` epochs.

    0.02 was calibrated as "~10x the epoch-to-epoch wobble of a converged run".
    That is right for a TRAINED arm, whose 29 constraint epochs are logged
    adjacently, and wrong for a POST-HOC one. `src/pipeline/warmup.py` logs
    `epoch < 3` and then every `max(1, warmup_epochs // 5)`-th epoch, so at the
    protocol's `warmup_epochs: 30` a post-hoc arm's log holds epochs
    1,2,3,6,12,18,24,30 and ITS LAST INTERVAL SPANS SIX EPOCHS. One constant
    across both judged the control (always the post-hoc arm) and the treatment
    (always the trained one) by different standards, which for a detector whose
    loudest output is "the CONTROL collapsed" is the wrong asymmetry to have.

    Fixing it in the LOGGER is not available. `compute_train_accuracy` iterates
    the `shuffle=True` training loader, and a DataLoader iteration draws its
    permutation seed from the global RNG -- so logging a different SET of
    epochs changes every later epoch's batch order, and therefore the result
    and every cached warm-up. Logging density is part of the numerics here.

    MEASURED over the 4,862 `training_log.csv` files in this repository,
    restricted to the converged tail (acc >= 0.9), the per-interval spread of a
    NON-collapsing run grows about as sqrt(gap):

        gap 1   n = 15,464   sd 0.00152   worst drop 0.0214
        gap 5   n = 43,785   sd 0.00300   worst drop 0.0113   (1.97x; sqrt 5 = 2.24)

    so sqrt scaling holds the design constant -- roughly 13x the wobble -- at
    every logging density. At gap 6 that is 0.049, against a worst observed
    non-collapse of 0.011 and the 0.082 drop of the run this detector was
    written for (`dosefix` clip seed 4, 0.9934 -> 0.9116, itself logged at
    gap 6), which it still catches by a factor of 1.7.
    """
    return COLLAPSE_DROP_AT_GAP_1 * math.sqrt(max(1, int(gap)))


def _provenance_key(cfg):
    """One run's provenance, as `((version, data_fingerprint), stamped)`.

    PREFERS THE RUNNER'S STAMP. `code_version` is written by
    `configs/gen_campaign` when the config is CREATED and is never revisited --
    `main()` skips any config already marked completed and rewrites only the
    pending ones -- so it describes the generator, not the run. Run half a
    campaign, land a change to a training file, resume the rest, and every
    config still carries the ORIGINAL value: the gate sees one provenance
    across two pipelines and scores them as one comparison, which is the exact
    thing it exists to refuse. `run_code_version` is stamped by
    `src/experiments/runner.py` at execution time and describes the weights.

    `stamped` is False when the run carries no runner stamp. Every one of the
    14,524 archived runs is in that state, so a missing value must NOT crash
    and must NOT invalidate: it degrades to the old, generator-level check --
    which can still separate two generations -- and `main` says loudly that
    those runs cannot be checked for mid-campaign drift.
    """
    rcv = cfg.get("run_code_version")
    return ((rcv or cfg.get("code_version"), cfg.get("data_fingerprint")),
            bool(rcv))


DOSE_FRACTION_TOLERANCE = 0.05


def _constraint_dose_check(rows):
    """Did every trained arm actually TAKE the constraint steps it was given?

    `finish_constraint_step` returns `applied`, which is False when the
    constraint gradient came back NaN or inf -- on the FP16 path a NaN norm
    fails the `> 0` gate and an inf norm is skipped inside `scaler.step`, so
    either way no update lands. All four trainers used to bind that to
    `_applied` and drop it, and `fioretto` consequently ran a 62%-length
    constraint phase -- 10 of 29 epochs lost, 6 NaN and 4 inf -- while writing
    `status: completed`. Two arms in one campaign can take 29 and 19 steps and
    be reported as the same treatment at the same dose.

    Nothing here can be recovered from a metric: a dropped step leaves no trace
    in the predictions except the effect it did not have.

    Runs from before this was recorded carry no counts. They are named rather
    than skipped -- "we cannot tell" is a different answer from "they agree".
    """
    per = collections.defaultdict(lambda: [0, 0, 0, 0])   # app, att, runs, blind
    for r in rows:
        app, att = r.get("steps_applied"), r.get("steps_attempted")
        cell = per[r["arm"]]
        cell[2] += 1
        if app is None or att is None:
            cell[3] += 1
            continue
        cell[0] += int(app)
        cell[1] += int(att)
    trained = {a: v for a, v in per.items() if v[1] > 0}
    blind = {a: v for a, v in per.items() if v[3] and v[1] == 0}
    if not trained and not blind:
        return
    print("")
    print("CONSTRAINT DOSE -- steps that LANDED, against steps attempted")
    fracs = {}
    for arm, (app, att, n, nb) in sorted(trained.items()):
        frac = app / float(att)
        fracs[arm] = frac
        flag = "" if app == att else "   *** %d STEP(S) LOST" % (att - app)
        print("  %-14s %5d / %-5d applied  (%.1f%%, %d run(s))%s"
              % (arm, app, att, 100.0 * frac, n, flag))
    for arm, (_a, _t, n, nb) in sorted(blind.items()):
        print("  %-14s      no counts recorded (%d of %d run(s) predate the "
              "field)" % (arm, nb, n))
    if any(v[0] != v[1] for v in trained.values()):
        print("    A lost step is a silent dose reduction: the epoch ran, the")
        print("    gradient was non-finite, no update landed, and the run still")
        print("    reports `status: completed`. `constraint_fp32: true`")
        print("    decouples the constraint pass from the CE loss scale.")
    if len(fracs) > 1 and (max(fracs.values()) - min(fracs.values())
                           > DOSE_FRACTION_TOLERANCE):
        lo = min(fracs, key=fracs.get)
        hi = max(fracs, key=fracs.get)
        print("    *** THESE ARMS DID NOT RUN AT THE SAME DOSE: `%s` landed "
              "%.1f%% of its" % (hi, 100.0 * fracs[hi]))
        print("        attempted steps and `%s` landed %.1f%%. An arm-vs-arm "
              "delta across" % (lo, 100.0 * fracs[lo]))
        print("        that gap is confounded with how much constraint phase "
              "each one got.")


def _terminal_collapse(run_dir):
    """Did this run's LAST epoch fall off its own trajectory?

    The pipeline keeps the final epoch's weights unconditionally -- no LR
    schedule, no best-checkpoint (`enable_checkpoint_restore: false` is a
    deliberate protocol choice, since restore was tested as an ARM and
    rejected). So one bad terminal epoch is baked into whatever that run
    becomes, and if the run is a CONTROL it is baked into every comparison
    against it.

    Measured 2026-08-21 on `results/dosefix`: `clip` seed 4 ended at train
    accuracy 0.9116 after 0.9934, while every other control run ended
    0.9935-1.0000. It scored ~15 items below its siblings, so EVERY arm "beat"
    clip at that seed -- and it flipped the 4-seed tralo_null-vs-clip delta
    from -5 items to zero. A single collapsed control reversed the sign of the
    headline number.

    THREE ANSWERS, NOT TWO:

        ("collapse", (last, prev, gap))
        ("ok",       (last, prev, gap))
        ("nolog",    why)

    `None` used to mean both "healthy" and "this run wrote no trajectory at
    all", and the second is the commoner case on exactly the arm that matters.
    A post-hoc arm that LOADS a cached warm-up writes no `training_log.csv`
    whatsoever -- `src/pipeline/warmup.py` returns early on a cache hit and the
    five post-hoc trainers log nothing -- and `clip` + `lp` share one
    `base_model_id`, as do `focal_clip` + `focal_lp`, so whichever of each pair
    the dispatcher runs SECOND is structurally invisible here. When that one is
    the `--control`, the "ONE OF THESE IS THE CONTROL" warning cannot fire even
    though its weights are byte-identical to an arm that did collapse.
    `_collapse_report` resolves that case through the shared warm-up, and
    reports whatever is left over, loudly.
    """
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        # A COMPLETED run with no log is not a healthy run.
        return ("nolog", "no training_log.csv")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return ("nolog", "unreadable training_log.csv (%s)" % type(exc).__name__)
    # Two spellings, because the dual arms write their own log schema. They
    # recorded NO accuracy at all until 2026-08-22, so every fioretto/hounie/alm
    # run was invisible to this detector -- in an 80-run campaign that is 48
    # runs where a terminal collapse could not be seen, on the arms the
    # campaign exists to test.
    col = next((c for c in ("Train_Acc", "train_acc") if c in df.columns), None)
    if col is None:
        return ("nolog", "no train-accuracy column")
    ecol = next((c for c in ("Epoch", "epoch") if c in df.columns), None)
    acc = pd.to_numeric(df[col], errors="coerce")
    ep = (pd.to_numeric(df[ecol], errors="coerce") if ecol
          else pd.Series(range(len(acc)), index=acc.index))
    keep = acc.notna() & ep.notna()
    acc, ep = acc[keep], ep[keep]
    if len(acc) < 2:
        return ("nolog", "fewer than 2 logged epochs")
    last, prev = float(acc.iloc[-1]), float(acc.iloc[-2])
    # THE GAP IS READ, NOT ASSUMED: the warm-up logger writes every sixth epoch
    # at warmup_epochs=30, so "the previous row" is six epochs back for a
    # post-hoc arm and one epoch back for a trained one.
    gap = max(1, int(round(float(ep.iloc[-1]) - float(ep.iloc[-2]))))
    detail = (last, prev, gap)
    return (("collapse" if last < prev - _collapse_threshold(gap) else "ok"),
            detail)


def _collapse_report(rows, control):
    """Terminal-epoch collapse across the campaign, INCLUDING the log-less arms.

    A post-hoc arm that loaded a cached warm-up wrote no trajectory, but its
    final weights ARE that warm-up's final epoch and `base_model_id` names the
    warm-up exactly. So a sibling post-hoc arm on the same `base_model_id` and
    seed answers the question for it: same weights, same verdict. Inheritance
    is restricted to post-hoc arms because for a TRAINED arm the warm-up is one
    epoch out of thirty, and its terminal epoch is not the scored model's.

    Anything still undetermined is printed rather than skipped. A completed run
    whose collapse status cannot be established is a hole in the check, and a
    hole is what let one collapsed control through.
    """
    own, warm = {}, collections.defaultdict(list)
    for i, r in enumerate(rows):
        rd = r.get("run_dir")
        own[i] = _terminal_collapse(rd) if rd else ("nolog", "no run_dir")
        if own[i][0] != "nolog" and r.get("posthoc") and r.get("base_model_id"):
            warm[(r["base_model_id"], r["seed"])].append((r["arm"], own[i]))

    collapsed, unresolved = [], []
    for i, r in enumerate(rows):
        status, detail = own[i]
        if status == "collapse":
            collapsed.append((r["arm"], r["cap"], r["seed"]) + detail + (None,))
            continue
        if status == "ok":
            continue
        sibs = (warm.get((r.get("base_model_id"), r.get("seed")), [])
                if r.get("posthoc") else [])
        hit = next((sb for sb in sibs if sb[1][0] == "collapse"), None)
        if hit:
            collapsed.append((r["arm"], r["cap"], r["seed"]) + hit[1][1]
                             + (hit[0],))
        elif not sibs:
            unresolved.append((r["arm"], r["cap"], r["seed"], detail))

    if unresolved:
        print("")
        print("*** %d COMPLETED RUN(S) WROTE NO TRAINING TRAJECTORY, so a terminal"
              % len(unresolved))
        print("    collapse cannot be ruled out for them. A post-hoc arm that")
        print("    loads a cached warm-up writes no training_log.csv at all, and")
        print("    the pipeline keeps the last epoch unconditionally.")
        for arm, cap, seed, why in sorted(unresolved, key=lambda t: tuple(map(str, t[:3]))):
            print("      %-12s %-10s seed %-4s  %s" % (arm, cap, seed, why))
        if any(a == control for a, _c, _s, _w in unresolved):
            print("    >>> ONE OF THESE IS THE CONTROL `%s`, so the check that"
                  % control)
            print("        matters most is the one that could not run.")

    if collapsed:
        print("")
        print("*** %d RUN(S) COLLAPSED ON THEIR FINAL EPOCH -- and the pipeline"
              % len(collapsed))
        print("    keeps the last epoch unconditionally, so that is the model scored.")
        for arm, cap, seed, last, prev, gap, via in sorted(
                collapsed, key=lambda t: tuple(map(str, t[:3]))):
            print("      %-12s %-10s seed %-4s train acc %.4f -> %.4f "
                  "over %d logged epoch(s)%s"
                  % (arm, cap, seed, prev, last, gap,
                     "  [via the shared warm-up, logged by `%s`]" % via
                     if via else ""))
        if any(a == control for a, _c, _s, _l, _p, _g, _v in collapsed):
            print("    >>> ONE OF THESE IS THE CONTROL `%s`. Every arm will appear to"
                  % control)
            print("        beat it at that seed, and a 4-seed mean can change SIGN on it.")


def _treatment_weight_keys():
    """Every knob whose ZEROING is what makes a null arm a null.

    DERIVED from configs/protocol.yml, not hardcoded, because a hardcoded tuple
    silently stops covering the next null someone adds -- which it already did:
    it listed lambda and `select_eta` only, so `fioretto_null`, `hounie_null`
    and `alm_null` fell through to the treated-arm branch and got the "one run
    counted twice" false alarm this mechanism exists to prevent.

    The rule is a key that is 0 in an `X_null` block AND NON-ZERO in its `X`
    twin. Both halves are needed. `fioretto_lambda_init` is 0.0 in BOTH, so it
    is not what distinguishes them -- taking every zero in the null block would
    classify the TREATED fioretto as untreated, which is worse than the bug it
    replaces. `hounie_eta_u` is 0.01 in both and is excluded the same way.
    """
    keys = {"lambda_global", "lambda_local", "select_eta"}   # floor
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with io.open(os.path.join(root, "configs", "protocol.yml"),
                     encoding="utf-8") as fh:
            blocks = (yaml.safe_load(fh).get("blocks") or {})
        for name, null_block in blocks.items():
            if not (str(name).endswith("_null") and isinstance(null_block, dict)):
                continue
            twin = blocks.get(str(name)[:-len("_null")]) or {}
            for k, v in null_block.items():
                try:
                    zeroed = float(v) == 0.0
                except (TypeError, ValueError):
                    continue
                try:
                    live = float(twin.get(k, 1.0)) != 0.0
                except (TypeError, ValueError):
                    live = True
                if zeroed and live:
                    keys.add(k)
    except Exception:
        pass
    return tuple(sorted(keys))


TREATMENT_WEIGHT_KEYS = _treatment_weight_keys()


def _zero_lambda_arms(rows):
    """Arms whose treatment weight is zero, read from the run CONFIG.

    Not from the `_null` name suffix. The suffix is a convention, and a
    convention is exactly the thing that drifts from the config without
    anything failing. An arm qualifies only if every run of it that could be
    read carries at least one treatment key and every key it carries is 0.
    """
    def _find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                got = _find(v, key)
                if got is not None:
                    return got
        return None

    seen = collections.defaultdict(list)
    for r in rows:
        rd = r.get("run_dir")
        if not rd:
            continue
        try:
            with open(os.path.join(rd, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        vals = [_find(cfg, k) for k in TREATMENT_WEIGHT_KEYS]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        seen[r["arm"]].append(all(float(v) == 0.0 for v in vals))
    return {a for a, flags in seen.items() if flags and all(flags)}


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
                if a.endswith("_null") or b.endswith("_null"):
                    # Under straight_through the penalty is relu(hard - K), so
                    # a seed already UNDER budget at warm-up 1 takes no step
                    # for the entire run and lands bit-identical to its null.
                    # That is correct behaviour and a real zero -- but it is
                    # not a TREATED seed, and averaging it in pulls the effect
                    # toward zero while looking like a null result. `clip`
                    # binds on 63-84% of runs, so one or two per cell is
                    # expected, not a bug.
                    print("      ^ identical to the null means the cap never "
                          "bound on those seeds, so no")
                    print("        constraint step was taken. Those are "
                          "UNTREATED seeds: they are real zeros,")
                    print("        but averaging them in dilutes the effect. "
                          "Report the treated count.")
    for a, b, n in dead:
        print("  *** %s and %s emit BIT-IDENTICAL raw predictions on all %d "
              "cell-seeds. Whatever separates them in the config is INERT. "
              "Any delta below is allocator-only." % (a, b, n))
    # A cap level that changes nothing is the same failure wearing a different
    # hat: the baseline runs in the multiclass campaign were bit-identical
    # across caps, so 12 cells rested on 6 models.
    zero_lam = _zero_lambda_arms(rows)
    for arm in arms:
        if arm in POSTHOC_ARMS:
            continue
        by_cap = collections.defaultdict(dict)
        for (ds, mdl, cap, capped, seed), h in per_arm[arm].items():
            by_cap[(ds, mdl, capped, seed)][cap] = h
        multi = {k: v for k, v in by_cap.items() if len(v) > 1}
        collapsed = [k for k, v in multi.items() if len(set(v.values())) == 1]
        if arm in zero_lam:
            # A lambda=0 arm has the cap REMOVED from its loss, so the cap
            # cannot reach training and its predictions MUST be identical
            # across cap levels. Here identity is the POSITIVE CONTROL and its
            # absence is the defect -- reporting it as "one run counted twice",
            # which is what this check means for a treated arm, sends the
            # reader hunting a bug in the one place there cannot be one.
            if not multi:
                continue
            if len(collapsed) == len(multi):
                print("  ok  %s (treatment weight 0): identical across cap levels in "
                      "%d groups, as it must be -- the cap is not in its loss. "
                      "It is ONE run per (dataset, backbone, seed), so its "
                      "effective n is cells / n_cap_levels, not cells."
                      % (arm, len(multi)))
            else:
                print("  *** %s has treatment weight 0, yet its predictions DIFFER "
                      "across cap levels in %d of %d groups. The cap is not in "
                      "its loss, so it cannot legitimately change the model: "
                      "either the cap is leaking into training or the run is "
                      "nondeterministic. Every paired delta against this "
                      "control is unattributable until that is settled."
                      % (arm, len(multi) - len(collapsed), len(multi)))
            continue
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
    crashed = collections.Counter()
    unscorable = []
    prov = collections.Counter()
    prov_src = collections.defaultdict(set)
    unstamped = 0
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
                # A run that CRASHED is reset to `pending` by the dispatcher,
                # which makes it indistinguishable from one that never started
                # -- so a campaign whose entire TREATMENT arm died reads as
                # "merely unfinished". Measured 2026-08-21: all 8 `tralo` runs
                # of `results/dosefix` died of CUDA OOM in the transductive
                # forward while every control completed (the lambda=0 arm skips
                # that pass entirely), and the panel reported only the controls.
                # The tell is an error_log.json sitting beside the config.
                st = cfg.get("status", "no status")
                # error_log*.json, not error_log.json. Renaming a crash log
                # aside after fixing the cause is the obvious tidy-up, and it
                # silently restores the exact blindness this block exists to
                # remove: the run goes back to looking merely unstarted.
                errs = sorted(glob.glob(os.path.join(os.path.dirname(p),
                                                     "error_log*.json")))
                if errs:
                    err = errs[0]
                    try:
                        e = json.load(open(err))
                        e = e[-1] if isinstance(e, list) else e
                        kind = e.get("exception_type", "unknown")
                    except Exception:
                        kind = "unreadable error_log.json"
                    crashed[(cfg.get("arm", "?"), kind)] += 1
                    st = "%s (CRASHED)" % st
                skipped[st] += 1
                continue
            r = panel(os.path.dirname(p), cfg)
            if r:
                # carried so the terminal-collapse check can reach
                # training_log.csv without re-globbing the campaign
                r["run_dir"] = os.path.dirname(p)
                # and so it can answer for a post-hoc run that wrote NO log:
                # its scored weights are the cached warm-up named by
                # base_model_id, which a sibling post-hoc arm on the same id
                # and seed did log.
                r["base_model_id"] = cfg.get("base_model_id")
                r["posthoc"] = (cfg.get("hyperparams") or {}).get(
                    "constraint_epochs") == 0
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
                _res = cfg.get("results") or {}
                r["steps_applied"] = _res.get("constraint_steps_applied")
                r["steps_attempted"] = _res.get("constraint_steps_attempted")
                r["reordering"] = cfg.get("reordering") or {}
                key, stamped = _provenance_key(cfg)
                prov[key] += 1
                prov_src[key].add("runner" if stamped else "generator")
                if not stamped:
                    unstamped += 1
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
    if unstamped:
        print("")
        print("*** %d of %d scorable run(s) carry NO `run_code_version`, so the"
              % (unstamped, sum(prov.values())))
        print("    provenance gate below reads the GENERATOR's commit for them.")
        print("    That stamp is written when the config is created and never")
        print("    updated, so it can separate two GENERATIONS but CANNOT see a")
        print("    code change landed while the campaign was running -- resume a")
        print("    half-finished campaign after editing a training file and both")
        print("    halves still agree. Re-run them to get the runner's stamp.")
    if len(prov) > 1:
        print("REFUSED: these runs do not share a provenance --")
        for (cv, df), n in sorted(prov.items(), key=lambda kv: -kv[1]):
            print("   %4d run(s)  version=%s (%s)  data_fingerprint=%s"
                  % (n, cv, "/".join(sorted(prov_src[(cv, df)])), df))
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
    _collapse_report(rows, args.control)
    _constraint_dose_check(rows)
    _allocator_check(rows)
    _identity_check(rows)
    _reordering_check(rows)
    if skipped:
        print("skipped %d run(s) that are not completed: %s"
              % (sum(skipped.values()), dict(skipped)))
    if crashed:
        print("")
        print("*** %d SKIPPED RUN(S) DID NOT MERELY FAIL TO START -- THEY CRASHED."
              % sum(crashed.values()))
        for (arm, kind), n in sorted(crashed.items()):
            print("      %-16s %-24s %d run(s)" % (arm, kind, n))
        print("    The dispatcher resets an interrupted run to `pending`, so a dead")
        print("    arm looks identical to an unstarted one. If the crashed arm is the")
        print("    TREATMENT, every comparison below is between controls only.")
        for arm, _k in {(a, k) for a, k in crashed}:
            if arm in RAW_MD5 or any(r.get("arm") == arm for r in rows):
                continue
            print("    >>> `%s` contributed NO scorable run at all." % arm)
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
    RUN_DIRS.clear()
    if "run_dir" in df:
        RUN_DIRS.update(df["run_dir"].to_dict())
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
        perseed_ccf1 = None   # pre-collapse pairs, for the resolution readout
        perseed_free = {}     # ditto for the allocation-free family
        perseed_eq = {}       # ditto for the budget-equalized family
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
                if m == "ccF1":
                    # Keep the PRE-COLLAPSE pairs. The cell mean is the right
                    # unit to TEST on, but it cannot say how many seeds the
                    # contrast needs, and that question is what decides whether
                    # a null is a finding or an underpowered read.
                    perseed_ccf1 = q[arm] - q[args.control]
                if m in FREE_RESOLUTION:
                    perseed_free[m] = q[arm] - q[args.control]
                if m in EQ_RESOLUTION:
                    perseed_eq[m] = q[arm] - q[args.control]
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

        _resolution_readout(perseed_ccf1, df, arm, args.control)
        _resolution_free_readout(perseed_free, arm, args.control)
        _resolution_eq_readout(perseed_eq, arm, args.control)

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

    _items_scale(rows)


def _items_scale(rows):
    """Print how many ITEMS one unit of capped-class F1 is worth, per cell.

    `F1 = 2TP/(K+n)` is linear in TP, so a capped-class F1 delta converts
    exactly into items -- and the effect space here is small enough that the
    conversion changes how a number reads. Measured on dermmnist, the gap from
    `clip` to the ANALYTIC ceiling `2K/(K+n)` is 1.9 items at class 1 / 30% and
    9.9 at class 2 / 50%, against a paired seed sd worth ~2.7 items.

    Printed after every panel because a reader who does not convert will read
    0.02 as a small effect when it can be the entire headroom.
    """
    per = {}
    for r in rows:
        v = r.get("items_per_001")
        if v is None or not np.isfinite(v):
            continue
        per.setdefault((r["dataset"], r["model"], r["cap"], r["capped"]),
                       []).append(v)
    if not per:
        return
    print("=" * 100)
    print("ITEMS PER 0.01 capF1  --  convert before believing a delta")
    print("=" * 100)
    print("  %-64s %14s" % ("cell (dataset/model/cap/capped)", "items / 0.01"))
    for k, v in sorted(per.items()):
        print("  %-64s %14.2f" % ("/".join(str(x) for x in k), float(np.mean(v))))
    print()
    print("  ccF1 is MACRO-averaged over the capped classes, so this is the TOTAL")
    print("  across them: items = d(ccF1) * sum_c (K_c + n_c) / 2. The measured gap")
    print("  from `clip` to the analytic ceiling 2K/(K+n) is 1.9 to 9.9 items PER")
    print("  CLASS, and the paired seed sd is worth ~2.7. A delta smaller than one")
    print("  item is not a delta -- it is a different allocation of the same")
    print("  predictions, and several archived results sit below that line.")
    print()


if __name__ == "__main__":
    main()

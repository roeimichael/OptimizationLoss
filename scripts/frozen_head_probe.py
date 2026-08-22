"""Price a LOSS FAMILY in minutes on CPU instead of a week on GPU.

WHY THIS EXISTS. A GPU campaign takes days and has repeatedly been consumed by
a pipeline defect rather than by the question it was launched to answer. Every
arm this project has rejected shared one property: it changed the training
objective and was then measured through the whole pipeline, so a null could
always be read as "the loss does nothing" OR "the harness ate it".

This harness removes the pipeline. It loads the penultimate features the model
already produced (`src/pipeline/features.py` -> `test_embeddings.npz`), refits
ONLY the final linear head under several losses on frozen features, and scores
the result through the project's REAL endpoint -- the same top-K allocation
`full_panel.py` applies, converted to ITEMS. What survives is the loss family
and nothing else.

    CUDA_VISIBLE_DEVICES="" python -m scripts.frozen_head_probe --synthetic matched
    CUDA_VISIBLE_DEVICES="" python -m scripts.frozen_head_probe --synthetic tailnoise
    CUDA_VISIBLE_DEVICES="" python -m scripts.frozen_head_probe --run-dir results/<c>/<...>/seed_1

================================================================================
SUCCESS CRITERIA -- pre-registered, before any number was read
================================================================================

BASELINE.  The `ce` arm. It is the exact null: identical features, identical
    split, identical initialisation, identical step count, and its refinement
    stage is more cross-entropy. Any arm that ties it has produced nothing.

TARGET.  A loss family earns a GPU campaign only if, against `ce`, on the
    held-out half:
      (a) the paired mean `d ccF1` is worth **>= 1.0 item** -- `full_panel.py`
          states that a delta below one item "is not a delta, it is a different
          allocation of the same predictions";
      (b) the paired two-sided **sign test** clears p <= 0.01;
      (c) the paired mean exceeds **2x its own standard error**.
    All three, or the answer is "no difference". This is a SCREEN, not a
    result: passing it buys a campaign, never a claim.

    STOP -- (c) WAS WRITTEN AS "2x the paired standard DEVIATION" AND THAT
    VERSION IS WRONG. The liveness control caught it before any treatment
    was read:
    in the `tailnoise` regime, where `topk` MUST win by construction, it won
    by +14.94 items on 7 of 8 seeds and the rule returned "NO DIFFERENCE"
    because the sd was 7.53 and 2 x 7.53 = 15.06. The defect is structural,
    not a near miss -- when an effect is itself seed-dependent, its sd grows
    with it, so `mean >= 2*sd` gets HARDER as the effect gets larger. It is
    the right standard for a small effect sitting on the noise floor (it is
    what certified `linear` at L50_G30: mean +0.0078, sd 0.0017, a 4.6x
    separation) and the wrong one for anything else.

    The correction is made on the CONTROL, never on a treatment: the control
    is designed to pass, so its failure is a fact about the rule. The
    conservative reading is not discarded -- every arm that clears the bar
    but has `|mean| < 2*sd` is printed with `[fragile]`, meaning the effect
    is smaller than one seed's swing and a 4-seed campaign could miss it.

BUDGET.  Minutes of CPU, one sitting, one documented hyperparameter default per
    loss. A family that needs a search to clear a screen has failed the screen.

LIVENESS.  A probe that cannot detect a difference is unfalsifiable and its
    null means nothing. Two controls, both reported every run:
      L1  a graded head corruption (`--corrupt-alphas`), which measures the
          probe's RESOLUTION in items rather than merely proving it can see a
          catastrophe;
      L2  the `tailnoise` synthetic regime, built so a cut-local loss MUST beat
          cross-entropy by construction (see `make_synthetic`).
    If L1 cannot separate a corruption worth ~1 item, no ~1-item null from this
    harness is readable, and the run says so.

================================================================================
WHAT THIS PROBE CANNOT DO
================================================================================

The resampling unit here is the SPLIT, not the dataset. `docs/FRAMEWORK.md`
section 0 fixes the generalization unit at three datasets and shows the exact
sign-flip floor is 2^(1-D) = 0.25 there. Eight split seeds on one feature set
buy resolution on THAT feature set and nothing above it. A positive here is a
reason to run a campaign; it is not a result, and it must never be quoted as
one.

The fit half is drawn from the test set. That is the project's own transductive
setting, not a leak introduced here -- but it means the held-out half is the
only scorable population, and the fit half is never scored. Feature
standardisation is fit on the fit half alone and applied unchanged to the
held-out half.

The budget K is computed from held-out labels via the pipeline's own
`compute_global_constraints` / `compute_local_constraints`. That is what a
transductive cap IS in this project; the probe imports those functions rather
than reimplementing them so the probe and the trainer cannot disagree.
"""
import argparse
import json
import os
import random
import sys

# CPU ONLY, and before torch is imported: a GPU visible here would make the
# probe non-reproducible against its own documented numbers for no benefit --
# a linear head on 2k frozen features is milliseconds either way.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
import torch.nn.functional as F                                   # noqa: E402
from sklearn.metrics import f1_score                              # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.full_panel import equalize_multi                     # noqa: E402
from scripts.score_arm import equalize                            # noqa: E402
from src.pipeline.features import EMBEDDING_FILE                  # noqa: E402
from src.training.constraints import (                            # noqa: E402
    compute_global_constraints, compute_local_constraints,
    normalize_constrained_classes)
from src.utils.constants import UNLIMITED                         # noqa: E402

# The real dermmnist test slice, measured from
# `evidence/predictions_mcbar_multiclass_2026-08-18.tar.gz`
# (mcbar/MobileNetV3/dermmnist/L30_G30/clip/seed_1/final_predictions_raw.csv).
# The synthetic regimes reproduce this composition exactly so that a synthetic
# number is at least in the right REGIME -- same n, same class imbalance, same
# genuinely-skewed group 2 -- while remaining honestly synthetic. Provenance
# lives here rather than in prose because a prevalence quoted from memory is
# how this project has been wrong before.
DERM_GROUP_CLASS = np.array([
    [6, 48, 79, 1, 98, 783, 15],     # group 0, n=1030
    [27, 28, 53, 22, 85, 517, 6],    # group 1, n=738
    [32, 27, 88, 0, 40, 41, 7],      # group 2, n=235  (the real skewed site)
], dtype=int)
DERM_CAPPED = [1, 2, 4]


# --------------------------------------------------------------- the inputs --

class ProbeData:
    """Everything a probe run needs, from either the real or synthetic path.

    `ref_probs` is the reference the CE head is checked against: the pipeline's
    own probabilities on the real path, the generative Bayes posterior on the
    synthetic one. They are NOT interchangeable and `ref_name` says which.
    """

    def __init__(self, features, y, groups, classes, local_pct, global_pct,
                 ref_probs, ref_name, label, ref_is_ceiling=False):
        self.features = np.asarray(features, dtype=np.float32)
        self.y = np.asarray(y, dtype=int)
        self.groups = np.asarray(groups, dtype=int)
        self.classes = list(classes)
        self.local_pct = float(local_pct)
        self.global_pct = float(global_pct)
        self.ref_probs = None if ref_probs is None else np.asarray(ref_probs, float)
        self.ref_name = ref_name
        self.label = label
        # A CEILING is not a null. In `tailnoise` the reference is the CLEAN
        # posterior of a population whose labels were then corrupted, so no
        # head fit on those labels can reach it and "the CE head is far below
        # the reference" is the regime working as designed. Gating the null
        # check on a ceiling would report a broken harness on every run of the
        # one regime built to prove the harness works.
        self.ref_is_ceiling = bool(ref_is_ceiling)
        self.n_classes = int(self.y.max()) + 1
        if self.ref_probs is not None:
            self.n_classes = max(self.n_classes, self.ref_probs.shape[1])
        if len(self.features) != len(self.y) or len(self.y) != len(self.groups):
            raise ValueError(
                "features/labels/groups disagree on length (%d/%d/%d)"
                % (len(self.features), len(self.y), len(self.groups)))


def load_real(run_dir):
    """A finished run directory: embeddings + raw predictions + config.

    Reads `final_predictions_raw.csv` for the labels, the groups and the
    model's own probabilities, exactly as `full_panel.panel` does -- the raw
    file, never the allocated one, because the allocated file has already had
    the endpoint applied to it.
    """
    emb = os.path.join(run_dir, EMBEDDING_FILE)
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    cfg_path = os.path.join(run_dir, "config.json")
    missing = [p for p in (emb, raw, cfg_path) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "cannot probe %s -- missing %s.\n"
            "`%s` is written by src/pipeline/features.py at the end of a run, "
            "so only runs finished AFTER that landed carry one. Runs that "
            "predate it cannot be probed and must not be substituted for with "
            "synthetic data."
            % (run_dir, ", ".join(os.path.basename(p) for p in missing),
               EMBEDDING_FILE))

    with np.load(emb) as z:
        if "features" not in z:
            raise SystemExit(
                "%s has no `features` array (keys: %s). src/pipeline/features.py "
                "writes exactly that key." % (emb, list(z.keys())))
        feats = z["features"]
    t = pd.read_csv(raw)
    if "Group_ID" not in t.columns:
        raise SystemExit(
            "%s has no Group_ID column. Every local cap would collapse into "
            "the global one and the probe would score a different constraint "
            "from the one the run was trained under." % raw)
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = t[cols].to_numpy(float)
    if not np.isfinite(P).all():
        raise SystemExit("%s holds non-finite probabilities -- a diverged run. "
                         "full_panel drops these; so does this." % raw)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    classes = normalize_constrained_classes(
        (cfg.get("dataset_config") or {}).get("constrained_class"))
    lp, gp = cfg["constraint"]
    return ProbeData(feats, t["True_Label"].to_numpy(int),
                     t["Group_ID"].to_numpy(int), classes, lp, gp,
                     P, "pipeline probabilities",
                     "REAL %s/%s/%s %s" % (cfg.get("dataset_mode"),
                                           cfg.get("model_name"),
                                           cfg.get("constraint_tag"),
                                           cfg.get("arm")))


def make_synthetic(regime, seed, dim=16, sep=2.1, tail_noise_frac=0.35,
                   local_pct=0.30, global_pct=0.30):
    """Documented, reproducible synthetic features in this project's regime.

    Both regimes draw class-conditional Gaussians in `dim` dimensions with the
    real dermmnist group x class composition above, so a linear head is exactly
    the right model family and cross-entropy is a consistent estimator of the
    Bayes ranking. That is deliberate: the interesting question is whether a
    cut-local loss can beat CE when CE is NOT handicapped.

    `matched`   -- clean labels. CE is asymptotically optimal for the ranking,
                   so every loss SHOULD tie. This is the expected-outcome
                   regime and the one a null is read from.

    `tailnoise` -- LIVENESS CONTROL L2. A fraction of the LOWEST-scoring
                   non-capped items is relabelled INTO a capped class. Those
                   items sit far below the budget cut, so:
                     * cross-entropy must fit them -- it weights every item
                       equally and they are numerous -- which rotates the head
                       away from the direction that separates at the cut;
                     * a loss whose gradient is confined to a window around the
                       K-th ranked item cannot see them at all.
                   So a cut-local loss MUST beat CE here, by construction. If
                   it does not, the harness is broken and no null from it is
                   readable.
    """
    rng = np.random.default_rng(seed)
    n_classes = DERM_GROUP_CLASS.shape[1]
    y, groups = [], []
    for gi, row in enumerate(DERM_GROUP_CLASS):
        for c, n in enumerate(row):
            y.extend([c] * int(n))
            groups.extend([gi] * int(n))
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=int)

    # Class means on a shared random basis, separation chosen so the capped
    # classes land at a realistic difficulty rather than a separable toy: the
    # matched regime's CE head reaches ccF1 in the 0.3-0.5 band the real
    # campaigns report, not 0.99.
    mu = rng.normal(scale=1.0, size=(n_classes, dim)).astype(np.float32)
    mu *= sep / np.linalg.norm(mu, axis=1, keepdims=True)
    X = mu[y] + rng.normal(scale=1.0, size=(len(y), dim)).astype(np.float32)

    # The Bayes posterior of this generative model, in closed form: equal
    # isotropic covariances make it a softmax of -||x-mu||^2/2 plus the log
    # prior. It is the reference the CE head is checked against.
    d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    prior = np.bincount(y, minlength=n_classes) / len(y)
    logit = -0.5 * d2 + np.log(np.clip(prior, 1e-12, None))[None, :]
    ref = np.exp(logit - logit.max(1, keepdims=True))
    ref /= ref.sum(1, keepdims=True)

    if regime == "tailnoise":
        # Score every non-capped item by how strongly it is NOT a capped class,
        # then relabel the least capped-looking tail. These items are the
        # furthest from any budget cut that exists, which is exactly what makes
        # them invisible to a cut-local loss and unavoidable for CE.
        capped_mass = ref[:, DERM_CAPPED].sum(1)
        pool = np.where(~np.isin(y, DERM_CAPPED))[0]
        pool = pool[np.argsort(capped_mass[pool])]          # least capped first
        k = int(round(tail_noise_frac * len(pool)))
        flip = pool[:k]
        y = y.copy()
        y[flip] = np.asarray(DERM_CAPPED)[rng.integers(0, len(DERM_CAPPED), k)]
        # `ref` stays the CLEAN posterior on purpose: it is the ranking a loss
        # would produce if it ignored the corrupted tail perfectly, so it is
        # the ceiling this regime is measured against, not a null.
    elif regime != "matched":
        raise SystemExit("unknown regime %r -- expected matched or tailnoise"
                         % regime)

    return ProbeData(X, y, groups, DERM_CAPPED, local_pct, global_pct,
                     ref,
                     ("clean-label Bayes CEILING" if regime == "tailnoise"
                      else "generative Bayes posterior"),
                     "SYNTHETIC %s (seed %d, d=%d, sep=%.2f)"
                     % (regime, seed, dim, sep),
                     ref_is_ceiling=(regime == "tailnoise"))


# --------------------------------------------------------------- the splits --

def stratified_halves(y, groups, seed, fit_frac=0.5):
    """Deterministic stratified fit / held-out split.

    Stratified on the (class, group) PAIR, not the class: dermmnist's group 2
    is a genuinely different population (TV distance 0.507 in
    `docs/FRAMEWORK.md`), so a class-stratified split would let the group
    composition -- and therefore every LOCAL budget -- drift between halves.
    A class with any (class, group) cell too small to split is demoted to a
    class-only stratum ENTIRELY -- demoting just the rare cell would leave a
    one-member stratum behind, which is the same error one step later, and
    dermmnist has exactly that shape (class 3 is 1 item in group 0 and 0 in
    group 2).
    """
    strata = y.astype(np.int64) * 1000 + groups.astype(np.int64)
    counts = pd.Series(strata).value_counts()
    rare_classes = {int(s) // 1000 for s in counts[counts < 2].index}
    if rare_classes:
        demote = np.isin(y, sorted(rare_classes))
        strata = np.where(demote, -(y.astype(np.int64) + 1), strata)
    left = pd.Series(strata).value_counts()
    if (left < 2).any():
        raise SystemExit(
            "classes %s have a single instance in the whole set, so no "
            "stratified split exists. This population cannot be halved."
            % sorted({abs(int(s)) - 1 for s in left[left < 2].index}))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fit_frac,
                                 random_state=seed)
    fit_idx, held_idx = next(sss.split(np.zeros(len(y)), strata))
    return np.sort(fit_idx), np.sort(held_idx)


def budgets(y_sub, groups_sub, classes, local_pct, global_pct, n_classes):
    """The pipeline's own budgets, on a subset. Never reimplemented here."""
    df = pd.DataFrame({"label": y_sub, "grp": groups_sub})
    G = compute_global_constraints(df, "label", global_pct,
                                   constrained_class=classes,
                                   num_classes=n_classes)
    L = compute_local_constraints(df, "label", local_pct, "grp",
                                  constrained_class=classes,
                                  num_classes=n_classes)
    return G, L


# ----------------------------------------------------------- the allocation --

def allocate(P, groups, G, L, classes):
    """The project's endpoint, not a copy of it.

    `equalize` / `equalize_multi` are imported from the scorer, so a change
    there reaches the probe automatically. `full_panel.panel` branches the same
    way on the number of capped classes.
    """
    if len(classes) == 1:
        return equalize(P, groups, G, L, classes[0])
    return equalize_multi(P, groups, G, L, classes)


def cc_f1(y, alloc, classes):
    return float(f1_score(y, alloc, labels=list(classes), average="macro",
                          zero_division=0))


def items_per_001(y, alloc, classes):
    """`full_panel.py`'s own conversion, character for character in effect.

    ccF1 is MACRO-averaged over the capped classes, so a delta d costs
    sum_c d*(K_c + n_c)/2 items -- summed across classes, not averaged.
    """
    return float(np.sum([0.01 * (int((alloc == c).sum()) + int((y == c).sum())) / 2
                         for c in classes]))


# ------------------------------------------------------------- the polytope --

def _matroid_topk(theta, K, group_idx_list, group_caps):
    """argmax of <theta, y> over {y in [0,1]^n : sum y = K, group budgets}.

    The budget polytope is a partition matroid, so the greedy -- scan
    descending, take while the class has global AND local room -- is exactly
    optimal, and it is the same rule `apply_allocation_heuristic` follows for
    one capped class. Vectorised over a leading sample dimension because the
    perturbed estimator needs M of these per step.

    Equivalence used: an item is taken iff it is inside the top-`cap_g` of its
    own group AND inside the top-K of that survivor set. Proof: only survivors
    are ever taken, and the survivors of group g ranked above item i number
    exactly i's within-group rank, so i's group is never full when reached.
    """
    theta = theta if theta.dim() == 2 else theta.unsqueeze(0)
    M, n = theta.shape
    if group_caps is not None:
        keep = torch.zeros(M, n, dtype=torch.bool)
        for idx, cap in zip(group_idx_list, group_caps):
            if len(idx) == 0:
                continue
            cap = int(min(cap, len(idx)))
            if cap <= 0:
                continue
            sub = theta[:, idx]
            top = sub.topk(cap, dim=1).indices
            keep[:, idx] = keep[:, idx].scatter(1, top, True)
        theta = theta.masked_fill(~keep, float("-inf"))
    K = int(min(K, n))
    out = torch.zeros(M, n)
    if K > 0:
        vals, idxs = theta.topk(K, dim=1)
        # `topk` always returns K entries, so when the local caps admit FEWER
        # than K survivors it hands back masked -inf positions and they would
        # be selected past their own group budget. The greedy simply stops
        # there. Caught by
        # test_the_probe_polytope_argmax_equals_the_allocators_own_greedy,
        # which was verified to FAIL on the version without this line -- and it
        # is reachable on real cap tags, where docs/FRAMEWORK.md measures the
        # local caps summing to exactly the global one (derm L30_G30: global
        # 67, local sum 67), so a single rounding step puts the sum below K.
        out.scatter_(1, idxs, torch.isfinite(vals).to(out.dtype))
    return out


def _group_view(groups, G_c, L_c_by_group):
    """Index lists and per-group caps for one capped class, or (None, None)."""
    if not L_c_by_group:
        return None, None
    gids = sorted(L_c_by_group)
    return ([np.where(groups == g)[0] for g in gids],
            [int(min(L_c_by_group[g], G_c)) for g in gids])


# ----------------------------------------------------------------- the loss --

def _cut(scores, K):
    """The K-th largest score: the operating point, detached.

    Detached on purpose. The cut is where the budget falls, not a parameter to
    be optimised -- letting the gradient move `t` would let a loss lower its
    own bar instead of moving items across it.
    """
    K = int(max(1, min(K, len(scores))))
    return scores.detach().topk(K).values[-1]


def topk_loss(logp_c, pos_mask, K, temp, surrogate):
    """precision@K surrogate: misplacements across the cut, and nothing else.

    With exactly K predictions emitted, ccF1 = 2TP/(K+n) is an affine function
    of the true positives inside the budget, so the endpoint IS precision@K.
    This counts the two ways to lose one: a positive below the cut, a negative
    above it.

    `sigmoid` is the smoothed 0/1 count -- bounded, so its gradient vanishes
    away from the cut, which is the whole point of a cut-local loss.
    `softplus` is the convex alternative and does NOT localise: an item far
    below the cut keeps a constant gradient, so it re-weights the whole ranking
    the way cross-entropy already does. Offered so the localisation itself can
    be attributed rather than assumed.
    """
    t = _cut(logp_c, K)
    z = (logp_c - t) / temp
    below = torch.sigmoid(-z) if surrogate == "sigmoid" else F.softplus(-z)
    above = torch.sigmoid(z) if surrogate == "sigmoid" else F.softplus(z)
    return (below[pos_mask].sum() + above[~pos_mask].sum()) / max(1, int(K))


def pauc_loss(logp_c, pos_mask, neg_frac, temp, max_pairs=200000):
    """Partial AUC / CVaR over the negatives: the false positives inside the budget.

    Ordinary AUC integrates over every false-positive rate, and almost all of
    that range is irrelevant here -- the allocator only ever emits K positives,
    so only the highest-ranked negatives can cost anything. This restricts the
    pair set to the top `neg_frac` of negatives BY CURRENT SCORE, which is the
    CVaR (worst-case tail) formulation of pAUC(0, neg_frac): the tail is
    re-selected every step, so the loss chases whichever negatives have climbed
    into the budget.
    """
    pos = logp_c[pos_mask]
    neg = logp_c[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return logp_c.sum() * 0.0
    m = int(max(1, round(neg_frac * len(neg))))
    hard = neg[neg.detach().topk(m).indices]
    if len(pos) * len(hard) > max_pairs:
        # Subsample the POSITIVES only. Truncating the negative tail would
        # change which negatives the loss is defined over, i.e. the alpha of
        # the partial AUC -- a different loss wearing the same name.
        step = int(np.ceil(len(pos) * len(hard) / max_pairs))
        pos = pos[::step]
    return torch.sigmoid(-(pos[:, None] - hard[None, :]) / temp).mean()


def perturbed_topk_loss(logp_c, pos_mask, K, eps, n_samples, generator,
                        group_idx_list, group_caps):
    """Fenchel-Young loss for a perturbed argmax over the budget polytope.

    y_eps(theta) = E_Z[ argmax_{y in C} <theta + eps Z, y> ] is a soft
    membership vector: the probability, under the perturbation, that each item
    falls inside the budget. The Fenchel-Young loss for that perturbed
    maximiser has gradient exactly

        dL/dtheta = y_eps(theta) - y_target

    (Berthet et al., "Learning with Differentiable Perturbed Optimizers"). It is
    implemented here as the loss VALUE

        L = mean_m <theta + eps Z_m, y_m*> - <theta, y_target>,   y_m* detached

    whose derivative in theta is `mean_m y_m* - y_target = y_eps - y_target` by
    construction, so no gradient is taken through a sort and no external
    dependency is needed. The value is the Monte-Carlo FY loss itself, not a
    straight-through stand-in.

    THE TARGET VERTEX, and it is a real choice. FY needs a target IN the
    polytope, so it must hold exactly K ones -- but the label indicator holds
    n_pos of them, and here K < n_pos by construction (the cap is a BUDGET,
    `K << n_true`, see docs/FRAMEWORK.md). Every K-subset of true positives
    gives the same precision@K, so the target is the feasible all-positive
    vertex nearest the current iterate: the K highest-scoring positives under
    the same matroid greedy, detached. Choosing the highest-scoring ones rather
    than a fixed subset is what keeps the target from fighting the model over
    WHICH positives to rank, a distinction the endpoint does not make.
    """
    n = len(logp_c)
    theta = logp_c
    Z = torch.randn(n_samples, n, generator=generator)
    pert = theta.detach().unsqueeze(0) + eps * Z
    y_star = _matroid_topk(pert, K, group_idx_list, group_caps)      # [M, n]

    masked = torch.where(pos_mask, theta.detach(),
                         torch.full_like(theta, float("-inf")))
    y_target = _matroid_topk(masked, K, group_idx_list, group_caps)[0]
    if not torch.isfinite(masked).any():
        return theta.sum() * 0.0

    lhs = ((theta.unsqueeze(0) + eps * Z) * y_star).sum(1).mean()
    rhs = (theta * y_target).sum()
    return (lhs - rhs) / max(1, int(K))


LOSSES = ("ce", "topk", "pauc", "ptopk")


def special_term(name, logits, y_t, classes, Kg, group_views, args, generator):
    """The added term, summed over capped classes. `ce` adds nothing."""
    if name == "ce":
        return logits.sum() * 0.0
    logp = F.log_softmax(logits, dim=1)
    total = logits.sum() * 0.0
    for c in classes:
        # THE score the allocator thresholds. It ranks by p_ic, and
        # log p_ic = z_ic - logsumexp_j z_ij is monotone in it -- so acting on
        # this quantity is acting at the operating point exactly, while acting
        # on the bare logit z_ic is not (softmax makes the other classes move
        # the ranking).
        s = logp[:, c]
        pos = (y_t == c)
        K = int(Kg[c])
        gidx, gcaps = group_views[c]
        if name == "topk":
            total = total + topk_loss(s, pos, K, args.temp, args.topk_surrogate)
        elif name == "pauc":
            total = total + pauc_loss(s, pos, args.pauc_neg_frac, args.temp)
        elif name == "ptopk":
            total = total + perturbed_topk_loss(
                s, pos, K, args.ptopk_eps, args.ptopk_samples, generator,
                gidx, gcaps)
        else:
            raise SystemExit("unknown loss %r -- expected one of %s"
                             % (name, ", ".join(LOSSES)))
    return total / max(1, len(classes))


# ------------------------------------------------------------------ the fit --

def fit_head(X, y, n_classes, classes, Kg, group_views, loss_name, args, seed,
             warm_start=None):
    """Refit ONLY a linear head, full batch, fixed steps, no early stopping.

    EQUAL COMPUTE BY CONSTRUCTION, and it is the same shape as the project's
    own protocol: every arm shares one CE stage (`--ce-steps`, the analogue of
    the shared warm-up) and then takes exactly `--refine-steps` more. The `ce`
    arm's refinement is more cross-entropy, so the ONLY difference between arms
    is the added term. Nothing here stops early, so "it is training, we will
    see" cannot happen: the stop criterion is the step count, fixed above.
    """
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    if warm_start is None:
        W = (torch.randn(n_classes, X.shape[1], generator=g) * 0.01).requires_grad_(True)
        b = torch.zeros(n_classes, requires_grad=True)
        steps, extra = args.ce_steps, None
    else:
        W = warm_start[0].detach().clone().requires_grad_(True)
        b = warm_start[1].detach().clone().requires_grad_(True)
        steps, extra = args.refine_steps, loss_name

    opt = torch.optim.Adam([W, b], lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = Xt @ W.T + b
        loss = F.cross_entropy(logits, y_t)
        if extra is not None and extra != "ce":
            loss = loss + args.special_weight * special_term(
                extra, logits, y_t, classes, Kg, group_views, args, g)
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def head_probs(W, b, X):
    with torch.no_grad():
        return F.softmax(torch.from_numpy(X) @ W.T + b, dim=1).numpy().astype(float)


def corrupt_head(W, classes, alpha, seed):
    """LIVENESS CONTROL L1: rotate the capped-class rows by a known amount.

    Each capped class's weight row is mixed with a random direction of equal
    norm, so the head keeps its scale and loses only ordering information. The
    strength is SWEPT rather than set to something catastrophic: a control that
    only proves the probe can see a disaster says nothing about whether it can
    see the 1.9-9.9 items that are the entire effect space here. The sweep
    reports the smallest alpha the probe separates, which IS the probe's
    resolution in items.
    """
    g = torch.Generator().manual_seed(seed)
    W = W.clone()
    for c in classes:
        row = W[c]
        noise = torch.randn(row.shape, generator=g)
        noise = noise * (row.norm() / noise.norm().clamp_min(1e-12))
        W[c] = (1.0 - alpha) * row + alpha * noise
    return W


# --------------------------------------------------------------- the report --

def paired(deltas):
    """Mean, spread and sign of a paired vector. Never a single number.

    Both spreads are printed on purpose. `sd` is what the pre-registered bar
    uses -- it asks whether the effect is large against a SINGLE seed's swing,
    which is the question "would this survive a 4-seed campaign". `sem`
    answers the weaker question "is the mean distinguishable from zero", and
    the two disagree by sqrt(n), so quoting only one of them decides the
    verdict by choice of statistic. The exact two-sided sign test is added
    because at 8 seeds its floor is 2^-7 = 0.0078, and a reader needs to know
    the floor before reading a p-value near it.
    """
    d = np.asarray(deltas, float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    pos = int((d > 0).sum())
    nz = int((d != 0).sum())
    k = max(pos, nz - pos)
    p = float(min(1.0, 2.0 * sum(_choose(nz, i) for i in range(k, nz + 1))
                  / (2.0 ** nz))) if nz else 1.0
    return {"mean": float(d.mean()), "sd": sd,
            "sem": sd / np.sqrt(n) if n > 1 else float("nan"),
            # `neg` is counted, never derived as n - pos: an exactly-zero
            # delta is neither, and deriving it reported "8/8 seeds negative"
            # for a corruption that changed nothing at all.
            "pos": pos, "neg": int((d < 0).sum()), "zero": n - nz,
            "n": n, "sign_p": p}


def _choose(n, k):
    from math import comb
    return comb(n, k)


def seeds_needed(effect_items, sd_items, power=0.80, alpha=0.05):
    """Paired seeds a GPU campaign needs to detect `effect_items` at `power`.

    The probe resamples SPLITS and can afford dozens; a campaign resamples
    training seeds and affords four. An effect can therefore be real here and
    structurally invisible there, and that is a decision, not a footnote -- so
    it is priced in the campaign's own unit. Normal approximation, two-sided:
    n = (z_a/2 + z_b)^2 * sd^2 / d^2.
    """
    if not (np.isfinite(effect_items) and np.isfinite(sd_items))             or effect_items <= 0 or sd_items <= 0:
        return float("nan")
    z_a, z_b = 1.959963985, 0.8416212336   # alpha=0.05 two-sided, power=0.80
    if (alpha, power) != (0.05, 0.80):     # anything else is computed, not guessed
        from statistics import NormalDist
        z_a = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        z_b = NormalDist().inv_cdf(power)
    return int(np.ceil((z_a + z_b) ** 2 * sd_items ** 2 / effect_items ** 2))


def verdict(st, n_seeds, min_items, max_sign_p):
    """The pre-registered bar, applied mechanically. See the module docstring
    for why (c) reads standard ERROR and what the `[fragile]` tag means.

    WHY (b) IS A SIGNIFICANCE LEVEL AND NOT A SIGN FRACTION. It was written as
    `7.0/8.0` for an eight-seed run, where it is a p<=0.07 bar. A FRACTION does
    not hold its meaning as seeds are added: the identical 87.5% demands
    p=0.0703 at n=8, p=2.77e-4 at n=24 and p=5.56e-10 at n=64. Adding data made
    the bar eight orders of magnitude harder, so the harness punished its own
    precision and no amount of CPU could ever clear it. A level is invariant.
    0.01 is STRICTER than the old rule at the size it was calibrated for -- at
    n=8 it demands 8/8 (p=0.0078) where the fraction accepted 7/8 -- so this
    tightens the screen where it was designed and only relaxes it where the
    fraction had drifted into absurdity. Directional CONSISTENCY, the other job
    the fraction was doing, is not lost: it is clause (c) plus the `[fragile]`
    tag, which is the question "would a 4-seed campaign see this".
    """
    m = abs(st["mean"])
    ok_size = m >= min_items
    ok_sign = st["sign_p"] <= max_sign_p
    ok_noise = (np.isfinite(st["sem"]) and st["sem"] > 0
                and m >= 2.0 * st["sem"])
    if not ok_size:
        return "NO DIFFERENCE (%.2f < %.1f item)" % (m, min_items)
    if not ok_sign:
        return "NO DIFFERENCE (sign %d/%d, p=%.3g > %.3g)" % (
            st["pos"], n_seeds, st["sign_p"], max_sign_p)
    if not ok_noise:
        return "NO DIFFERENCE (mean %.2f < 2 sem = %.2f items)" % (
            m, 2.0 * st["sem"])
    tag = ""
    if not (np.isfinite(st["sd"]) and st["sd"] > 0 and m >= 2.0 * st["sd"]):
        # The conservative reading, kept visible rather than kept as the rule --
        # priced in SEEDS, because "worth a campaign" and "fragile" otherwise
        # contradict each other and leave the reader to do the power
        # calculation by hand. A campaign that cannot see its own effect is
        # the expensive way to reproduce this line.
        tag = "  [fragile: %.2f < 2 sd = %.2f; a GPU campaign needs ~%d seeds "               "per cell to see this, vs the standard 4]" % (
                  m, 2.0 * st["sd"], seeds_needed(m, st["sd"]))
    return ("WORTH A CAMPAIGN" if st["mean"] > 0 else "WORSE, decisively") + tag


# -------------------------------------------------------------- determinism --

def determinism_digest(seed=1):
    """A digest of a miniature end-to-end probe run. Must be a CONSTANT.

    WHY THIS EXISTS AS A FUNCTION AND NOT AS A TEST BODY. A probe whose answer
    depends on ambient state cannot be trusted to report "no difference",
    which is the answer it will return most often. In-process seeding is not
    enough to prove that: the failure mode is a draw from a GLOBAL generator,
    and whether that changes an answer depends on what ran before -- inside
    one interpreter, and across interpreters (hash randomisation, import
    order, thread count). So the digest is callable from a fresh process and
    the gate compares the two.

    `python -m scripts.frozen_head_probe --selfcheck` prints it.
    """
    import argparse
    import hashlib
    data = make_synthetic("matched", 0, dim=8)
    fit_idx, held_idx = stratified_halves(data.y, data.groups, seed)
    Xf, yf, gf = data.features[fit_idx], data.y[fit_idx], data.groups[fit_idx]
    Xh, yh, gh = data.features[held_idx], data.y[held_idx], data.groups[held_idx]
    mu, sd = Xf.mean(0, keepdims=True), Xf.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xf = ((Xf - mu) / sd).astype(np.float32)
    Xh = ((Xh - mu) / sd).astype(np.float32)
    Gf, Lf = budgets(yf, gf, data.classes, data.local_pct, data.global_pct, 7)
    Gh, Lh = budgets(yh, gh, data.classes, data.local_pct, data.global_pct, 7)
    views = {c: _group_view(gf, Gf[c],
                            {g: lim[c] for g, lim in Lf.items()
                             if c < len(lim) and lim[c] < UNLIMITED})
             for c in data.classes}
    args = argparse.Namespace(
        ce_steps=20, refine_steps=10, lr=0.05, weight_decay=1e-4,
        special_weight=1.0, temp=0.5, topk_surrogate="sigmoid",
        pauc_neg_frac=0.05, ptopk_eps=0.5, ptopk_samples=4)

    h = hashlib.md5()
    shared = fit_head(Xf, yf, 7, data.classes, Gf, views, "ce", args, seed)
    for name in LOSSES:
        W, b = fit_head(Xf, yf, 7, data.classes, Gf, views, name, args, seed,
                        warm_start=shared)
        P = head_probs(W, b, Xh)
        alloc = allocate(P, gh, Gh, Lh, data.classes)
        h.update(W.numpy().tobytes())
        h.update(b.numpy().tobytes())
        h.update(np.asarray(alloc, dtype=np.int64).tobytes())
        h.update(("%.12f" % cc_f1(yh, alloc, data.classes)).encode())
    # the liveness control is part of the contract, so it is part of the digest
    for alpha in (0.1, 0.4):
        h.update(corrupt_head(shared[0], data.classes, alpha, seed)
                 .numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------- driver --

def run_seed(data, seed, args):
    """One split seed: shared CE stage, then every arm, scored on held-out."""
    fit_idx, held_idx = stratified_halves(data.y, data.groups, seed,
                                          args.fit_frac)
    Xf, yf, gf = data.features[fit_idx], data.y[fit_idx], data.groups[fit_idx]
    Xh, yh, gh = data.features[held_idx], data.y[held_idx], data.groups[held_idx]

    # Standardisation is fit on the fit half ONLY and applied unchanged to the
    # held-out half. Fitting it on everything is the textbook leak and would
    # let held-out geometry into the head's training.
    mu, sd = Xf.mean(0, keepdims=True), Xf.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xf = ((Xf - mu) / sd).astype(np.float32)
    Xh = ((Xh - mu) / sd).astype(np.float32)

    n_classes = data.n_classes
    Gf, Lf = budgets(yf, gf, data.classes, data.local_pct, data.global_pct,
                     n_classes)
    Gh, Lh = budgets(yh, gh, data.classes, data.local_pct, data.global_pct,
                     n_classes)
    group_views = {c: _group_view(gf, Gf[c],
                                  {g: lim[c] for g, lim in Lf.items()
                                   if c < len(lim) and lim[c] < UNLIMITED})
                   for c in data.classes}
    for c in data.classes:
        n_pos = int((yf == c).sum())
        if n_pos < Gf[c]:
            raise SystemExit(
                "class %d has %d positives on the fit half but a budget of %d. "
                "The perturbed-argmax target vertex is all-positive, so it "
                "cannot be filled -- this cell is outside the K << n_true "
                "regime the probe assumes." % (c, n_pos, Gf[c]))

    shared = fit_head(Xf, yf, n_classes, data.classes, Gf, group_views, "ce",
                      args, seed)

    out = {}
    for name in args.losses:
        W, b = fit_head(Xf, yf, n_classes, data.classes, Gf, group_views, name,
                        args, seed, warm_start=shared)
        P = head_probs(W, b, Xh)
        alloc = allocate(P, gh, Gh, Lh, data.classes)
        out[name] = {"ccF1": cc_f1(yh, alloc, data.classes),
                     "scale": items_per_001(yh, alloc, data.classes),
                     "W": W, "b": b}

    for alpha in args.corrupt_alphas:
        W = corrupt_head(out["ce"]["W"], data.classes, alpha, seed)
        P = head_probs(W, out["ce"]["b"], Xh)
        alloc = allocate(P, gh, Gh, Lh, data.classes)
        out["corrupt@%g" % alpha] = {"ccF1": cc_f1(yh, alloc, data.classes),
                                     "scale": items_per_001(yh, alloc, data.classes)}

    if data.ref_probs is not None:
        Pr = data.ref_probs[held_idx]
        alloc = allocate(Pr, gh, Gh, Lh, data.classes)
        out["_reference"] = {"ccF1": cc_f1(yh, alloc, data.classes),
                             "scale": items_per_001(yh, alloc, data.classes)}
    out["_n_held"] = len(held_idx)
    out["_budgets"] = {int(c): int(Gh[c]) for c in data.classes}
    return out


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--selfcheck", action="store_true",
                     help="print the determinism digest and exit. It must be "
                          "the same value in every process, on every machine, "
                          "whatever ran before it.")
    src.add_argument("--run-dir", help="a finished run holding %s" % EMBEDDING_FILE)
    src.add_argument("--synthetic", choices=("matched", "tailnoise"),
                     help="synthetic features; results are NOT about any dataset")
    a.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                   help="split seeds; the resampling unit is the SPLIT")
    a.add_argument("--losses", nargs="+", default=list(LOSSES), choices=LOSSES)
    a.add_argument("--fit-frac", type=float, default=0.5)
    a.add_argument("--ce-steps", type=int, default=400,
                   help="shared CE stage, the analogue of the warm-up")
    a.add_argument("--refine-steps", type=int, default=400,
                   help="equal compute for every arm, ce included")
    a.add_argument("--lr", type=float, default=0.05)
    a.add_argument("--weight-decay", type=float, default=1e-4)
    a.add_argument("--special-weight", type=float, default=1.0)
    a.add_argument("--temp", type=float, default=0.5,
                   help="surrogate temperature, in log-probability units")
    a.add_argument("--topk-surrogate", choices=("sigmoid", "softplus"),
                   default="sigmoid")
    a.add_argument("--pauc-neg-frac", type=float, default=0.05)
    a.add_argument("--ptopk-eps", type=float, default=0.5)
    a.add_argument("--ptopk-samples", type=int, default=32)
    a.add_argument("--corrupt-alphas", type=float, nargs="*",
                   default=[0.02, 0.05, 0.10, 0.20, 0.40, 0.80],
                   help="LIVENESS L1: graded head damage, measures resolution")
    a.add_argument("--synthetic-dim", type=int, default=16,
                   help="feature width. Small on purpose: a wide synthetic "
                        "feature space makes the linear head's ESTIMATION "
                        "error dominate, and the probe would then be measuring "
                        "sample size rather than the loss.")
    a.add_argument("--synthetic-sep", type=float, default=2.1,
                   help="class-mean norm. Calibrated ON THE `ce` ARM ALONE, "
                        "before any treatment was run, so that the Bayes ccF1 "
                        "lands at 0.436 and the CE head at 0.422 -- the band "
                        "the real campaigns report (dosefix ccF1 0.3931), not "
                        "a separable toy. Raising it further only makes every "
                        "loss agree sooner.")
    a.add_argument("--tail-noise-frac", type=float, default=0.35)
    a.add_argument("--items-per-001", type=float, default=None,
                   help="items per 0.01 ccF1. Default: computed exactly per "
                        "seed with full_panel's own formula and printed. Pass "
                        "the value full_panel prints for a cell to override.")
    a.add_argument("--min-items", type=float, default=1.0,
                   help="pre-registered bar (a)")
    a.add_argument("--max-sign-p", type=float, default=0.01,
                   help="pre-registered bar (b), a significance level rather "
                        "than a sign fraction -- see verdict()")
    a.add_argument("--headroom-items", type=float, default=9.9,
                   help="the top of the measured gap from `clip` to a PERFECT "
                        "allocator (docs/FRAMEWORK.md: 1.9-9.9 items). A probe "
                        "whose resolution is coarser than this cannot read a "
                        "null on this project's question at all.")
    a.add_argument("--json-out", default=None,
                   help="write raw per-seed ccF1 for post-hoc analysis")
    args = a.parse_args(argv)

    if args.selfcheck:
        print(determinism_digest())
        return None

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    print("=" * 96)
    print("FROZEN-HEAD PROBE -- prices a loss family on CPU, never a result")
    print("=" * 96)

    per_seed, data_label, ref_name = {}, None, None
    for s in args.seeds:
        # Seeded per split, and the seed is logged: the split, the head init,
        # the perturbation stream and the corruption all derive from it, so a
        # row here is reproducible from the number printed beside it.
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if args.synthetic:
            # The generative draw is FIXED across split seeds. Re-drawing it
            # would vary the population and the split together, and the paired
            # spread would then be measuring two things at once.
            data = make_synthetic(args.synthetic, 0, args.synthetic_dim,
                                  args.synthetic_sep, args.tail_noise_frac)
        else:
            data = load_real(args.run_dir)
        data_label, ref_name = data.label, data.ref_name
        per_seed[s] = run_seed(data, s, args)

    print("data          : %s" % data_label)
    print("capped classes: %s   budgets on held-out: %s"
          % (data.classes, per_seed[args.seeds[0]]["_budgets"]))
    print("held-out n    : %d of %d" % (per_seed[args.seeds[0]]["_n_held"],
                                        len(data.y)))
    print("seeds         : %s  (SPLIT seeds -- the unit is the split, not a "
          "dataset)" % args.seeds)
    print("compute       : %d shared CE steps + %d refinement steps, identical "
          "for every arm" % (args.ce_steps, args.refine_steps))
    scales = [per_seed[s]["ce"]["scale"] for s in args.seeds]
    scale = args.items_per_001 if args.items_per_001 else float(np.mean(scales))
    print("items / 0.01  : %.3f  (%s; per-seed %.3f-%.3f)"
          % (scale,
             "given on the command line" if args.items_per_001
             else "computed with full_panel's formula",
             min(scales), max(scales)))
    print()

    ref = [per_seed[s]["_reference"]["ccF1"] for s in args.seeds
           if "_reference" in per_seed[s]]
    if ref:
        d_ref = [per_seed[s]["ce"]["ccF1"] - per_seed[s]["_reference"]["ccF1"]
                 for s in args.seeds]
        st = paired([d / 0.01 * scale for d in d_ref])
        ceiling = getattr(data, "ref_is_ceiling", False)
        print("THE %s -- the CE head against the %s"
              % ("CEILING CHECK" if ceiling else "NULL CHECK", ref_name))
        print("  ccF1  reference %.4f   CE head %.4f   d = %+.4f = %+.2f items "
              "(sd %.2f over %d seeds)"
              % (float(np.mean(ref)),
                 float(np.mean([per_seed[s]["ce"]["ccF1"] for s in args.seeds])),
                 float(np.mean(d_ref)), st["mean"], st["sd"], st["n"]))
        if ceiling:
            print("  This reference is a CEILING, not a null: the labels were")
            print("  corrupted after it was computed, so the CE head is")
            print("  SUPPOSED to sit below it. The gap is the room the regime")
            print("  leaves a cut-local loss to recover -- if it were small,")
            print("  the liveness control would be testing nothing.")
        else:
            print("  TOLERANCE: |d| <= 2.7 items, the paired seed sd this project")
            print("  measures (docs/FRAMEWORK.md section 9). Chosen because the")
            print("  harness is a credible null only if refitting the head moves")
            print("  the endpoint by less than changing the seed does -- above")
            print("  that the harness moves the answer more than the treatment can.")
            print("  -> %s" % ("PASS, the CE arm is a credible null"
                               if abs(st["mean"]) <= 2.7 else
                               "FAIL -- do not read any delta below this size"))
        print()

    print("-" * 96)
    print("PER-SEED ccF1 ON THE HELD-OUT HALF")
    print("-" * 96)
    arms = list(args.losses) + ["corrupt@%g" % x for x in args.corrupt_alphas]
    print("  %-14s %s" % ("arm", " ".join("%8s" % ("s%d" % s) for s in args.seeds)))
    for name in arms:
        print("  %-14s %s" % (name, " ".join(
            "%8.4f" % per_seed[s][name]["ccF1"] for s in args.seeds)))
    print()

    print("-" * 96)
    print("PAIRED AGAINST `ce`, SAME SPLIT AND SAME INITIALISATION, IN ITEMS")
    print("-" * 96)
    print("  %-14s %9s %8s %8s %8s %7s %8s  %s"
          % ("arm", "d items", "sd", "sem", "min", "sign", "sign p", "verdict"))
    summary = {}
    for name in arms:
        if name == "ce":
            continue
        d = [(per_seed[s][name]["ccF1"] - per_seed[s]["ce"]["ccF1"]) / 0.01 * scale
             for s in args.seeds]
        st = paired(d)
        summary[name] = {"per_seed_items": d, **st,
                         "verdict": verdict(st, len(args.seeds), args.min_items,
                                            args.max_sign_p)}
        print("  %-14s %+9.2f %8.2f %8.2f %+8.2f %4d/%d %8.4f  %s"
              % (name, st["mean"], st["sd"], st["sem"], min(d), st["pos"],
                 len(args.seeds), st["sign_p"], summary[name]["verdict"]))
    print()

    live = [(x, summary["corrupt@%g" % x]) for x in args.corrupt_alphas]
    detected = [x for x, s in live if "NO DIFFERENCE" not in s["verdict"]]
    print("-" * 96)
    print("LIVENESS L1 -- graded head corruption. The probe's RESOLUTION.")
    print("-" * 96)
    if detected:
        smallest = min(detected)
        st = summary["corrupt@%g" % smallest]
        res = abs(st["mean"])
        print("  damage vs items, the whole curve (alpha is a knob; items are")
        print("  the unit every other number in this project is quoted in):")
        for x in args.corrupt_alphas:
            row = summary["corrupt@%g" % x]
            print("    alpha %-5g %+8.2f items  %d/%d seeds negative   %s"
                  % (x, row["mean"], row["neg"], len(args.seeds),
                     "resolved" if "NO DIFFERENCE" not in row["verdict"] else "-"))
        print("  -> the probe RESOLVES a %.2f-item effect on this data. A null"
              % res)
        print("     above that size is a measurement; below it, it is silence.")
        if res > args.headroom_items:
            print()
            print("  *** AND THAT IS COARSER THAN THE ENTIRE QUESTION. The gap from")
            print("     `clip` to a PERFECT allocator is 1.9-%.1f items, so a probe"
                  % args.headroom_items)
            print("     that only resolves %.2f cannot tell a real effect from" % res)
            print("     nothing HERE. Every `NO DIFFERENCE` above is UNREADABLE on")
            print("     this input -- not a null, an absence of measurement. The")
            print("     resolution is a property of the FEATURE SPACE, not of the")
            print("     harness, so it must be re-read on every embedding file.")
    else:
        print("  NOTHING DETECTED AT ANY ALPHA. The probe cannot see damage it")
        print("  inflicted itself, so every null above is unfalsifiable and")
        print("  none of them may be reported. Widen --corrupt-alphas or add")
        print("  seeds before reading a single row of this run.")
    print()

    if args.synthetic:
        print("  *** SYNTHETIC. These numbers describe a generative model whose")
        print("  *** group x class COMPOSITION was copied from dermmnist. The")
        print("  *** features are Gaussian draws, not images, and dermmnist is")
        print("  *** itself a REMOVED dataset (FRAMEWORK 2(n)). They license a")
        print("  *** statement about the LOSSES and the HARNESS, and no")
        print("  *** statement whatsoever about this project's data.")
        print()

    if args.json_out:
        payload = {"data": data_label, "seeds": args.seeds,
                   "resolution_items": (abs(summary["corrupt@%g" % min(detected)]["mean"])
                                        if detected else None),
                   "items_per_001": scale, "args": vars(args),
                   "per_seed_ccF1": {name: {str(s): per_seed[s][name]["ccF1"]
                                            for s in args.seeds}
                                     for name in arms},
                   "summary": summary}
        if ref:
            payload["per_seed_ccF1"]["_reference"] = {
                str(s): per_seed[s]["_reference"]["ccF1"] for s in args.seeds}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print("raw per-seed ccF1 -> %s" % args.json_out)
    return summary


if __name__ == "__main__":
    main()

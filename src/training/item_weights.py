"""Per-item weights for the transductive soft count.

WHY THIS EXISTS. Read from source, the reference constraint builds its soft count
as an UNWEIGHTED sum of probabilities (`chunk_eff = chunk_proba`), so

    dL / dp_i(c) = psi'(S_c)      -- IDENTICAL for every item in the scope

and the only per-item differentiation left is the softmax Jacobian p_i(1 - p_i).
Everything the constraint knows about item i is p_i(c) -- the very quantity the
post-hoc allocator already ranks by. That is why it cannot re-order, and it is
one line of code (LEDGER PART 2.1).

The proved harm lemma (LEDGER PART 2.5) leaves exactly one door open: it assumes
the current scores already exhaust the available label information. A weighted
count S_c = sum_i w_i p_i(c) escapes the theorem ONLY IF w_i carries correctness
information that p_i does not. `knn_agree` does: among items the model places in
a capped class at p(c) >= 0.99, neighbourhood agreement predicts WRONG at AUC
0.869 / 0.874, against 0.678 for a margin control that is a pure function of p.

So the weight here is neighbourhood DISAGREEMENT. An item whose neighbours in
embedding space mostly predict something else is the item most likely to be
occupying a capped slot it does not deserve, and it receives the larger share of
the push. It is computed with no labels and no test-time supervision: argmax
agreement among embedding neighbours is available exactly when the transductive
count itself is.

Renormalisation is PER GROUP to mean 1. The local group is the scope that binds
(the global cap has never bound, LEDGER PART 3), and holding the mean at 1 keeps
the weighted count on the same scale as the unweighted one, so the penalty
bounds, the dual updates and the logged counts stay comparable across arms. A
uniform weight vector is exactly all-ones, which reproduces the reference arm
bit-for-bit -- that identity is what `tests/test_item_weights.py` pins.
"""
import torch
import torch.nn.functional as F

from src.pipeline.features import head_and_feature_dim

UNIFORM = "uniform"
KNN_DISAGREE = "knn_disagree"
MODES = (UNIFORM, KNN_DISAGREE)


def read_weight_config(hp):
    """The weighting knob, validated. Absent means the reference unweighted sum."""
    mode = str(hp.get("constraint_weight", UNIFORM))
    if mode not in MODES:
        raise ValueError(
            "constraint_weight must be one of %s, got %r" % (list(MODES), mode))
    k = int(hp.get("constraint_weight_k", 20))
    if mode != UNIFORM and k < 1:
        raise ValueError("constraint_weight_k must be >= 1, got %r" % k)
    floor = float(hp.get("constraint_weight_floor", 0.05))
    if mode != UNIFORM and not (0.0 < floor <= 1.0):
        raise ValueError(
            "constraint_weight_floor must be in (0, 1], got %r" % floor)
    return {"mode": mode, "k": k, "floor": floor}


def _knn_disagreement(feats, preds, k, chunk=2048):
    """Fraction of an item's k nearest neighbours whose argmax differs from its own.

    Cosine similarity on the penultimate features, self excluded. Chunked over
    rows so the n x n similarity is never materialised.
    """
    n = feats.shape[0]
    fn = F.normalize(feats.float(), dim=1)
    k_eff = min(k, max(n - 1, 1))
    out = torch.empty(n, device=feats.device, dtype=torch.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = fn[start:end] @ fn.t()
        rows = torch.arange(start, end, device=feats.device)
        sims[rows - start, rows] = float("-inf")            # exclude self
        nn = sims.topk(k_eff, dim=1).indices
        out[start:end] = (preds[nn] != preds[start:end, None]).float().mean(dim=1)
    return out


def features_and_preds(model, X_test, chunk, device):
    """One no_grad pass returning (penultimate features, argmax preds), both on device.

    The reference arm never calls this. It exists so the weighting arm costs ONE
    extra pass over the test set per constraint epoch rather than two -- the head
    pre-hook gives the features and the same forward gives the logits.
    """
    head, _ = head_and_feature_dim(model)
    grabbed = []

    def _pre_hook(_module, args):
        grabbed.append(args[0].detach().float())

    handle = head.register_forward_pre_hook(_pre_hook)
    was_training = model.training
    preds = []
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test), chunk):
                xb = X_test[i:i + chunk]
                if device is not None:
                    xb = xb.to(device)
                preds.append(model(xb).argmax(dim=1))
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not grabbed:
        raise RuntimeError("the head hook never fired; no embeddings captured")
    feats = torch.cat(grabbed, dim=0)
    if feats.dim() > 2:                       # some heads receive [n, d, 1, 1]
        feats = feats.flatten(1)
    return feats, torch.cat(preds, dim=0)


def item_weights(cfg, feats, preds, group_ids):
    """Per-item weights for the soft count, mean 1 within every group.

    Returns a float32 tensor [n] on `feats.device`. In UNIFORM mode this is
    exactly ones, so the weighted count is bit-identical to the reference sum.
    """
    n = feats.shape[0]
    if cfg["mode"] == UNIFORM:
        return torch.ones(n, device=feats.device, dtype=torch.float32)

    raw = _knn_disagreement(feats, preds, cfg["k"]) + cfg["floor"]
    w = torch.ones(n, device=feats.device, dtype=torch.float32)
    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        mean = raw[mask].mean()
        # A group in which every item agrees with all its neighbours has raw ==
        # floor everywhere; the ratio is then exactly 1 and the group falls back
        # to the reference sum, which is the right behaviour, not a failure.
        w[mask] = raw[mask] / mean if mean > 0 else 1.0
    return w


def apply(w, proba, start, end):
    """`w[start:end] * proba`, in proba's OWN dtype.

    The cast is load-bearing, not tidiness. The counting pass can hand us a
    half-precision `proba` under AMP; `w` is float32, and float32 * float16
    PROMOTES to float32 in torch. The reference arm would then accumulate its
    soft count at a different precision than before this module existed, and
    every stored tralo result would stop reproducing -- from a feature that is
    supposed to be a no-op when switched off. Multiplying by exactly 1.0 within
    a single dtype is bit-exact, so after the cast `uniform` is free.
    """
    return w[start:end, None].to(proba.dtype) * proba


def weight_spread(w):
    """Coefficient of variation of the weights -- 0.0 exactly when they are inert.

    This is the number `gate:weight_bites` reads. A weighting arm that logs 0.0
    here did nothing at all, which is how five previous flags turned out to be
    inert without anyone noticing.
    """
    if w.numel() == 0:
        return 0.0
    mean = float(w.mean())
    if mean == 0.0:
        return 0.0
    return float(w.std(unbiased=False)) / abs(mean)

"""Penultimate features for the test set, saved beside the predictions.

WHY THIS EXISTS. Every stored run holds a 2014x7 score matrix and nothing else,
so every post-hoc question that needs the model's REPRESENTATION -- rather than
its scores -- requires retraining. That is the difference between a ten-minute
offline sweep and a week of GPU, and it is what currently blocks the two
cheapest open probes:

  * refit ONLY the last linear layer under a different loss on frozen features,
    which isolates the loss from the whole pipeline and prices a loss family in
    minutes on CPU rather than in a campaign;
  * any transductive method that reads pairwise structure among test items
    (a k-NN graph over the embedding) instead of the per-item score. That is the
    one input a post-hoc top-K allocator provably does not have: the allocator
    consumes 2014x7 scores, this is 2014xd geometry.

The cap makes the second unusually safe, which is worth stating: an estimator
that improves the ORDERING but drifts in calibration is normally dangerous, and
here the allocator re-imposes the exact budget afterwards, so only the ordering
survives to the metric.

WHY A HOOK AND NOT A MODEL CHANGE. The warm-up cache is keyed on the model
definition, so editing any of the four backbone classes would silently
invalidate every cached warm-up. A forward-pre-hook on the head reads the same
tensor without touching the definition.
"""
import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

EMBEDDING_FILE = "test_embeddings.npz"
# A no_grad inference chunk, deliberately NOT a config key. It cannot change a
# number -- no gradient flows here -- so putting it in protocol.yml would only
# give check_parity another value to compare across arms, which is what made
# `constraint_chunk_size` fail on every campaign while naming two knobs.
EMBEDDING_CHUNK = 128


def head_and_feature_dim(model):
    """The final head module, and the width of whatever feeds it.

    The four backbones name it differently -- `heads` on ViT, `fc` on RegNet,
    `classifier` on both MobileNets -- so it is looked up rather than hardcoded.
    A backbone whose head is not found raises instead of guessing a width: a
    wrong feature dim would produce a silently meaningless embedding file.
    """
    bb = getattr(model, "backbone", model)
    for name in ("heads", "fc", "classifier"):
        head = getattr(bb, name, None)
        if isinstance(head, nn.Module):
            linear = [m for m in head.modules() if isinstance(m, nn.Linear)]
            if linear:
                return head, linear[0].in_features
    raise ValueError(
        "could not find a head module on %s. Add this backbone's head attribute "
        "to the lookup rather than guessing a feature width."
        % type(model).__name__)


def extract_test_embeddings(model, X_test, chunk, device=None):
    """Penultimate features for every test item, as float32 [n, d].

    Pure inference under no_grad, and the hook is removed in a finally block so
    a failure here cannot leave a live hook on a model the caller keeps using.
    """
    head, _ = head_and_feature_dim(model)
    grabbed = []

    def _pre_hook(_module, args):
        grabbed.append(args[0].detach().float().cpu())

    handle = head.register_forward_pre_hook(_pre_hook)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test), chunk):
                xb = X_test[i:i + chunk]
                if device is not None:
                    xb = xb.to(device)
                model(xb)
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not grabbed:
        raise RuntimeError("the head hook never fired; no embeddings captured")
    feats = torch.cat(grabbed, dim=0)
    if feats.dim() > 2:                       # some heads receive [n, d, 1, 1]
        feats = feats.flatten(1)
    return feats.numpy().astype(np.float32)


def save_test_embeddings(experiment_path, model, X_test, chunk, device=None):
    """Write `test_embeddings.npz` beside the predictions. Never fatal.

    A failure here must not lose a finished training run: the embedding is an
    extra artefact for offline analysis, not part of the result. It is logged
    loudly instead, because a silently absent file would be read as "this
    backbone has no features" by whatever tries to load it later.
    """
    try:
        feats = extract_test_embeddings(model, X_test, chunk, device)
    except Exception as exc:                  # noqa: BLE001 - never lose a run
        log.warning("could not save test embeddings (%s: %s). The run is "
                    "unaffected; offline feature analysis will not be "
                    "available for it.", type(exc).__name__, exc)
        return None
    path = experiment_path / EMBEDDING_FILE
    np.savez_compressed(path, features=feats)
    log.info("[final] saved test embeddings %s -> %s", feats.shape, path.name)
    return path

"""heuristic methodology: greedy top-K allocation post-hoc on the warmup model.

No constraint-phase training. Allocates predictions class-by-class
(constrained classes first, sorted by tightest limit) using softmax
probabilities. Remaining samples get their best feasible class.
"""

import logging
import time

import numpy as np
import torch

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.utils.constants import UNLIMITED, CONSTRAINT_CHUNK_SIZE

log = logging.getLogger(__name__)


def _build_hierarchy(num_classes, global_constraints, constrained_classes):
    """Capped classes first (tightest global budget first), then the rest."""
    constrained_sorted = sorted(constrained_classes,
                                key=lambda c: global_constraints[c])
    unconstrained = [c for c in range(num_classes) if c not in constrained_classes]
    return constrained_sorted + unconstrained


def apply_allocation_heuristic(probs, groups, hierarchy, global_constraints,
                               local_constraints, num_classes):
    """Greedy allocation: threshold the ranking at the budget.

    Capped classes are filled by one JOINT pass over every (item, capped class)
    pair in descending probability, so several capped classes compete for the
    same items instead of being served in sequence. With a single capped class
    this is identical to taking that class's top-K.
    """
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}
    argmax_preds = np.argmax(probs, axis=1)

    def _local_limit(group_id, c):
        lim = local_constraints.get(group_id, [UNLIMITED] * num_classes)[c]
        if lim is None or (isinstance(lim, float) and np.isnan(lim)):
            return UNLIMITED
        return lim

    def _slots(group_id):
        if group_id not in current_local:
            current_local[group_id] = {c: 0 for c in range(n_classes)}
        return current_local[group_id]

    def _has_room(group_id, c):
        """Both scopes, independently -- a class may be capped in either."""
        local = _slots(group_id)
        g_limit = global_constraints[c]
        if g_limit < UNLIMITED and current_global[c] >= g_limit:
            return False
        l_limit = _local_limit(group_id, c)
        return not (l_limit < UNLIMITED and local[c] >= l_limit)

    def _take(idx, c):
        local = _slots(groups[idx])
        y_pred[idx] = c
        assigned_mask[idx] = True
        current_global[c] += 1
        local[c] += 1

    capped = [c for c in range(n_classes)
              if global_constraints[c] < UNLIMITED
              or any(_local_limit(g, c) < UNLIMITED for g in local_constraints)]

    # ---- pass 1: every (item, capped class) pair, highest probability first --
    if capped:
        cols = np.array(capped)
        flat = probs[:, cols].ravel()
        order = np.argsort(flat)[::-1]
        items, classes = np.divmod(order, len(cols))
        for idx, ci in zip(items, cols[classes]):
            if assigned_mask[idx]:
                continue
            if _has_room(groups[idx], ci):
                _take(idx, ci)

    # ---- pass 2: uncapped classes take the items that already prefer them ----
    for class_idx in hierarchy:
        if class_idx in capped:
            continue
        unassigned = np.where(~assigned_mask)[0]
        if len(unassigned) == 0:
            break
        candidates = unassigned[argmax_preds[unassigned] == class_idx]
        if len(candidates) == 0:
            continue
        for idx in candidates[np.argsort(probs[candidates, class_idx])[::-1]]:
            if _has_room(groups[idx], class_idx):
                _take(idx, class_idx)

    # ---- pass 3: leftovers go to their best FEASIBLE class ------------------
    forced = 0
    for idx in np.where(~assigned_mask)[0]:
        group_id = groups[idx]
        feasible = [c for c in range(n_classes) if _has_room(group_id, c)]
        if feasible:
            _take(idx, max(feasible, key=lambda c: probs[idx, c]))
        else:
            # Every class is full: sum of budgets < number of items, so the
            # instance is infeasible and SOMETHING must be violated. Say so --
            # the old code silently took argmax of an all -1 vector, which
            # returns 0, and assigned class 0 past its cap.
            forced += 1
            _take(idx, int(np.argmax(probs[idx])))
    if forced:
        log.warning("Allocation infeasible: %d item(s) had no class with room "
                    "left. Budgets sum to less than the test set, so these "
                    "assignments VIOLATE a cap.", forced)
    return y_pred, time.time() - start_time


def verify_allocation(y_pred, groups, global_constraints, local_constraints,
                      num_classes):
    """Return the list of caps `y_pred` violates. Empty list means feasible."""
    bad = []
    for c in range(num_classes):
        limit = global_constraints[c]
        n = int((y_pred == c).sum())
        if limit < UNLIMITED and n > limit:
            bad.append("global class %d: %d > %d" % (c, n, limit))
    for group_id, bounds in local_constraints.items():
        mask = (groups == group_id)
        for c in range(num_classes):
            limit = bounds[c]
            if limit is None or (isinstance(limit, float) and np.isnan(limit)):
                continue
            n = int((y_pred[mask] == c).sum())
            if limit < UNLIMITED and n > limit:
                bad.append("local group %s class %d: %d > %d"
                           % (group_id, c, n, limit))
    return bad


def _infer_probs(model, X_test, chunk_size):
    model.eval()
    with torch.no_grad():
        chunks = [model(X_test[i:i + chunk_size])
                  for i in range(0, len(X_test), chunk_size)]
        probs = torch.softmax(torch.cat(chunks, dim=0), dim=1).cpu().numpy()
    return probs


def train(inputs: TrainInputs) -> TrainOutputs:
    device = inputs.device
    X_test = inputs.X_test.to(device)
    chunk_size = int(inputs.hyperparams.get("constraint_chunk_size",
                                            CONSTRAINT_CHUNK_SIZE))
    probs = _infer_probs(inputs.model, X_test, chunk_size)

    hierarchy = _build_hierarchy(
        inputs.num_classes, inputs.global_con, inputs.constrained_classes)
    y_pred, exec_time = apply_allocation_heuristic(
        probs, inputs.group_ids, hierarchy,
        inputs.global_con, inputs.local_con, inputs.num_classes,
    )
    violations = verify_allocation(
        y_pred, inputs.group_ids, inputs.global_con, inputs.local_con,
        inputs.num_classes)
    if violations:
        raise RuntimeError(
            "heuristic produced predictions that violate %d cap(s): %s. This "
            "arm sets skip_targeted_correction=True, so nothing downstream "
            "would have caught it." % (len(violations), violations[:5]))
    log.info("Heuristic allocation: %.3fs, all caps satisfied", exec_time)

    return TrainOutputs(
        model=inputs.model,
        summary={"allocation_time": exec_time},
        skip_targeted_correction=True,
        precomputed_predictions=y_pred,
    )

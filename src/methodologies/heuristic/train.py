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
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _build_hierarchy(num_classes, global_constraints, constrained_classes):
    constrained_sorted = sorted(constrained_classes,
                                key=lambda c: global_constraints[c])
    unconstrained = [c for c in range(num_classes) if c not in constrained_classes]
    return constrained_sorted + unconstrained


def apply_allocation_heuristic(probs, groups, hierarchy, global_constraints,
                               local_constraints, num_classes):
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}
    argmax_preds = np.argmax(probs, axis=1)
    for class_idx in hierarchy:
        g_limit = global_constraints[class_idx]
        is_constrained = g_limit < UNLIMITED
        unassigned = np.where(~assigned_mask)[0]
        if len(unassigned) == 0:
            break
        if is_constrained:
            class_probs = probs[unassigned, class_idx]
            sorted_indices = unassigned[np.argsort(class_probs)[::-1]]
        else:
            prefer = argmax_preds[unassigned] == class_idx
            candidates = unassigned[prefer]
            if len(candidates) == 0:
                continue
            class_probs = probs[candidates, class_idx]
            sorted_indices = candidates[np.argsort(class_probs)[::-1]]
        for idx in sorted_indices:
            group_id = groups[idx]
            if group_id not in current_local:
                current_local[group_id] = {c: 0 for c in range(n_classes)}
            if is_constrained and current_global[class_idx] >= g_limit:
                break
            l_limit = local_constraints.get(group_id, [UNLIMITED] * num_classes)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = UNLIMITED
            if l_limit < UNLIMITED and current_local[group_id][class_idx] >= l_limit:
                continue
            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1
    remaining = np.where(~assigned_mask)[0]
    for idx in remaining:
        sample_probs = probs[idx].copy()
        group_id = groups[idx]
        if group_id not in current_local:
            current_local[group_id] = {c: 0 for c in range(n_classes)}
        for c in range(n_classes):
            if global_constraints[c] < UNLIMITED and current_global[c] >= global_constraints[c]:
                sample_probs[c] = -1
            if global_constraints[c] < UNLIMITED:
                l_limit = local_constraints.get(group_id, [UNLIMITED] * n_classes)[c]
                if l_limit < UNLIMITED and current_local[group_id].get(c, 0) >= l_limit:
                    sample_probs[c] = -1
        best = np.argmax(sample_probs)
        y_pred[idx] = best
        current_global[best] = current_global.get(best, 0) + 1
        current_local[group_id][best] = current_local[group_id].get(best, 0) + 1
    return y_pred, time.time() - start_time


def _infer_probs(model, X_test, device, chunk_size=256):
    model.eval()
    with torch.no_grad():
        chunks = [model(X_test[i:i + chunk_size])
                  for i in range(0, len(X_test), chunk_size)]
        probs = torch.softmax(torch.cat(chunks, dim=0), dim=1).cpu().numpy()
    return probs


def train(inputs: TrainInputs) -> TrainOutputs:
    device = inputs.device
    X_test = inputs.X_test.to(device)
    probs = _infer_probs(inputs.model, X_test, device)

    hierarchy = _build_hierarchy(
        inputs.num_classes, inputs.global_con, inputs.constrained_classes)
    y_pred, exec_time = apply_allocation_heuristic(
        probs, inputs.group_ids, hierarchy,
        inputs.global_con, inputs.local_con, inputs.num_classes,
    )
    log.info("Heuristic allocation: %.3fs", exec_time)

    return TrainOutputs(
        model=inputs.model,
        summary={"allocation_time": exec_time},
        skip_targeted_correction=True,
        precomputed_predictions=y_pred,
    )

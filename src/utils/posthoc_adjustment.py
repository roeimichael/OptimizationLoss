"""Post-hoc constraint adjustment: flip borderline predictions to satisfy limits."""

import logging

import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger(__name__)


def compute_constraint_delta(predictions, constraint_limit, constrained_class):
    """Positive = over limit, negative = under limit."""
    return int((predictions == constrained_class).sum() - constraint_limit)


def adjust_predictions_to_constraint(predictions, probabilities, constraint_limit,
                                     constrained_class):
    """Flip borderline samples to satisfy constraint. Returns (adjusted_preds, info)."""
    predictions = predictions.copy()
    current_count = (predictions == constrained_class).sum()
    delta = current_count - constraint_limit
    info = {
        'original_count': int(current_count),
        'constraint_limit': constraint_limit,
        'delta': int(delta),
        'samples_adjusted': 0,
        'adjustment_type': 'none',
    }

    if delta == 0:
        return predictions, info

    constrained_probs = probabilities[:, constrained_class]

    if delta > 0:
        indices = np.where(predictions == constrained_class)[0]
        sorted_order = np.argsort(constrained_probs[indices])
        for idx in indices[sorted_order[:delta]]:
            probs = probabilities[idx].copy()
            probs[constrained_class] = -1
            predictions[idx] = np.argmax(probs)
        info['adjustment_type'] = 'drop'
        info['samples_adjusted'] = int(delta)
    else:
        indices = np.where(predictions != constrained_class)[0]
        sorted_order = np.argsort(constrained_probs[indices])[::-1]
        for idx in indices[sorted_order[:abs(delta)]]:
            predictions[idx] = constrained_class
        info['adjustment_type'] = 'add'
        info['samples_adjusted'] = int(abs(delta))

    info['final_count'] = int((predictions == constrained_class).sum())
    info['constraint_satisfied'] = (info['final_count'] <= constraint_limit)
    return predictions, info


def enforce_local_constraints(y_pred, y_proba, group_ids, local_con, constrained_class):
    """Enforce per-group local constraints by flipping lowest-confidence predictions."""
    total_adjusted = 0
    for gid, group_limits in local_con.items():
        g_limit = group_limits[constrained_class]
        if g_limit >= 1e9:
            continue
        g_limit = int(g_limit)
        g_mask = (group_ids == gid)
        g_pred_count = ((y_pred == constrained_class) & g_mask).sum()

        if g_pred_count > g_limit:
            g_constrained = np.where(g_mask & (y_pred == constrained_class))[0]
            g_probs = y_proba[g_constrained, constrained_class]
            sorted_order = np.argsort(g_probs)
            n_to_flip = g_pred_count - g_limit
            flip_indices = g_constrained[sorted_order[:n_to_flip]]

            for idx in flip_indices:
                probs = y_proba[idx].copy()
                probs[constrained_class] = -1
                y_pred[idx] = np.argmax(probs)

            total_adjusted += n_to_flip
            log.info("Local adj group %d: flipped %d (limit=%d)", gid, n_to_flip, g_limit)

    if total_adjusted > 0:
        log.info("Total local adjustments: %d", total_adjusted)
    return y_pred, total_adjusted


def apply_posthoc_adjustment(model, X_test, global_constraints, constrained_class, device='cpu'):
    """Get model predictions and adjust to satisfy constraint."""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        logits = model(X_test)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        original = logits.argmax(dim=1).cpu().numpy()

    constraint_limit = int(global_constraints[constrained_class])
    if constraint_limit >= 1e9:
        return original, original, {'adjustment_type': 'none'}

    adjusted, info = adjust_predictions_to_constraint(
        original, probabilities, constraint_limit, constrained_class)
    return original, adjusted, info

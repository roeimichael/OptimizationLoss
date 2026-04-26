# Classification metrics: accuracy, F1, ECE, calibration, and uncertainty.
# Shared by both optimization and heuristic experiment runners.

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.utils.inference import chunked_forward
from src.utils.constants import UNLIMITED


def compute_prediction_statistics(model, X_test, group_ids, num_classes=7):
    model.eval()
    with torch.no_grad():
        logits = chunked_forward(model, X_test)
        preds = torch.argmax(logits, dim=1)
        proba = torch.softmax(logits, dim=1)
        hard_counts = torch.bincount(preds, minlength=num_classes)
        soft_counts = proba.sum(dim=0)
        global_hard = {c: int(hard_counts[c]) for c in range(num_classes)}
        global_soft = {c: float(soft_counts[c]) for c in range(num_classes)}
        local_hard, local_soft = {}, {}
        for gid in torch.unique(group_ids):
            g = gid.item()
            mask = (group_ids == g)
            g_preds = preds[mask]
            g_proba = proba[mask]
            g_hard = torch.bincount(g_preds, minlength=num_classes)
            g_soft = g_proba.sum(dim=0)
            local_hard[g] = {c: int(g_hard[c]) for c in range(num_classes)}
            local_soft[g] = {c: float(g_soft[c]) for c in range(num_classes)}
    model.train()
    return global_hard, local_hard, global_soft, local_soft


def compute_train_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def get_predictions_with_probabilities(model, X_test):
    model.eval()
    with torch.no_grad():
        logits = chunked_forward(model, X_test)
        preds = logits.argmax(dim=1).cpu().numpy()
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    model.train()
    return preds, proba


def compute_ece(y_true, y_proba, n_bins=15):
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correctness = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        ece += (bin_size / n_samples) * abs(correctness[in_bin].mean() - confidences[in_bin].mean())
    return ece


def compute_uncertainty_metrics(y_true, y_proba):
    n_samples, n_classes = y_proba.shape
    predictions = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    correct_mask = (predictions == y_true)
    clipped = np.clip(y_proba, 1e-10, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(n_samples), y_true] = 1.0
    brier = np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))
    conf_correct = confidences[correct_mask].mean() if correct_mask.any() else 0.0
    conf_incorrect = confidences[~correct_mask].mean() if (~correct_mask).any() else 0.0
    return {
        'mean_entropy': float(normalized_entropy.mean()),
        'mean_confidence': float(confidences.mean()),
        'confidence_correct': float(conf_correct),
        'confidence_incorrect': float(conf_incorrect),
        'confidence_gap': float(conf_correct - conf_incorrect),
        'pct_high_confidence': float((confidences > 0.8).mean()),
        'pct_low_confidence': float((confidences < 0.6).mean()),
        'brier_score': float(brier),
    }


def compute_flips(y_raw, y_adjusted):
    """Count how many predictions changed from raw argmax to post-hoc adjusted."""
    return int((y_raw != y_adjusted).sum())


def compute_raw_constraint_satisfaction(y_raw, global_con, local_con, group_ids,
                                       constrained_classes):
    """Pre-post-hoc constraint satisfaction.

    AUDIT B4: this metric is named `raw_*` for backward compatibility with
    existing CSVs, but its semantics differ across methodologies and the
    name is misleading:

    - For our_approach / fioretto_ldf: y_raw is the argmax of a model
      trained with constraint pressure -- it had a chance to satisfy.
    - For heuristic / po_lp / danits_lp: y_raw is the argmax of the raw
      WARMUP model -- no constraint pressure, will look terrible here even
      when the post-hoc result is identical to our_approach.

    The metric is therefore an honest measure of "did the training phase
    pull predictions toward feasibility" but a DISHONEST head-to-head if
    framed as "constraint satisfaction without post-hoc".

    Outputs are also written to evaluation_metrics.csv under
    `pre_posthoc_*` aliases so downstream readers can use the clearer
    name without breaking older parsers.
    """
    n_global_constrained = 0
    n_global_satisfied = 0
    total_excess = 0

    for c in constrained_classes:
        if c < len(global_con) and global_con[c] < UNLIMITED:
            n_global_constrained += 1
            count = int((y_raw == c).sum())
            limit = int(global_con[c])
            if count <= limit:
                n_global_satisfied += 1
            else:
                total_excess += count - limit

    n_local_constrained = 0
    n_local_satisfied = 0
    if local_con:
        unique_groups = set(group_ids) if not hasattr(group_ids, 'unique') else group_ids.unique()
        for g in unique_groups:
            g_key = int(g) if hasattr(g, 'item') else g
            if g_key not in local_con:
                continue
            g_mask = (group_ids == g)
            g_preds = y_raw[g_mask]
            bounds = local_con[g_key]
            for c in constrained_classes:
                if c < len(bounds) and bounds[c] < UNLIMITED:
                    n_local_constrained += 1
                    count = int((g_preds == c).sum())
                    limit = int(bounds[c])
                    if count <= limit:
                        n_local_satisfied += 1
                    else:
                        total_excess += count - limit

    g_pct = n_global_satisfied / max(n_global_constrained, 1)
    l_pct = n_local_satisfied / max(n_local_constrained, 1)
    all_sat = (n_global_satisfied == n_global_constrained and
               n_local_satisfied == n_local_constrained)

    out = {
        'raw_global_satisfied_pct': float(g_pct),
        'raw_local_satisfied_pct': float(l_pct),
        'raw_all_satisfied': bool(all_sat),
        'raw_total_excess': int(total_excess),
    }
    # B4: emit aliases under the clearer name so downstream code can
    # transition without breaking older parsers.
    out['pre_posthoc_global_satisfied_pct'] = out['raw_global_satisfied_pct']
    out['pre_posthoc_local_satisfied_pct'] = out['raw_local_satisfied_pct']
    out['pre_posthoc_all_satisfied'] = out['raw_all_satisfied']
    out['pre_posthoc_total_excess'] = out['raw_total_excess']
    return out


def compute_metrics(y_true, y_pred, y_proba=None):
    accuracy = np.mean(y_true == y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    result = {
        'accuracy': accuracy,
        'precision_per_class': precision, 'recall_per_class': recall,
        'f1_per_class': f1, 'support_per_class': support,
        'precision_macro': p_macro, 'recall_macro': r_macro, 'f1_macro': f1_macro,
        'precision_weighted': p_weighted, 'recall_weighted': r_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    if y_proba is not None:
        result['ece'] = compute_ece(y_true, y_proba)
        result.update(compute_uncertainty_metrics(y_true, y_proba))
    return result

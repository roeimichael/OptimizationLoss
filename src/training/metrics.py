"""Metrics computation utilities.

Labels and predictions are 0-indexed (0-4) throughout.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def compute_prediction_statistics(model, X_test, group_ids, num_classes=2):
    """Compute hard and soft prediction counts per class and per group.

    Labels and predictions are 0-indexed (0-4).
    Returned counts use 0-indexed keys matching constraint array indices.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)  # 0-4
        proba = torch.softmax(logits, dim=1)

        # Global counts (keys 0-4)
        global_hard = {c: (preds == c).sum().item() for c in range(num_classes)}
        global_soft = {c: proba[:, c].sum().item() for c in range(num_classes)}

        # Local counts per group
        local_hard = {}
        local_soft = {}
        for gid in torch.unique(group_ids):
            g = gid.item()
            mask = (group_ids == g)
            local_hard[g] = {c: (preds[mask] == c).sum().item() for c in range(num_classes)}
            local_soft[g] = {c: proba[mask, c].sum().item() for c in range(num_classes)}

    model.train()
    return global_hard, local_hard, global_soft, local_soft


def compute_train_accuracy(model, loader, device):
    """Compute accuracy on training data."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def get_predictions_with_probabilities(model, X_test):
    """Get predictions and probabilities for test data.

    Returns predictions as 0-indexed (0-4).
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()  # 0-4
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    model.train()
    return preds, proba


def compute_metrics(y_true, y_pred):
    """Compute classification metrics.

    Expects 0-indexed labels (0-4) for both y_true and y_pred.
    """
    accuracy = np.mean(y_true == y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': p_macro,
        'recall_macro': r_macro,
        'f1_macro': f1_macro,
        'precision_weighted': p_weighted,
        'recall_weighted': r_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

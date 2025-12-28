"""
Metrics and evaluation utilities for transductive learning.

This module provides functions for computing prediction statistics
and evaluating model performance.
"""

import torch
import numpy as np


def compute_prediction_statistics(model, X_test_tensor, group_ids_test):
    """
    Compute prediction statistics for test data.

    Args:
        model: Trained neural network model
        X_test_tensor: Test features (torch.Tensor)
        group_ids_test: Group/course IDs for test samples (torch.Tensor)

    Returns:
        tuple: (global_counts, local_counts, global_soft_counts, local_soft_counts)
            - global_counts: Dict {class_id: hard_count}
            - local_counts: Dict {group_id: {class_id: hard_count}}
            - global_soft_counts: Dict {class_id: soft_sum}
            - local_soft_counts: Dict {group_id: {class_id: soft_sum}}
    """
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
        test_proba = torch.nn.functional.softmax(test_logits, dim=1)

        # Global statistics
        global_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count

        global_soft_counts = {}
        for class_id in range(3):
            soft_count = test_proba[:, class_id].sum().item()
            global_soft_counts[class_id] = soft_count

        # Local (per-group) statistics
        unique_groups = torch.unique(group_ids_test)
        local_counts = {}
        local_soft_counts = {}

        for group_id in unique_groups:
            group_id_val = group_id.item()
            group_mask = (group_ids_test == group_id_val)
            group_preds = test_preds[group_mask]
            group_proba = test_proba[group_mask]

            course_counts = {}
            course_soft_counts = {}
            for class_id in range(3):
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count
                soft_count = group_proba[:, class_id].sum().item()
                course_soft_counts[class_id] = soft_count

            local_counts[group_id_val] = course_counts
            local_soft_counts[group_id_val] = course_soft_counts

    model.train()
    return global_counts, local_counts, global_soft_counts, local_soft_counts


def compute_train_accuracy(model, train_loader, device):
    """
    Compute accuracy on training data.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        device: torch device (cpu/cuda)

    Returns:
        float: Training accuracy (0.0 to 1.0)
    """
    model.eval()
    train_correct = 0
    train_total = 0

    with torch.no_grad():
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

    model.train()
    return train_correct / train_total


def evaluate_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    return np.mean(y_true == y_pred)

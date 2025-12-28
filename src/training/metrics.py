import torch
import numpy as np

def compute_prediction_statistics(model, X_test_tensor, group_ids_test):
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
        test_proba = torch.nn.functional.softmax(test_logits, dim=1)

        global_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count

        global_soft_counts = {}
        for class_id in range(3):
            soft_count = test_proba[:, class_id].sum().item()
            global_soft_counts[class_id] = soft_count

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
    return np.mean(y_true == y_pred)

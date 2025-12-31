import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def greedy_constraint_selection(model, X_test_tensor, group_ids_test, y_test,
                                 global_constraints, local_constraints_dict,
                                 experiment_folder):
    """
    FIXED: Greedy constraint-based selection benchmark that RESPECTS constraints.

    Algorithm:
    1. Get predictions and probabilities from the model
    2. For each course, greedily select top predictions for each constrained class
    3. For remaining samples, assign to highest probability class that doesn't violate constraints
    4. Save benchmark results for comparison
    """

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_proba = torch.nn.functional.softmax(test_logits, dim=1).cpu().numpy()
    model.train()

    n_samples = len(test_proba)
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    course_ids = group_ids_test.cpu().numpy()

    final_predictions = np.full(n_samples, -1, dtype=int)

    unique_courses = np.unique(course_ids)

    print("\n" + "="*80)
    print("GREEDY CONSTRAINT-BASED SELECTION (Benchmark)")
    print("="*80)
    print("Algorithm: For each course, select top N samples per class based on probability")
    print("Then assign remaining samples while RESPECTING constraints")
    print("="*80 + "\n")

    global_counts = {0: 0, 1: 0, 2: 0}

    # Phase 1: Assign constrained classes greedily
    for course_id in sorted(unique_courses):
        course_mask = (course_ids == course_id)
        course_indices = np.where(course_mask)[0]
        course_proba = test_proba[course_mask]

        local_cons = local_constraints_dict.get(course_id, [float('inf')] * 3)

        # Process constrained classes (Dropout, Enrolled)
        for class_id in range(3):
            constraint = local_cons[class_id]

            if constraint >= 1e9:
                continue

            class_probs = course_proba[:, class_id]
            top_indices_local = np.argsort(class_probs)[::-1]

            assigned = 0
            for local_idx in top_indices_local:
                global_idx = course_indices[local_idx]

                if final_predictions[global_idx] == -1:
                    if global_counts[class_id] < global_constraints[class_id]:
                        final_predictions[global_idx] = class_id
                        global_counts[class_id] += 1
                        assigned += 1

                        if assigned >= constraint:
                            break

    # Phase 2: Assign remaining samples while respecting constraints
    unassigned_indices = np.where(final_predictions == -1)[0]

    print(f"Phase 1 complete: {len(final_predictions) - len(unassigned_indices)} samples assigned")
    print(f"Phase 2: Assigning {len(unassigned_indices)} remaining samples with constraint checking...")

    for idx in unassigned_indices:
        sample_course = course_ids[idx]
        sample_proba = test_proba[idx]
        local_cons = local_constraints_dict.get(sample_course, [float('inf')] * 3)

        # Try to assign to classes in order of probability
        classes_by_prob = np.argsort(sample_proba)[::-1]

        assigned = False
        for class_id in classes_by_prob:
            # Check global constraint
            if global_counts[class_id] >= global_constraints[class_id]:
                continue

            # Check local constraint
            # Count current predictions for this course and class
            course_mask = (course_ids == sample_course)
            course_predictions = final_predictions[course_mask]
            current_class_count = np.sum(course_predictions == class_id)

            if current_class_count < local_cons[class_id]:
                # Safe to assign!
                final_predictions[idx] = class_id
                global_counts[class_id] += 1
                assigned = True
                break

        # If all classes violate constraints, assign to unlimited class (Graduate)
        if not assigned:
            final_predictions[idx] = 2  # Graduate (unlimited)
            global_counts[2] += 1

    # Verify no samples left unassigned
    assert np.all(final_predictions != -1), "Some samples were not assigned!"

    accuracy = np.mean(y_true == final_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, final_predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, final_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, final_predictions, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, final_predictions)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }

    print("\n" + "-"*80)
    print("BENCHMARK RESULTS")
    print("-"*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print("-"*80)

    print("\nGlobal Constraint Satisfaction:")
    class_names = ['Dropout', 'Enrolled', 'Graduate']
    for class_id in range(3):
        constraint = global_constraints[class_id]
        predicted = global_counts[class_id]
        if constraint < 1e9:
            status = "OK" if predicted <= constraint else "VIOLATED"
            print(f"  {class_names[class_id]}: {predicted}/{int(constraint)} - {status}")
        else:
            print(f"  {class_names[class_id]}: {predicted} (Unlimited)")
    print("="*80 + "\n")

    # Verify constraints are satisfied
    print("\nConstraint Verification:")
    global_violated = False
    for class_id in range(3):
        if global_counts[class_id] > global_constraints[class_id]:
            print(f"  WARNING: Global constraint violated for class {class_id}")
            global_violated = True

    if not global_violated:
        print("  All global constraints satisfied")

    os.makedirs(experiment_folder, exist_ok=True)

    save_benchmark_predictions(
        os.path.join(experiment_folder, 'benchmark_predictions.csv'),
        y_true, final_predictions, test_proba, course_ids
    )

    save_benchmark_metrics(
        os.path.join(experiment_folder, 'benchmark_metrics.csv'),
        metrics
    )

    save_benchmark_constraint_comparison(
        os.path.join(experiment_folder, 'benchmark_constraint_comparison.csv'),
        final_predictions, course_ids, global_constraints, local_constraints_dict
    )

    return final_predictions, metrics

def save_benchmark_predictions(save_path, y_true, y_pred, y_proba, course_ids):
    df_data = {
        'Sample_Index': list(range(len(y_true))),
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'Prob_Dropout': y_proba[:, 0],
        'Prob_Enrolled': y_proba[:, 1],
        'Prob_Graduate': y_proba[:, 2],
        'Correct': (y_true == y_pred).astype(int),
        'Course_ID': course_ids
    }
    df = pd.DataFrame(df_data)
    df.to_csv(save_path, index=False)
    print(f"Benchmark predictions saved to: {save_path}")

def save_benchmark_metrics(save_path, metrics):
    import csv
    class_names = ['Dropout', 'Enrolled', 'Graduate']

    rows = []
    rows.append(['Metric', 'Value'])
    rows.append(['Overall Accuracy', f"{metrics['accuracy']:.4f}"])
    rows.append([''])
    rows.append(['Macro Averaged Metrics', ''])
    rows.append(['Precision (Macro)', f"{metrics['precision_macro']:.4f}"])
    rows.append(['Recall (Macro)', f"{metrics['recall_macro']:.4f}"])
    rows.append(['F1-Score (Macro)', f"{metrics['f1_macro']:.4f}"])
    rows.append([''])
    rows.append(['Weighted Averaged Metrics', ''])
    rows.append(['Precision (Weighted)', f"{metrics['precision_weighted']:.4f}"])
    rows.append(['Recall (Weighted)', f"{metrics['recall_weighted']:.4f}"])
    rows.append(['F1-Score (Weighted)', f"{metrics['f1_weighted']:.4f}"])
    rows.append([''])
    rows.append(['Per-Class Metrics', ''])
    rows.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    for i, class_name in enumerate(class_names):
        rows.append([
            class_name,
            f"{metrics['precision_per_class'][i]:.4f}",
            f"{metrics['recall_per_class'][i]:.4f}",
            f"{metrics['f1_per_class'][i]:.4f}",
            int(metrics['support_per_class'][i])
        ])
    rows.append([''])
    rows.append(['Confusion Matrix', ''])
    rows.append([''] + class_names)
    cm = metrics['confusion_matrix']
    for i, class_name in enumerate(class_names):
        rows.append([class_name] + [int(cm[i][j]) for j in range(3)])

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Benchmark metrics saved to: {save_path}")

def save_benchmark_constraint_comparison(save_path, predictions, course_ids,
                                         global_constraints, local_constraints_dict):
    unique_courses = np.unique(course_ids)
    rows = []

    class_names = ['Dropout', 'Enrolled', 'Graduate']

    for course_id in sorted(unique_courses):
        course_mask = (course_ids == course_id)
        course_preds = predictions[course_mask]

        local_cons = local_constraints_dict.get(course_id, [float('inf')] * 3)

        for class_id in range(3):
            predicted = np.sum(course_preds == class_id)
            constraint = local_cons[class_id]

            if constraint < 1e9:
                overprediction = max(0, predicted - constraint)
                status = "OK" if predicted <= constraint else "OVER"
            else:
                overprediction = 0
                constraint = 'Unlimited'
                status = "N/A"

            rows.append({
                'Course_ID': course_id,
                'Class': class_names[class_id],
                'Constraint': constraint,
                'Predicted': predicted,
                'Overprediction': overprediction,
                'Status': status
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Benchmark constraint comparison saved to: {save_path}")

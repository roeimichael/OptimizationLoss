#!/usr/bin/env python3
"""
Setup and run saturation on all optimization experiments.

This script:
1. Creates results/saturated_approach/ folder structure
2. Copies experiment configs from our_approach
3. Applies saturation to all experiments
4. Saves saturated results in the new folder

This gives you 3 complete result sets for comparison:
- results/our_approach/     : Pure optimization (constraint training)
- results/heuristic/         : Pure heuristic (warmup + greedy allocation)
- results/saturated_approach/: Hybrid (optimization + saturation)
"""

import json
import csv
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete
from src.training.metrics import compute_metrics
from src.utils.data_loader import load_experiment_data


def load_optimization_predictions(experiment_path: Path):
    """Load predictions from optimization results."""

    predictions_file = experiment_path / 'final_predictions.csv'

    if not predictions_file.exists():
        raise FileNotFoundError(f"No predictions found at {predictions_file}")

    y_true = []
    y_pred = []
    probs = []
    groups = []

    with open(predictions_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row['True_Label']))
            y_pred.append(int(row['Predicted_Label']))
            probs.append([
                float(row['Prob_Dropout']),
                float(row['Prob_Enrolled']),
                float(row['Prob_Graduate'])
            ])
            groups.append(int(row['Course_ID']))

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(probs),
        np.array(groups)
    )


def compute_current_allocations(y_pred, groups):
    """Compute current global and local allocations."""

    # Global counts
    global_counts = {c: int(np.sum(y_pred == c)) for c in range(3)}

    # Local counts per group
    local_counts = {}
    for group_id in np.unique(groups):
        group_mask = groups == group_id
        group_preds = y_pred[group_mask]
        local_counts[group_id] = {c: int(np.sum(group_preds == c)) for c in range(3)}

    return global_counts, local_counts


def apply_saturation(y_pred, probs, groups, global_constraint, local_constraint):
    """Apply heuristic saturation to fill unused constraint budget."""

    y_saturated = y_pred.copy()
    current_global, current_local = compute_current_allocations(y_pred, groups)

    # Process constrained classes [0, 1]
    for class_idx in [0, 1]:
        g_limit = global_constraint[class_idx]

        if g_limit >= 1e8:
            continue  # Skip unlimited

        current_count = current_global[class_idx]
        remaining_budget = int(g_limit - current_count)

        if remaining_budget <= 0:
            continue

        # Find candidates: currently class 2, sorted by probability for class_idx
        candidates = []
        for idx in range(len(y_saturated)):
            if y_saturated[idx] == 2:  # Currently Graduate
                prob = probs[idx, class_idx]
                group_id = groups[idx]

                # Check local constraint
                if group_id in local_constraint:
                    local_limit = local_constraint[group_id][class_idx]
                    if local_limit < 1e8:
                        local_current = current_local[group_id][class_idx]
                        if local_current >= local_limit:
                            continue

                candidates.append((idx, prob, group_id))

        # Sort by probability (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Reassign top candidates
        reassigned = 0
        for idx, prob, group_id in candidates:
            if reassigned >= remaining_budget:
                break

            # Double-check local constraint
            if group_id in local_constraint:
                local_limit = local_constraint[group_id][class_idx]
                if local_limit < 1e8:
                    if current_local[group_id][class_idx] >= local_limit:
                        continue
                    current_local[group_id][class_idx] += 1

            # Reassign
            y_saturated[idx] = class_idx
            current_global[class_idx] += 1
            reassigned += 1

    return y_saturated


def save_saturated_predictions(output_path, y_true, y_pred_original, y_pred_saturated, probs, groups):
    """Save saturated predictions to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Sample_Index', 'True_Label', 'Predicted_Original', 'Predicted_Saturated',
            'Prob_Dropout', 'Prob_Enrolled', 'Prob_Graduate', 'Correct', 'Course_ID'
        ])

        for i in range(len(y_true)):
            correct = 1 if y_pred_saturated[i] == y_true[i] else 0
            writer.writerow([
                i, y_true[i], y_pred_original[i], y_pred_saturated[i],
                f"{probs[i, 0]:.6f}", f"{probs[i, 1]:.6f}", f"{probs[i, 2]:.6f}",
                correct, groups[i]
            ])


def save_evaluation_metrics(output_path, metrics):
    """Save evaluation metrics to CSV."""

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

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def setup_and_run_saturations():
    """Main function to setup structure and run all saturations."""

    print("=" * 80)
    print("BATCH SATURATION: OPTIMIZATION + HEURISTIC SATURATION")
    print("=" * 80)

    source_dir = Path('results/our_approach')
    target_dir = Path('results/saturated_approach')

    if not source_dir.exists():
        print(f"ERROR: {source_dir} does not exist!")
        return False

    # Get all experiment directories
    all_configs = sorted(list(source_dir.rglob('config.json')))

    print(f"\nFound {len(all_configs)} experiments in our_approach/")
    print(f"Will apply saturation and save to: {target_dir}/\n")

    # Setup progress tracking
    success_count = 0
    error_count = 0
    errors = []

    # Track statistics
    stats = {
        'total': len(all_configs),
        'improvements': [],
        'no_improvement': 0,
        'saturated_samples': []
    }

    for i, source_config_path in enumerate(all_configs, 1):
        source_exp_dir = source_config_path.parent
        rel_path = source_exp_dir.relative_to(source_dir)

        # Create target directory
        target_exp_dir = target_dir / rel_path
        target_exp_dir.mkdir(parents=True, exist_ok=True)

        # Load source config
        try:
            config = load_config_from_path(source_exp_dir)

            # Extract metadata
            model_name = config.get('model_name', 'unknown')
            constraint = config.get('constraint', [])
            constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"
            variation = config.get('variation_name', 'unknown')

            print(f"[{i}/{len(all_configs)}] {model_name} | Constraint {constraint_str} | {variation}")

            # Load optimization predictions
            y_true, y_pred_opt, probs, groups = load_optimization_predictions(source_exp_dir)

            # Load constraints
            _, _, _, _, _, global_constraint, local_constraint = load_experiment_data(config)

            # Apply saturation
            y_pred_saturated = apply_saturation(y_pred_opt, probs, groups, global_constraint, local_constraint)

            # Count saturated samples
            num_changed = np.sum(y_pred_saturated != y_pred_opt)
            stats['saturated_samples'].append(num_changed)

            # Compute metrics
            opt_metrics = compute_metrics(y_true, y_pred_opt)
            sat_metrics = compute_metrics(y_true, y_pred_saturated)

            improvement = sat_metrics['accuracy'] - opt_metrics['accuracy']

            if improvement > 0:
                stats['improvements'].append(improvement)
                print(f"  ✓ Accuracy: {opt_metrics['accuracy']:.4f} → {sat_metrics['accuracy']:.4f} "
                      f"(+{improvement:.4f}, {num_changed} saturated)")
            else:
                stats['no_improvement'] += 1
                print(f"  → Accuracy: {opt_metrics['accuracy']:.4f} (no saturation needed)")

            # Update config for saturated approach
            config['methodology'] = 'saturated_approach'
            config['experiment_path'] = str(target_exp_dir).replace('\\', '/')

            # Store both original optimization and saturated results
            config['optimization_results'] = config.get('results', {})
            config['results'] = {
                'accuracy': float(sat_metrics['accuracy']),
                'precision_macro': float(sat_metrics['precision_macro']),
                'recall_macro': float(sat_metrics['recall_macro']),
                'f1_macro': float(sat_metrics['f1_macro']),
                'methodology': 'saturated',
                'samples_saturated': int(num_changed)
            }
            config['status'] = 'completed'

            # Save saturated predictions
            save_saturated_predictions(
                target_exp_dir / 'final_predictions.csv',
                y_true, y_pred_opt, y_pred_saturated, probs, groups
            )

            # Save evaluation metrics
            save_evaluation_metrics(target_exp_dir / 'evaluation_metrics.csv', sat_metrics)

            # Save config
            save_config_to_path(config, target_exp_dir)
            mark_experiment_complete(target_exp_dir)

            success_count += 1

        except Exception as e:
            error_count += 1
            error_msg = f"{model_name}/{constraint_str}/{variation}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"  ✗ ERROR: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("SATURATION COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {stats['total']}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    if stats['improvements']:
        avg_improvement = np.mean(stats['improvements'])
        max_improvement = np.max(stats['improvements'])
        improved_count = len(stats['improvements'])
        print(f"\nSaturation improved {improved_count}/{stats['total']} experiments:")
        print(f"  Average improvement: +{avg_improvement:.4f} accuracy")
        print(f"  Max improvement: +{max_improvement:.4f} accuracy")
        print(f"  Average samples saturated: {np.mean(stats['saturated_samples']):.1f}")

    if stats['no_improvement'] > 0:
        print(f"\n{stats['no_improvement']} experiments already at full budget utilization")

    if errors:
        print(f"\n⚠️  Errors encountered:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nResults saved to: {target_dir}/")
    print("\nYou now have 3 complete result sets:")
    print("  1. results/our_approach/      - Pure optimization")
    print("  2. results/heuristic/          - Pure heuristic")
    print("  3. results/saturated_approach/ - Optimization + saturation")
    print("=" * 80)

    return success_count == stats['total']


def main():
    try:
        success = setup_and_run_saturations()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

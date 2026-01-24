#!/usr/bin/env python3
"""
Test: Does constraint training help or hurt?
Compare warmup model vs constraint-trained model, both with heuristic allocation.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path
from src.training.trainer import ConstraintTrainer
from src.training.metrics import compute_metrics
from run_heuristic import apply_allocation_heuristic


def test_model(config_path: str, use_warmup_only: bool = False):
    """Test model with heuristic allocation."""

    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cpu')

    # Load data
    X_train_clean, X_test_clean, _, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # Load model
    trainer = ConstraintTrainer(config, str(experiment_path), device)

    if use_warmup_only:
        # Load ONLY the warmup model (cached)
        trainer.setup_model(X_train_clean.shape[1], config['base_model_id'])
        print(f"✓ Using WARMUP-ONLY model: {config['base_model_id']}")
    else:
        # Load the constraint-trained model
        # First setup with warmup, then would load constraint-trained weights
        # For this test, we'll just use the predictions from the saved file
        print(f"✓ Using CONSTRAINT-TRAINED model from experiment results")
        return None  # Will use saved predictions

    # Get predictions
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Apply heuristic allocation
    class_hierarchy = [0, 1, 2]
    y_pred, _ = apply_allocation_heuristic(
        probs, groups_test.values, class_hierarchy, global_constraint, local_constraint
    )

    # Compute metrics
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    metrics = compute_metrics(y_true, y_pred)

    return metrics


def main():
    print("=" * 80)
    print("WARMUP-ONLY vs CONSTRAINT-TRAINED MODEL")
    print("=" * 80)

    config_path = 'results/our_approach/BasicNN/constraint_0.5_0.3/lr_lambda_test/lr_0.0001_lambda_balanced/config.json'

    # Test 1: Warmup-only + heuristic (should match pure heuristic)
    print("\n[TEST 1] Warmup Model + Heuristic Allocation")
    print("-" * 80)
    warmup_metrics = test_model(config_path, use_warmup_only=True)

    # Test 2: Get constraint-trained results from saved predictions
    print("\n[TEST 2] Constraint-Trained Model Results")
    print("-" * 80)
    from apply_saturation import load_optimization_predictions
    experiment_path = Path(config_path).parent

    y_true, y_pred_opt, probs_opt, groups = load_optimization_predictions(experiment_path)
    opt_metrics = compute_metrics(y_true, y_pred_opt)

    print(f"✓ Loaded optimization predictions from: {experiment_path}")

    # Test 3: Load pure heuristic results
    print("\n[TEST 3] Pure Heuristic Results")
    print("-" * 80)
    heuristic_path = Path('results/heuristic/BasicNN/constraint_0.5_0.3/lr_lambda_test/lr_0.0001_lambda_balanced/config.json')

    if heuristic_path.exists():
        from src.utils.filesystem_manager import load_config_from_path
        heur_config = load_config_from_path(heuristic_path.parent)
        heur_acc = heur_config['results']['accuracy']
        print(f"✓ Pure heuristic accuracy: {heur_acc:.4f}")
    else:
        heur_acc = None
        print("⚠️  Heuristic results not available")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\n1. Warmup-Only + Heuristic:")
    print(f"   Accuracy: {warmup_metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {warmup_metrics['precision_macro']:.4f}")
    print(f"   F1 (macro): {warmup_metrics['f1_macro']:.4f}")

    print(f"\n2. Constraint-Trained (before saturation):")
    print(f"   Accuracy: {opt_metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {opt_metrics['precision_macro']:.4f}")
    print(f"   F1 (macro): {opt_metrics['f1_macro']:.4f}")

    if heur_acc:
        print(f"\n3. Pure Heuristic (warmup + greedy):")
        print(f"   Accuracy: {heur_acc:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    diff = warmup_metrics['accuracy'] - opt_metrics['accuracy']
    if diff > 0:
        print(f"\n⚠️  CONSTRAINT TRAINING HURTS PERFORMANCE!")
        print(f"    Warmup-only: {warmup_metrics['accuracy']:.4f}")
        print(f"    After constraint training: {opt_metrics['accuracy']:.4f}")
        print(f"    Degradation: {diff:.4f} ({diff/warmup_metrics['accuracy']*100:.1f}%)")
        print("\n    Recommendation: Consider using warmup model directly")
    else:
        print(f"\n✓ Constraint training improves performance")
        print(f"    Improvement: {-diff:.4f}")

    if heur_acc and abs(warmup_metrics['accuracy'] - heur_acc) < 0.005:
        print(f"\n✓ Warmup-only matches pure heuristic (both ~{warmup_metrics['accuracy']:.2f})")
        print("    This confirms both use the same warmup model + greedy allocation")

    print("=" * 80)


if __name__ == '__main__':
    main()

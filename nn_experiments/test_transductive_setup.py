import sys
import torch
import pandas as pd

print("Testing transductive training setup...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from config import *
    print("Config imported successfully")
    print(f"  Lambda global: {NN_CONFIGS[0]['lambda_global']}")
    print(f"  Lambda local: {NN_CONFIGS[0]['lambda_local']}")
    print(f"  Test size: {TRAINING_PARAMS['test_size']}")
    print(f"  Constraints: {len(CONSTRAINTS)} pairs")

    from data_loader import load_and_preprocess_data, split_data
    print("Data loader imported successfully")

    from constraints import compute_global_constraints, compute_local_constraints
    print("Constraints module imported successfully")

    from model import NeuralNetClassifier
    print("Model imported successfully")

    from transductive_loss import MulticlassTransductiveLoss
    print("Loss function imported successfully")

    from trainer import train_model_transductive, predict, evaluate_accuracy
    print("Trainer imported successfully")

    print("\nTesting transductive loss function...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_logits = torch.randn(100, 3).to(device)
    test_groups = torch.randint(2, 10, (100,)).to(device)

    criterion = MulticlassTransductiveLoss(
        global_constraints=[50.0, 30.0, None],
        local_constraints={2: [5.0, 3.0, None], 3: [6.0, 4.0, None]},
        lambda_global=1.0,
        lambda_local=1.0,
        use_ce=False
    ).to(device)

    loss_constraint, _, loss_global, loss_local = criterion(
        test_logits, y_true=None, group_ids=test_groups
    )

    print(f"  Constraint loss: {loss_constraint.item():.4f}")
    print(f"  Global loss: {loss_global.item():.4f}")
    print(f"  Local loss: {loss_local.item():.4f}")

    print("\nAll tests passed!")
    print("\nTransductive training approach:")
    print("1. Split data 90-10 train-test")
    print("2. Compute constraints on TEST set")
    print("3. For each training batch:")
    print("   - Forward pass on train batch -> compute BCE loss")
    print("   - Forward pass on FULL test set -> compute constraint loss")
    print("   - Combine losses and backpropagate")
    print("\nReady to run: python run_experiments.py")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

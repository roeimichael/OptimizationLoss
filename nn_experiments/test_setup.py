import torch
import sys

print("Testing environment setup...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("\nTesting imports...")
try:
    from config import *
    print("  config.py - OK")

    from data_loader import load_and_preprocess_data
    print("  data_loader.py - OK")

    from constraints import compute_global_constraints, compute_local_constraints
    print("  constraints.py - OK")

    from transductive_loss import MulticlassTransductiveLoss
    print("  transductive_loss.py - OK")

    from model import NeuralNetClassifier
    print("  model.py - OK")

    from dataset import StudentDataset
    print("  dataset.py - OK")

    from trainer import train_model, predict
    print("  trainer.py - OK")

    print("\nAll imports successful!")

    print("\nTesting loss function on device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = MulticlassTransductiveLoss(
        global_constraints=[114.0, 64.0, None],
        local_constraints={2: [7.0, 3.0, None]},
        lambda_global=1.0,
        lambda_local=0.5,
        use_ce=True
    ).to(device)

    logits = torch.randn(16, 3).to(device)
    labels = torch.randint(0, 3, (16,)).to(device)
    groups = torch.randint(2, 4, (16,)).to(device)

    loss_total, loss_ce, loss_global, loss_local = criterion(logits, labels, groups)
    print(f"  Loss computation successful!")
    print(f"  Total: {loss_total.item():.4f}, CE: {loss_ce.item():.4f}, Global: {loss_global.item():.4f}, Local: {loss_local.item():.4f}")

    print("\nSetup test PASSED!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

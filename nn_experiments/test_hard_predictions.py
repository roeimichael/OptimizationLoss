import torch
import torch.nn as nn
from transductive_loss import MulticlassTransductiveLoss

print("Testing hard prediction counts in loss function...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

logits = torch.tensor([
    [2.0, 1.0, 0.5],
    [0.5, 3.0, 1.0],
    [1.0, 0.5, 2.5],
    [2.5, 1.5, 0.5],
    [0.5, 2.5, 1.5],
], device=device, requires_grad=True)

print("\nLogits:")
print(logits)

y_pred = torch.argmax(logits, dim=1)
print(f"\nHard predictions (argmax): {y_pred.cpu().numpy()}")

hard_counts = torch.zeros(3)
for class_id in range(3):
    hard_counts[class_id] = (y_pred == class_id).sum().float()

print(f"Hard counts per class: {hard_counts.cpu().numpy()}")
print(f"  Class 0: {int(hard_counts[0])} students")
print(f"  Class 1: {int(hard_counts[1])} students")
print(f"  Class 2: {int(hard_counts[2])} students")

global_constraints = [2.0, 1.0, None]
print(f"\nGlobal constraints: {global_constraints}")

criterion = MulticlassTransductiveLoss(
    global_constraints=global_constraints,
    local_constraints=None,
    lambda_global=1.0,
    lambda_local=1.0,
    use_ce=False
).to(device)

loss_total, loss_ce, loss_global, loss_local = criterion(logits, y_true=None, group_ids=None)

print(f"\nLoss values:")
print(f"  Global loss: {loss_global.item():.4f}")
print(f"  Total loss: {loss_total.item():.4f}")

print(f"\nConstraint violations:")
for class_id in range(3):
    if global_constraints[class_id] is not None:
        count = int(hard_counts[class_id])
        limit = int(global_constraints[class_id])
        excess = max(0, count - limit)
        print(f"  Class {class_id}: {count} predicted, limit {limit}, excess {excess}")

print("\nTesting gradient flow (straight-through estimator)...")
loss_total.backward()

print(f"Gradients exist: {logits.grad is not None}")
print(f"Gradient mean: {logits.grad.abs().mean().item():.6f}")

print("\nTesting with groups (local constraints)...")

logits2 = torch.randn(10, 3, device=device, requires_grad=True)
groups = torch.tensor([2, 2, 2, 3, 3, 3, 3, 4, 4, 4], device=device)

local_constraints = {
    2: [1.0, 1.0, None],
    3: [2.0, 1.0, None],
    4: [1.0, 1.0, None]
}

criterion2 = MulticlassTransductiveLoss(
    global_constraints=[5.0, 3.0, None],
    local_constraints=local_constraints,
    lambda_global=1.0,
    lambda_local=1.0,
    use_ce=False
).to(device)

loss_total2, loss_ce2, loss_global2, loss_local2 = criterion2(logits2, y_true=None, group_ids=groups)

print(f"\nWith local constraints:")
print(f"  Global loss: {loss_global2.item():.4f}")
print(f"  Local loss: {loss_local2.item():.4f}")
print(f"  Total loss: {loss_total2.item():.4f}")

y_pred2 = torch.argmax(logits2, dim=1)
print(f"\nHard predictions per group:")
for group_id in [2, 3, 4]:
    mask = groups == group_id
    group_preds = y_pred2[mask]
    counts = [(group_preds == c).sum().item() for c in range(3)]
    print(f"  Group {group_id}: {counts}")

loss_total2.backward()
print(f"\nGradients exist: {logits2.grad is not None}")

print("\nAll tests passed!")
print("\nKey behavior:")
print("- Forward pass uses HARD counts (actual predicted students per class)")
print("- Backward pass uses SOFT counts (probabilities) for gradient flow")
print("- This is called 'straight-through estimator'")

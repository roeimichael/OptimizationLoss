"""TraLO on a capacity-limited TinyCNN.

Hypothesis: TraLO's constraint-loss benefits from headroom in the warmup model
(the model hasn't memorized, so the constraint loss has something to push
against). Headroom probe (headroom_probe.py) confirmed TinyCNN on CIFAR-100
plateaus at train_acc=0.42, test_acc=0.41 — no saturation.

This script applies TraLO on the SAME TinyCNN + same recipe. Compares against
the CE-only baseline (headroom_probe.py).

Constraint: cap class 0 predictions at K=30 (out of ~100 natural per class
on the 10k test set). Forces the model to suppress class-0 predictions while
preserving F1.

Imports MulticlassTransductiveLoss directly from src.losses (no reimplementation).
Reuses the bookkeeping pattern (lambda ratchet, rho ramp, freeze-on-satisfy,
reset_optimizer_at_sat, undershoot hinge) from src/methodologies/tralo/train.py.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# Import the authoritative TraLO loss
sys.path.insert(0, os.path.expanduser("~/OL-replication"))
from src.losses import MulticlassTransductiveLoss  # noqa: E402
from src.utils.constants import UNLIMITED  # noqa: E402

NUM_CLASSES = 100
CONSTRAINED_CLASS = 0
K_GLOBAL = 30  # cap class 0 predictions on test set


class TinyCNN(nn.Module):
    """30,084 params — same as headroom_probe.py (capacity-limited, provably
    cannot interpolate 50k CIFAR-100 samples per Zhang et al. 2017)."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def chunked_forward_with_grad(model, X, chunk):
    """Forward pass over a large tensor in chunks, with grad enabled.
    Concatenates softmax outputs. Used for the penalty pass."""
    parts = []
    n = X.size(0)
    for i in range(0, n, chunk):
        logits = model(X[i:i + chunk])
        parts.append(F.softmax(logits, dim=1))
    return torch.cat(parts, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup_epochs", type=int, default=100)
    ap.add_argument("--constraint_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr_constraint", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.path.expanduser("~/data/cifar100_torchvision"))
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--pred_out", default="predictions.npy")
    # TraLO hyperparams (matching our paper recipe)
    ap.add_argument("--initial_rho", type=float, default=5.0)
    ap.add_argument("--rho_target", type=float, default=100.0)
    ap.add_argument("--lambda_global_init", type=float, default=0.05)
    ap.add_argument("--lambda_step", type=float, default=0.002)
    ap.add_argument("--fior_beta", type=float, default=0.50)
    ap.add_argument("--penalty_mode", default="both")
    ap.add_argument("--reset_optimizer_at_sat", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR100(root=args.data_root, train=True,
                                             download=True, transform=tf_train)
    test_ds = torchvision.datasets.CIFAR100(root=args.data_root, train=False,
                                            download=True, transform=tf_test)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Pre-load test set as a single tensor for the constraint passes
    print("Pre-loading test set as tensor for constraint passes...")
    X_test_list, y_test_list = [], []
    for x, y in test_loader:
        X_test_list.append(x)
        y_test_list.append(y)
    X_test = torch.cat(X_test_list).to(device)
    y_test = torch.cat(y_test_list).to(device)
    print(f"  X_test shape: {tuple(X_test.shape)}")
    n_test = X_test.size(0)

    model = TinyCNN(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TinyCNN params={n_params:,}")
    print(f"Constraint: class {CONSTRAINED_CLASS} cap K={K_GLOBAL} "
          f"(natural class freq ~{n_test // NUM_CLASSES})")

    # --- Build constraint loss machine ---
    global_con = [float(UNLIMITED)] * NUM_CLASSES
    global_con[CONSTRAINED_CLASS] = float(K_GLOBAL)
    local_con = {}
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_con,
        local_constraints=local_con,
        num_classes=NUM_CLASSES,
        initial_rho=args.initial_rho,
        alpha_kl=0.0,
        penalty_mode=args.penalty_mode,
    ).to(device)
    criterion_constraint.set_lambda_per_class(
        CONSTRAINED_CLASS, args.lambda_global_init, scope="global")

    # --- Optimizer + cosine schedule (matching headroom_probe.py recipe) ---
    total_epochs = args.warmup_epochs + args.constraint_epochs
    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)

    rho_step = (args.rho_target - args.initial_rho) / max(args.constraint_epochs, 1)
    rho_frozen = False
    satisfaction_epoch = None
    first_satisfaction = False

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "phase", "epoch", "train_loss", "train_acc", "test_acc",
            "class0_hard", "class0_soft", "lambda_c0", "rho",
            "satisfied", "penalty_loss",
        ])
        t0 = time.time()

        for ep in range(1, total_epochs + 1):
            phase = "warmup" if ep <= args.warmup_epochs else "constraint"

            # ---- CE pass ----
            model.train()
            if phase == "constraint":
                for pg in opt.param_groups:
                    pg["lr"] = args.lr_constraint
            loss_sum = correct = total = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss_ce = F.cross_entropy(logits, y)
                loss_ce.backward()
                opt.step()
                loss_sum += loss_ce.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            train_loss = loss_sum / total
            train_acc = correct / total

            # ---- Test soft+hard counts (no_grad) ----
            model.eval()
            with torch.no_grad():
                logits_test = []
                for i in range(0, n_test, 256):
                    logits_test.append(model(X_test[i:i + 256]))
                logits_test = torch.cat(logits_test)
                proba_test = F.softmax(logits_test, dim=1)
                preds_test = logits_test.argmax(1)
                soft_counts = proba_test.sum(0)
                hard_counts = torch.bincount(preds_test, minlength=NUM_CLASSES).float()
                test_correct = (preds_test == y_test).sum().item()
                test_acc = test_correct / n_test
                class0_hard = hard_counts[CONSTRAINED_CLASS].item()
                class0_soft = soft_counts[CONSTRAINED_CLASS].item()

            # Satisfaction check based on hard counts (matches TraLO's behaviour)
            satisfied = (hard_counts[CONSTRAINED_CLASS].item() <= K_GLOBAL)

            penalty_val = 0.0
            if phase == "constraint":
                # ---- Penalty pass: forward through test set WITH grad ----
                model.train()
                opt.zero_grad(set_to_none=True)
                soft_counts_grad = torch.zeros(NUM_CLASSES, device=device)
                for i in range(0, n_test, 256):
                    p = F.softmax(model(X_test[i:i + 256]), dim=1)
                    soft_counts_grad = soft_counts_grad + p.sum(0)

                # Standard bounded penalty
                penalty_global = criterion_constraint.compute_global_from_counts(soft_counts_grad)

                # Undershoot hinge: push soft_count UP when below K to park near K
                # from below (matches hybrid_mode='undershoot_hinge' in TraLO)
                lam_c0 = criterion_constraint.get_lambda_per_class(
                    CONSTRAINED_CLASS, scope="global")
                K_t = torch.tensor(float(K_GLOBAL), device=device)
                undershoot = F.relu(K_t - soft_counts_grad[CONSTRAINED_CLASS]) / K_t
                hinge = lam_c0 * args.fior_beta * undershoot

                total_penalty = penalty_global + hinge
                if total_penalty.requires_grad:
                    total_penalty.backward()
                    opt.step()
                penalty_val = float(total_penalty.detach().item())

                # ---- Lambda ratchet + rho ramp (post-step bookkeeping) ----
                if satisfied:
                    if not first_satisfaction:
                        first_satisfaction = True
                        satisfaction_epoch = ep
                        rho_frozen = True
                        print(f"  *** First satisfaction at epoch {ep} — "
                              f"freezing rho, resetting optimizer ***", flush=True)
                        if args.reset_optimizer_at_sat:
                            # Rebuild SGD with fresh momentum buffers
                            opt = torch.optim.SGD(
                                model.parameters(), lr=args.lr_constraint,
                                momentum=0.9, weight_decay=5e-4)
                else:
                    # Violation: increment lambda (the ratchet)
                    new_lam = lam_c0 + args.lambda_step
                    criterion_constraint.set_lambda_per_class(
                        CONSTRAINED_CLASS, new_lam, scope="global")
                    # Ramp rho until satisfaction
                    if not rho_frozen:
                        criterion_constraint.increment_rho(rho_step)

            lam_now = criterion_constraint.get_lambda_per_class(
                CONSTRAINED_CLASS, scope="global")
            rho_now = criterion_constraint.get_rho()
            sched.step()

            elapsed = time.time() - t0
            print(f"{phase[:4]} ep {ep:3d}  loss={train_loss:.3f}  "
                  f"tr_acc={train_acc:.3f}  te_acc={test_acc:.3f}  "
                  f"c0_hard={int(class0_hard):3d}  c0_soft={class0_soft:6.2f}  "
                  f"lam={lam_now:.4f}  rho={rho_now:.2f}  "
                  f"{'SAT' if satisfied else 'viol'}  "
                  f"pen={penalty_val:.4f}  t={elapsed:.0f}s", flush=True)
            w.writerow([
                phase, ep, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{test_acc:.6f}", int(class0_hard), f"{class0_soft:.4f}",
                f"{lam_now:.6f}", f"{rho_now:.4f}", int(satisfied),
                f"{penalty_val:.6f}",
            ])
            f.flush()

    # --- Save final test predictions for paired comparison ---
    model.eval()
    with torch.no_grad():
        logits_test = []
        for i in range(0, n_test, 256):
            logits_test.append(model(X_test[i:i + 256]))
        logits_test = torch.cat(logits_test)
        preds = logits_test.argmax(1).cpu().numpy()
        y_true = y_test.cpu().numpy()
    np.save(args.pred_out, np.stack([y_true, preds], axis=1))

    # --- Final summary ---
    final_c0_hard = int(np.sum(preds == CONSTRAINED_CLASS))
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    print(f"\nFINAL  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
    print(f"  class {CONSTRAINED_CLASS} hard count = {final_c0_hard} "
          f"(K={K_GLOBAL}, {'SATISFIED' if final_c0_hard <= K_GLOBAL else 'VIOLATED'})")
    print(f"  macro F1 = {f1:.4f}")
    print(f"  satisfaction_epoch = {satisfaction_epoch}")


if __name__ == "__main__":
    main()

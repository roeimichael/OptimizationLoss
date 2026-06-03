"""Fioretto-LDF baseline on the capacity-limited TinyCNN.

Ports the core Fioretto Algorithm 1/2 from src/methodologies/fioretto_ldf/train.py
to standalone form for direct comparison with tralo_headroom.py.

Key differences from TraLO (whose script lives in tralo_headroom.py):
  - Linear penalty: L = lambda_c * sum_i softmax(logits[i])[c]   (for violated c)
  - No rho, no bounded saturation, no undershoot hinge.
  - Dual update: lambda_c += step_size * excess  (subgradient ascent on dual).

Same TinyCNN, same CIFAR-100 recipe, same K=30 cap on class 0. Per-epoch
log identical schema to tralo_headroom for paired analysis.
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

sys.path.insert(0, os.path.expanduser("~/OL-replication"))
from src.utils.constants import UNLIMITED  # noqa: E402

NUM_CLASSES = 100
CONSTRAINED_CLASS = 0
K_GLOBAL = 30


class TinyCNN(nn.Module):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup_epochs", type=int, default=100)
    ap.add_argument("--constraint_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr_constraint", type=float, default=0.005)
    ap.add_argument("--step_size", type=float, default=0.01,
                    help="Fioretto dual step size (matches our paper recipe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.path.expanduser("~/data/cifar100_torchvision"))
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--pred_out", default=None)
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

    X_test_list, y_test_list = [], []
    for x, y in test_loader:
        X_test_list.append(x)
        y_test_list.append(y)
    X_test = torch.cat(X_test_list).to(device)
    y_test = torch.cat(y_test_list).to(device)
    n_test = X_test.size(0)
    print(f"X_test shape: {tuple(X_test.shape)}")

    model = TinyCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Model: TinyCNN params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Constraint: class {CONSTRAINED_CLASS} cap K={K_GLOBAL}")
    print(f"Fioretto step_size={args.step_size}")

    total_epochs = args.warmup_epochs + args.constraint_epochs
    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)

    lambda_g = 0.0  # single global multiplier for class 0
    satisfaction_epoch = None

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

            # ---- Test soft+hard counts ----
            model.eval()
            with torch.no_grad():
                logits_test = []
                for i in range(0, n_test, 256):
                    logits_test.append(model(X_test[i:i + 256]))
                logits_test = torch.cat(logits_test)
                proba_test = F.softmax(logits_test, dim=1)
                preds_test = logits_test.argmax(1)
                soft_c0 = proba_test[:, CONSTRAINED_CLASS].sum().item()
                hard_c0 = int((preds_test == CONSTRAINED_CLASS).sum().item())
                test_acc = (preds_test == y_test).float().mean().item()

            satisfied = (hard_c0 <= K_GLOBAL)

            penalty_val = 0.0
            if phase == "constraint":
                # ---- Fioretto linear penalty pass ----
                excess_soft = max(0.0, soft_c0 - K_GLOBAL)
                violated = (soft_c0 > K_GLOBAL)

                if violated and lambda_g > 0:
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    # Linear penalty: lambda * sum of softmax[c0]
                    constraint_loss = torch.zeros((), device=device)
                    for i in range(0, n_test, 256):
                        p = F.softmax(model(X_test[i:i + 256]), dim=1)
                        constraint_loss = constraint_loss + lambda_g * p[:, CONSTRAINED_CLASS].sum()
                    if constraint_loss.requires_grad:
                        constraint_loss.backward()
                        # Grad clip mirror TraLO
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        opt.step()
                    penalty_val = float(constraint_loss.detach().item())

                # ---- Subgradient dual ascent on excess (post-step) ----
                # Use soft excess as the violation measure (matches Fioretto Eq. 5)
                lambda_g = lambda_g + args.step_size * excess_soft

                if satisfied and satisfaction_epoch is None:
                    satisfaction_epoch = ep
                    print(f"  *** First satisfaction at epoch {ep} ***", flush=True)

            sched.step()
            elapsed = time.time() - t0
            print(f"{phase[:4]} ep {ep:3d}  loss={train_loss:.3f}  "
                  f"tr_acc={train_acc:.3f}  te_acc={test_acc:.3f}  "
                  f"c0_hard={hard_c0:3d}  c0_soft={soft_c0:6.2f}  "
                  f"lam={lambda_g:.4f}  "
                  f"{'SAT' if satisfied else 'viol'}  "
                  f"pen={penalty_val:.4f}  t={elapsed:.0f}s", flush=True)
            w.writerow([
                phase, ep, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{test_acc:.6f}", hard_c0, f"{soft_c0:.4f}",
                f"{lambda_g:.6f}", "0.0", int(satisfied),
                f"{penalty_val:.6f}",
            ])
            f.flush()

    if args.pred_out:
        model.eval()
        with torch.no_grad():
            logits_test = []
            for i in range(0, n_test, 256):
                logits_test.append(model(X_test[i:i + 256]))
            preds = torch.cat(logits_test).argmax(1).cpu().numpy()
            y_true = y_test.cpu().numpy()
        np.save(args.pred_out, np.stack([y_true, preds], axis=1))

    from sklearn.metrics import f1_score
    preds_final = preds if args.pred_out else preds_test.cpu().numpy()
    y_final = y_test.cpu().numpy()
    final_c0 = int((preds_final == CONSTRAINED_CLASS).sum())
    f1 = f1_score(y_final, preds_final, average="macro", zero_division=0)
    print(f"\nFINAL  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
    print(f"  class {CONSTRAINED_CLASS} hard count = {final_c0} "
          f"(K={K_GLOBAL}, {'SATISFIED' if final_c0 <= K_GLOBAL else 'VIOLATED'})")
    print(f"  macro F1 = {f1:.4f}")
    print(f"  satisfaction_epoch = {satisfaction_epoch}")


if __name__ == "__main__":
    main()

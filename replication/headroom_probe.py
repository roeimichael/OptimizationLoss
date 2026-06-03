"""Capacity-limit headroom probe (Zhang et al. 2017, Arpit et al. 2017).

CLAIM (Zhang Theorem 1): a NN cannot interpolate n samples in d input
dimensions when params < 2n + d. CIFAR-100 has n=50000, d=3072 -> need
>=103072 params to interpolate. Our TinyCNN has ~30k -> guaranteed plateau.

Recipe: vanilla CE, SGD lr=0.02 mom=0.9 wd=5e-4, cosine over 200 epochs,
batch 128, standard CIFAR-100 augmentation (crop pad=4 + hflip).

Expected: train_acc plateaus ~0.50-0.70, test_acc ~0.25-0.35. No noise.

Fully standalone: no imports from src/. Logs per-epoch CSV.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.path.expanduser("~/data/cifar100_torchvision"))
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--pred_out", default=None,
                    help="path to save (y_true, y_pred) array (.npy)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=True, transform=tf_train,
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=tf_test,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = TinyCNN(num_classes=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_samples = len(train_ds)
    threshold = 2 * n_samples + 32 * 32 * 3
    print(f"Model: TinyCNN params={n_params:,}")
    print(f"Data:  CIFAR-100 train_n={n_samples:,}  d_input={32*32*3}")
    print(f"Zhang interpolation threshold (2n+d): {threshold:,}")
    print(f"  -> params/threshold = {n_params/threshold:.3f} "
          f"(< 1.0 means cannot interpolate)")
    print(f"Recipe: SGD lr={args.lr} mom=0.9 wd=5e-4, cosine over {args.epochs} epochs")
    print()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "elapsed_s"])
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            loss_sum = correct = total = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            train_loss = loss_sum / total
            train_acc = correct / total
            test_acc = evaluate(model, test_loader, device)
            lr_now = sched.get_last_lr()[0]
            sched.step()
            elapsed = time.time() - t0
            print(f"epoch {ep:3d}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
                  f"lr={lr_now:.4f}  t={elapsed:.1f}s", flush=True)
            w.writerow([ep, f"{train_loss:.6f}", f"{train_acc:.6f}",
                        f"{test_acc:.6f}", f"{lr_now:.6f}", f"{elapsed:.1f}"])
            f.flush()


    if args.pred_out:
        model.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                all_preds.append(model(x).argmax(1).cpu().numpy())
                all_y.append(y.numpy())
        preds = np.concatenate(all_preds)
        ys = np.concatenate(all_y)
        np.save(args.pred_out, np.stack([ys, preds], axis=1))
        c0 = int((preds == 0).sum())
        from sklearn.metrics import f1_score
        f1 = f1_score(ys, preds, average="macro", zero_division=0)
        print(f"  class 0 hard count = {c0}  macro_F1 = {f1:.4f}")
    print()
    print(f"FINAL  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
    print(f"PLATEAU CHECK (train_acc < 0.95): "
          f"{'PASS - non-saturating' if train_acc < 0.95 else 'FAIL - saturated'}")


if __name__ == "__main__":
    main()

"""Deep diagnostic plot for one SmallCNN TraLO cell with WARMUP included.

Reads from results/pending_runs/_diag_smallcnn/SmallCNN/tralo/seed_1/
(this cell is run with cache deleted so warmup phase IS logged).

4 panels stacked, x-axis = epoch (warmup 0-29 + constraint 30-129):
  1. Losses (L_CE / L_Global / L_Local) + Train_Acc on secondary axis
  2. Hard predicted counts per class (all 7), MEL bold
  3. Soft predicted counts per class (all 7), MEL bold + cap line
  4. Lambdas + satisfaction strip
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "results/pending_runs/_diag_smallcnn/SmallCNN/tralo/seed_1/training_log.csv"
OUT = "scripts/_paper_agg/smallcnn_diag.png"
NUM_CLASSES = 7
CONSTRAINED_CLASS = 4  # MEL
CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


def fnum(r, k):
    v = r.get(k, "")
    try:
        return float(v) if v not in ("", "nan") else None
    except (ValueError, TypeError):
        return None


def main():
    rows = []
    with open(PATH) as f:
        for r in csv.DictReader(f):
            try:
                int(r["Epoch"])
            except (ValueError, TypeError):
                continue
            rows.append(r)
    rows.sort(key=lambda r: int(r["Epoch"]))
    ep = [int(r["Epoch"]) for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

    # Row 1: losses + train_acc
    ax = axes[0]
    for k, c in [("L_CE", "C0"), ("L_Global", "C1"), ("L_Local", "C2")]:
        vals = [fnum(r, k) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(ep, vals, label=k, color=c, lw=1.6)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("loss component")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0, 30, alpha=0.08, color="blue", label="warmup")
    ax.axvspan(30, max(ep), alpha=0.08, color="orange", label="constraint")
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    tracc = [fnum(r, "Train_Acc") for r in rows]
    ax2.plot(ep, tracc, color="gray", ls=":", lw=1.4, label="Train_Acc")
    ax2.axhline(0.995, color="red", ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Train_Acc", color="gray")
    ax2.tick_params(labelcolor="gray")
    ax.set_title("SmallCNN derm L30_G30 TraLO seed 1 — warmup (blue) + constraint (orange)")

    # Row 2: per-class hard counts
    ax = axes[1]
    for c in range(NUM_CLASSES):
        vals = [fnum(r, f"Hard_Class{c}") for r in rows]
        lw = 2.5 if c == CONSTRAINED_CLASS else 1.0
        alpha = 1.0 if c == CONSTRAINED_CLASS else 0.6
        ax.plot(ep, vals, label=f"{c}:{CLASS_NAMES[c]}", lw=lw, alpha=alpha)
    # MEL cap line
    cap = [fnum(r, f"Limit_Class{CONSTRAINED_CLASS}") for r in rows]
    if any(v is not None for v in cap):
        ax.plot(ep, cap, color="black", ls=":", lw=1.5, label="MEL cap")
    ax.set_ylabel("Hard count (argmax)")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0, 30, alpha=0.08, color="blue")
    ax.axvspan(30, max(ep), alpha=0.08, color="orange")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # Row 3: per-class soft counts
    ax = axes[2]
    for c in range(NUM_CLASSES):
        vals = [fnum(r, f"Soft_Class{c}") for r in rows]
        lw = 2.5 if c == CONSTRAINED_CLASS else 1.0
        alpha = 1.0 if c == CONSTRAINED_CLASS else 0.6
        ax.plot(ep, vals, label=f"{c}:{CLASS_NAMES[c]}", lw=lw, alpha=alpha)
    ax.set_ylabel("Soft count (Σ prob)")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0, 30, alpha=0.08, color="blue")
    ax.axvspan(30, max(ep), alpha=0.08, color="orange")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # Row 4: lambdas + satisfaction strip
    ax = axes[3]
    lamg = [fnum(r, "Lambda_Global") for r in rows]
    laml = [fnum(r, "Lambda_Local") for r in rows]
    ax.plot(ep, lamg, label="Lambda_Global", color="C5", lw=1.5)
    ax.plot(ep, laml, label="Lambda_Local", color="C6", lw=1.5)
    ax.set_ylabel("Lambda")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.axvspan(0, 30, alpha=0.08, color="blue")
    ax.axvspan(30, max(ep), alpha=0.08, color="orange")
    ax2 = ax.twinx()
    gsat = [fnum(r, "Global_Satisfied") for r in rows]
    lsat = [fnum(r, "Local_Satisfied") for r in rows]
    if any(v is not None for v in gsat):
        ax2.fill_between(ep, 0, [g or 0 for g in gsat], alpha=0.2,
                         color="green", label="Global satisfied")
    if any(v is not None for v in lsat):
        ax2.fill_between(ep, 0, [l or 0 for l in lsat], alpha=0.15,
                         color="purple", label="Local satisfied")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("satisfaction (0/1)")
    ax2.legend(loc="upper left", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

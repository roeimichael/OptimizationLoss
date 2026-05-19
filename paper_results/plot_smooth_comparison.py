"""2x2 grid of EuroSAT L50 band=5 seed_1 trajectories under smooth ∈ {0,3,5,10}.

Compares the spike-up/slam-down of instant lambda transitions (smooth=0) to
progressively smoother ramps. Each panel shows prediction count vs epoch
with K=111 and K±band=5 marked.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load(path, key):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get(key)
            try:
                out.append(float(v) if v not in (None, "", "inf") else None)
            except ValueError:
                out.append(None)
    return out


def load_epoch(path):
    with open(path) as f:
        return [int(r["Epoch"]) for r in csv.DictReader(f)]


K = 111
BAND = 5
CLS = 4
PATHS = {s: Path(f"paper_results/_log_smooth{s}_eurosat.csv") for s in [0, 3, 5, 10]}

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False, sharey=True)
order = [0, 3, 5, 10]
for ax, smooth in zip(axes.flat, order):
    path = PATHS[smooth]
    ep = load_epoch(path)
    hard = load(path, f"Hard_Class{CLS}")
    lam = load(path, "Lambda_Global")

    ax.axhspan(K - BAND, K + BAND, color="#fff3cd", alpha=0.55,
               label=f"K ± band ({BAND})")
    ax.axhline(K, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"K = {K}")
    # Filter only constraint phase (epoch >= 51) for clean view
    pairs = [(e, h) for e, h in zip(ep, hard) if h is not None]
    px, py = zip(*pairs) if pairs else ([], [])
    ax.plot(px, py, marker="o", color="#2c4a7a", linewidth=1.4,
            markersize=2.5, label="Hard count")

    # Twin axis for lambda
    ax2 = ax.twinx()
    lam_pairs = [(e, l) for e, l in zip(ep, lam) if l is not None]
    if lam_pairs:
        lx, ly = zip(*lam_pairs)
        ax2.plot(lx, ly, color="#2a7", linestyle=":", linewidth=1.0,
                 alpha=0.7, label="λ_global")
        ax2.set_ylabel("λ_global", color="#2a7", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#2a7", labelsize=8)
        ax2.set_ylim(0, max(0.3, max(ly) * 1.1))

    title = f"smooth={smooth}" + (" (instant, control)" if smooth == 0 else f" (linear ramp over {smooth} ep)")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(f"Predictions of class {CLS}")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    if smooth == 0:
        ax.legend(loc="upper right", fontsize=8)

fig.suptitle("EuroSAT · MobileNetV3 · cls_4 · L50 (K=111) · band=5 · cycles=3 · seed_1\n"
             "Effect of oscillation_lambda_smooth on prediction-count trajectory",
             fontsize=12)
plt.tight_layout()
out = Path("paper/figures/fig_osc_smooth_comparison.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Wrote {out}")
plt.close(fig)

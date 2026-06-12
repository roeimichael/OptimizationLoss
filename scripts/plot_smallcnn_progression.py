"""Plot TraLO training progression across 3 backbones on derm L30_G30.

Three columns (TinyCNN, SmallCNN, MediumCNN), each with:
  Top:    L_CE, L_Global, L_Local vs epoch  +  Train_Acc on secondary axis
  Middle: Lambda_Global, Lambda_Local + Global/Local_Satisfied band
  Bottom: Hard_Class4 (predicted MEL count) vs epoch  +  Limit cap line

Only TraLO logs all these columns; Fioretto/Hounie use different schemas
so they're plotted separately below if data is present.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/pending_runs/derm_smallcnn_full"
BACKBONES = ["TinyCNN", "SmallCNN", "MediumCNN"]
OUT = "scripts/_paper_agg/smallcnn_progression.png"


def load_log(backbone, method):
    path = os.path.join(ROOT, backbone, method, "seed_1", "training_log.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            ep_key = "Epoch" if "Epoch" in r else ("epoch" if "epoch" in r else None)
            if ep_key is None:
                continue
            try:
                int(r[ep_key])
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def fnum(r, k):
    v = r.get(k, "")
    try:
        return float(v) if v not in ("", "nan", None) else None
    except (ValueError, TypeError):
        return None


def plot_tralo_panel(rows, axes_col, backbone):
    ep = [int(r["Epoch"]) for r in rows]
    lce = [fnum(r, "L_CE") for r in rows]
    lg = [fnum(r, "L_Global") for r in rows]
    ll = [fnum(r, "L_Local") for r in rows]
    lamg = [fnum(r, "Lambda_Global") for r in rows]
    laml = [fnum(r, "Lambda_Local") for r in rows]
    gsat = [fnum(r, "Global_Satisfied") for r in rows]
    lsat = [fnum(r, "Local_Satisfied") for r in rows]
    hc4 = [fnum(r, "Hard_Class4") for r in rows]
    sc4 = [fnum(r, "Soft_Class4") for r in rows]
    lim4 = [fnum(r, "Limit_Class4") for r in rows]
    tracc = [fnum(r, "Train_Acc") for r in rows]

    ax = axes_col[0]
    ax.plot(ep, lce, label="L_CE", color="C0", lw=1.5)
    ax.plot(ep, lg, label="L_Global", color="C1", lw=1.5)
    ax.plot(ep, ll, label="L_Local", color="C2", lw=1.5)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title(f"TraLO on {backbone}")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(ep, tracc, color="gray", ls=":", lw=1.2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Train_Acc", color="gray", fontsize=8)
    ax2.axhline(0.995, color="red", ls=":", lw=0.8, alpha=0.6)
    ax2.tick_params(labelcolor="gray", labelsize=7)

    ax = axes_col[1]
    ax.plot(ep, lamg, label="Lambda_Global", color="C5", lw=1.5)
    ax.plot(ep, laml, label="Lambda_Local", color="C6", lw=1.5)
    ax.set_ylabel("Lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    # shade satisfaction
    ax2 = ax.twinx()
    sat_band = [(g or 0) * (l or 0) for g, l in zip(gsat, lsat)]
    ax2.fill_between(ep, 0, sat_band, alpha=0.15, color="green",
                     label="both satisfied")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([])

    ax = axes_col[2]
    ax.plot(ep, hc4, label="Hard count", color="C3", lw=1.5)
    ax.plot(ep, sc4, label="Soft count", color="C4", lw=1.2, ls="--",
            alpha=0.8)
    ax.plot(ep, lim4, label="Cap (limit)", color="black", lw=1.5, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MEL pred count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main():
    fig, axes = plt.subplots(3, len(BACKBONES), figsize=(16, 10), sharex=True)
    for j, bb in enumerate(BACKBONES):
        rows = load_log(bb, "tralo")
        if not rows:
            axes[0, j].set_title(f"TraLO on {bb}\n(no log)")
            continue
        plot_tralo_panel(rows, axes[:, j], bb)

    fig.suptitle(
        "TraLO training progression on DermMNIST L30_G30 "
        "(constrained class = MEL, seed 1)",
        fontsize=13,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

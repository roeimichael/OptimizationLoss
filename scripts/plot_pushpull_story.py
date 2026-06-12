"""Paper-style figures for the push-pull story.

Figure 1: TraLO training dynamics, saturated vs push-pull (2 panels).
Figure 2: F1 by gradient method (TraLO/Fioretto/Hounie), saturated vs push-pull.

Captions live in the paper doc, NOT on the figures.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("scripts/_audit/figures")
OUT_DIR.mkdir(exist_ok=True)

LOG_PUSH = "scripts/_audit/training_logs/demo_warmup1.csv"
LOG_SAT = "scripts/_audit/training_logs/demo_warmup50.csv"


def parse_tralo_log(path):
    seen = set()
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ep = int(r["Epoch"])
            except (KeyError, ValueError):
                continue
            if ep in seen:
                continue
            seen.add(ep)
            try:
                ta = float(r.get("Train_Acc", "nan"))
                ce = float(r.get("L_CE", "nan"))
            except ValueError:
                continue
            rows.append((ep, ta, ce))
    rows.sort()
    return rows


def fig1_dynamics():
    sat = parse_tralo_log(LOG_SAT)
    push = parse_tralo_log(LOG_PUSH)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    x_max = max(max(r[0] for r in sat), max(r[0] for r in push))

    for ax, data, title, warmup_end in [
        (axes[0], sat, "Saturated (warmup=50)", 50),
        (axes[1], push, "Push-pull (warmup=1)", 1),
    ]:
        eps = [r[0] for r in data]
        tas = [r[1] for r in data]
        ces = [r[2] for r in data]
        ax2 = ax.twinx()
        l1, = ax.plot(eps, tas, "b-", lw=1.8, label="train accuracy")
        l2, = ax2.plot(eps, ces, "r-", lw=1.8, label="CE loss")
        ax.axvspan(0, warmup_end, color="grey", alpha=0.10, zorder=0)
        vl = ax.axvline(warmup_end, color="black", ls="--", lw=1.0,
                        label="constraint phase start")
        # warmup/phase-2 region labels
        if warmup_end >= 5:
            ax.text(warmup_end / 2, 1.02, "warmup",
                    ha="center", fontsize=8, color="dimgrey")
        ax.text((warmup_end + x_max) / 2, 1.02, "constraint",
                ha="center", fontsize=8, color="dimgrey")
        ax.set_xlim(0, x_max + 2)
        ax.set_xlabel("epoch")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax2.set_ylim(0.0, 1.1)
        ax.tick_params(axis="y", colors="b")
        ax2.tick_params(axis="y", colors="r")
        if ax is axes[0]:
            ax.set_ylabel("train accuracy", color="b")
        if ax is axes[1]:
            ax2.set_ylabel("CE loss", color="r")
        ax.grid(alpha=0.25)
        ax.legend(handles=[l1, l2, vl],
                  loc="center right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "fig1_dynamics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def collect_f1(ds, model, cls, warmup, tag):
    rows = list(csv.DictReader(open("scripts/_audit/saturation_audit_v2.csv")))
    by_method = defaultdict(list)
    for r in rows:
        if (r["dataset"] == ds and r["model"] == model
                and r["constrained_class"] == cls
                and r["warmup_epochs"] == warmup
                and r["constraint_tag"] == tag
                and r["f1_macro"]):
            try:
                by_method[r["method"]].append(float(r["f1_macro"]))
            except ValueError:
                pass
    return {m: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))),
                len(v)) for m, v in by_method.items()}


def fig2_f1_bars():
    methods = ["tralo", "fioretto_ldf", "hounie_rcl"]
    labels = ["TraLO", "Fioretto", "Hounie"]
    colors = ["#1f77b4", "#aaaaaa", "#aaaaaa"]

    sat = collect_f1("dermmnist", "MobileNetV2", "4", "50", "L30_G30")
    push = collect_f1("dermmnist", "MobileNetV2", "4", "1", "L50_G50")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)

    for ax, data, title in [
        (axes[0], sat, "Saturated (warmup=50)"),
        (axes[1], push, "Push-pull (warmup=1)"),
    ]:
        means = [data.get(m, (0, 0, 0))[0] for m in methods]
        sems = [data.get(m, (0, 0, 0))[1] for m in methods]
        ns = [data.get(m, (0, 0, 0))[2] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=sems, capsize=4, color=colors,
               edgecolor="black", linewidth=0.6, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for xi, m in enumerate(means):
            ax.text(xi, m + 0.008, f"{m:.3f}", ha="center", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("F1 (macro)")
        ax.set_ylim(0.55, 0.72)

    fig.tight_layout()
    out = OUT_DIR / "fig2_f1_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_dynamics()
    fig2_f1_bars()

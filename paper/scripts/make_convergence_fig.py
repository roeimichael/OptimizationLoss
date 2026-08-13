"""Regenerate fig_convergence (single column, figure) for the AAAI paper.

One clean panel: BEST-SO-FAR constraint excess (running minimum of E = max(0, hard count -
cap), so the curve is a monotone descent toward the quota with no confusing rise-then-fall)
over the constraint phase for the three constraint-trained methods (TraLO, Fioretto-LDF,
Hounie-RCL) on DermMNIST with the RegNetY-400MF backbone at the single tight symmetric cap
L=G=30, seeds 1-4. Line = median over seeds; band = interquartile range (25-75%). A filled
marker sits at each method's median TIME-TO-FEASIBILITY (median over seeds of the first
epoch whose best-so-far excess reaches the cap); the epoch is also folded into the legend
so nothing is repeated.

Why this cell: DermMNIST on RegNet is where the three methods separate most cleanly,
because RegNet's cross-entropy saturates slowly, so the two objectives visibly compete
(same regime as the App-A mechanism probe). All three reach feasibility here, so there is
no "never feasible" glyph; the story is speed. Grid-wide (paper backbones x datasets) the
medians are: TraLO 15 ep at 98% feasible, Fioretto 14 ep but only 91% feasible (its dual
can stall), Hounie 55 ep at 100% -- reliable but ~3.6x slower.

Epoch alignment: TraLO logs at GLOBAL epochs (from 51, after the 50-epoch warmup),
baselines at CONSTRAINT epochs from 0; both are normalized to the start of the constraint
phase (x - x[0]) so the axis is "epochs into the constraint phase" for every method.

Data source: paper/data/dynamics/dermmnist/*/symmetric/RegNetY400MF/L30_G30/
(self-contained; see data/README_DATA.md). Constrained class 4 (melanoma).

Run:  python paper/scripts/make_convergence_fig.py
"""
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import apply_style, savefig_dual, C_TRALO, C_FIORETTO, C_HOUNIE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
RESULTS = ROOT / "paper" / "data" / "dynamics"
OUT.mkdir(parents=True, exist_ok=True)

apply_style()

SEEDS = [1, 2, 3, 4]
CLS = 4          # DermMNIST constrained class (melanoma)
SWEEP = "RegNetY400MF"
XMAX = 52

STYLE = {
    "tralo":        {"color": C_TRALO,    "ls": "-",           "lw": 2.2, "z": 10,
                     "marker": "o", "ms": 7.0, "label": "TraLO (ours)"},
    "fioretto_ldf": {"color": C_FIORETTO, "ls": (0, (6, 2.2)), "lw": 1.8, "z": 8,
                     "marker": "^", "ms": 7.5, "label": "Fioretto-LDF"},
    "hounie_rcl":   {"color": C_HOUNIE,   "ls": (0, (1, 1.5)), "lw": 1.8, "z": 6,
                     "marker": "s", "ms": 6.5, "label": "Hounie-RCL"},
}


def _log_path(method, tag, seed):
    return (RESULTS / "dermmnist" / method / "symmetric" / SWEEP
            / tag / f"seed_{seed}" / "training_log.csv")


def _excess_tralo(df):
    epochs = df["Epoch"].to_numpy(dtype=float)
    g_hard = df[f"Hard_Class{CLS}"].to_numpy(dtype=float)
    g_lim = df[f"Limit_Class{CLS}"].to_numpy(dtype=float)
    excess = np.where(np.isfinite(g_lim), np.maximum(0.0, g_hard - g_lim), 0.0)
    gi = 0
    while True:
        h_col, l_col = f"Group{gi}_Hard_Class{CLS}", f"Group{gi}_Limit_Class{CLS}"
        if h_col not in df.columns or l_col not in df.columns:
            break
        h = df[h_col].to_numpy(dtype=float)
        l = df[l_col].to_numpy(dtype=float)
        excess += np.where(np.isfinite(l), np.maximum(0.0, h - l), 0.0)
        gi += 1
    return epochs, excess


def _load_curve(method, tag, seed):
    p = _log_path(method, tag, seed)
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    if method == "tralo":
        x, y = _excess_tralo(df)
    else:
        x, y = df["epoch"].to_numpy(dtype=float), df["total_excess"].to_numpy(dtype=float)
    if len(x):
        x = x - x[0]
    return np.asarray(x, float), np.asarray(y, float)


NAME = {"tralo": "TraLO", "fioretto_ldf": "Fioretto-LDF", "hounie_rcl": "Hounie-RCL"}


def _series(method, tag):
    """Best-so-far excess per seed on grid 0..XMAX: a MONOTONE descent (the lowest
    excess reached so far), so the curve reads simply as 'distance to the quota over
    training', with no confusing rise-then-fall. Returns median + a faint IQR band and
    the per-seed median first-feasible epoch."""
    grid = np.arange(0, XMAX + 1)
    curves, feas = [], []
    for seed in SEEDS:
        x, y = _load_curve(method, tag, seed)
        if x is None or len(x) == 0:
            continue
        bsf = np.minimum.accumulate(y)                 # best-so-far -> monotone
        curves.append(np.interp(grid, x, bsf, left=bsf[0], right=bsf[-1]))
        hit = np.where(bsf <= 0.5)[0]
        feas.append(float(x[hit[0]]) if len(hit) else None)
    if not curves:
        return None
    stack = np.vstack(curves)
    med = np.median(stack, 0)
    q1, q3 = np.percentile(stack, [25, 75], axis=0)
    fe = [f for f in feas if f is not None]
    conv = float(np.median(fe)) if fe else None
    return grid, med, q1, q3, conv


def make_convergence():
    """One clean panel. Three monotone curves fall to the quota line; the epoch each
    method needs to get there is folded into the legend, so nothing is repeated."""
    tag = "L30_G30"
    fig, ax = plt.subplots(figsize=(3.7, 3.15))
    order = ["hounie_rcl", "fioretto_ldf", "tralo"]    # draw TraLO last, on top
    data = {m: _series(m, tag) for m in order}
    convs = [data[m][4] for m in order if data[m] and data[m][4] is not None]
    xview = (max(convs) + 5) if convs else XMAX     # crop the empty post-convergence tail
    for m in order:
        if data[m] is None:
            continue
        st = STYLE[m]
        grid, med, q1, q3, conv = data[m]
        lab = NAME[m] + (f" ($\\approx${conv:.0f} ep)" if conv is not None else "")
        # Once a method reaches the quota it holds it (flat at zero); stop drawing there so
        # the descents stay legible and no long flat tail clutters the axis. The filled
        # marker sits at the convergence epoch, and the curve is drawn down into it.
        if conv is not None:
            keep = grid < conv
            gx = np.append(grid[keep], conv)
            gm = np.append(np.maximum(med[keep], 0), 0.0)
            b1 = np.append(np.maximum(q1[keep], 0), 0.0)
            b3 = np.append(np.maximum(q3[keep], 0), 0.0)
        else:
            gx, gm = grid, np.maximum(med, 0)
            b1, b3 = np.maximum(q1, 0), np.maximum(q3, 0)
        ax.fill_between(gx, b1, b3, color=st["color"], alpha=0.08, linewidth=0,
                        zorder=st["z"] - 6)
        ax.plot(gx, gm, color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], zorder=st["z"], label=lab)
        if conv is not None:
            ax.plot(conv, 0.0, marker=st["marker"], color=st["color"], markeredgecolor="white",
                    markersize=st["ms"] + 1.0, markeredgewidth=1.2, clip_on=False, zorder=20)
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.5, zorder=1)
    ax.text(xview * 0.99, 7, "quota met", ha="right", va="bottom",
            fontsize=7.2, color="0.4", style="italic")
    ax.set_xlim(0, xview)
    ax.set_ylim(-6, 250)
    ax.set_xlabel("epochs into the constraint phase")
    ax.set_ylabel("predictions over the quota")
    ax.set_title("Reaching the quota during training\n"
                 r"(DermMNIST, RegNetY-400MF, $L{=}G{=}30$)", fontsize=8.6)
    ax.grid(alpha=0.15, zorder=0)
    h, l = ax.get_legend_handles_labels()          # TraLO (ours) first
    ax.legend(h[::-1], l[::-1], loc="upper right", frameon=False, fontsize=8.4,
              handlelength=2.0, borderaxespad=0.4, labelspacing=0.45)
    fig.tight_layout()
    pdf, png = savefig_dual(fig, str(OUT), "fig_convergence")
    plt.close(fig)
    print("median time-to-feasibility:", {m: (data[m][4] if data[m] else None) for m in order})
    return png


if __name__ == "__main__":
    p = make_convergence()
    print("WROTE", p, os.path.getsize(p))

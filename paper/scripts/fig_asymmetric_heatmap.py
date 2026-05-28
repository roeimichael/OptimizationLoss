"""Phase 2 asymmetric tightness heatmap (polished v2).

Renders a 5x5 grid showing TraLO's macro_f1 advantage (delta) and post-hoc
flips advantage over the mean of {fioretto_ldf, hounie_rcl} for each (L, G)
tightness pair on DermMNIST / MobileNetV3 / MEL / loc_group.

Style choices (matching the headline figure discipline):
  - Diverging palettes centered on zero so blue = good-for-TraLO and
    red = bad-for-TraLO in BOTH panels:
      * F1 delta uses RdBu_r  (positive delta = TraLO higher F1 -> blue).
      * Flips delta uses RdBu (negative delta = TraLO fewer flips -> blue).
  - Per-cell numeric annotations (signed) with black/white contrast.
  - Symmetric color limits (vmin = -|max|, vmax = +|max|) so white = tie.
  - Diagonal cells (L = G, which is the symmetric Table A regime) are
    rendered with a grey hatch and an em-dash annotation to make it
    visually clear the table is off-diagonal only.

Output:
    paper/figures/fig_asymmetric_heatmap_v2.png

Source:
    paper/tables/table_B_phase2_asymmetric_derm.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
CSV_PATH = PAPER_DIR / "tables" / "table_B_phase2_asymmetric_derm.csv"
OUT_PATH = PAPER_DIR / "figures" / "fig_asymmetric_heatmap_v2.png"

TIGHTS = [20, 30, 50, 70, 80]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_tag(tag: str) -> tuple[int, int]:
    # e.g. "L20_G30" -> (20, 30)
    L = int(tag.split("_")[0].lstrip("L"))
    G = int(tag.split("_")[1].lstrip("G"))
    return L, G


def _annotate(ax, grid, fmt, vmax):
    """Write numeric annotations into each non-NaN cell of grid."""
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if np.isnan(v):
                continue
            # White text on saturated cells, black otherwise.
            color = "black" if abs(v) < 0.6 * vmax else "white"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")


def _mark_diagonal(ax, n):
    """Render diagonal cells with grey hatch + em-dash label to indicate
    they are not measured in Table B (symmetric tightness lives in Table A)."""
    for i in range(n):
        rect = Rectangle(
            (i - 0.5, i - 0.5), 1.0, 1.0,
            facecolor="#D9D9D9",
            edgecolor="black",
            linewidth=0.6,
            hatch="////",
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(i, i, "—", ha="center", va="center",
                fontsize=11, color="black", fontweight="bold",
                zorder=3)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df[["L", "G"]] = df["constraint_tag"].apply(
        lambda t: pd.Series(parse_tag(t))
    )

    tralo = df[df["method"] == "tralo"].set_index(["L", "G"])
    fio = df[df["method"] == "fioretto_ldf"].set_index(["L", "G"])
    hou = df[df["method"] == "hounie_rcl"].set_index(["L", "G"])

    # Build a 5x5 grid: rows=L (y-axis), cols=G (x-axis).
    f1_delta = np.full((len(TIGHTS), len(TIGHTS)), np.nan, dtype=float)
    flip_delta = np.full((len(TIGHTS), len(TIGHTS)), np.nan, dtype=float)
    for i, L in enumerate(TIGHTS):
        for j, G in enumerate(TIGHTS):
            key = (L, G)
            if key not in tralo.index:
                continue
            t_f1 = tralo.loc[key, "macro_f1_mean"]
            t_flips = tralo.loc[key, "flips_mean"]
            baseline_f1 = np.nanmean([
                fio.loc[key, "macro_f1_mean"] if key in fio.index else np.nan,
                hou.loc[key, "macro_f1_mean"] if key in hou.index else np.nan,
            ])
            baseline_flips = np.nanmean([
                fio.loc[key, "flips_mean"] if key in fio.index else np.nan,
                hou.loc[key, "flips_mean"] if key in hou.index else np.nan,
            ])
            f1_delta[i, j] = t_f1 - baseline_f1
            flip_delta[i, j] = t_flips - baseline_flips

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    # --- Macro F1 heatmap (TraLO - mean(Fio, Hou)) --------------------------
    # Positive delta = TraLO higher F1 -> we want BLUE (good for TraLO).
    # In matplotlib's RdBu colormap, LOW values -> red and HIGH values -> blue,
    # so RdBu (NOT _r) gives positive -> blue, negative -> red. Centered at 0.
    ax = axes[0]
    vmax = float(np.nanmax(np.abs(f1_delta)))
    im = ax.imshow(
        f1_delta,
        cmap="RdBu",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        zorder=1,
    )
    ax.set_xticks(range(len(TIGHTS)))
    ax.set_xticklabels([f"G{g}" for g in TIGHTS])
    ax.set_yticks(range(len(TIGHTS)))
    ax.set_yticklabels([f"L{l}" for l in TIGHTS])
    ax.set_xlabel("Global tightness $G$")
    ax.set_ylabel("Local tightness $L$")
    ax.set_title(r"Macro F1 advantage: TraLO $-$ mean(Fio, Hou)"
                 "\n(blue = TraLO better)")
    _annotate(ax, f1_delta, "{:+.3f}", vmax)
    _mark_diagonal(ax, len(TIGHTS))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$ macro F1  (TraLO $-$ baseline)")

    # --- Flips heatmap (TraLO - mean(Fio, Hou)) -----------------------------
    # MORE flips = WORSE.  Positive delta = TraLO needed more flips (bad for
    # TraLO) -> RED.  Negative delta = TraLO needed fewer flips (good for
    # TraLO) -> BLUE.  We plot the negated delta ("flip advantage") so that
    # high values mean "good for TraLO", and use the same RdBu palette as
    # the F1 panel (high -> blue) so the visual convention is consistent.
    ax = axes[1]
    flip_advantage = -flip_delta  # positive = TraLO fewer flips = good
    vmax_f = float(np.nanmax(np.abs(flip_advantage)))
    im2 = ax.imshow(
        flip_advantage,
        cmap="RdBu",
        origin="lower",
        vmin=-vmax_f,
        vmax=vmax_f,
        aspect="equal",
        zorder=1,
    )
    ax.set_xticks(range(len(TIGHTS)))
    ax.set_xticklabels([f"G{g}" for g in TIGHTS])
    ax.set_yticks(range(len(TIGHTS)))
    ax.set_yticklabels([f"L{l}" for l in TIGHTS])
    ax.set_xlabel("Global tightness $G$")
    ax.set_ylabel("Local tightness $L$")
    ax.set_title(r"Flips delta: TraLO $-$ mean(Fio, Hou)"
                 "\n(blue = TraLO fewer flips)")
    # Annotations show the original signed delta (TraLO - baseline).
    _annotate(ax, flip_delta, "{:+.1f}", vmax_f)
    _mark_diagonal(ax, len(TIGHTS))
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Flip advantage: $-\,\Delta$ flips  (higher = TraLO better)")

    fig.suptitle(
        "Asymmetric tightness landscape on DermMNIST "
        r"(diagonal $L\!=\!G$ excluded; see Table A)",
        fontsize=11.5,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print a short numerical summary for the paper.
    n_cells = int(np.count_nonzero(~np.isnan(f1_delta)))
    n_f1_wins = int(np.nansum(f1_delta > 0))
    n_f1_ties = int(np.nansum(np.abs(f1_delta) <= 0.001))
    n_flip_wins = int(np.nansum(flip_delta < 0))
    n_flip_ties = int(np.nansum(np.abs(flip_delta) <= 0.5))
    print(
        f"cells={n_cells}; F1 wins (TraLO>0)={n_f1_wins}; "
        f"F1 ties (|delta|<=0.001)={n_f1_ties}; "
        f"flip wins (TraLO<0)={n_flip_wins}; "
        f"flip ties (|delta|<=0.5)={n_flip_ties}; "
        f"mean F1 delta={np.nanmean(f1_delta):+.4f}; "
        f"mean flip delta={np.nanmean(flip_delta):+.2f}"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Regenerate fig_octmnist (single column): the OctMNIST constrained-class F1 win.

Design (2026-07-05, per-backbone gap): TWO stacked panels sharing x.
  top    -- absolute cc-F1 of every method as the cap tightens, on a representative
            backbone (MobileNetV3). TraLO is the bold "hero" line; the other
            constraint-trained methods are thin solid; the two post-hoc clippers are
            faded dashed/dotted and daggered (they post a higher RAW cc-F1 by
            over-predicting then clipping, so they are reference only). The six curves
            sit within ~0.02 of each other -- the win is invisible without the gap panel.
  bottom -- the headline gap, TraLO minus the BEST constraint-trained baseline (paired
            by seed), plotted PER BACKBONE: MobileNetV3, RegNetY-400MF, and ViT-B/16.
            This is the point of the figure: the win is backbone-general and GROWS with
            backbone capacity -- ViT-B/16 peaks at +0.081 at L30 -- and everywhere it
            traces the same honest inverted-U (near-zero at L10/L20 and at loose caps,
            positive only in the tight-binding band).
The tight-binding band (L30/L40) is shaded in both panels.

Data: paper/data/corpus/corpus_final.csv. Top panel = sweep 'paper_final',
octmnist, MobileNetV3. Bottom panel = canonical OctMNIST cells (constrained class 2,
group synth_group) across all three backbones and their per-backbone sweeps, deduped and
paired by seed -- so the plotted gaps match the numbers in the text and Table 2 exactly.

Run:  python paper/scripts/make_octmnist_fig.py
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (apply_style, savefig_dual, OKABE, C_TRALO, C_FIORETTO,
                       C_HOUNIE, BACKBONE_COLOR)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "paper", "data", "corpus", "corpus_final.csv")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

apply_style()

BACKBONE = "MobileNetV3"
TAGS = ["L10_G10", "L20_G20", "L30_G30", "L40_G40", "L50_G50",
        "L60_G60", "L70_G70", "L80_G80", "L90_G90"]
XLAB = [t.split("_")[0] for t in TAGS]

# ---- bottom panel: per-backbone gap (paper_final only, paired-by-seed) ----
DUALS = ["fioretto_ldf", "hounie_rcl"]
# (model key, label, marker) -- ViT-B/16 is the hero (thickest, drawn last/on top)
GAP_BACKBONES = [
    ("MobileNetV3",  "MobileNetV3",   "o", 1.1, 3, False),
    ("RegNetY400MF", "RegNetY-400MF", "s", 1.1, 4, False),
    ("ViTB16",       "ViT-B/16",      "D", 1.9, 6, True),
]


def _paired(metric):
    """paper_final octmnist pivot with per-seed best trained dual and TraLO's paired gap."""
    df = pd.read_csv(CSV, low_memory=False)
    df = df[(df.sweep == "paper_final") & (df.dataset == "octmnist")
            & df.constraint_tag.isin(TAGS)]
    df = df.groupby(["model", "constraint_tag", "method", "seed"],
                    as_index=False)[metric].mean()
    piv = df.pivot_table(index=["model", "constraint_tag", "seed"],
                         columns="method", values=metric).reset_index()
    piv["best"] = piv[DUALS].max(axis=1)
    piv = piv.dropna(subset=["tralo", "best"])
    piv["gap"] = piv.tralo - piv.best
    return piv


def backbone_gaps(metric="cc_f1"):
    """Per (backbone, cap) paired gap of TraLO over the best trained dual, on `metric`.

    Source is the SINGLE 'paper_final' sweep (identical to Table 1/2), NOT a cross-sweep
    pool: octmnist also lives in separate vit_octmnist_s*/octmnist_* sweeps, and pooling
    them would average two distinct runs that share a seed number. paper_final already
    holds the full octmnist grid (9 caps x 6 methods x 4 seeds per backbone).
    """
    piv = _paired(metric)
    out = {}
    for key, *_ in GAP_BACKBONES:
        g = [float(piv[(piv.model == key) & (piv.constraint_tag == t)].gap.mean())
             for t in TAGS]
        out[key] = np.array(g)
    return out


def backbone_gap_cis(metric="cc_f1", n_boot=20000, seed=0):
    """Per (backbone, cap) 95% bootstrap CI of the paired gap.

    The panel plots means over four seeds and the peak carries a std 55% of its
    own value (+0.081 +/- 0.045, Table 7), so without intervals the inverted-U
    is not visually separable from noise. Resamples the four paired per-seed
    gaps, which is the unit the comparison is made on.
    """
    piv = _paired(metric)
    rng = np.random.default_rng(seed)
    out = {}
    for key, *_ in GAP_BACKBONES:
        lo, hi = [], []
        for t in TAGS:
            v = piv[(piv.model == key) & (piv.constraint_tag == t)].gap.to_numpy(float)
            if len(v) == 0:
                lo.append(np.nan); hi.append(np.nan); continue
            draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
            lo.append(float(np.percentile(draws, 2.5)))
            hi.append(float(np.percentile(draws, 97.5)))
        out[key] = (np.array(lo), np.array(hi))
    return out


def tight_cap_table():
    """Print the tab_oct_backbone numbers: paired gap mean +/- sample-std + seed-winrate
    at L30/L40 for cc_f1 AND f1_macro (the two Delta columns of the table)."""
    for metric in ["cc_f1", "f1_macro"]:
        piv = _paired(metric)
        print(f"\n[tab_oct_backbone] {metric}: Delta mean +/- std (ddof=1), seeds-won")
        for key, lab, *_ in GAP_BACKBONES:
            for t in ["L30_G30", "L40_G40"]:
                s = piv[(piv.model == key) & (piv.constraint_tag == t)].gap
                print(f"    {lab:12s} {t[:3]}  {s.mean():+.3f}  +/-{s.std(ddof=1):.3f}  "
                      f"{int((s > 0).sum())}/{len(s)}")

# (method key, label, color, linestyle, marker, lw, ms, zorder, markerfacecolor, alpha)
# Post-hoc clippers are deliberately recessive (alpha .38, low zorder): inside the shaded
# band their RAW cc-F1 sits above TraLO's, and at full strength they hijack the 3-second
# read of the headline win (which is vs the constraint-TRAINED methods). Styles + daggers
# keep them fully identifiable; nothing is hidden.
METHODS = [
    ("tralo",          "TraLO",        C_TRALO,          "-",         "o", 1.35, 3.8, 7, None,    1.00),
    ("tralo_bounded",  "TraLO-b",      OKABE["skyblue"], "-",         "o", 0.85, 2.7, 5, "white", 0.95),
    ("fioretto_ldf",   "Fioretto",     C_FIORETTO,       "-",         "^", 0.85, 2.9, 5, None,    0.95),
    ("hounie_rcl",     "Hounie",       C_HOUNIE,         "-",         "s", 0.85, 2.7, 5, None,    0.95),
    ("danits_lp",      r"LP-LG$^\dagger$", OKABE["purple"], (0, (4, 2)), "D", 0.80, 2.4, 2, None, 0.38),
    ("heuristic",      r"Heur.$^\dagger$",  "#7a7a7a",     (0, (1, 1.6)), "v", 0.80, 2.5, 2, None, 0.38),
]


def method_means():
    df = pd.read_csv(CSV)
    d = df[(df["sweep"] == "paper_final") & (df["dataset"] == "octmnist")
           & (df["model"] == BACKBONE)]
    out = {}
    for key, *_ in METHODS:
        y = []
        for tag in TAGS:
            sub = d[(d["constraint_tag"] == tag) & (d["method"] == key)]["cc_f1"]
            y.append(float(np.mean(sub)) if len(sub) else np.nan)
        out[key] = np.array(y)
    return out


BAND = (1.6, 3.4)          # x-extent of the shaded tight-binding band (around L30/L40)
C_BAND = "#f5deb0"          # band fill (a touch stronger than before; still recessive)


def make_fig(means, gaps, cis=None):
    x = np.arange(len(TAGS), dtype=float)
    fig, (ax, axg) = plt.subplots(
        2, 1, figsize=(3.7, 3.7), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.35], "hspace": 0.09})

    # ----- top: absolute cc-F1, all six methods (MobileNetV3 example) -----
    ax.axvspan(*BAND, color=C_BAND, alpha=0.45, zorder=0)
    for key, lab, col, ls, mk, lw, ms, z, mfc, al in METHODS:
        ax.plot(x, means[key], color=col, ls=ls, lw=lw, marker=mk, ms=ms, zorder=z,
                label=lab, alpha=al, mfc=(mfc if mfc else col), mec=col, mew=0.7)
    ax.set_ylabel("constrained-class F1")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, ncol=2,
              columnspacing=1.0, handlelength=2.0)

    # ----- bottom: the headline gap PER BACKBONE (TraLO - best trained, paired) -----
    axg.axvspan(*BAND, color=C_BAND, alpha=0.45, zorder=0)
    axg.axhline(0.0, color="#888888", lw=0.7, zorder=1)
    i30 = TAGS.index("L30_G30")
    for key, lab, mk, lw, z, hero in GAP_BACKBONES:
        col = BACKBONE_COLOR[key]
        g = gaps[key]
        # 95% bootstrap band over the four paired per-seed gaps. Drawn under the
        # lines: at the peak the interval is wide enough that the inverted-U has
        # to be read as a trend, not as a set of separated points.
        if cis is not None and key in cis:
            lo, hi = cis[key]
            axg.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=z - 1)
        axg.plot(x, g, color=col, lw=lw, marker=mk, ms=(3.6 if hero else 2.8),
                 zorder=z, label=lab, mec="white" if hero else col, mew=0.5)
    axg.set_ylabel("$\\Delta$ cc-F1", fontsize=8)
    # Headroom for the ViT bootstrap band, whose upper bound reaches +0.116 --
    # the old limit of 0.094 clipped it, which would have hidden exactly the
    # uncertainty the band exists to show.
    axg.set_ylim(-0.030, 0.125)
    axg.set_yticks([0.00, 0.04, 0.08, 0.12])
    axg.legend(loc="upper right", frameon=False, fontsize=7.2, handlelength=1.6,
               labelspacing=0.25, borderaxespad=0.3,
               title="TraLO $-$ best trained", title_fontsize=7.2)
    axg.set_xlim(-0.4, len(TAGS) - 0.6)
    axg.set_xticks(x)
    axg.set_xticklabels(XLAB)
    axg.set_xlabel(r"cap level $L$ (%)")

    fig.tight_layout()
    pdf, png = savefig_dual(fig, OUT, "fig_octmnist")
    plt.close(fig)
    return png


if __name__ == "__main__":
    means = method_means()
    gaps = backbone_gaps()
    cis = backbone_gap_cis()
    p = make_fig(means, gaps, cis)
    print("WROTE", p, os.path.getsize(p))
    i30, i40 = TAGS.index("L30_G30"), TAGS.index("L40_G40")
    for key, lab, *_ in GAP_BACKBONES:
        print(f"  {lab:12s} gap @L30={gaps[key][i30]:+.3f} "
              f"[{cis[key][0][i30]:+.3f},{cis[key][1][i30]:+.3f}]  "
              f"@L40={gaps[key][i40]:+.3f} "
              f"[{cis[key][0][i40]:+.3f},{cis[key][1][i40]:+.3f}]")
    tight_cap_table()

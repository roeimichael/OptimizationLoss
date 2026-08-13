"""Generate fig_loss_shape and fig_mechanism for the AAAI paper.

  - fig_loss_shape: the bounded transductive penalty L(E)=E/(E+K)+rho*(E/K)^2/(1+(E/K)^2)
    vs a naive UNBOUNDED dual-ascent penalty (linear lambda*E), over excess E for a few rho.
    SINGLE-COLUMN analytic illustration (no training data).
  - fig_mechanism: empirical trajectories on a ONE-EPOCH-WARMUP probe cell
    (dermmnist RegNetY400MF pushpull_derm_w1 L50_G50, seed 1) -- FULL-WIDTH (figure*).
    Warmup=1 keeps the classifier LEARNING through the constraint phase, so the
    CE-vs-constraint tug-of-war is visible: (a) lambda 53 vs 0.18 (~300x),
    (b) identical CE traces, (c) excess oscillates while CE is live and falls to
    the cap only after the CE-saturation gate fires -- for BOTH methods.
    Reads the self-contained copy under paper/data/dynamics/ (see
    paper/data/README_DATA.md for provenance).

Run:  python paper/scripts/make_figs.py
"""
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (apply_style, savefig_dual, legend_clear,
                       C_TRALO, C_FIORETTO, OKABE)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The AAAI tree was retired to archive/legacy in the 2026-07 reorg; the paper is
# self-contained under paper/ now. Both paths kept resolvable so the script runs
# from a clean clone.
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(PAPER, "figures")
DATA = os.path.join(PAPER, "data")
os.makedirs(OUT, exist_ok=True)

apply_style()

# Sequential blue ramp for the TraLO rho curves (luminance-distinct -> grayscale-safe).
# Lightest step darkened one notch (was #9ecae1) so the rho=0.5 curve survives grayscale.
C_RHO = ["#6baed6", "#3182bd", "#08519c"]


# ----------------------------------------------------------------------------
# fig_loss_shape  -- single column (~3.4 in)
# ----------------------------------------------------------------------------
def make_loss_shape():
    K = 10.0   # cap (constrained-class count limit), illustrative
    E_MAX = 40.0
    E = np.linspace(0.0, E_MAX, 800)  # soft excess over cap -- runs the full axis
    e = E / K

    fig, ax = plt.subplots(figsize=(3.45, 2.7))

    # TraLO bounded penalty for a few rho: saturation term + rho*bounded-quad.
    # Curves run to the axis edge and each label sits at its OWN curve's endpoint
    # (color-matched), so curve<->label association is unambiguous.
    sat = E / (E + K)
    for c, rho in zip(C_RHO, [0.5, 1.0, 2.0]):
        quad = rho * (e ** 2) / (1.0 + e ** 2)
        L = sat + quad
        ax.plot(E, L, color=c, lw=2.0, zorder=3)
        ax.axhline(1.0 + rho, color=c, ls=":", lw=1.0, alpha=0.75, zorder=1)
        ax.annotate(rf"$\rho{{=}}{rho:g}$  ($1{{+}}\rho{{=}}{1.0 + rho:g}$)",
                    xy=(E_MAX, L[-1]), xytext=(E_MAX + 0.8, L[-1]),
                    ha="left", va="center", fontsize=8.5, color=c)

    # Naive UNBOUNDED dual-ascent effective penalty: lambda*E (grows without bound).
    slope = 0.18
    Lin = slope * E
    ax.plot(E, Lin, color=C_FIORETTO, lw=2.2, ls="--", zorder=3)
    # Label where the line exits the top of the axes (ylim 5.2 -> E = 5.2/slope).
    ax.annotate(r"unbounded $\lambda E$", xy=(5.2 / slope, 5.05),
                xytext=(5.2 / slope + 1.0, 5.05),
                ha="left", va="center", fontsize=8.5, color=C_FIORETTO)

    ax.set_xlabel(r"soft excess over cap $E$  (cap $K{=}10$)")
    ax.set_ylabel(r"penalty $L(E)$")
    ax.set_ylim(0, 5.2)
    ax.set_xlim(0, 53)  # data to 40 + room for the endpoint labels

    fig.tight_layout()
    pdf, png = savefig_dual(fig, OUT, "fig_loss_shape")
    plt.close(fig)
    return png


# ----------------------------------------------------------------------------
# fig_mechanism  -- FULL WIDTH (figure*, ~7.0 in), three panels
# ----------------------------------------------------------------------------
def _tralo_excess(tra, cls=4):
    """Constraint excess (global + per-group local) from a TraLO log, mirroring
    make_convergence_fig."""
    lim = tra[f"Limit_Class{cls}"]
    g = np.where(np.isfinite(lim), np.maximum(0.0, tra[f"Hard_Class{cls}"] - lim), 0.0)
    i = 0
    while f"Group{i}_Hard_Class{cls}" in tra.columns:
        h, l = tra[f"Group{i}_Hard_Class{cls}"], tra[f"Group{i}_Limit_Class{cls}"]
        g = g + np.where(np.isfinite(l), np.maximum(0.0, h - l), 0.0)
        i += 1
    return np.asarray(g, dtype=float)


def make_mechanism():
    """One-epoch-warmup probe (derm L50/G50, RegNetY-400MF, seed 1): the classifier
    is still LEARNING through the whole constraint phase, so the CE-vs-constraint
    tug-of-war is visible (with the paper's 50-epoch warmup, CE is pre-saturated
    and the saturation gate switches it off within ~2 epochs in every method).
    Causal chain, left to right:
      (a) dual ascent escalates lambda every violated epoch (1 -> 53) while
          TraLO's ratchet stays <= 0.18 (~300x smaller);
      (b) the escalation buys nothing -- the two CE traces are numerically
          indistinguishable (same seed; constraint step is one clipped update
          per epoch). Open circles mark the CE-saturation gate (train acc >=
          0.995) after which CE updates stop;
      (c) NEITHER method can move the count while CE is live (excess oscillates
          ~60 epochs); it falls to the cap only after CE stops -- and then both
          methods satisfy within a few epochs. Same outcome, 300x the pressure."""
    base = os.path.join(DATA, "dynamics", "dermmnist")

    def _seed_logs(method):
        """Every logged seed of the probe, seed_1 first. The probe was always run
        at three seeds; the figure used to plot only seed_1, which made a
        three-seed result read as n=1 (blind review r1)."""
        pat = os.path.join(base, method, "w1_probe", "pushpull_derm_w1",
                           "L50_G50", "seed_*", "training_log.csv")
        return [pd.read_csv(p) for p in sorted(glob.glob(pat))]

    fio_all = _seed_logs("fioretto_ldf")
    tra_all = _seed_logs("tralo")
    assert fio_all and tra_all, "no probe logs found under %s" % base
    fio, tra = fio_all[0], tra_all[0]
    print("mechanism probe: %d Fioretto seeds, %d TraLO seeds"
          % (len(fio_all), len(tra_all)))

    fio_ep = fio["epoch"].to_numpy(dtype=float)  # constraint epochs 0..70
    # TraLO logs absolute epochs (2, 5, 10, ...); rel epoch 0 aligns with fio
    # epoch 0 (same seed: CE matches to 4 decimals at every shared epoch).
    tra_ep = tra["Epoch"].to_numpy(dtype=float) - tra["Epoch"].min()

    fio_ce = fio["ce_loss"].to_numpy(dtype=float)
    fio_lg = fio["max_lambda_g"].to_numpy(dtype=float)
    fio_ex = fio["total_excess"].to_numpy(dtype=float)
    tra_ce = tra["L_CE"].to_numpy(dtype=float)
    tra_lg = tra["Lambda_Global"].to_numpy(dtype=float)
    tra_ex = _tralo_excess(tra)

    # CE-saturation gate epochs. Fioretto's log goes NaN when the gate stops the
    # CE loop (np.mean of an empty list); TraLO's logs 0.0 by convention. Neither
    # value is a real loss -- truncate both CE traces at the last REAL sample.
    fio_real = ~np.isnan(fio_ce)
    fio_gate = float(fio_ep[np.argmax(~fio_real)])      # first gated epoch (58)
    tra_real = tra_ce > 0.0
    tra_gate = float(tra_ep[np.argmax(~tra_real)])      # first logged 0.0 (63)

    xmax = float(max(fio_ep.max(), tra_ep.max()))       # 76
    ratio = fio_lg.max() / tra_lg.max()                 # ~297

    FIO = dict(color=C_FIORETTO, lw=1.6)
    TRA = dict(color=C_TRALO, lw=1.4, ls="--", marker="s", ms=3.0)
    # Seeds 2..n as faint traces behind the seed-1 curve: enough to show the
    # spread without turning three panels into spaghetti.
    GHOST_F = dict(color=C_FIORETTO, lw=0.7, alpha=0.30, zorder=1)
    GHOST_T = dict(color=C_TRALO, lw=0.7, alpha=0.30, ls="--", zorder=1)

    def _ghosts(ax, logs, ep_col, val_col, style, transform=None):
        """Plot the non-primary seeds. Silently skips a seed missing the column,
        so a partially-logged probe degrades to fewer ghosts, never to a crash."""
        for df in logs[1:]:
            if ep_col not in df.columns:
                continue
            ep = df[ep_col].to_numpy(dtype=float)
            ep = ep - ep.min()
            if transform is not None:
                val = transform(df)
            elif val_col in df.columns:
                val = df[val_col].to_numpy(dtype=float)
            else:
                continue
            n = min(len(ep), len(val))
            ax.plot(ep[:n], val[:n], **style)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 1.40))

    # ---- (a) the cause: the constraint weight each method applies ----
    # Log scale: TraLO's ratchet (0.05 -> 0.18, then frozen) is invisible next to
    # lambda=53 on a linear axis -- it reads as "flat zero" when it in fact rises
    # 3.6x and then freezes. Both methods keep one multiplier per constraint; the
    # global-cap one is plotted. On this cell the local caps are never violated,
    # so every local multiplier stays at exactly zero (TraLO logs Lambda_Local=0
    # throughout; Fioretto's local duals get zero subgradient and never move).
    _ghosts(axA, fio_all, "epoch", "max_lambda_g", GHOST_F)
    _ghosts(axA, tra_all, "Epoch", "Lambda_Global", GHOST_T)
    axA.plot(fio_ep, fio_lg, label="Fioretto-LDF", **FIO)
    axA.plot(tra_ep, tra_lg, label="TraLO (ours)", **TRA)
    axA.set_yscale("log")
    axA.set_ylim(0.04, 95)
    axA.set_yticks([0.05, 0.5, 5, 50])
    axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axA.yaxis.set_minor_locator(mticker.NullLocator())
    # House rule (2026-07-03): NO text or arrows inside the axes. The panel
    # titles and the caption carry the reading; the curves stand alone.
    axA.set_title("(a) the response: $\\lambda$ escalates", fontsize=9)
    axA.set_ylabel("multiplier $\\lambda$")
    # Placed by collision check, not by hand: the escalating Fioretto curve
    # sweeps through the upper-left corner this legend used to sit in.
    legend_clear(axA, frameon=False, fontsize=7,
                 handlelength=1.6, borderaxespad=0.2)

    # ---- (b) the non-effect: classification loss is identical either way ----
    _ghosts(axB, fio_all, "epoch", "ce_loss", GHOST_F)
    _ghosts(axB, tra_all, "Epoch", "L_CE", GHOST_T)
    axB.plot(fio_ep[fio_real], fio_ce[fio_real], **FIO)
    axB.plot(tra_ep[tra_real], tra_ce[tra_real], **TRA)
    yB = float(np.nanmax(fio_ce)) * 1.14
    axB.set_ylim(0, yB)
    # Open circles = last real CE sample before the shared saturation gate
    # (train acc >= 0.995) stops CE updates in each method.
    fio_last = int(np.argmax(~fio_real)) - 1
    tra_last = int(np.argmax(~tra_real)) - 1
    axB.plot([fio_ep[fio_last]], [fio_ce[fio_last]], marker="o", ms=6.5,
             mfc="none", mec=C_FIORETTO, mew=1.4, zorder=4)
    axB.plot([tra_ep[tra_last]], [tra_ce[tra_last]], marker="o", ms=6.5,
             mfc="none", mec=C_TRALO, mew=1.4, zorder=4)
    axB.set_title("(b) effect on the classifier: none", fontsize=9)
    axB.set_ylabel("cross-entropy")

    # ---- (c) the outcome: the count moves only after CE stops ----
    _ghosts(axC, fio_all, "epoch", "total_excess", GHOST_F)
    _ghosts(axC, tra_all, "Epoch", None, GHOST_T, transform=_tralo_excess)
    axC.plot(fio_ep, fio_ex, **{**FIO, "lw": 1.3})
    axC.plot(tra_ep, tra_ex, **TRA)
    yC = float(max(fio_ex.max(), tra_ex.max())) * 1.24
    axC.set_ylim(-8, yC)
    # Method-colored dotted lines: where each method's CE gate fires. The
    # monotone crash to 0 starts exactly there for both.
    # Method-colored dotted lines mark each method's CE-gate epoch (caption).
    axC.axvline(fio_gate, color=C_FIORETTO, ls=":", lw=1.1,
                alpha=0.85, zorder=1)
    axC.axvline(tra_gate, color=C_TRALO, ls=":", lw=1.1,
                alpha=0.85, zorder=1)
    axC.set_title("(c) the count waits for CE, not $\\lambda$", fontsize=9)
    axC.set_ylabel("excess (count $-$ cap)")

    for ax_ in (axA, axB, axC):
        ax_.set_xlabel("constraint epoch")
        ax_.set_xlim(0, xmax + 1)
        ax_.set_xticks(np.arange(0, xmax + 1, 10))

    fig.tight_layout()
    pdf, png = savefig_dual(fig, OUT, "fig_mechanism")
    plt.close(fig)
    return png, fio_gate, tra_gate, float(fio_lg.max()), float(tra_lg.max())


if __name__ == "__main__":
    # fig_loss_shape is generated by make_loss_shape_fig.py (2-panel penalty+gradient,
    # no in-plot text); the make_loss_shape() below is the superseded single-panel
    # design and is intentionally NOT called so it cannot clobber the current figure.
    p2, fgate, tgate, fmax, tmax = make_mechanism()
    print("WROTE", p2, os.path.getsize(p2))
    print(f"mechanism: gates fio@{fgate:.0f} tra@{tgate:.0f}; "
          f"max lambda fio={fmax:.2f} tralo={tmax:.3f} ({fmax/tmax:.0f}x)")

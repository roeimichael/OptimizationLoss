"""Rebuild fig_loss_shape: what the penalty does, and what its gradient does.

Two panels stacked vertically so each is a full column wide (3.3in) and readable, rather
than two cramped 1.5in panels side by side. No in-plot text labels (house rule; they also
overflowed the axes): every curve is named in a legend. A light band under TraLO's curve
marks the region the bounded penalty lives in.

Panel (a): the effective per-cap penalty lambda*l against the violation, at the multipliers
each method actually runs at (warmup-1 probe: TraLO ratchets to lambda<=0.19 and freezes;
Fioretto-LDF's dual ascent escalates to lambda~53). TraLO flattens at its ceiling
lambda(1+rho); the dual's linear penalty grows without bound in E and in lambda.

Panel (b): the count-gradient d(lambda*l)/dS, the quantity that reaches the parameters.
TraLO's is capped and vanishes both at the cap and far from it; the dual's is constant in E
and scales with the escalating lambda.

The caption's honest point: this is loss-surface geometry, not update geometry. The
constraint term is a scalar on a fixed direction (Sec. 3), so the optimizer discards the
penalty's scale, and the shape is tested empirically (Sec. 5.1 / Supp. B.2).

Regenerate:  python scripts/make_loss_shape_fig.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from fig_style import apply_style, savefig_dual, C_TRALO, C_FIORETTO, C_WASH

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")

K = 10.0                 # illustrative cap
RHO = 100.0              # TraLO's rho ceiling (ramps 5 -> 100)
LAM_TRALO = 0.19         # measured max ratcheted multiplier (w1 probe)
LAM_FIORETTO = 53.0      # measured escalated dual multiplier (w1 probe)


def penalty(E, rho=RHO):
    u = E / K
    return E / (E + K) + rho * u ** 2 / (1 + u ** 2)


def count_grad(E, rho=RHO):
    """d(penalty)/dS in units of 1/K (S = soft count, E = relu(S-K))."""
    u = E / K
    return (1.0 / (1 + u) ** 2 + rho * 2 * u / (1 + u ** 2) ** 2) / K


def main():
    apply_style()
    plt.rcParams.update({"axes.titlesize": 9.5, "legend.fontsize": 8.5,
                         "axes.labelsize": 9.5})
    # Taller than the 2.12in original: the legend now carries four entries on two
    # rows, and the log minor ticks need room to read as ticks rather than a smear.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.55), sharex=True)
    E = np.linspace(0, 60, 800)
    ceiling = LAM_TRALO * (1 + RHO)

    # One shared legend at the top (both panels contrast the same two methods), so the
    # plot areas stay clear of labels even when short.

    # ---- (a) effective penalty: ours flattens under a ceiling, theirs runs away -------
    pen = LAM_TRALO * penalty(E)
    ax1.fill_between(E, 1e-1, pen, color=C_TRALO, alpha=0.10, lw=0)
    l_tralo, = ax1.plot(E, pen, color=C_TRALO, lw=2.6)
    l_dual, = ax1.plot(E, LAM_FIORETTO * E / K, color=C_FIORETTO, ls="--", lw=2.6)
    l_ceil = ax1.axhline(ceiling, color=C_WASH, ls=":", lw=1.4)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-1, 1e3)
    ax1.set_xlim(0, 60)
    ax1.set_ylabel(r"penalty $\lambda\ell$")
    ax1.set_title("(a) the penalty stays bounded")

    # ---- (b) what reaches the weights: capped and vanishing vs a constant -------------
    # The dual penalty of panel (a) is lambda*E/K, so the count-gradient it applies is
    # lambda/K -- NOT lambda. Plotting the constant at lambda made panel (b) disagree
    # with its own panel (a) by a factor of K; blind review round 1 caught it.
    g = LAM_TRALO * count_grad(E)
    ax2.fill_between(E, 1e-2, g, color=C_TRALO, alpha=0.10, lw=0)
    ax2.plot(E, g, color=C_TRALO, lw=2.6)
    ax2.axhline(LAM_FIORETTO / K, color=C_FIORETTO, ls="--", lw=2.6)
    # Same dual rule at TraLO's OWN multiplier. Without it the panel reads as a
    # scale argument, which Sec. 6 shows is immaterial -- the honest contrast is
    # the shape at matched lambda, and it is small.
    l_match = ax2.axhline(LAM_TRALO / K, color=C_FIORETTO, ls=":", lw=1.6)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 3e1)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel(r"constraint violation $E$")
    ax2.set_ylabel(r"gradient $\partial(\lambda\ell)/\partial S$")
    ax2.set_title("(b) so does the gradient it applies")

    fig.legend([l_tralo, l_dual, l_ceil, l_match],
               [r"TraLO ($\lambda\!\leq\!0.19$)", r"dual ascent ($\lambda\!\to\!53$)",
                r"ceiling $\lambda(1{+}\rho)$", r"dual at matched $\lambda$"],
               loc="upper center", ncol=2, frameon=False, handlelength=1.5,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.5, 1.02))
    # Log minor ticks on a 3-decade axis this short render as a solid black band.
    for ax in (ax1, ax2):
        ax.minorticks_off()
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout(pad=0.3, h_pad=0.55, rect=[0, 0, 1, 0.88])
    pdf, png = savefig_dual(fig, OUT, "fig_loss_shape")
    print("wrote", pdf)
    print(f"TraLO: ceiling lam(1+rho)={ceiling:.1f}; count-grad peak={g.max():.2f} "
          f"at E={E[np.argmax(g)]:.1f}; at E=0 -> {g[0]:.4f}")
    print(f"dual:  count-grad={LAM_FIORETTO/K:.2f} constant (lambda/K); "
          f"ratio at TraLO's peak = {LAM_FIORETTO/K/g.max():.1f}x")
    print(f"dual at matched lambda: {LAM_TRALO/K:.4f} -> TraLO's peak is "
          f"{g.max()/(LAM_TRALO/K):.1f}x it (shape alone, scale removed)")
    # Consistency check: panel (b) must be d(panel a)/dS for BOTH curves.
    num = np.gradient(LAM_TRALO * penalty(E), E)
    assert np.allclose(num[1:], g[1:], rtol=2e-3), "TraLO: (b) is not d(a)/dS"
    num_d = np.gradient(LAM_FIORETTO * E / K, E)
    assert np.allclose(num_d, LAM_FIORETTO / K, rtol=1e-9), "dual: (b) is not d(a)/dS"
    print("panel consistency: (b) == d(a)/dS for both curves")


if __name__ == "__main__":
    main()

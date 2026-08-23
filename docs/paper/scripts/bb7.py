"""Backbone interaction, step 7: variance decomposition of the reported gap.

tralo - bestdual = (tralo - clipCE) - (bestdual - clipCE). Both terms are
measured here against the same untreated 30-epoch CE control in the same seed,
so the question "is the gap TraLO gaining or the duals losing?" is answerable by
comparing how much of the gap's variance each term carries.
"""
import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]

S = pd.read_csv("paper/scripts/out_bb_decomp_perseed.csv")
S = S[S.metric == "ccF1eq"]
g = S.groupby(CELL).agg(TD=("tralo_m_bestdual", "mean"),
                        TC=("tralo_m_clip", "mean"),
                        DC=("bestdual_m_clip", "mean")).reset_index()

print("=" * 100)
print("VARIANCE DECOMPOSITION of  (tralo - bestdual)  over the 12 cells")
print("=" * 100)
print("  mean  TraLO - plainCE   = %+0.4f   sd %.4f   range [%+0.4f, %+0.4f]"
      % (g.TC.mean(), g.TC.std(ddof=1), g.TC.min(), g.TC.max()))
print("  mean  duals - plainCE   = %+0.4f   sd %.4f   range [%+0.4f, %+0.4f]"
      % (g.DC.mean(), g.DC.std(ddof=1), g.DC.min(), g.DC.max()))
print("  mean  TraLO - duals     = %+0.4f   sd %.4f"
      % (g.TD.mean(), g.TD.std(ddof=1)))
print()
print("  corr(TraLO-duals, TraLO-plainCE) = %+0.3f  r2=%.3f"
      % (g.TD.corr(g.TC), g.TD.corr(g.TC) ** 2))
print("  corr(TraLO-duals, duals-plainCE) = %+0.3f  r2=%.3f"
      % (g.TD.corr(g.DC), g.TD.corr(g.DC) ** 2))
print()
print("  var(TraLO-plainCE)/var(TraLO-duals) = %.2f"
      % (g.TC.var(ddof=1) / g.TD.var(ddof=1)))
print("  var(duals-plainCE)/var(TraLO-duals) = %.2f"
      % (g.DC.var(ddof=1) / g.TD.var(ddof=1)))
print()
print("  cells where TraLO beats plain CE by >0.005 : %d of 12"
      % int((g.TC > 0.005).sum()))
print("  cells where TraLO loses to plain CE by >0.005: %d of 12"
      % int((g.TC < -0.005).sum()))
print("  cells where TraLO is within +-0.005 of plain CE: %d of 12"
      % int((g.TC.abs() <= 0.005).sum()))
print()
print("  sign of (duals - plainCE) by dataset:")
for ds, gg in g.groupby("dataset"):
    print("    %-12s %s   (TraLO-duals: %s)"
          % (ds, np.round(gg.DC.to_numpy(), 4).tolist(),
             np.round(gg.TD.to_numpy(), 4).tolist()))

# seed-level, n=48
print()
print("  seed-level (n=48): corr(T-D, D-C) = %+0.3f ; corr(T-D, T-C) = %+0.3f"
      % (S.tralo_m_bestdual.corr(S.bestdual_m_clip),
         S.tralo_m_bestdual.corr(S.tralo_m_clip)))
print("  seed-level sd: T-C %.4f   D-C %.4f   T-D %.4f"
      % (S.tralo_m_clip.std(ddof=1), S.bestdual_m_clip.std(ddof=1),
         S.tralo_m_bestdual.std(ddof=1)))

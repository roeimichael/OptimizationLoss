"""Backbone interaction, step 6: the overshoot variable is measured on the
WRONG MODEL, plus count mobility and the analytic penalty gradient.

The brief's overshoot (clipper raw count / K) is the count of a 30-epoch plain-CE
model. TraLO never sees that model: it starts from a ONE-epoch warm-up. The first
row TraLO logs (Epoch=2, i.e. constraint epoch 1) is the closest recorded state to
its actual starting point, so the correlation is recomputed against that too.

Also computed here:
  * count mobility on the every-5 epoch grid that BOTH backbones log, so a denser
    TraLO log (extra rows on satisfaction) cannot inflate it,
  * the analytic dL/dS of the penalty at each cell's own operating point, using
    the logged lambda and soft count and the rho schedule read from
    src/methodologies/tralo/train.py:126 (rho_step=(100-5)/29, incremented every
    epoch until first satisfaction, then frozen).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")

CELL = ["dataset", "model", "cap"]
GRID = [4, 9, 14, 19, 24, 29]          # logged by every run: (epoch+1)%5==0


def corr(x, y, lab):
    ok = ~(np.isnan(x) | np.isnan(y))
    n = int(ok.sum())
    if n < 4:
        print("    %-42s n<4" % lab)
        return
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rs = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    print("    %-42s n=%2d  r=%+0.3f  r2=%.3f  spearman=%+0.3f  t=%+0.2f"
          % (lab, n, r, r * r, rs, t))


def main():
    T = pd.read_csv("paper/scripts/out_bb_traj_full.csv")
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    fb = fb[fb.campaign == "lrc0.0001_noceskip"]
    K = fb.groupby(CELL)["K"].first()
    edge = fb[fb.method == "tralo"].set_index(CELL)["d_vs_bestdual"]
    clipraw = fb[fb.method == "tralo"].set_index(CELL)["clip_raw"]

    rows = []
    for (ds, mo, cap, sd), g in T.groupby(CELL + ["seed"]):
        g = g.sort_values("cep")
        kk = float(K.loc[(ds, mo, cap)])
        start = g[g.cep == 1]
        gr = g[g.cep.isin(GRID)].sort_values("cep")
        rows.append({
            "dataset": ds, "model": mo, "cap": cap, "seed": sd, "K": kk,
            "hard_cep1": float(start.hard.iloc[0]) if len(start) else np.nan,
            "soft_cep1": float(start.soft.iloc[0]) if len(start) else np.nan,
            "n_grid": len(gr),
            "grid_absdelta": float(np.abs(np.diff(gr.hard.to_numpy())).mean())
                             if len(gr) > 1 else np.nan,
            "grid_sd": float(gr.hard.std(ddof=1)) if len(gr) > 1 else np.nan,
            "grid_min": float(gr.hard.min()) if len(gr) else np.nan,
            "hard_last": float(g.hard.iloc[-1]), "soft_last": float(g.soft.iloc[-1]),
            "lam_last": float(g.lam.iloc[-1]),
            "ever_sat": int((g.sat > 0).any()),
            "first_sat": float(g.cep[g.sat > 0].min()) if (g.sat > 0).any() else np.nan,
        })
    R = pd.DataFrame(rows)
    C = R.groupby(CELL).agg(
        K=("K", "first"), hard_cep1=("hard_cep1", "mean"),
        grid_absdelta=("grid_absdelta", "mean"), grid_sd=("grid_sd", "mean"),
        grid_min=("grid_min", "mean"), hard_last=("hard_last", "mean"),
        soft_last=("soft_last", "mean"), lam_last=("lam_last", "mean"),
        ever_sat=("ever_sat", "sum"), first_sat=("first_sat", "mean")).reset_index()
    C["clip_raw"] = [float(clipraw.loc[(a, b, c)]) for a, b, c in
                     zip(C.dataset, C.model, C.cap)]
    C["edge"] = [float(edge.loc[(a, b, c)]) for a, b, c in
                 zip(C.dataset, C.model, C.cap)]
    C["overshoot_clip"] = C.clip_raw / C.K            # brief's variable
    C["overshoot_start"] = C.hard_cep1 / C.K          # what TraLO actually faced
    C["mobility"] = C.grid_absdelta / C.K

    # rho: 5 + 3.2759*e until first satisfaction, then frozen (train.py:126,424)
    rho_step = (100.0 - 5.0) / 29.0
    C["rho_end"] = np.where(C.first_sat.notna(),
                            5.0 + rho_step * C.first_sat.fillna(29),
                            5.0 + rho_step * 29)
    u = np.maximum(0.0, (C.soft_last - C.K) / C.K)
    C["u_end"] = u
    C["dLdS_sat"] = (1.0 / C.K) * 1.0 / (1 + u) ** 2
    C["dLdS_quad"] = (1.0 / C.K) * C.rho_end * 2 * u / (1 + u ** 2) ** 2
    C["dLdS"] = C.lam_last * (C.dLdS_sat + C.dLdS_quad)

    print("=" * 128)
    print("THE OVERSHOOT VARIABLE IS MEASURED ON A MODEL TraLO NEVER SEES.")
    print("clip_raw = 30-epoch plain-CE count. hard_cep1 = TraLO's own count at")
    print("its first logged constraint epoch (warm-up is 1 epoch here).")
    print("=" * 128)
    print(C[CELL + ["K", "clip_raw", "overshoot_clip", "hard_cep1",
                    "overshoot_start", "hard_last", "ever_sat", "edge"]]
          .sort_values(["dataset", "model", "cap"])
          .to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n  tissuemnist, the cell the question is about:")
    tm = C[C.dataset == "tissuemnist"].set_index(["model", "cap"])
    for cap in ["L30_G30", "L50_G50"]:
        a = tm.loc[("MobileNetV3", cap)]
        b = tm.loc[("RegNetY400MF", cap)]
        print("    %s  clipper overshoot MNV3 %.2f > RegNet %.2f, but TraLO's OWN"
              " start is MNV3 %.2f < RegNet %.2f  -> the ordering REVERSES"
              % (cap, a.overshoot_clip, b.overshoot_clip,
                 a.overshoot_start, b.overshoot_start))

    print("\n" + "=" * 128)
    print("CORRELATION with TraLO's edge (ccF1eq vs best dual), 12 cells")
    print("=" * 128)
    y = C.edge.to_numpy(float)
    for c in ["overshoot_clip", "overshoot_start", "mobility", "grid_sd",
              "ever_sat", "dLdS", "lam_last", "rho_end", "u_end"]:
        corr(C[c].to_numpy(float), y, c)
    print("  dataset-demeaned:")
    D = C.copy()
    for c in ["edge", "overshoot_clip", "overshoot_start", "mobility", "ever_sat"]:
        D[c] = D[c] - D.groupby("dataset")[c].transform("mean")
    for c in ["overshoot_clip", "overshoot_start", "mobility", "ever_sat"]:
        corr(D[c].to_numpy(float), D.edge.to_numpy(float), c + " (demeaned)")

    print("\n" + "=" * 128)
    print("COUNT MOBILITY on the shared every-5 grid (epochs 4,9,14,19,24,29),")
    print("so a denser TraLO log cannot inflate it. mobility = mean |d hard| / K.")
    print("=" * 128)
    print(C[CELL + ["K", "n_seeds" if "n_seeds" in C else "grid_absdelta",
                    "grid_sd", "mobility", "ever_sat"]]
          .sort_values(["model", "dataset", "cap"])
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  mobility by backbone (mean over its 6 cells):")
    for mo, g in C.groupby("model"):
        print("    %-14s mean|d hard| per 5 epochs = %6.1f   /K = %.2f   "
              "cells where the count ever reached the cap: %d/6"
              % (mo, g.grid_absdelta.mean(), g.mobility.mean(),
                 int((g.ever_sat > 0).sum())))

    print("\n" + "=" * 128)
    print("ANALYTIC PENALTY GRADIENT dL/dS at each cell's END-OF-RUN operating")
    print("point.  L = lam*[E/(E+K) + rho*(E/K)^2/(1+(E/K)^2)],  E=relu(S-K).")
    print("rho ratchets by (100-5)/29 per epoch until first satisfaction.")
    print("=" * 128)
    print(C[CELL + ["K", "soft_last", "u_end", "lam_last", "rho_end",
                    "dLdS_sat", "dLdS_quad", "dLdS"]]
          .sort_values(["model", "dataset", "cap"])
          .to_string(index=False, float_format=lambda x: "%.5f" % x))
    print("\n  mean dL/dS   MobileNetV3 %.5f   RegNetY400MF %.5f   ratio R/M %.2f"
          % (C[C.model == "MobileNetV3"].dLdS.mean(),
             C[C.model == "RegNetY400MF"].dLdS.mean(),
             C[C.model == "RegNetY400MF"].dLdS.mean() /
             C[C.model == "MobileNetV3"].dLdS.mean()))
    C.to_csv("paper/scripts/out_bb_final.csv", index=False)
    print("\nwrote paper/scripts/out_bb_final.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

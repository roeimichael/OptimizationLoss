"""Part 2: does the bb6 claim survive seed-level inspection and cell counting?"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 260)
CELL = ["dataset", "model", "cap"]

cell = pd.read_csv("paper/scripts/out_refute_bb6.csv")
T = pd.read_csv("paper/scripts/out_refute_bb6_traj.csv")

print("=" * 132)
print("A. THE SEED SPREAD THE 4-SEED MEAN HIDES")
print("=" * 132)
s1 = T[T.cep == 1].groupby(["dataset", "model", "cap", "seed"]).hard.first().reset_index()
s1 = s1[s1.cap == "L30_G30"]
piv = s1.pivot_table(index=["dataset", "seed"], columns="model", values="hard")
piv["MNV3_lower"] = piv.MobileNetV3 < piv.RegNetY400MF
print(piv.to_string())
print("\n  per-seed 'MNV3 starts LOWER than RegNet' (the claim's reversal):")
for ds, g in piv.groupby("dataset"):
    print("    %-12s %d of %d seeds   MNV3 range [%.0f,%.0f]  RegNet range [%.0f,%.0f]"
          % (ds, int(g.MNV3_lower.sum()), len(g),
             g.MobileNetV3.min(), g.MobileNetV3.max(),
             g.RegNetY400MF.min(), g.RegNetY400MF.max()))
print("\n  seed spread ratio max/min of hard_cep1 vs of clip_raw:")
for _, r in cell[cell.cap == "L30_G30"].iterrows():
    print("    %-12s %-13s  cep1 [%3.0f,%3.0f] = %4.1fx     clip [%3.0f,%3.0f]"
          % (r.dataset, r.model, r.cep1_min, r.cep1_max,
             r.cep1_max / max(r.cep1_min, 1e-9), 0, 0))

print("\n" + "=" * 132)
print("B. DOES TraLO 'NEVER SEE' THE CLIPPER-LEVEL COUNT?")
print("hard_max = the highest hard count TraLO's OWN model reaches on the pool")
print("=" * 132)
c = cell.copy()
c["max_over_clip"] = c.hard_max / c.clip_raw
print(c[CELL + ["clip_raw", "hard_max", "max_over_clip", "hard_last"]]
      .sort_values(CELL).to_string(index=False, float_format=lambda x: "%.2f" % x))
print("\n  cells where TraLO's own count REACHES >= 0.9 x the clipper's raw count:"
      " %d of %d" % (int((c.max_over_clip >= 0.9).sum()), len(c)))
print("  cells where it EXCEEDS the clipper's raw count: %d of %d"
      % (int((c.max_over_clip >= 1.0).sum()), len(c)))
print("  tissuemnist MobileNetV3: TraLO's own max %.1f vs clipper raw %.1f (%.1f%%)"
      % (c[(c.dataset == "tissuemnist") & (c.model == "MobileNetV3")].hard_max.max(),
         231.25,
         100 * c[(c.dataset == "tissuemnist") & (c.model == "MobileNetV3")].hard_max.max() / 231.25))

print("\n" + "=" * 132)
print("C. BACKBONE ORDERING: is it CONSTANT across datasets for both variables?")
print("(the edge's backbone ordering FLIPS on tissuemnist -- a constant-sign")
print(" variable cannot explain a sign flip in either direction)")
print("=" * 132)
w = c.pivot_table(index=["dataset", "cap"],
                  columns="model", values=["os_clip", "os_start", "os_max", "edge"])
rows = []
for (ds, cap), r in w.iterrows():
    rows.append(dict(
        dataset=ds, cap=cap,
        os_clip_M=r[("os_clip", "MobileNetV3")], os_clip_R=r[("os_clip", "RegNetY400MF")],
        os_start_M=r[("os_start", "MobileNetV3")], os_start_R=r[("os_start", "RegNetY400MF")],
        edge_M=r[("edge", "MobileNetV3")], edge_R=r[("edge", "RegNetY400MF")]))
W = pd.DataFrame(rows)
W["clip_says_M"] = W.os_clip_M > W.os_clip_R
W["start_says_M"] = W.os_start_M > W.os_start_R
W["edge_favors_M"] = W.edge_M > W.edge_R
print(W.to_string(index=False, float_format=lambda x: "%.3f" % x))
print("\n  os_clip  ranks MNV3 above RegNet in %d of 6 (dataset,cap) pairs"
      % int(W.clip_says_M.sum()))
print("  os_start ranks MNV3 above RegNet in %d of 6 (dataset,cap) pairs"
      % int(W.start_says_M.sum()))
print("  the EDGE  favours MNV3 in %d of 6 -- and only on tissuemnist"
      % int(W.edge_favors_M.sum()))
print("  -> both variables are CONSTANT in backbone sign across all 3 datasets;")
print("     the outcome is not. Neither can produce the tissuemnist flip.")
print("\n  agreement counted per (dataset,cap) pair:")
print("    os_clip  agrees with the edge ordering in %d of 6"
      % int((W.clip_says_M == W.edge_favors_M).sum()))
print("    os_start agrees with the edge ordering in %d of 6"
      % int((W.start_says_M == W.edge_favors_M).sum()))

print("\n" + "=" * 132)
print("D. MAGNITUDE ALIGNMENT (dataset level, n=3 -- reported, not believed)")
print("log backbone ratio of the overshoot vs the backbone gap in the edge")
print("=" * 132)
g = c.groupby(["dataset", "model"]).agg(edge=("edge", "mean"),
                                        os_clip=("os_clip", "mean"),
                                        os_start=("os_start", "mean"),
                                        clip_raw=("clip_raw", "first"),
                                        cep1=("hard_cep1", "first")).reset_index()
p = g.pivot(index="dataset", columns="model")
out = pd.DataFrame({
    "edge_gap_M_minus_R": p[("edge", "MobileNetV3")] - p[("edge", "RegNetY400MF")],
    "log_ratio_clip": np.log(p[("clip_raw", "MobileNetV3")] / p[("clip_raw", "RegNetY400MF")]),
    "log_ratio_cep1": np.log(p[("cep1", "MobileNetV3")] / p[("cep1", "RegNetY400MF")]),
})
print(out.to_string(float_format=lambda x: "%+.4f" % x))
for k in ["log_ratio_clip", "log_ratio_cep1"]:
    r = np.corrcoef(out[k], out.edge_gap_M_minus_R)[0, 1]
    print("  corr(%s, edge gap) over 3 datasets = %+.3f   [n=3, DF=1: no evidence]"
          % (k, r))

print("\n" + "=" * 132)
print("E. POOLED CORRELATIONS (n=12, violates atomic-cell rule) + CELL COUNTS")
print("=" * 132)


def corr(x, y, lab, spent=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    df = n - 2 - spent
    t = r * np.sqrt(max(df, 1) / max(1e-12, 1 - r * r))
    print("    %-32s n=%d df=%d r=%+0.3f r2=%.3f t=%+0.2f" % (lab, n, df, r, r * r, t))


for v in ["os_clip", "os_start", "os_max"]:
    corr(c[v], c.edge, v + " raw")
D = c.copy()
for v in ["edge", "os_clip", "os_start", "os_max"]:
    D[v] = D[v] - D.groupby("dataset")[v].transform("mean")
print("  dataset-demeaned (3 group means fitted -> 3 df spent):")
for v in ["os_clip", "os_start", "os_max"]:
    corr(D[v], D.edge, v + " demeaned", spent=3)
print("\n  variance decomposition of the demeaned x (which axis carries it?):")
for v in ["os_clip", "os_start"]:
    dd = D[[v]].copy()
    dd["cap"], dd["model"] = c.cap.values, c.model.values
    vc = dd.groupby("cap")[v].mean().var(ddof=0)
    vm = dd.groupby("model")[v].mean().var(ddof=0)
    print("    %-9s between-CAP var %.4f   between-BACKBONE var %.4f"
          "   -> cap axis carries %.0f%%"
          % (v, vc, vm, 100 * vc / (vc + vm)))

for v in ["os_clip", "os_start"]:
    w_ = c[c.edge > 0][v]
    l_ = c[c.edge <= 0][v]
    vs = np.sort(c[v].unique())
    best = max(((sign * (c[v] - thr) > 0) == (c.edge > 0)).mean()
               for thr in (vs[:-1] + vs[1:]) / 2 for sign in (1, -1))
    print("  %-9s WIN cells [%.2f,%.2f]  LOSS cells [%.2f,%.2f]  overlap=%s"
          "  best threshold rule %d/12"
          % (v, w_.min(), w_.max(), l_.min(), l_.max(),
             not (w_.min() > l_.max() or l_.min() > w_.max()), round(best * 12)))

"""Part 2: the overshoot that the OPTIMIZER actually faces.

bb3 defines overshoot from the post-hoc clipper's count -- a model trained 30
epochs of plain CE. The trained arms never see that model: they start from a
ONE-epoch warm-up. If overshoot is a mechanism, the load-bearing quantity is
the count at the first constraint epoch, not the count of a 30x-better-trained
network. This script builds that predictor from the raw training logs, with all
four schema traps handled explicitly, and re-runs the same correlations.

Traps handled:
  1. TraLO's log is SPARSE -> never len(df); select by the Epoch VALUE.
  2. "Epoch" (TraLO) vs "epoch" (duals).
  3. duals log no Hard_Class* at all -> only TraLO can supply the count, and it
     is a valid proxy only if all three arms share one warm-up cache.
  4. headers repeat mid-file -> pd.to_numeric(errors="coerce").dropna().
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

TR = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CELL = ["dataset", "model", "cap"]


def pearson(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def spear(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    return pearson(pd.Series(x[ok]).rank().to_numpy(float),
                   pd.Series(y[ok]).rank().to_numpy(float))[0]


def tp(r, n):
    if n < 3 or abs(r) >= 1:
        return float("nan"), float("nan")
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    from math import lgamma
    df = n - 2
    xx = df / (df + t * t)

    def betacf(a, b, x, itmax=300, eps=3e-12):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, dd = 1.0, 1.0 - qab * x / qap
        dd = 1.0 / (dd if abs(dd) > 1e-30 else 1e-30)
        h = dd
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            dd = 1.0 / max(1e-30, abs(1.0 + aa * dd)) * np.sign(1.0 + aa * dd)
            c = 1.0 + aa / c
            h *= dd * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            dd = 1.0 / max(1e-30, abs(1.0 + aa * dd)) * np.sign(1.0 + aa * dd)
            c = 1.0 + aa / c
            de = dd * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        bt = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                      + a * math.log(x) + b * math.log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1 - x) / b
    return t, betai(df / 2.0, 0.5, xx)


def rep(x, y, lab):
    r, n = pearson(np.asarray(x, float), np.asarray(y, float))
    if n == 0:
        print("    %-44s degenerate" % lab)
        return
    t, p = tp(r, n)
    print("    %-44s n=%2d r=%+0.3f r2=%.3f rho=%+0.3f t=%+0.2f p=%.3f"
          % (lab, n, r, r * r, spear(np.asarray(x, float), np.asarray(y, float)), t, p))


def start_counts():
    """Constrained-class hard count at TraLO's FIRST logged constraint epoch."""
    rows = []
    for cfg_path in sorted(glob.glob(TR + "/**/config.json", recursive=True)):
        d = os.path.dirname(cfg_path)
        cfg = json.load(open(cfg_path))
        if cfg.get("methodology") != "tralo":
            continue
        lg = os.path.join(d, "training_log.csv")
        if not os.path.exists(lg):
            continue
        df = pd.read_csv(lg)
        col = "Epoch" if "Epoch" in df.columns else ("epoch" if "epoch" in df.columns else None)
        if col is None:
            continue
        dc = cfg.get("dataset_config") or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        hcol = "Hard_Class%d" % cls
        if hcol not in df.columns:
            continue
        ep = pd.to_numeric(df[col], errors="coerce")
        hd = pd.to_numeric(df[hcol], errors="coerce")
        lim = pd.to_numeric(df.get("Limit_Class%d" % cls), errors="coerce")
        sat = pd.to_numeric(df.get("Global_Satisfied"), errors="coerce")
        ok = ep.notna() & hd.notna()
        ep, hd = ep[ok], hd[ok]
        lim = lim[ok] if lim is not None else None
        sat = sat[ok] if sat is not None else None
        if len(ep) == 0:
            continue
        i0 = ep.idxmin()
        hp = cfg.get("hyperparams") or {}
        rows.append({
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
            "n_log_rows": int(len(ep)),          # NOT an epoch count
            "first_epoch": float(ep.loc[i0]),
            "last_epoch": float(ep.max()),
            "start_hard": float(hd.loc[i0]),
            "K_log": float(lim.loc[i0]) if lim is not None else float("nan"),
            "n_sat_rows": int((sat > 0).sum()) if sat is not None else -1,
            "base_model_id": cfg.get("base_model_id"),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 112)
    print("A. warm-up cache shared by all three trained arms?")
    print("=" * 112)
    ids = []
    for cfg_path in sorted(glob.glob(TR + "/**/config.json", recursive=True)):
        cfg = json.load(open(cfg_path))
        hp = cfg.get("hyperparams") or {}
        ids.append({"dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
                    "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
                    "method": cfg.get("methodology"),
                    "base_model_id": cfg.get("base_model_id")})
    I = pd.DataFrame(ids)
    g = I.groupby(["dataset", "model", "seed"])["base_model_id"].nunique()
    print("  distinct base_model_id per (dataset,model,seed): max=%d  -> %s"
          % (g.max(), "SHARED" if g.max() == 1 else "NOT SHARED"))
    g2 = I.groupby(["dataset", "model"])["base_model_id"].nunique()
    print("  distinct base_model_id per (dataset,model) across seeds: %s"
          % g2.to_dict())

    print("\n" + "=" * 112)
    print("B. TraLO log schema audit (the trap that retracted a finding)")
    print("=" * 112)
    S = start_counts()
    print("  tralo runs with a usable log: %d of 48" % len(S))
    print("  n_log_rows distribution : %s" % S.n_log_rows.value_counts().to_dict())
    print("  first_epoch values      : %s" % sorted(S.first_epoch.unique()))
    print("  last_epoch  values      : %s" % sorted(S.last_epoch.unique()))
    print("  -> len(df) would claim %s epochs; Epoch.max() says %s"
          % (sorted(S.n_log_rows.unique())[:5], sorted(S.last_epoch.unique())[:5]))

    print("\n" + "=" * 112)
    print("C. the overshoot the optimizer actually faces vs the one bb3 used")
    print("=" * 112)
    tr = pd.read_csv("paper/scripts/out_refute_trained.csv")
    cl = pd.read_csv("paper/scripts/out_refute_clip.csv")
    piv = tr.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    pc = cl[cl.method.isin(["heuristic", "danits_lp"])].pivot_table(
        index=CELL + ["seed"], columns="method", values="ccF1eq")
    s = piv.copy()
    s["T_D"] = s["tralo"] - s[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    s["T_C"] = s["tralo"] - pc.max(axis=1).reindex(s.index)
    s = s.reset_index()
    S = S.merge(s[CELL + ["seed", "T_D", "T_C"]], on=CELL + ["seed"], how="left")
    kt = tr.groupby(CELL).agg(K=("K", "first"), n_pool=("n_pool", "first"),
                              n_true=("n_true", "first")).reset_index()
    nat = cl[cl.method == "heuristic"].groupby(CELL)["count_raw"].mean() \
            .rename("clip_raw").reset_index()
    c = S.groupby(CELL).agg(start_hard=("start_hard", "mean"),
                            start_sd=("start_hard", "std"),
                            T_D=("T_D", "mean"), T_C=("T_C", "mean")).reset_index()
    c = c.merge(kt, on=CELL).merge(nat, on=CELL)
    c["K_log_ok"] = True
    c["start_ratio"] = c.start_hard / c.K
    c["start_abs"] = c.start_hard - c.K
    c["clip_ratio"] = c.clip_raw / c.K
    c = c.sort_values(["dataset", "cap", "model"])
    print(c[["dataset", "model", "cap", "K", "start_hard", "start_sd", "clip_raw",
             "start_ratio", "clip_ratio", "T_D"]]
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  correlation between the two overshoot definitions: r=%+0.3f"
          % pearson(c.start_ratio.to_numpy(float), c.clip_ratio.to_numpy(float))[0])

    print("\n  target = T_D (ccF1eq), 12 cells")
    rep(c.start_ratio, c.T_D, "START overshoot ratio (warm-up-1 model)")
    rep(c.start_abs, c.T_D, "START overshoot abs")
    rep(np.log(c.start_ratio), c.T_D, "log START ratio")
    rep(1 - c.K / c.start_hard, c.T_D, "START cut_frac")
    rep(c.start_hard / c.n_true, c.T_D, "START count / n_true")
    rep(c.start_hard / c.n_pool, c.T_D, "START rate / pool")
    rep(c.clip_ratio, c.T_D, "CLIPPER overshoot ratio (bb3's version)")
    print("\n  dataset-demeaned")
    for nm, v in [("START ratio", c.start_ratio), ("START abs", c.start_abs),
                  ("CLIP ratio", c.clip_ratio), ("CLIP abs", c.clip_raw - c.K)]:
        xv = v - v.groupby(c.dataset).transform("mean")
        yv = c.T_D - c.groupby("dataset")["T_D"].transform("mean")
        rep(xv, yv, nm + " (-dataset)")

    print("\n  SEED-LEVEL n=48, per-seed starting count")
    S["start_ratio"] = S.start_hard / S.merge(kt, on=CELL, how="left").K.to_numpy()
    rep(S.start_ratio, S.T_D, "START ratio per seed vs T_D")
    xv = S.start_ratio - S.groupby(CELL)["start_ratio"].transform("mean")
    yv = S.T_D - S.groupby(CELL)["T_D"].transform("mean")
    rep(xv, yv, "START ratio per seed (within cell)")

    print("\n" + "=" * 112)
    print("D. cell COUNTING with the start-overshoot")
    print("=" * 112)
    win = (c.T_D > 0).to_numpy()
    for nm, v in [("start_ratio", c.start_ratio), ("clip_ratio", c.clip_ratio)]:
        x = v.to_numpy(float)
        print("  %-12s win-cells mean %.3f  loss-cells mean %.3f   "
              "ranks(win)=%s" % (nm, x[win].mean(), x[~win].mean(),
                                 sorted(pd.Series(x).rank()[win].astype(int).tolist())))

    print("\n" + "=" * 112)
    print("E. how many of the 50 encodings would be expected to clear r2>=0.15")
    print("=" * 112)
    R = pd.read_csv("paper/scripts/out_refute_overshoot.csv")
    n12 = R[R.n == 12]
    r2crit = 0.15
    # r2 for which p=0.05 at n=12
    from math import sqrt
    tcrit = 2.228
    r_at_05 = tcrit / sqrt(tcrit ** 2 + 10)
    print("  at n=12, p=0.05 needs |r|>=%.3f i.e. r2>=%.3f -- the 0.15 bar the"
          % (r_at_05, r_at_05 ** 2))
    print("  claim uses is LOWER than the significance threshold, so clearing it")
    print("  is not evidence of prediction.")
    print("  encodings tested at n=12: %d ; clearing r2>=0.15: %d ; p<0.05: %d"
          % (len(n12), int((n12.r2 >= r2crit).sum()), int((n12.p < 0.05).sum())))
    print(n12[n12.r2 >= r2crit][["encoding", "r", "r2", "p"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())

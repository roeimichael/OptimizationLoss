"""INDEPENDENT re-derivation of the bb6 'overshoot measured on the wrong model'
claim. Rebuilds hard_cep1 and clip_raw from RAW files only (config.json,
training_log.csv, final_predictions_raw.csv). Does not read out_bb_traj_full.csv
or out_bb_final.csv.

Schema traps handled explicitly and REPORTED:
  T1 sparse TraLO log      -> never use len(df); use Epoch max/min
  T2 column case           -> TraLO 'Epoch'
  T3 duals log no counts   -> only tralo used here
  T4 repeated headers      -> to_numeric(errors=coerce) + dropna
  T5 constrained class     -> config['dataset_config']['constrained_class'][0]
plus two traps bb5/bb6 do NOT check:
  T6 append-mode log       -> duplicate Epoch values from a re-run in place
  T7 pool identity         -> sum(Hard_Class*) must equal n_pool of the preds
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)

CELL = ["dataset", "model", "cap"]
NOCESKIP = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CLIPROOT = "results/headroom/headroom_b30"


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def cfg_of(d):
    return json.load(open(os.path.join(d, "config.json")))


def cls_of(c):
    v = (c.get("dataset_config") or {}).get("constrained_class")
    return int(v[0] if isinstance(v, (list, tuple)) else v)


def scan(root):
    out = []
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cp)
        c = json.load(open(cp))
        hp = c.get("hyperparams") or {}
        out.append(dict(dir=d, dataset=c.get("dataset_mode"),
                        model=c.get("model_name"), cap=c.get("constraint_tag"),
                        method=c.get("methodology"), seed=hp.get("seed"),
                        warmup=hp.get("warmup_epochs"),
                        cepochs=hp.get("constraint_epochs"),
                        lr=hp.get("lr"), lrc=hp.get("lr_constraint"),
                        ce_skip=hp.get("enable_ce_skip"), cls=cls_of(c)))
    return pd.DataFrame(out)


def main():
    print("=" * 132)
    print("STEP 0  campaign inventory + regime check (raw configs)")
    print("=" * 132)
    A = scan(NOCESKIP)
    B = scan(CLIPROOT)
    print("  noceskip rows=%d  methods=%s" % (len(A), sorted(A.method.unique())))
    print("  warmup values in noceskip : %s" % sorted(A.warmup.dropna().unique()))
    print("  cepochs values            : %s" % sorted(A.cepochs.dropna().unique()))
    print("  ce_skip values            : %s" % sorted(map(str, A.ce_skip.unique())))
    print("  lr / lr_constraint        : %s / %s"
          % (sorted(A.lr.dropna().unique()), sorted(A.lrc.dropna().unique())))
    print("  constrained class by ds   : %s"
          % A.groupby("dataset").cls.unique().to_dict())
    print("  clip root rows=%d methods=%s  warmup=%s cepochs=%s"
          % (len(B), sorted(B.method.unique()),
             sorted(B.warmup.dropna().unique()),
             sorted(B.cepochs.dropna().unique())))

    # ---------------- STEP 1: raw TraLO logs, trap-checked ----------------
    print("\n" + "=" * 132)
    print("STEP 1  raw TraLO training_log.csv -> per-seed hard count at EVERY")
    print("logged epoch. Trap audit printed per run-group.")
    print("=" * 132)
    rows, traj, bad = [], [], []
    for r in A[A.method == "tralo"].itertuples():
        p = os.path.join(r.dir, "training_log.csv")
        if not os.path.exists(p):
            bad.append((r.dir, "no log"))
            continue
        lg = pd.read_csv(p)
        hc, sc = "Hard_Class%d" % r.cls, "Soft_Class%d" % r.cls
        if hc not in lg.columns:
            bad.append((r.dir, "no %s" % hc))
            continue
        E = num(lg, "Epoch")
        H = num(lg, hc)
        t = pd.DataFrame({"E": E, "hard": H, "soft": num(lg, sc),
                          "lam": num(lg, "Lambda_Global"),
                          "sat": num(lg, "Global_Satisfied")}).dropna(subset=["E", "hard"])
        # T7: does the row's full hard histogram sum to the pool size?
        hcols = [c for c in lg.columns if c.startswith("Hard_Class")]
        tot = sum(num(lg, c) for c in hcols)
        pool_from_log = pd.Series(tot).dropna().unique()
        # T6: duplicated epochs?
        dup = int(t.E.duplicated().sum())
        t["cep"] = t.E - 1                       # 0-based loop index
        t = t.sort_values("cep")
        first = t[t.cep >= 1]
        if not len(first):
            bad.append((r.dir, "no cep>=1"))
            continue
        rows.append(dict(dataset=r.dataset, model=r.model, cap=r.cap, seed=r.seed,
                         warmup=r.warmup, n_rows=len(lg), n_num=len(t),
                         E_min=float(t.E.min()), E_max=float(t.E.max()),
                         dup_E=dup,
                         pool=float(pool_from_log[0]) if len(pool_from_log) == 1 else -1,
                         pool_uniq=len(pool_from_log),
                         first_cep=float(first.cep.iloc[0]),
                         hard_first=float(first.hard.iloc[0]),
                         hard_at_cep1=float(t.hard[t.cep == 1].iloc[0])
                         if (t.cep == 1).any() else np.nan,
                         hard_max=float(t.hard.max()),
                         hard_last=float(t.hard.iloc[-1])))
        g = t.copy()
        g["dataset"], g["model"], g["cap"], g["seed"] = r.dataset, r.model, r.cap, r.seed
        traj.append(g)
    R = pd.DataFrame(rows)
    T = pd.concat(traj, ignore_index=True)
    print("  tralo runs parsed : %d   failures: %s" % (len(R), bad[:5]))
    print("  T1 len(df) vs Epoch.max : mean n_numeric_rows=%.1f  mean Epoch.max=%.1f"
          " -> len(df) would understate epochs by %.1fx"
          % (R.n_num.mean(), R.E_max.mean(), R.E_max.mean() / R.n_num.mean()))
    print("  T6 runs with DUPLICATE Epoch rows (append contamination): %d"
          % int((R.dup_E > 0).sum()))
    print("  T7 runs where sum(Hard_Class*) is not a single pool size : %d"
          % int((R.pool_uniq != 1).sum()))
    print("  T7 pool size per dataset from the LOG: %s"
          % R.groupby("dataset").pool.unique().to_dict())
    print("  first logged cep per run (should be 1 when warmup=1): %s"
          % sorted(R.first_cep.unique()))
    print("  runs missing a cep==1 row: %d" % int(R.hard_at_cep1.isna().sum()))

    # ---------------- STEP 2: clip_raw from raw predictions ---------------
    print("\n" + "=" * 132)
    print("STEP 2  clip_raw re-derived from the post-hoc arm's")
    print("final_predictions_raw.csv (heuristic, warmup=30, cepochs=0)")
    print("=" * 132)
    crows = []
    for r in B[B.method == "heuristic"].itertuples():
        p = os.path.join(r.dir, "final_predictions_raw.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        crows.append(dict(dataset=r.dataset, model=r.model, cap=r.cap, seed=r.seed,
                          n_pool=len(df),
                          count_raw=int((df.Predicted_Label == r.cls).sum())))
    C0 = pd.DataFrame(crows)
    clip = C0.groupby(CELL).agg(clip_raw=("count_raw", "mean"),
                                clip_min=("count_raw", "min"),
                                clip_max=("count_raw", "max"),
                                n_pool=("n_pool", "first"),
                                nseed=("seed", "size")).reset_index()
    print(clip.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ---------------- STEP 3: cell table ---------------------------------
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    fb = fb[fb.campaign == "lrc0.0001_noceskip"]
    K = fb.groupby(CELL)["K"].first()
    edge = fb[fb.method == "tralo"].set_index(CELL)["d_vs_bestdual"]

    cell = R.groupby(CELL).agg(hard_cep1=("hard_at_cep1", "mean"),
                               cep1_min=("hard_at_cep1", "min"),
                               cep1_max=("hard_at_cep1", "max"),
                               hard_max=("hard_max", "mean"),
                               hard_last=("hard_last", "mean"),
                               nseed=("seed", "size")).reset_index()
    cell["K"] = [float(K.loc[tuple(x)]) for x in cell[CELL].values]
    cell["edge"] = [float(edge.loc[tuple(x)]) for x in cell[CELL].values]
    cell = cell.merge(clip[CELL + ["clip_raw", "n_pool"]], on=CELL)
    cell["os_clip"] = cell.clip_raw / cell.K
    cell["os_start"] = cell.hard_cep1 / cell.K
    cell["os_max"] = cell.hard_max / cell.K
    cell["win"] = np.where(cell.edge > 0, "W", "L")

    print("\n" + "=" * 132)
    print("STEP 3  the 12 atomic cells, re-derived")
    print("=" * 132)
    print(cell[CELL + ["K", "n_pool", "nseed", "clip_raw", "os_clip",
                       "hard_cep1", "cep1_min", "cep1_max", "os_start",
                       "hard_max", "os_max", "hard_last", "edge", "win"]]
          .sort_values(["dataset", "model", "cap"])
          .to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n  IS hard_cep1 IDENTICAL ACROSS CAPS within (dataset,backbone)?")
    for (ds, mo), g in cell.groupby(["dataset", "model"]):
        v = sorted(g.hard_cep1.unique())
        print("    %-12s %-13s hard_cep1 = %s   identical=%s"
              % (ds, mo, v, len(v) == 1))
    print("\n  ... and PER SEED (the mean can hide disagreement):")
    ps = R.pivot_table(index=["dataset", "model", "seed"], columns="cap",
                       values="hard_at_cep1")
    ps["same"] = np.isclose(ps.iloc[:, 0], ps.iloc[:, 1])
    print("    per-seed L30 == L50 at cep1 : %d of %d"
          % (int(ps["same"].sum()), len(ps)))

    # ---------------- STEP 4: does one CE epoch move the count? ----------
    print("\n" + "=" * 132)
    print("STEP 4  is cep=1 'effectively the warm-up state'?  The loop order is")
    print("  for epoch in range(warmup,total): [full CE pass] -> [count pass] ->")
    print("  [one clipped constraint step] -> [log].  So the cep=1 COUNT is taken")
    print("  AFTER an extra full CE epoch and BEFORE any constraint step ever.")
    print("=" * 132)
    print("  count movement from cep1 to the next logged epoch (pure CE + <=3 steps):")
    mv = []
    for (ds, mo, cap, sd), g in T.groupby(CELL + ["seed"]):
        g = g.sort_values("cep")
        g = g[g.cep >= 1]
        if len(g) < 2:
            continue
        mv.append(dict(dataset=ds, model=mo, cap=cap, seed=sd,
                       cep1=float(g.hard.iloc[0]), nxt_cep=float(g.cep.iloc[1]),
                       nxt=float(g.hard.iloc[1])))
    MV = pd.DataFrame(mv)
    MV["dabs"] = (MV.nxt - MV.cep1).abs()
    print(MV.groupby(["dataset", "model"]).agg(
        cep1=("cep1", "mean"), next_cep=("nxt_cep", "mean"),
        next_hard=("nxt", "mean"), mean_abs_move=("dabs", "mean")).to_string(
        float_format=lambda x: "%.1f" % x))
    print("\n  cep1 -> fully-trained (clip_raw) for the SAME dataset/backbone:")
    z = cell.groupby(["dataset", "model"]).agg(
        cep1=("hard_cep1", "first"), clip=("clip_raw", "first")).reset_index()
    z["ratio"] = z["clip"].astype(float) / z["cep1"].astype(float)
    print(z.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ---------------- STEP 5: correlations, and CELL COUNTING -------------
    print("\n" + "=" * 132)
    print("STEP 5  correlation over the 12 pooled cells (violates the atomic-cell")
    print("rule; reported only to check the claim's arithmetic), then CELL COUNTS")
    print("=" * 132)

    def corr(x, y, lab, n_demean_groups=0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = ~(np.isnan(x) | np.isnan(y))
        n = int(ok.sum())
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        df_ = n - 2 - n_demean_groups
        t = r * np.sqrt(max(df_, 1) / max(1e-12, 1 - r * r))
        print("    %-34s n=%2d df=%2d r=%+0.3f r2=%.3f t=%+0.2f"
              % (lab, n, df_, r, r * r, t))
        return r

    y = cell.edge.to_numpy(float)
    for c in ["os_clip", "os_start", "os_max"]:
        corr(cell[c], y, c + "  (raw, pooled)")
    D = cell.copy()
    for c in ["edge", "os_clip", "os_start", "os_max"]:
        D[c] = D[c] - D.groupby("dataset")[c].transform("mean")
    print("  dataset-demeaned (3 datasets -> 3 df spent):")
    for c in ["os_clip", "os_start", "os_max"]:
        corr(D[c], D.edge, c + "  (demeaned)", n_demean_groups=3)

    print("\n  CELL COUNTING -- does either variable SEPARATE wins from losses?")
    for c in ["os_clip", "os_start", "os_max"]:
        w = cell[cell.win == "W"][c]
        l = cell[cell.win == "L"][c]
        overlap = not (w.min() > l.max() or l.min() > w.max())
        print("    %-9s  WIN cells range [%.2f, %.2f] (n=%d)   LOSS cells range"
              " [%.2f, %.2f] (n=%d)   ranges overlap=%s"
              % (c, w.min(), w.max(), len(w), l.min(), l.max(), len(l), overlap))
        # threshold rule: best achievable accuracy of a single split on this var
        vs = np.sort(cell[c].unique())
        best = 0
        for thr in (vs[:-1] + vs[1:]) / 2:
            for sign in (1, -1):
                acc = ((sign * (cell[c] - thr) > 0) == (cell.win == "W")).mean()
                best = max(best, acc)
        print("        best single-threshold rule classifies %.0f%% of the 12"
              " cells (%d/12)" % (100 * best, round(best * 12)))

    print("\n  the tissuemnist contrast the claim is built on:")
    tm = cell[cell.dataset == "tissuemnist"].set_index(["model", "cap"])
    for cap in ["L30_G30", "L50_G50"]:
        a, b = tm.loc[("MobileNetV3", cap)], tm.loc[("RegNetY400MF", cap)]
        print("    %s  os_clip  MNV3 %.2f vs RegNet %.2f  (MNV3 higher: %s)"
              % (cap, a.os_clip, b.os_clip, a.os_clip > b.os_clip))
        print("           os_start MNV3 %.2f vs RegNet %.2f  (MNV3 higher: %s)"
              % (a.os_start, b.os_start, a.os_start > b.os_start))
        print("           os_max   MNV3 %.2f vs RegNet %.2f  (MNV3 higher: %s)"
              % (a.os_max, b.os_max, a.os_max > b.os_max))

    cell.to_csv("paper/scripts/out_refute_bb6.csv", index=False)
    T.to_csv("paper/scripts/out_refute_bb6_traj.csv", index=False)
    print("\nwrote paper/scripts/out_refute_bb6.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""INDEPENDENT re-derivation of the 'derm == oct in TraLO's trajectory' claim.

Rebuilt from raw files, not from traj_ds.py.  Adds the two things the claim's
evidence cannot see:

  (A) EARLY STOP.  src/methodologies/tralo/train.py line 426:
          if stable_count >= stable_count_threshold: break
      comes BEFORE the logging block at line 430.  So (i) a converged run is
      TRUNCATED, and (ii) the epoch that triggers convergence is NEVER logged.
      "TraLO logs every satisfied epoch" is therefore false for converged runs,
      and "final X" is read at different wall-clock epochs across runs.

  (B) A FIXED EPOCH GRID.  Epochs 5,10,15,20,25,30 are logged unconditionally
      (the (epoch+1)%5==0 arm).  Every other logged row exists only because the
      run was satisfied there.  So log density is a FUNCTION of satisfaction,
      and any per-logged-row statistic (oscillation, mean |delta|, frac above K)
      is confounded with it.  Recompute those on the fixed grid only.

usage: python paper/scripts/refute_traj.py --root results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def cls_of(cfg):
    c = (cfg.get("dataset_config") or {}).get("constrained_class")
    return int(c[0] if isinstance(c, (list, tuple)) else c)


def ev_of(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"].astype(str), t["Value"].astype(str)))


def fnum(ev, k):
    try:
        return float(ev.get(k, ""))
    except Exception:
        return np.nan


def raw_count(d, cls):
    p = os.path.join(d, "final_predictions_raw.csv")
    if not os.path.exists(p):
        return np.nan
    return int((pd.read_csv(p)["Predicted_Label"].to_numpy(int) == cls).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="paper/scripts/out_refute_traj.csv")
    ap.add_argument("--trajout", default="paper/scripts/out_refute_traj_epochs.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 90)

    rows, traj = [], []
    for cfgp in sorted(glob.glob(args.root + "/**/config.json", recursive=True)):
        d = os.path.dirname(cfgp)
        cfg = json.load(open(cfgp))
        if cfg.get("methodology") != "tralo":
            continue
        lgp = os.path.join(d, "training_log.csv")
        if not os.path.exists(lgp):
            continue
        hp = cfg.get("hyperparams") or {}
        c = cls_of(cfg)
        lg = pd.read_csv(lgp)
        e = num(lg, "Epoch")
        lg = lg[e.notna()].copy()
        lg["ep"] = e[e.notna()].astype(float)
        lg = lg.sort_values("ep")
        h = num(lg, "Hard_Class%d" % c).to_numpy(float)
        s = num(lg, "Soft_Class%d" % c).to_numpy(float)
        K = float(num(lg, "Limit_Class%d" % c).iloc[0])
        gs = num(lg, "Global_Satisfied").to_numpy(float)
        ls = num(lg, "Local_Satisfied").to_numpy(float)
        sat = (gs == 1) & (ls == 1)
        lam = num(lg, "Lambda_Global").to_numpy(float)
        ce = num(lg, "L_CE").to_numpy(float)
        ep = lg["ep"].to_numpy(float)
        ev = ev_of(d)

        budget = int(cfg.get("epoch_budget", 30))
        warm = int(hp.get("warmup_epochs", 1))
        cep = int(hp.get("constraint_epochs", 29))
        last = float(ep.max())

        # sign changes exactly as traj_ds.py does it (drop zeros)
        sgn = np.sign(h - K)
        nz = sgn[sgn != 0]
        cross = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0

        # --- fixed grid: epochs logged unconditionally, comparable across runs
        grid = np.array([5, 10, 15, 20, 25, 30], dtype=float)
        gm = np.isin(ep, grid)
        hg, eg = h[gm], ep[gm]
        if len(hg) > 1:
            swing_g = float(np.abs(np.diff(hg)).mean() / K)
        else:
            swing_g = np.nan
        sg = np.sign(hg - K)
        nzg = sg[sg != 0]
        cross_g = int((np.diff(nzg) != 0).sum()) if len(nzg) > 1 else 0
        above_g = float((hg > K).mean()) if len(hg) else np.nan
        # all-logged-row swing (what collapse.py does)
        swing_all = float(np.abs(np.diff(h)).mean() / K) if len(h) > 1 else np.nan
        above_all = float((h > K).mean())

        rows.append(dict(
            dataset=cfg.get("dataset_mode"), model=cfg.get("model_name"),
            cap=cfg.get("constraint_tag"), seed=hp.get("seed"), dir=d,
            K=K, budget=budget, warm=warm, cep=cep,
            last_ep=last, n_rows=len(lg),
            truncated=int(last < budget),
            n_sat=int(sat.sum()), n_gsat=int((gs == 1).sum()),
            ever_sat=int(sat.any()),
            first_sat=float(ep[sat][0]) if sat.any() else np.nan,
            last_sat=float(ep[sat][-1]) if sat.any() else np.nan,
            r_last=float(h[-1] / K), r_min=float(h.min() / K),
            r_max=float(h.max() / K), r_first=float(h[0] / K),
            cnt_last=float(h[-1]), cnt_min=float(h.min()),
            cross=cross, cross_grid=cross_g,
            swing_all=swing_all, swing_grid=swing_g,
            above_all=above_all, above_grid=above_g,
            n_grid=int(gm.sum()),
            lam_last=float(lam[-1]), lam_first=float(lam[0]),
            ce_last=float(ce[-1]), ce_first=float(ce[0]),
            soft_last=float(s[-1]),
            raw_cnt=raw_count(d, c),
            sat_ep_ev=fnum(ev, "Satisfaction Epoch"),
            best_sat_ep=fnum(ev, "Best Satisfied Epoch"),
            min_exc_ep=fnum(ev, "Min Excess Epoch"),
            min_exc=fnum(ev, "Min Total Excess"),
            restored_ep=fnum(ev, "Restored From Epoch"),
            restore_kind=ev.get("Restore Kind", "") or "none",
            raw_exc=fnum(ev, "Raw Total Excess"),
            flips=fnum(ev, "Flips Required"),
            raw_all_sat=fnum(ev, "Raw All Satisfied"),
        ))
        for i in range(len(lg)):
            traj.append(dict(dataset=cfg.get("dataset_mode"),
                             model=cfg.get("model_name"), cap=cfg.get("constraint_tag"),
                             seed=hp.get("seed"), ep=ep[i], hard=h[i], soft=s[i], K=K,
                             gsat=gs[i], lsat=ls[i], lam=lam[i], ce=ce[i],
                             ongrid=int(ep[i] in set(grid.tolist()))))

    t = pd.DataFrame(rows)
    t["unspent"] = t["K"] - t["raw_cnt"]
    t.to_csv(args.out, index=False)
    pd.DataFrame(traj).to_csv(args.trajout, index=False)
    print("n tralo runs = %d   datasets=%s" % (len(t), sorted(t.dataset.unique())))

    order = ["MobileNetV3", "RegNetY400MF"]
    capo = ["L30_G30", "L50_G50"]
    t["mo_i"] = t.model.map({m: i for i, m in enumerate(order)})
    t["cap_i"] = t.cap.map({c: i for i, c in enumerate(capo)})

    agg = t.groupby(CELL + ["mo_i", "cap_i"]).agg(
        n=("seed", "size"), K=("K", "mean"),
        last_ep=("last_ep", "mean"), min_last_ep=("last_ep", "min"),
        n_trunc=("truncated", "sum"),
        n_sat=("n_sat", "mean"), ever=("ever_sat", "sum"),
        first_sat=("first_sat", "mean"),
        r_last=("r_last", "mean"), r_min=("r_min", "mean"),
        cross=("cross", "mean"), cross_g=("cross_grid", "mean"),
        swing_all=("swing_all", "mean"), swing_g=("swing_grid", "mean"),
        above_all=("above_all", "mean"), above_g=("above_grid", "mean"),
        lam_last=("lam_last", "mean"), ce_last=("ce_last", "mean"),
        raw_cnt=("raw_cnt", "mean"), unspent=("unspent", "mean"),
        n_restore_sat=("restore_kind", lambda s: int((s == "fully_satisfied").sum())),
        n_restore_min=("restore_kind", lambda s: int((s == "min_excess").sum())),
        n_restore_none=("restore_kind", lambda s: int((s == "none").sum())),
        restored_ep=("restored_ep", "mean"),
        flips=("flips", "mean"), raw_exc=("raw_exc", "mean"),
    ).reset_index().sort_values(["dataset", "mo_i", "cap_i"])

    print("\n" + "=" * 150)
    print("PER-CELL (4 seeds each).  cell order per dataset = MNV3-L30, MNV3-L50, RegNet-L30, RegNet-L50")
    print("=" * 150)
    print(agg.drop(columns=["mo_i", "cap_i"]).to_string(
        index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 150)
    print("CLAIMED-VALUE CHECK  (derm 4 cells | oct 4 cells, in the claim's stated order)")
    print("=" * 150)
    fields = [("n_sat", "fully-satisfied epochs"), ("first_sat", "first satisfied epoch"),
              ("r_last", "final count/K"), ("r_min", "min count/K"),
              ("cross", "sign changes (all logged rows)"),
              ("lam_last", "final Lambda_Global"), ("ce_last", "final L_CE"),
              ("unspent", "K - final raw count"), ("last_ep", "LAST LOGGED EPOCH")]
    for f, lbl in fields:
        line = "  %-32s" % lbl
        for ds in ["dermmnist", "octmnist"]:
            g = agg[agg.dataset == ds]
            line += "  %-9s " % ds[:4] + "/".join(
                ("%.3f" % v if np.isfinite(v) else "-") for v in g[f])
        print(line)
    for ds in ["dermmnist", "octmnist", "tissuemnist"]:
        g = agg[agg.dataset == ds]
        print("  %-12s seeds ever satisfied = %d/16   runs truncated by convergence = %d/16"
              % (ds, g["ever"].sum(), g["n_trunc"].sum()))

    print("\n" + "=" * 150)
    print("COUNT THE CELLS, do not average them:  derm-vs-oct paired by (backbone,cap)")
    print("=" * 150)
    dd = agg[agg.dataset == "dermmnist"].set_index(["mo_i", "cap_i"])
    oo = agg[agg.dataset == "octmnist"].set_index(["mo_i", "cap_i"])
    for f, lbl in fields + [("above_g", "frac grid epochs above K"),
                            ("swing_g", "swing on fixed grid"),
                            ("above_all", "frac ALL logged rows above K"),
                            ("swing_all", "swing on all logged rows"),
                            ("n_restore_sat", "#seeds restored from a SAT ckpt"),
                            ("raw_cnt", "raw count of restored model"),
                            ("flips", "post-hoc flips")]:
        d_ = dd[f].to_numpy(float)
        o_ = oo[f].to_numpy(float)
        delta = d_ - o_
        ok = np.isfinite(delta)
        npos = int((delta[ok] > 0).sum())
        nneg = int((delta[ok] < 0).sum())
        rng_d = "[%.3f, %.3f]" % (np.nanmin(d_), np.nanmax(d_))
        rng_o = "[%.3f, %.3f]" % (np.nanmin(o_), np.nanmax(o_))
        sep = "SEPARATED" if (np.nanmax(d_) < np.nanmin(o_) or np.nanmax(o_) < np.nanmin(d_)) else ""
        print("  %-32s derm>oct in %d/%d cells, derm<oct in %d/%d   derm %s  oct %s  %s"
              % (lbl, npos, int(ok.sum()), nneg, int(ok.sum()), rng_d, rng_o, sep))

    print("\n" + "=" * 150)
    print("PER-SEED trajectories (E<epoch>:<count>, * = fully satisfied, ^ = on the unconditional /5 grid)")
    print("=" * 150)
    J = pd.DataFrame(traj)
    J["mo_i"] = J.model.map({m: i for i, m in enumerate(order)})
    J["cap_i"] = J.cap.map({c: i for i, c in enumerate(capo)})
    for ds in ["dermmnist", "octmnist"]:
        for (mo, cap), g in J[J.dataset == ds].sort_values(["mo_i", "cap_i"]).groupby(
                ["model", "cap"], sort=False):
            print("\n--- %s %s %s   K=%d" % (ds, mo, cap, g.K.iloc[0]))
            for sd, gs in g.groupby("seed"):
                gs = gs.sort_values("ep")
                lastl = gs.ep.max()
                print("  seed %s (last logged E%d%s): " % (
                    sd, lastl, "  <-- TRUNCATED" if lastl < 30 else "") + " ".join(
                    "E%d:%d%s" % (e, hh, "*" if (a == 1 and b == 1) else "")
                    for e, hh, a, b in zip(gs.ep, gs.hard, gs.gsat, gs.lsat)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

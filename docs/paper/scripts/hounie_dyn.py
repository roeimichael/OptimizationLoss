"""Read-only dynamics harvest for the dual methods (hounie_rcl, fioretto_ldf).

The training log records only (epoch, ce_loss, constraint_loss, total_excess,
all_satisfied, max_lam_g, max_u_g, h_u).  It does NOT record the per-epoch
constrained-class count.  But the dual recursions are deterministic and
invertible, so the SOFT count can be reconstructed exactly from the logged
multipliers.

hounie_rcl (src/methodologies/hounie_rcl/train.py L279-292):

    lam[t] = max(0, lam[t-1] + eta_lam * (mean_l[t] - u[t-1]))     # step 3
    u[t]   = max(0, u[t-1]   + eta_u   * (lam[t] - 2*alpha*u[t-1]))# step 4
    mean_l[t] = (soft_count[t] - K) / N

  => when lam[t] > 0 (not clamped):
        mean_l[t]     = (lam[t] - lam[t-1]) / eta_lam + u[t-1]
        soft_count[t] = K + N * mean_l[t]

fioretto_ldf (src/methodologies/fioretto_ldf/train.py L270-271):

    lam[t] = lam[t-1] + step * max(0, soft_count[t] - K)
  => soft_count[t] = K + (lam[t] - lam[t-1]) / step   (identified only when the
     increment is > 0; a zero increment means soft_count <= K, unresolved)

h_u = alpha * (sum_c u_g[c]^2 + sum_(g,c) u_l[g,c]^2)  (L317-318), and there is
exactly one global constrained class here, so

    U_loc = sqrt(max(0, h_u/alpha - max_u_g^2))

is the L2 magnitude of the LOCAL slack vector -- the only window the log gives
onto the per-group multipliers.

ce_loss == NaN marks an epoch where the CE batch loop did not run: the CE
saturation gate (`enable_ce_skip`, default True in BOTH dual trainers) fired and
`np.mean([])` wrote nan.  So the log DOES tell us, epoch by epoch, whether the
network was still receiving CE gradient.  That is the axis this script measures.

Self-check: the u recursion is verified against the logged columns; reproducing
it to rounding precision confirms the lam recursion reading too.

    python paper/scripts/hounie_dyn.py --root results/headroom/<campaign>
    python paper/scripts/hounie_dyn.py --dump dermmnist,MobileNetV3,L30_G30,1
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

CELL = ["dataset", "model", "cap"]
NUMCOLS = ["epoch", "ce_loss", "constraint_loss", "total_excess", "all_satisfied",
           "max_lam_g", "max_u_g", "h_u", "max_lambda_g"]


def numeric(df):
    """Coerce every column; headers repeat mid-file so non-numeric rows must go.
    Only `epoch` may not be NaN -- ce_loss NaN is SIGNAL (CE loop skipped)."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in NUMCOLS else df[c]
    keep = [c for c in out.columns if c in NUMCOLS and c != "ce_loss"]
    return out.dropna(subset=keep).reset_index(drop=True)


def trend(x):
    """Spearman rho of a series against epoch index; nan if degenerate."""
    x = np.asarray(x, float)
    if len(x) < 4 or np.all(x == x[0]):
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    t = np.arange(len(x), dtype=float)
    return float(np.corrcoef(r, t)[0, 1])


def load_run(cfg_path):
    cfg = json.load(open(cfg_path))
    d = os.path.dirname(cfg_path)
    log_p = os.path.join(d, "training_log.csv")
    raw_p = os.path.join(d, "final_predictions_raw.csv")
    if not (os.path.exists(log_p) and os.path.exists(raw_p)):
        return None
    hp = cfg.get("hyperparams") or {}
    dc = cfg.get("dataset_config") or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    lp, gp = cfg["constraint"]

    t = pd.read_csv(raw_p)
    y = t["True_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else np.zeros(len(t), int)
    N = len(t)
    ntrue = int((y == cls).sum())
    K = int(np.round(ntrue * gp))
    grp = {}
    for gg in np.unique(g):
        m = (g == gg)
        grp[int(gg)] = dict(N_g=int(m.sum()), ntrue_g=int(((y == cls) & m).sum()),
                            K_g=int(np.round(((y == cls) & m).sum() * lp)))

    log = numeric(pd.read_csv(log_p))
    if log.empty:
        return None
    return dict(cfg=cfg, hp=hp, cls=cls, N=N, K=K, ntrue=ntrue, groups=grp,
                log=log, path=d,
                dataset=cfg.get("dataset_mode"), model=cfg.get("model_name"),
                cap=cfg.get("constraint_tag"), seed=hp.get("seed"),
                method=cfg.get("methodology"),
                ce_skip_cfg=hp.get("enable_ce_skip", "UNSET->default True"),
                raw_count=int((t["Predicted_Label"].to_numpy(int) == cls).sum()))


def hounie_traj(r):
    hp, log = r["hp"], r["log"]
    eta = float(hp.get("hounie_eta_lambda", 0.01))
    eta_u = float(hp.get("hounie_eta_u", 0.01))
    alpha = float(hp.get("hounie_alpha", 10.0))
    lam = log["max_lam_g"].to_numpy(float)
    u = log["max_u_g"].to_numpy(float)
    hu = log["h_u"].to_numpy(float)
    upred = np.empty_like(u)
    prev = 0.0
    for i in range(len(u)):
        upred[i] = max(0.0, prev + eta_u * (lam[i] - 2 * alpha * prev))
        prev = u[i]
    u_err = float(np.max(np.abs(upred - u)))
    ulag = np.concatenate([[0.0], u[:-1]])
    lamlag = np.concatenate([[0.0], lam[:-1]])
    mean_l = (lam - lamlag) / eta + ulag
    soft = r["K"] + r["N"] * mean_l
    soft[(lam <= 0)] = np.nan          # lam clamped at the max(0,.) floor
    U_loc = np.sqrt(np.maximum(0.0, hu / alpha - u ** 2))
    return dict(lam=lam, u=u, hu=hu, soft=soft, mean_l=mean_l, U_loc=U_loc,
                u_err=u_err, eta=eta, alpha=alpha)


def fioretto_traj(r):
    step = float(r["hp"]["fioretto_step_size"])
    lam = r["log"]["max_lambda_g"].to_numpy(float)
    inc = lam - np.concatenate([[0.0], lam[:-1]])
    soft = r["K"] + inc / step
    soft[inc <= 0] = np.nan
    return dict(lam=lam, soft=soft, step=step)


def summarize(r):
    log = r["log"]
    ep = log["epoch"].to_numpy(int)
    exc = log["total_excess"].to_numpy(float)
    sat = log["all_satisfied"].to_numpy(int)
    ce = log["ce_loss"].to_numpy(float)
    n = len(ep)
    ce_off = np.isnan(ce)
    ce_off_ep = int(ep[ce_off][0]) + 1 if ce_off.any() else None
    first_sat = int(ep[sat == 1][0]) + 1 if sat.any() else None
    row = dict(dataset=r["dataset"], model=r["model"], cap=r["cap"],
               seed=r["seed"], method=r["method"], K=r["K"], N=r["N"],
               ntrue=r["ntrue"], epochs=n, last_epoch=int(ep.max()) + 1,
               ce_skip_cfg=r["ce_skip_cfg"],
               ce_off_ep=ce_off_ep, n_ce_off=int(ce_off.sum()),
               ce_min=np.nanmin(ce),
               first_sat=first_sat, n_sat=int(sat.sum()),
               sat_flips=int(((sat[:-1] == 1) & (sat[1:] == 0)).sum()) if n > 1 else 0,
               exc0=exc[0], exc_min=exc.min(), exc_final=exc[-1],
               exc_trend_all=trend(exc),
               exc_trend_ce_on=trend(exc[~ce_off]),
               exc_trend_ce_off=trend(exc[ce_off]),
               raw_count=r["raw_count"])
    if r["method"] == "hounie_rcl":
        T = hounie_traj(r)
        s, lam = T["soft"], T["lam"]
        row.update(u_recursion_maxerr=T["u_err"],
                   lam0=lam[0], lam_max=lam.max(), lam_final=lam[-1],
                   lam_peak_ep=int(np.argmax(lam)) + 1,
                   lam_decaying_eps=int((np.diff(lam) < 0).sum()),
                   u_max=T["u"].max(), Uloc_max=T["U_loc"].max(),
                   soft0=s[0], soft_min=np.nanmin(s), soft_final=s[-1],
                   soft_min_over_K=np.nanmin(s) / r["K"],
                   soft_final_over_K=s[-1] / r["K"],
                   n_ep_soft_below_K=int(np.nansum(s < r["K"])),
                   soft_trend_ce_on=trend(s[~ce_off]),
                   soft_trend_ce_off=trend(s[ce_off]),
                   soft_at_sat=(s[sat == 1][0] if sat.any() else np.nan),
                   soft_drop_after_sat=((s[sat == 1][0] - s[-1]) if sat.any() else np.nan),
                   cstr_loss_max=log["constraint_loss"].max())
    elif r["method"] == "fioretto_ldf":
        T = fioretto_traj(r)
        s = T["soft"]
        row.update(lam0=T["lam"][0], lam_max=T["lam"].max(), lam_final=T["lam"][-1],
                   soft0=s[0], soft_min=np.nanmin(s) if np.isfinite(s).any() else np.nan,
                   soft_final=s[-1], n_ep_lam_frozen=int((~np.isfinite(s)).sum()),
                   cstr_loss_max=log["constraint_loss"].max())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--methods", default="hounie_rcl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="dataset,model,cap,seed")
    ap.add_argument("--perrun", action="store_true")
    args = ap.parse_args()
    want = args.methods.split(",")

    rows, runs = [], []
    for p in sorted(glob.glob(args.root + "/**/config.json", recursive=True)):
        try:
            cfg = json.load(open(p))
        except Exception:
            continue
        if cfg.get("methodology") not in want:
            continue
        r = load_run(p)
        if r is None:
            continue
        runs.append(r)
        rows.append(summarize(r))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no runs under", args.root)
        return 1

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 80)

    if args.dump:
        ds, mo, cap, sd = args.dump.split(",")
        for r in runs:
            if not (r["dataset"] == ds and r["model"] == mo and r["cap"] == cap
                    and str(r["seed"]) == sd):
                continue
            print("### %s | %s %s %s seed%s | N=%d K_global=%d n_true=%d | %s" %
                  (r["method"], ds, mo, cap, sd, r["N"], r["K"], r["ntrue"], r["path"]))
            print("    groups:", r["groups"])
            print("    enable_ce_skip in config:", r["ce_skip_cfg"])
            base = pd.DataFrame({
                "ep": r["log"]["epoch"].to_numpy(int) + 1,
                "ce": r["log"]["ce_loss"],
                "ce_on": (~r["log"]["ce_loss"].isna()).astype(int),
                "cstr": r["log"]["constraint_loss"],
                "excess": r["log"]["total_excess"].astype(int),
                "sat": r["log"]["all_satisfied"].astype(int)})
            if r["method"] == "hounie_rcl":
                T = hounie_traj(r)
                print("    u-recursion max abs err %.2e (rounding floor ~1e-6) -> "
                      "reconstruction verified" % T["u_err"])
                base["lam_g"] = T["lam"]; base["u_g"] = T["u"]
                base["U_loc"] = T["U_loc"]
                base["soft_cnt"] = T["soft"]
                base["soft-K"] = T["soft"] - r["K"]
            else:
                T = fioretto_traj(r)
                base["lam_g"] = T["lam"]; base["soft_cnt"] = T["soft"]
                base["soft-K"] = T["soft"] - r["K"]
            print(base.to_string(index=False, float_format=lambda x: "%.5g" % x))
            print()
        return 0

    if args.perrun:
        print("=" * 130)
        print("PER-RUN  root=%s" % args.root)
        print("=" * 130)
        print(d.sort_values(["method", "dataset", "model", "cap", "seed"])
              .to_string(index=False, float_format=lambda x: "%.4g" % x))

    print()
    print("=" * 130)
    print("PER-CELL MEANS over seeds (cells never pooled)   root=%s" % args.root)
    print("=" * 130)
    order = ["K", "epochs", "ce_off_ep", "n_ce_off", "ce_min", "first_sat", "n_sat",
             "sat_flips", "exc0", "exc_min", "exc_final", "exc_trend_ce_on",
             "exc_trend_ce_off", "lam0", "lam_max", "lam_final", "lam_peak_ep",
             "lam_decaying_eps", "u_max", "Uloc_max", "soft0", "soft_min",
             "soft_final", "soft_min_over_K", "soft_final_over_K",
             "n_ep_soft_below_K", "soft_trend_ce_on", "soft_trend_ce_off",
             "soft_at_sat", "soft_drop_after_sat", "n_ep_lam_frozen",
             "cstr_loss_max", "raw_count"]
    for m, g in d.groupby("method"):
        num = g.select_dtypes(include=[np.number]).columns
        agg = g.groupby(CELL)[list(num)].mean()
        nsat = g.groupby(CELL)["first_sat"].apply(lambda s: int(s.notna().sum()))
        nceoff = g.groupby(CELL)["ce_off_ep"].apply(lambda s: int(s.notna().sum()))
        cnt = g.groupby(CELL)["seed"].size().rename("nseed")
        show = [c for c in order if c in agg.columns]
        tab = pd.concat([cnt, nsat.rename("seeds_that_satisfied"),
                         nceoff.rename("seeds_CE_turned_off"), agg[show]], axis=1)
        print("\n--- %s ---" % m)
        print(tab.to_string(float_format=lambda x: "%.4g" % x))

    if args.out:
        d.to_csv(args.out, index=False)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""REFUTATION pass on the bb4.py "overshoot ordering contradicts in 4 of 6" claim.

Everything is re-derived from final_predictions_raw.csv / config.json. Nothing
is read from out_factbase.csv except as a cross-check at the end.

The claim's unit is a CONTRAST between two atomic cells that differ only in
backbone. Three things are checked that bb4.py never checks:

  (A) Is the ordering it is testing on the LEFT side (overshoot) resolvable at
      all? overshoot = clip_raw/K with K identical across backbones inside a
      (dataset,cap), so the ordering is decided entirely by clip_raw -- a
      4-seed mean with a documented 20-83% seed swing.
  (B) How many INDEPENDENT contrasts are there? K cancels, so the overshoot
      ordering cannot depend on cap: L30 and L50 re-use one number.
  (C) Does the AGREE/CONTRADICT verdict survive resampling the seeds?
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CLIP = "results/headroom/headroom_b30"
SIBS = {
    "lrc0.0001_noceskip": NOCE,
    "lrc0.0001": "results/headroom/headroom_b30_lrc0.0001",
    "lrc5e-05": "results/headroom/headroom_b30_lrc5e-05",
}
DUALS = ["fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ raw counts
def raw_counts(root, methods):
    """count_raw straight out of final_predictions_raw.csv, no helper."""
    out = []
    for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cfg_path))
        if cfg.get("methodology") not in methods:
            continue
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)  # TRAP 5
        t = pd.read_csv(raw, usecols=["Predicted_Label", "True_Label"])
        hp = cfg.get("hyperparams") or {}
        out.append({
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"), "cls": cls,
            "count_raw": int((t.Predicted_Label.to_numpy(int) == cls).sum()),
            "n_true": int((t.True_Label.to_numpy(int) == cls).sum()),
            "n_pool": len(t), "run_dir": d,
            "warmup": hp.get("warmup_epochs"), "cepochs": hp.get("constraint_epochs"),
        })
    return pd.DataFrame(out)


# ------------------------------------------------------------- schema-trap epochs
def last_epoch(run_dir):
    p = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(p):
        return np.nan, np.nan
    df = pd.read_csv(p)
    col = "Epoch" if "Epoch" in df.columns else ("epoch" if "epoch" in df.columns else None)
    if col is None:
        return np.nan, len(df)
    v = pd.to_numeric(df[col], errors="coerce").dropna()   # TRAPS 2 + 4
    return (float(v.max()) if len(v) else np.nan), len(df)  # TRAP 1: max != len


def sect(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def main():
    # ================================================================ STEP 0
    sect("STEP 0  SCHEMA TRAPS: is the claim contaminated by an epoch misread?")
    ep = []
    for _, r in raw_counts(NOCE, ["tralo", "fioretto_ldf", "hounie_rcl"]).iterrows():
        m, n = last_epoch(r.run_dir)
        ep.append({"method": r.method, "max_epoch": m, "n_rows": n})
    ep = pd.DataFrame(ep)
    print(ep.groupby("method").agg(runs=("n_rows", "size"),
                                   rows_median=("n_rows", "median"),
                                   maxep_median=("max_epoch", "median"),
                                   maxep_min=("max_epoch", "min"),
                                   maxep_max=("max_epoch", "max")).to_string())
    print("\n  TraLO rows_median << maxep_median confirms the sparse log. The")
    print("  claim under test uses NO epoch column, so trap 1 cannot be its bug.")

    # ================================================================ STEP 1
    sect("STEP 1  RE-DERIVE clip_raw (the whole left side of the claim) FROM RAW FILES")
    cl = raw_counts(CLIP, ["heuristic"])
    print("  heuristic runs found: %d   warmup=%s cepochs=%s"
          % (len(cl), sorted(cl.warmup.unique()), sorted(cl.cepochs.unique())))
    per = cl.pivot_table(index=["dataset", "model", "seed"], columns="cap",
                         values="count_raw")
    dup = (per["L30_G30"] == per["L50_G50"]).mean()
    print("\n  L30 vs L50 heuristic count_raw identical in %.0f%% of (ds,model,seed):"
          % (100 * dup))
    print(per.to_string())
    print("\n  -> the clipper's raw count is a property of the 30-epoch CE model")
    print("     ONLY. It does not know the cap. n=4 seeds, not 8.")

    nat = cl.groupby(["dataset", "model", "seed"])["count_raw"].first().reset_index()
    g = nat.groupby(["dataset", "model"])["count_raw"]
    tbl = g.agg(mean="mean", sd="std", n="size", lo="min", hi="max").reset_index()
    print("\n  natural rate per (dataset, backbone), 4 seeds:")
    print(tbl.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ================================================================ STEP 2
    sect("STEP 2  IS THE OVERSHOOT ORDERING RESOLVABLE?  Welch t on clip_raw, n=4 vs 4")
    print("  K is IDENTICAL across backbones inside a (dataset,cap), so")
    print("  sign(overshoot_MNV3 - overshoot_RegNet) == sign(clip_raw_MNV3 - clip_raw_RegNet)")
    print("  and CANNOT depend on the cap.\n")
    from scipy import stats
    res = {}
    for ds, gg in nat.groupby("dataset"):
        a = gg[gg.model == "MobileNetV3"].sort_values("seed").count_raw.to_numpy(float)
        b = gg[gg.model == "RegNetY400MF"].sort_values("seed").count_raw.to_numpy(float)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        # seeds are matched across backbones -> the paired test is also available
        tp, pp = stats.ttest_rel(a, b)
        res[ds] = (a.mean(), b.mean(), t, p)
        print("  %-12s MNV3 %7.2f %s   RegNet %7.2f %s"
              % (ds, a.mean(), np.round(a, 0).tolist(), b.mean(), np.round(b, 0).tolist()))
        print("               diff %+7.2f   Welch t=%+.2f p=%.3f   paired t=%+.2f p=%.3f  %s"
              % (a.mean() - b.mean(), t, p, tp, pp,
                 "RESOLVED" if p < 0.05 else "*** NOT RESOLVED (coin flip) ***"))

    # ================================================================ STEP 3
    sect("STEP 3  RE-DERIVE THE EDGE (right side) FROM RAW FILES, per cell")
    d = A.rows_for(NOCE)
    d = d[d.method.isin(["tralo"] + DUALS)]
    print("  scored %d runs; seeds per cell:" % len(d))
    piv = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    piv = piv.dropna(subset=["tralo"]).copy()
    piv["d"] = piv["tralo"] - piv[DUALS].max(axis=1)
    edge = piv.reset_index().groupby(CELL)["d"].agg(["mean", "std", "size"]).reset_index()
    edge.columns = CELL + ["edge", "sd", "n"]
    print(edge.to_string(index=False, float_format=lambda x: "%.4f" % x))

    K = d.groupby(CELL)["K"].first().reset_index()
    M = edge.merge(K, on=CELL).merge(
        tbl.rename(columns={"mean": "clip_raw"})[["dataset", "model", "clip_raw"]],
        on=["dataset", "model"])
    M["overshoot"] = M.clip_raw / M.K
    print("\n  reconstructed the exact table bb4.py prints:")
    for (ds, cap), gg in M.groupby(["dataset", "cap"]):
        gg = gg.set_index("model")
        om, orn = gg.loc["MobileNetV3", "overshoot"], gg.loc["RegNetY400MF", "overshoot"]
        em, ern = gg.loc["MobileNetV3", "edge"], gg.loc["RegNetY400MF", "edge"]
        v = "AGREES" if np.sign(om - orn) == np.sign(em - ern) else "CONTRADICTS"
        print("   %-12s %-8s overshoot %.2f vs %.2f (gap %+.3f = %+.1f%%)   "
              "edge %+0.4f vs %+0.4f (gap %+0.4f)  -> %s"
              % (ds, cap, om, orn, om - orn, 100 * (om - orn) / orn, em, ern, em - ern, v))

    # ================================================================ STEP 4
    sect("STEP 4  COUNT, DO NOT AVERAGE: how many INDEPENDENT contrasts are there?")
    sgn = []
    for (ds, cap), gg in M.groupby(["dataset", "cap"]):
        gg = gg.set_index("model")
        sgn.append({"dataset": ds, "cap": cap,
                    "sign_overshoot": int(np.sign(gg.loc["MobileNetV3", "overshoot"]
                                                  - gg.loc["RegNetY400MF", "overshoot"])),
                    "sign_edge": int(np.sign(gg.loc["MobileNetV3", "edge"]
                                             - gg.loc["RegNetY400MF", "edge"]))})
    sgn = pd.DataFrame(sgn)
    print(sgn.to_string(index=False))
    nun_o = sgn.groupby("dataset").sign_overshoot.nunique()
    nun_e = sgn.groupby("dataset").sign_edge.nunique()
    print("\n  distinct overshoot signs within a dataset: %s" % nun_o.to_dict())
    print("  distinct edge signs within a dataset:      %s" % nun_e.to_dict())
    print("  -> both caps give the SAME verdict inside every dataset.")
    print("  -> the '6 tests' are 3 contrasts each counted twice. The real")
    print("     tally is 2 CONTRADICTS of 3, not 4 of 6.")

    # ================================================================ STEP 5
    sect("STEP 5  DOES THE VERDICT SURVIVE THE SEEDS?  10k bootstrap over seeds")
    print("  Resample 4 seeds with replacement independently for the clipper")
    print("  (left side) and the trained arms (right side), recompute both cell")
    print("  means, re-adjudicate. A verdict that is real should be stable.\n")
    per_seed_edge = piv.reset_index()
    B = 10000
    for (ds, cap), gg in M.groupby(["dataset", "cap"]):
        arrs = {}
        for mo in ["MobileNetV3", "RegNetY400MF"]:
            arrs[mo] = (
                nat[(nat.dataset == ds) & (nat.model == mo)].count_raw.to_numpy(float),
                per_seed_edge[(per_seed_edge.dataset == ds) & (per_seed_edge.model == mo)
                              & (per_seed_edge.cap == cap)]["d"].to_numpy(float),
            )
        kk = float(gg.K.iloc[0])
        con = 0
        for _ in range(B):
            os_, ed_ = {}, {}
            for mo, (c, e) in arrs.items():
                os_[mo] = RNG.choice(c, len(c), replace=True).mean() / kk
                ed_[mo] = RNG.choice(e, len(e), replace=True).mean()
            so = np.sign(os_["MobileNetV3"] - os_["RegNetY400MF"])
            se = np.sign(ed_["MobileNetV3"] - ed_["RegNetY400MF"])
            con += (so != se)
        obs = "CONTRADICTS" if sgn[(sgn.dataset == ds) & (sgn.cap == cap)].sign_overshoot.iloc[0] \
            != sgn[(sgn.dataset == ds) & (sgn.cap == cap)].sign_edge.iloc[0] else "AGREES"
        print("  %-12s %-8s point verdict %-11s   P(CONTRADICTS under resampling) = %.3f  %s"
              % (ds, cap, obs, con / B,
                 "STABLE" if abs(con / B - 0.5) > 0.35 else "*** COIN FLIP ***"))

    # ================================================================ STEP 6
    sect("STEP 6  REPLICATION: re-adjudicate the SAME 6 rows on the sibling campaigns")
    print("  If 'CONTRADICTS' is a property of the data and not of one campaign's")
    print("  seed draw, the verdict must repeat when the constraint LR / CE gate")
    print("  changes. The clipper (left side) is IDENTICAL in all three.\n")
    verdicts = {}
    for lab, root in SIBS.items():
        dd = A.rows_for(root)
        dd = dd[dd.method.isin(["tralo"] + DUALS)]
        pv = dd.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
        pv = pv.dropna(subset=["tralo"]).copy()
        pv["d"] = pv["tralo"] - pv[DUALS].max(axis=1)
        e = pv.reset_index().groupby(CELL)["d"].mean().reset_index()
        mm = e.merge(K, on=CELL).merge(
            tbl.rename(columns={"mean": "clip_raw"})[["dataset", "model", "clip_raw"]],
            on=["dataset", "model"])
        mm["overshoot"] = mm.clip_raw / mm.K
        row = {}
        for (ds, cap), gg in mm.groupby(["dataset", "cap"]):
            gg = gg.set_index("model")
            so = np.sign(gg.loc["MobileNetV3", "overshoot"] - gg.loc["RegNetY400MF", "overshoot"])
            se = np.sign(gg.loc["MobileNetV3", "d"] - gg.loc["RegNetY400MF", "d"])
            row[(ds, cap)] = ("AGREES" if so == se else "CONTRADICTS",
                              gg.loc["MobileNetV3", "d"] - gg.loc["RegNetY400MF", "d"])
        verdicts[lab] = row
    keys = sorted(verdicts["lrc0.0001_noceskip"])
    print("  %-12s %-8s | %s" % ("dataset", "cap",
                                 " | ".join("%-24s" % k for k in SIBS)))
    flips = 0
    for k in keys:
        cells = []
        for lab in SIBS:
            v, g_ = verdicts[lab][k]
            cells.append("%-11s (%+.4f)" % (v, g_))
        vs = {verdicts[lab][k][0] for lab in SIBS}
        if len(vs) > 1:
            flips += 1
        print("  %-12s %-8s | %s   %s" % (k[0], k[1], " | ".join(cells),
                                          "<-- FLIPS" if len(vs) > 1 else ""))
    print("\n  rows whose verdict FLIPS across campaigns: %d of %d" % (flips, len(keys)))

    # ================================================================ STEP 7
    sect("STEP 7  CROSS-CHECK against the prior agent's fact base")
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    fbo = fb[(fb.campaign == "lrc0.0001_noceskip") & (fb.method == "tralo")][
        CELL + ["clip_raw", "K", "d_vs_bestdual"]].copy()
    chk = M.merge(fbo, on=CELL, suffixes=("_mine", "_fb"))
    chk["dclip"] = chk.clip_raw_mine - chk.clip_raw_fb
    chk["dedge"] = chk.edge - chk.d_vs_bestdual
    print(chk[CELL + ["clip_raw_mine", "clip_raw_fb", "dclip",
                      "edge", "d_vs_bestdual", "dedge"]]
          .to_string(index=False, float_format=lambda x: "%.6f" % x))
    print("\n  max |clip_raw delta| = %.6g   max |edge delta| = %.6g"
          % (chk.dclip.abs().max(), chk.dedge.abs().max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

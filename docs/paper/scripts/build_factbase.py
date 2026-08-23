"""ONE fact base for every headroom campaign, so no agent re-derives a number.

Metric definitions are NOT redefined here. Every metric column comes from
analyze_headroom.rows_for() unchanged, so this file cannot drift from
stratify.py / compare_all.py. This script only

  (a) runs rows_for over every campaign,
  (b) adds pool context that rows_for does not carry (test-pool size, the
      number of TRUE constrained-class samples, so K can be read as a rate),
  (c) averages over SEED ONLY inside the atomic cell
      (campaign, dataset, model, cap, method) -- never across caps, backbones
      or datasets,
  (d) joins the post-hoc clipper's own raw count onto every row, because that
      count IS the unconstrained CE model's natural rate and it is the only
      thing that says whether the cap binds at all.

    python paper/scripts/build_factbase.py --out paper/scripts/out_factbase.csv
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CAMPAIGNS = [
    # label                     root
    ("headroom_b30",            "results/headroom/headroom_b30"),
    ("lrc0.0001",               "results/headroom/headroom_b30_lrc0.0001"),
    ("lrc0.0001_noceskip",      "results/headroom/headroom_b30_lrc0.0001_noceskip"),
    ("lrc0.0001_noceskip_full", "results/headroom/headroom_b30_lrc0.0001_noceskip_full"),
    ("lrc5e-05",                "results/headroom/headroom_b30_lrc5e-05"),
    ("fixsel_derm",             "newdirections/arm_fixsel/results/fixsel"),
]
CLIP_CAMPAIGN = "headroom_b30"
CELL = ["dataset", "model", "cap"]
KEY = ["dataset", "cap", "model", "seed", "method"]


def last_epoch(run_dir):
    """Highest epoch index the run actually reached.

    Three traps, all of which have already produced a wrong number once:
      - TraLO's log is SPARSE (a row only every 5th epoch, on satisfaction, or
        on the first constraint epoch), so len(df) is not the epoch count.
      - the column is "Epoch" for TraLO and "epoch" for the duals.
      - headers repeat mid-file, one per phase, so the column is object dtype
        and must be coerced before max().
    """
    p = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(p):
        return float("nan")
    try:
        df = pd.read_csv(p)
    except Exception:
        return float("nan")
    col = "Epoch" if "Epoch" in df.columns else ("epoch" if "epoch" in df.columns
                                                 else None)
    if col is None:
        return float("nan")
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.max()) if len(v) else float("nan")


def pool_stats(root):
    """Test-pool size and TRUE constrained-class count, per run.

    rows_for() does not return either, and without them K is an unreadable
    integer: K is round(pct * TRUE count of the class), so K only becomes
    interpretable next to the pool it is drawn from.
    """
    out = []
    for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        if cls is None:
            continue
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        try:
            t = pd.read_csv(raw, usecols=["True_Label"])
        except Exception:
            continue
        y = t["True_Label"].to_numpy(int)
        hp = cfg.get("hyperparams") or {}
        out.append({
            "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
            "model": cfg.get("model_name"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"),
            "cls": cls, "n_pool": len(y), "n_true_cls": int((y == cls).sum()),
            "last_epoch": last_epoch(d),
            "ce_skip": hp.get("enable_ce_skip"),
            "lr": hp.get("lr"), "lr_constraint": hp.get("lr_constraint"),
            "stable_thr": hp.get("stable_count_threshold"),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/scripts/out_factbase.csv")
    ap.add_argument("--per-run-out", default="paper/scripts/out_factbase_perrun.csv")
    args = ap.parse_args()

    per_run = []
    for label, root in CAMPAIGNS:
        if not os.path.isdir(root):
            print("MISSING %s" % root)
            continue
        d = A.rows_for(root)
        if d.empty:
            print("%-24s no scorable runs" % label)
            continue
        dup = int(d.duplicated(subset=KEY).sum())
        ps = pool_stats(root)
        d = d.merge(ps, on=KEY, how="left")
        d["campaign"] = label
        print("%-24s %4d scorable runs%s" % (label, len(d),
              ("   WARNING %d duplicate keys" % dup) if dup else ""))
        per_run.append(d)
    if not per_run:
        return 1
    R = pd.concat(per_run, ignore_index=True)
    # Collapse is judged on the model's OWN count, before post-hoc fills to K.
    R["collapsed"] = R["count_raw"] < (R["K"] / 3.0)
    # last_epoch is NOT comparable across methods as logged. Verified against
    # the raw logs: TraLO's `Epoch` is 1-indexed and spans warm-up + constraint
    # (max 30 = 1 warm-up + 29 constraint); the duals' `epoch` is 0-indexed and
    # covers the constraint phase only (max 28 = 29 constraint epochs).
    # Comparing the raw maxima understates the duals by two epochs.
    R["constraint_epochs_run"] = R["last_epoch"] + R["method"].map(
        {"tralo": -1, "fioretto_ldf": 1, "hounie_rcl": 1}).fillna(0)
    R.loc[R.method.isin(["heuristic", "danits_lp"]), "constraint_epochs_run"] = 0.0
    R.to_csv(args.per_run_out, index=False)

    # ---- average over SEED ONLY -------------------------------------------
    G = ["campaign", "dataset", "model", "cap", "method"]
    agg = R.groupby(G).agg(
        n_seeds=("seed", "size"),
        K=("K", "first"),
        n_pool=("n_pool", "first"),
        n_true_cls=("n_true_cls", "first"),
        ccF1eq=("ccF1eq", "mean"),
        AP=("AP", "mean"),
        macroEq=("macroEq", "mean"),
        count_raw=("count_raw", "mean"),
        count_adj=("count", "mean"),
        sat=("sat", "mean"),
        n_collapsed=("collapsed", "sum"),
        last_epoch=("last_epoch", "mean"),
        constraint_epochs_run=("constraint_epochs_run", "mean"),
        warmup=("warmup", "first"),
        cepochs=("cepochs", "first"),
        ce_skip=("ce_skip", "first"),
        lr=("lr", "first"),
        lr_constraint=("lr_constraint", "first"),
        stable_thr=("stable_thr", "first"),
    ).reset_index()

    # ---- the clipper's raw count = the unconstrained CE model's own rate ---
    clip = R[(R.campaign == CLIP_CAMPAIGN) & (R.method == "heuristic")]
    nat = clip.groupby(CELL).agg(clip_raw=("count_raw", "mean"),
                                 clip_raw_min=("count_raw", "min"),
                                 clip_raw_max=("count_raw", "max")).reset_index()
    agg = agg.merge(nat, on=CELL, how="left")
    # ---- the two adjudications, PAIRED within seed then averaged ----------
    # Cell-level, so they repeat on every method row of the cell: an agent
    # quoting a single row should not have to recompute who won it.
    clip_pr = R[(R.campaign == CLIP_CAMPAIGN) & (R.method.isin(["heuristic", "danits_lp"]))]
    dl = []
    for camp, g in R.groupby("campaign"):
        piv = g.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
        if "tralo" not in piv.columns:
            continue
        duals = [m for m in ["fioretto_ldf", "hounie_rcl"] if m in piv.columns]
        cp = clip_pr.pivot_table(index=CELL + ["seed"], columns="method",
                                 values="ccF1eq")
        s = piv.dropna(subset=["tralo"]).copy()
        if duals:
            s["d_vs_bestdual"] = s["tralo"] - s[duals].max(axis=1)
        s["d_vs_clip"] = s["tralo"] - cp.max(axis=1).reindex(s.index)
        s = s.reset_index()
        keep = [c for c in ["d_vs_bestdual", "d_vs_clip"] if c in s.columns]
        t = s.groupby(CELL)[keep].mean().reset_index()
        t["campaign"] = camp
        dl.append(t)
    if dl:
        agg = agg.merge(pd.concat(dl, ignore_index=True),
                        on=["campaign"] + CELL, how="left")

    agg["natural_rate"] = agg["clip_raw"] / agg["n_pool"]
    agg["K_frac_pool"] = agg["K"] / agg["n_pool"]
    agg["K_over_clip_raw"] = agg["K"] / agg["clip_raw"]
    agg["binds"] = (agg["clip_raw"] > agg["K"]).map({True: "yes", False: "NO"})

    cols = ["campaign", "dataset", "model", "cap", "method", "n_seeds",
            "K", "n_pool", "n_true_cls", "K_frac_pool",
            "clip_raw", "clip_raw_min", "clip_raw_max",
            "natural_rate", "K_over_clip_raw", "binds",
            "ccF1eq", "AP", "macroEq", "count_raw", "count_adj", "sat",
            "n_collapsed", "d_vs_bestdual", "d_vs_clip",
            "constraint_epochs_run", "last_epoch",
            "warmup", "cepochs", "ce_skip", "lr",
            "lr_constraint", "stable_thr"]
    for c in cols:
        if c not in agg.columns:
            agg[c] = float("nan")
    agg = agg[cols].sort_values(["campaign", "dataset", "model", "cap", "method"])
    agg.to_csv(args.out, index=False)
    print("\nwrote %s   rows=%d" % (os.path.abspath(args.out), len(agg)))
    print("wrote %s   rows=%d" % (os.path.abspath(args.per_run_out), len(R)))

    # ---- natural rate vs cap ---------------------------------------------
    print("\n" + "=" * 108)
    print("NATURAL RATE: what the UNCONSTRAINED CE model (30ep plain CE, the")
    print("post-hoc arms' own predictions) assigns to the constrained class,")
    print("against the cap K. K = round(pct * TRUE count of the class).")
    print("=" * 108)
    nb = agg[(agg.campaign == CLIP_CAMPAIGN) & (agg.method == "heuristic")][
        ["dataset", "model", "cap", "n_pool", "n_true_cls", "K", "K_frac_pool",
         "clip_raw", "clip_raw_min", "clip_raw_max",
         "natural_rate", "K_over_clip_raw", "binds"]]
    print(nb.to_string(index=False, float_format=lambda x: "%.4f" % x))

    print("\n" + "=" * 108)
    print("SANITY: K / n_true_cls must equal the cap percentage exactly")
    print("=" * 108)
    chk = nb.copy()
    chk["K_over_true"] = chk["K"] / chk["n_true_cls"]
    print(chk[["dataset", "model", "cap", "K", "n_true_cls", "K_over_true"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())

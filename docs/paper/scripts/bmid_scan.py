"""Re-derive the base_model_id sharing claim from raw config.json files.

Group = (dataset, backbone, seed).  A group is "shared" if every method in it
carries the same base_model_id.
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

ROOTS = {
    "paper_final": "results/pending_runs/paper_final",
    "warmup_ablation": "results/pending_runs/warmup_ablation",
    "warmup1_probe": "results/pending_runs/warmup1_probe",
    "headroom_b30": "results/headroom/headroom_b30",
    "headroom_noceskip": "results/headroom/headroom_b30_lrc0.0001_noceskip",
    "headroom_lrc0.0001": "results/headroom/headroom_b30_lrc0.0001",
    "headroom_lrc5e-05": "results/headroom/headroom_b30_lrc5e-05",
}


def scan(root):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.json" not in filenames:
            continue
        p = os.path.join(dirpath, "config.json")
        try:
            c = json.load(open(p))
        except Exception as e:
            rows.append({"path": p, "err": str(e)})
            continue
        hp = c.get("hyperparams", {}) or {}
        rows.append({
            "path": p,
            "dataset": c.get("dataset_mode"),
            "model": c.get("model_name"),
            "method": c.get("methodology"),
            "cap": c.get("constraint_tag"),
            "seed": hp.get("seed"),
            "warmup": hp.get("warmup_epochs"),
            "cepochs": hp.get("constraint_epochs"),
            "lr": hp.get("lr"),
            "lr_c": hp.get("lr_constraint"),
            "bmid": c.get("base_model_id"),
            "status": c.get("status"),
            "cached": (c.get("results") or {}).get("used_cached_model"),
        })
    return pd.DataFrame(rows)


def report(name, df):
    print("=" * 110)
    print("ROOT %s   n_config=%d" % (name, len(df)))
    if df.empty:
        return None
    print("  methods:", dict(df.method.value_counts()))
    print("  warmup_epochs by method:")
    for m, g in df.groupby("method"):
        print("     %-14s warmup=%s  constraint_epochs=%s  n=%d"
              % (m, sorted(set(g.warmup.dropna())), sorted(set(g.cepochs.dropna())), len(g)))
    key = ["dataset", "model", "seed"]
    shared, total, detail = 0, 0, []
    for k, g in df.groupby(key, dropna=False):
        total += 1
        n_ids = g.bmid.nunique()
        if n_ids == 1:
            shared += 1
        detail.append({
            "dataset": k[0], "model": k[1], "seed": k[2],
            "n_runs": len(g), "n_methods": g.method.nunique(),
            "n_bmid": n_ids,
            "n_warmup": g.warmup.nunique(),
            "shared": n_ids == 1,
        })
    print("  GROUPS (dataset,backbone,seed) with a SINGLE base_model_id: %d / %d" % (shared, total))
    d = pd.DataFrame(detail)
    print(d.to_string(index=False))
    # per-method bmid map for one illustrative group
    return df


def main():
    os.chdir(os.path.expanduser("~/OptimizationLoss"))
    all_df = {}
    for name, root in ROOTS.items():
        if not os.path.isdir(root):
            print("MISSING ROOT", name, root)
            continue
        df = scan(root)
        all_df[name] = df
        report(name, df)
        print()

    # Illustrative cell asked about in the claim
    for name in ["paper_final", "headroom_b30", "headroom_noceskip"]:
        if name not in all_df:
            continue
        df = all_df[name]
        g = df[(df.dataset == "dermmnist") & (df.model == "MobileNetV3") & (df.seed == 1)]
        print("### %s  dermmnist/MobileNetV3/seed_1 base_model_id by method" % name)
        if g.empty:
            print("   (empty)")
        else:
            print(g[["method", "cap", "warmup", "cepochs", "lr", "lr_c", "bmid", "status"]]
                  .sort_values(["method", "cap"]).to_string(index=False))
        print()

    # cross-campaign: do noceskip trained arms share the bmid of headroom_b30 trained arms?
    if "headroom_b30" in all_df and "headroom_noceskip" in all_df:
        a = all_df["headroom_b30"]
        b = all_df["headroom_noceskip"]
        a2 = a[a.method.isin(["heuristic", "danits_lp"])]
        merged = []
        for k, gb in b.groupby(["dataset", "model", "seed"], dropna=False):
            ga = a2[(a2.dataset == k[0]) & (a2.model == k[1]) & (a2.seed == k[2])]
            merged.append({
                "dataset": k[0], "model": k[1], "seed": k[2],
                "trained_bmid": sorted(set(gb.bmid)),
                "clip_bmid": sorted(set(ga.bmid)),
                "same": set(gb.bmid) == set(ga.bmid),
            })
        m = pd.DataFrame(merged)
        print("### ACTUAL HEADLINE COMPARISON (trained=noceskip vs clip=headroom_b30)")
        print("  groups where trained and clip share ALL base_model_id: %d / %d"
              % (int(m["same"].sum()), len(m)))
        print(m.to_string(index=False))
        print()

    # do trained arms in headroom_b30 vs noceskip share checkpoints?
    if "headroom_b30" in all_df and "headroom_noceskip" in all_df:
        a = all_df["headroom_b30"]
        b = all_df["headroom_noceskip"]
        at = a[a.method.isin(["tralo", "fioretto_ldf", "hounie_rcl"])]
        same = 0
        tot = 0
        for k, gb in b.groupby(["dataset", "model", "seed"], dropna=False):
            ga = at[(at.dataset == k[0]) & (at.model == k[1]) & (at.seed == k[2])]
            tot += 1
            if set(gb.bmid) == set(ga.bmid) and len(set(gb.bmid)) == 1:
                same += 1
        print("### trained arms: headroom_b30 vs noceskip share the SAME single bmid in %d/%d groups"
              % (same, tot))


if __name__ == "__main__":
    sys.exit(main())

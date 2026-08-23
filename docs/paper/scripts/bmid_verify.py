"""Decisive checks on the base_model_id claim.

1. Recompute every base_model_id from the config's own warm-up keys.  If the
   recomputation matches the stored string, the stored string is trustworthy.
2. Counterfactual: take a trained-arm config, flip ONLY warmup_epochs 1->30,
   recompute.  If it lands on the clipper's hash, warm-up depth is the sole
   cause of the split.
3. Count sharing restricted to the trained trio (tralo/fioretto/hounie) --
   the comparison the headline stratified result actually makes.
4. Count sharing restricted to (dataset, backbone, seed, cap) cells.
"""
import copy
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.expanduser("~/OptimizationLoss"))
from src.config_generators.generate_configs import compute_base_model_id  # noqa: E402

ROOTS = {
    "paper_final": "results/pending_runs/paper_final",
    "headroom_b30": "results/headroom/headroom_b30",
    "headroom_noceskip": "results/headroom/headroom_b30_lrc0.0001_noceskip",
}
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]


def load(root):
    rows = []
    for dp, _, fn in os.walk(root):
        if "config.json" not in fn:
            continue
        c = json.load(open(os.path.join(dp, "config.json")))
        hp = c["hyperparams"]
        dc = c.get("dataset_config") or {}
        recomputed = compute_base_model_id(
            c["model_name"], hp, c["dataset_mode"],
            dc.get("data_dir"), dc)
        rows.append({
            "dataset": c["dataset_mode"], "model": c["model_name"],
            "method": c["methodology"], "cap": c.get("constraint_tag"),
            "seed": hp.get("seed"), "warmup": hp.get("warmup_epochs"),
            "bmid": c.get("base_model_id"), "recomputed": recomputed,
            "match": c.get("base_model_id") == recomputed,
            "cfg": c,
        })
    return pd.DataFrame(rows)


def main():
    os.chdir(os.path.expanduser("~/OptimizationLoss"))
    D = {k: load(v) for k, v in ROOTS.items()}

    print("=" * 100)
    print("1. Is the stored base_model_id reproducible from the config's own warm-up keys?")
    print("=" * 100)
    for k, df in D.items():
        print("  %-20s  recomputed == stored in %d / %d configs" %
              (k, int(df["match"].sum()), len(df)))
    bad = pd.concat([df[~df["match"]].assign(root=k) for k, df in D.items()])
    if len(bad):
        print("  MISMATCHES:")
        print(bad[["root", "dataset", "model", "method", "seed", "warmup",
                   "bmid", "recomputed"]].to_string(index=False))
    print()

    print("=" * 100)
    print("2. COUNTERFACTUAL: flip ONLY warmup_epochs on a trained arm 1 -> 30.")
    print("   Does it reproduce the clipper's hash?  (tests the causal attribution)")
    print("=" * 100)
    hb = D["headroom_b30"]
    ok, tot = 0, 0
    for (ds, mo, sd), g in hb.groupby(["dataset", "model", "seed"]):
        tr = g[g.method.isin(TRAINED)].iloc[0]
        cl = g[g.method.isin(CLIP)].iloc[0]
        c = copy.deepcopy(tr["cfg"])
        c["hyperparams"]["warmup_epochs"] = int(cl["warmup"])
        dc = c.get("dataset_config") or {}
        h = compute_base_model_id(c["model_name"], c["hyperparams"],
                                  c["dataset_mode"], dc.get("data_dir"), dc)
        tot += 1
        ok += int(h == cl["bmid"])
        if tot <= 3:
            print("   %-12s %-13s seed%s  trained(w=%s) %s" %
                  (ds, mo, sd, tr["warmup"], tr["bmid"]))
            print("   %-12s %-13s        +warmup->%s   => %s   clipper actual %s   %s" %
                  ("", "", cl["warmup"], h, cl["bmid"], "MATCH" if h == cl["bmid"] else "DIFFER"))
    print("   warm-up depth alone explains the hash split in %d / %d groups" % (ok, tot))
    print()

    print("=" * 100)
    print("3. Sharing restricted to the TRAINED TRIO (the headline comparison)")
    print("=" * 100)
    for k, df in D.items():
        t = df[df.method.isin(TRAINED)]
        if t.empty:
            continue
        n_sh, n_tot = 0, 0
        for _, g in t.groupby(["dataset", "model", "seed"]):
            n_tot += 1
            n_sh += int(g.bmid.nunique() == 1)
        print("  %-20s  trained-trio groups with a SINGLE base_model_id: %d / %d"
              % (k, n_sh, n_tot))
    # and the clippers among themselves
    for k, df in D.items():
        c = df[df.method.isin(CLIP)]
        if c.empty:
            continue
        n_sh, n_tot = 0, 0
        for _, g in c.groupby(["dataset", "model", "seed"]):
            n_tot += 1
            n_sh += int(g.bmid.nunique() == 1)
        print("  %-20s  clipper-pair groups with a SINGLE base_model_id: %d / %d"
              % (k, n_sh, n_tot))
    print()

    print("=" * 100)
    print("4. Sharing counted per ATOMIC CELL x seed (dataset, backbone, cap, seed)")
    print("=" * 100)
    for k, df in D.items():
        n_sh, n_tot = 0, 0
        for _, g in df.groupby(["dataset", "model", "cap", "seed"]):
            n_tot += 1
            n_sh += int(g.bmid.nunique() == 1)
        print("  %-20s  %d / %d cell-seeds share one base_model_id" % (k, n_sh, n_tot))
    print()

    print("=" * 100)
    print("5. Compute actually spent: warmup_epochs + constraint_epochs by method")
    print("=" * 100)
    for k, df in D.items():
        print("  %s" % k)
        for m, g in df.groupby("method"):
            w = sorted({c["hyperparams"]["warmup_epochs"] for c in g.cfg})
            ce = sorted({c["hyperparams"]["constraint_epochs"] for c in g.cfg})
            print("     %-14s warmup=%s constraint_epochs=%s" % (m, w, ce))


if __name__ == "__main__":
    sys.exit(main())

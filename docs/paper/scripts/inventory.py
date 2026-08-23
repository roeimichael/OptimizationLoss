"""Inventory every campaign on disk: regime (warmup/cepochs/lr/ce_skip) x method mix.

Goal: find the OLD campaigns where the dual methods (fioretto_ldf, hounie_rcl)
were compared against the post-hoc clippers (heuristic, danits_lp) at
warmup_epochs == 50, so we can see whether the ordering really flipped.

    python paper/scripts/inventory.py
"""
import glob
import json
import os
import sys

import pandas as pd

ROOTS = ["results", "archive", "archive_experiments", "extra_experiments",
         "newdirections"]


def campaign_of(path, root):
    """Everything between the root and the first level that looks like a method."""
    rel = os.path.relpath(os.path.dirname(path), root)
    return root + "/" + "/".join(rel.split(os.sep)[:2])


def main():
    rows = []
    for root in ROOTS:
        for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                continue
            hp = cfg.get("hyperparams") or {}
            d = os.path.dirname(cfg_path)
            rows.append({
                "root": root,
                "campaign": campaign_of(cfg_path, root),
                "sweep": cfg.get("sweep"),
                "arm": cfg.get("arm"),
                "dataset": cfg.get("dataset_mode"),
                "model": cfg.get("model_name"),
                "cap": cfg.get("constraint_tag"),
                "method": cfg.get("methodology"),
                "warmup": hp.get("warmup_epochs"),
                "cepochs": hp.get("constraint_epochs"),
                "lr": hp.get("lr"),
                "lr_c": hp.get("lr_constraint"),
                "ce_skip": hp.get("enable_ce_skip"),
                "seed": hp.get("seed"),
                "has_raw": os.path.exists(os.path.join(d, "final_predictions_raw.csv")),
                "has_fin": os.path.exists(os.path.join(d, "final_predictions.csv")),
                "path": d,
            })
    t = pd.DataFrame(rows)
    t.to_csv("paper/scripts/out_inventory.csv", index=False)
    print("total configs: %d" % len(t))

    # Which campaigns contain BOTH a dual and a clipper, so a comparison exists?
    DUAL = {"fioretto_ldf", "hounie_rcl"}
    CLIP = {"heuristic", "danits_lp"}
    print("\n" + "=" * 118)
    print("CAMPAIGNS CONTAINING BOTH A DUAL AND A CLIPPER (a dual-vs-clip comparison exists)")
    print("=" * 118)
    out = []
    for camp, g in t.groupby("campaign"):
        ms = set(g.method.dropna())
        if not (ms & DUAL) or not (ms & CLIP):
            continue
        scor = g[g.has_raw & g.has_fin]
        out.append({
            "campaign": camp,
            "n": len(g),
            "n_scorable": len(scor),
            "warmups": ",".join(sorted({str(x) for x in g.warmup.dropna()})),
            "cepochs": ",".join(sorted({str(x) for x in g.cepochs.dropna()})),
            "ce_skip": ",".join(sorted({str(x) for x in g.ce_skip.dropna().unique()})) or "-",
            "lr_c": ",".join(sorted({str(x) for x in g.lr_c.dropna().unique()})) or "-",
            "datasets": ",".join(sorted(set(g.dataset.dropna()))),
            "methods": ",".join(sorted(ms)),
        })
    o = pd.DataFrame(out).sort_values("n", ascending=False)
    print(o.to_string(index=False))

    print("\n" + "=" * 118)
    print("ALL CAMPAIGNS, warmup mix (for orientation)")
    print("=" * 118)
    allc = t.groupby("campaign").agg(
        n=("method", "size"),
        warmups=("warmup", lambda s: ",".join(sorted({str(x) for x in s.dropna()}))),
        methods=("method", lambda s: ",".join(sorted(set(s.dropna())))),
    ).sort_values("n", ascending=False)
    print(allc.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

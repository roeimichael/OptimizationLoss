"""Does TraLO ever satisfy the cap DURING training, or does post-hoc do the work?

One cell's trajectory said the count runs 146-202 against K=67 for the whole run
and ends at 114 -- still violating -- so the constraint phase is a directional
nudge and post-hoc adjustment is what actually enforces the count. One cell is
an anecdote. This is the census.

Reading the sparse log is safe for exactly this question. TraLO writes a row
when `(epoch+1) % 5 == 0 OR is_satisfied OR epoch == warmup_epochs`, so every
satisfying epoch is guaranteed to be logged even though most epochs are not.
"Did it ever satisfy" is therefore answerable exactly; "how many epochs did it
run" is NOT, which is the trap that produced the retracted epoch-asymmetry
claim. Only the first question is asked here.

Counts are hard counts (argmax), which is what satisfaction is verified on --
not the soft counts the loss actually descends.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def census(root):
    rows = []
    for cfg_path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg_path)
        try:
            cfg = json.load(open(cfg_path))
        except ValueError:
            continue
        if cfg.get("methodology") != "tralo":
            continue          # only tralo writes Hard_Class*/Global_Satisfied
        log = os.path.join(d, "training_log.csv")
        if not os.path.exists(log):
            continue
        try:
            t = pd.read_csv(log)
        except Exception:
            continue
        if "Epoch" not in t.columns or t.empty:
            continue

        cc = cfg["dataset_config"]["constrained_class"]
        hard, lim = "Hard_Class%d" % cc, "Limit_Class%d" % cc
        if hard not in t.columns or lim not in t.columns:
            continue
        warm = cfg["hyperparams"].get("warmup_epochs", 0)
        c = t[t.Epoch >= warm].copy()          # constraint phase only
        if c.empty:
            continue

        K = pd.to_numeric(c[lim], errors="coerce").replace([float("inf")], pd.NA).dropna()
        K = float(K.iloc[0]) if len(K) else float("nan")
        h = pd.to_numeric(c[hard], errors="coerce")

        g = pd.to_numeric(c.get("Global_Satisfied", 0), errors="coerce").fillna(0)
        l = pd.to_numeric(c.get("Local_Satisfied", 0), errors="coerce").fillna(0)
        both = (g > 0) & (l > 0)

        rows.append({
            "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
            "cap": cfg["constraint_tag"], "arm": cfg.get("arm"),
            "seed": cfg["hyperparams"].get("seed"),
            "K": K,
            "rows_logged": len(c),
            "ever_global": bool((g > 0).any()),
            "ever_joint": bool(both.any()),
            "final_global": bool(g.iloc[-1] > 0),
            "final_joint": bool(both.iloc[-1]),
            "count_first": float(h.iloc[0]), "count_final": float(h.iloc[-1]),
            "count_min": float(h.min()), "count_max": float(h.max()),
            "fill_final": float(h.iloc[-1]) / K if K == K and K else float("nan"),
            "fill_min": float(h.min()) / K if K == K and K else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    d = census(args.root)
    if d.empty:
        print("no tralo runs with a readable log under", args.root)
        return 1
    print("%d TraLO runs under %s" % (len(d), args.root))

    print()
    print("=" * 92)
    print("DOES THE CONSTRAINT PHASE EVER REACH FEASIBILITY?")
    print("=" * 92)
    print("  ever satisfied GLOBAL cap during training : %3d / %3d  (%.0f%%)"
          % (d.ever_global.sum(), len(d), 100 * d.ever_global.mean()))
    print("  ever satisfied BOTH caps during training  : %3d / %3d  (%.0f%%)"
          % (d.ever_joint.sum(), len(d), 100 * d.ever_joint.mean()))
    print("  still feasible at the LAST logged epoch   : %3d / %3d  (%.0f%%)"
          % (d.final_joint.sum(), len(d), 100 * d.final_joint.mean()))
    print()
    print("  -> the gap between 'ever' and 'final' is the run touching feasibility")
    print("     and then leaving it; post-hoc adjustment ships whatever epoch 30 lands on.")

    print()
    print("=" * 92)
    print("BY CELL  (fill = hard count / K at the end of training; 1.0 = exactly at cap)")
    print("=" * 92)
    t = d.groupby(["dataset", "model", "cap"]).agg(
        n=("seed", "size"),
        ever_global=("ever_global", "mean"), ever_joint=("ever_joint", "mean"),
        final_joint=("final_joint", "mean"),
        K=("K", "first"),
        fill_final=("fill_final", "mean"), fill_min=("fill_min", "mean"),
        count_final=("count_final", "mean"), count_max=("count_max", "mean"))
    print(t.round(3).to_string())

    print()
    print("=" * 92)
    print("HOW FAR OVER THE CAP DOES TRAINING LEAVE IT?  (this is post-hoc's workload)")
    print("=" * 92)
    over = d[d.fill_final > 1.0]
    print("  runs ending ABOVE the cap : %d / %d" % (len(over), len(d)))
    if len(over):
        print("  median overshoot          : %.2fx K   (max %.2fx)"
              % (over.fill_final.median(), over.fill_final.max()))
    under = d[d.fill_final <= 1.0]
    print("  runs ending at or below   : %d / %d   median fill %.2f"
          % (len(under), len(d), under.fill_final.median() if len(under) else float("nan")))

    if args.csv:
        d.to_csv(args.csv, index=False)
        print("\nwrote", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

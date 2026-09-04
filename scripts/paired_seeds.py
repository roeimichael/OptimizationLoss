"""Paired-by-seed read of a scan: each arm against its OWN control, same seed.

WHY PAIRED. The seed is the dominant term in every comparison this project
makes, so an unpaired pooled-std read is banned here. Measured 2026-08-20 on a
multi-class 4-epoch scan: the same arm scored +0.0414 macro-F1 against its
control at seed 1 and -0.0302 at seed 2. The seed-1 number alone would have
justified an 8-hour campaign.

WHY THE CONTROL IS THE lambda=0 ARM AND NOT `clip`. The null shares the cached
warm-up, the allocator, the schedule and the seed, and differs only in lambda,
so the difference IS the constraint. `clip` differs in training path AND
allocator, and the allocator alone is worth ~0.0014 here.

READ d capF1 SEPARATELY FROM d macroF1. They have wildly different precision.
Paired over two seeds, d capF1 came out -0.0149 and -0.0150 -- sd 0.0000 --
while d macroF1 had sd 0.0507. macro-F1 is dominated by the uncapped classes,
which swing with the seed; the capped classes are what the method is actually
about and they are measurable. A mean whose sd exceeds it is not a result.

AND KNOW THE CEILING. Recall on a capped class is hard-limited to K/n_true, so
F1 there has an analytic maximum. At L30_G20 on dermmnist the entire headroom
is 0.038 and 0.052 -- smaller than the seed noise, which means that cell cannot
resolve a win even in principle. Check the ceiling before spending GPU on a cell.

    python -m scripts.paired_seeds /tmp/seedval --capped 2 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts import quarantine


def metrics(d, capped):
    p = pd.read_csv(d / "final_predictions.csv")
    y = p["True_Label"].to_numpy()
    pred = p["Predicted_Label"].to_numpy()
    prob = p[[c for c in p.columns if c.startswith("Prob_Class_")]].to_numpy()
    lab = sorted(set(y.tolist()))
    per = dict(zip(lab, f1_score(y, pred, average=None, labels=lab,
                                 zero_division=0)))
    unc = [c for c in lab if c not in capped]
    cap = [c for c in lab if c in capped]
    oh = np.eye(prob.shape[1])[y][:, lab]
    return {
        "macroF1": float(np.mean(list(per.values()))),
        "capF1": float(np.mean([per[c] for c in cap])) if cap else float("nan"),
        "uncF1": float(np.mean([per[c] for c in unc])) if unc else float("nan"),
        "auroc": float(roc_auc_score(oh, prob[:, lab], average="macro")),
    }


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root")
    a.add_argument("--capped", type=int, nargs="+", default=[2, 4])
    a.add_argument("--control", default="null")
    a.add_argument("--allow-quarantined", action="store_true",
                   help="read a campaign `scripts.quarantine` marked dead")
    args = a.parse_args()

    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE, so a
    # marker on a dead campaign prevented nothing here. No fallback import --
    # if the gate cannot load, the tool must break.
    from scripts.quarantine import gate
    blocked, dead = gate([args.root], args.allow_quarantined, "read")
    if blocked:
        return 1

    root = Path(args.root)
    runs = [d for d in root.iterdir()
            if d.is_dir() and d.name.startswith("seed")
            and (d / "final_predictions.csv").exists()]
    # A PARTIAL marker names arms whose contrasts are disqualified, and every
    # line this tool prints is `arm - control` at a fixed seed. Announcing is
    # not enforcing, so the enumerated run directories go through the filter.
    #
    # ⚠️ THIS ROOT IS THE FLAT `seed<N>_<arm>` LAYOUT the ad-hoc drivers write,
    # which `arm_of_run` cannot parse -- so on a partially quarantined root the
    # filter EXCLUDES everything rather than guessing, and says so. That is the
    # same fail-closed direction `score_scan` takes on the same layout, and it
    # is deliberate: keeping an unclassifiable run fails the marker open.
    runs = [Path(x) for x in quarantine.drop_dead_runs(
        [str(d) for d in runs], dead, label="seed run")]
    if not runs:
        # This tool reads the FLAT `seed<N>_<arm>` layout the ad-hoc server
        # drivers write. A gen_campaign tree is
        # <root>/<model>/<dataset>/<cap>/<arm>/seed_<N>, and pointing this at
        # one gives "no completed runs" -- which reads as "nothing finished"
        # rather than "wrong tool", and that is the kind of message this
        # project has abandoned arms over.
        if list(root.rglob("final_predictions.csv")):
            print("no seed<N>_<arm> directories under %s, but there ARE" % root)
            print("completed runs deeper in the tree -- this looks like a")
            print("gen_campaign campaign, which this tool cannot read.")
            print()
            print("Use the campaign scorer, which keys on the CELL and will not")
            print("pool across cap levels:")
            print("    python -m scripts.full_panel %s --control clip" % root)
            print("    python -m scripts.full_panel %s --control tralo_null" % root)
            print()
            print("`clip` is the stronger quality bar and is the headline read;")
            print("`tralo_null` attributes a delta to the constraint rather")
            print("than to the regime. Run BOTH.")
            return 1
        print("no completed runs under %s" % root)
        return 1
    seeds = sorted({int(d.name.split("_")[0][4:]) for d in runs})
    arms = sorted({d.name.split("_", 1)[1] for d in runs} - {args.control})

    print("PAIRED BY SEED -- arm minus its own `%s` at the SAME seed." % args.control)
    print("root: %s   capped classes: %s\n"
          % (root, ",".join(str(c) for c in args.capped)))
    keys = ("macroF1", "capF1", "uncF1", "auroc")
    for arm in arms:
        rows = []
        for s in seeds:
            x = root / ("seed%d_%s" % (s, arm))
            c = root / ("seed%d_%s" % (s, args.control))
            if not ((x / "final_predictions.csv").exists()
                    and (c / "final_predictions.csv").exists()):
                continue
            mx, mc = metrics(x, args.capped), metrics(c, args.capped)
            rows.append((s, {k: mx[k] - mc[k] for k in keys}))
        if not rows:
            continue
        print("%s  (n=%d seeds)" % (arm.upper(), len(rows)))
        print("  %-6s %11s %11s %11s %11s"
              % ("seed", "d macroF1", "d capF1", "d uncF1", "d AUROC"))
        for s, d in rows:
            print("  %-6d %+11.4f %+11.4f %+11.4f %+11.4f"
                  % (s, d["macroF1"], d["capF1"], d["uncF1"], d["auroc"]))
        mean = {k: float(np.mean([d[k] for _, d in rows])) for k in keys}
        sd = {k: (float(np.std([d[k] for _, d in rows], ddof=1))
                  if len(rows) > 1 else float("nan")) for k in keys}
        print("  %-6s %+11.4f %+11.4f %+11.4f %+11.4f"
              % ("MEAN", *[mean[k] for k in keys]))
        print("  %-6s %11.4f %11.4f %11.4f %11.4f"
              % ("sd", *[sd[k] for k in keys]))
        for k in ("macroF1", "capF1"):
            pos = sum(1 for _, d in rows if d[k] > 0)
            flag = "" if len(rows) < 2 or abs(mean[k]) > sd[k] else "   <- sd exceeds the mean"
            print("  %-9s positive in %d/%d seeds%s" % (k, pos, len(rows), flag))
        print()
    print("Count seeds, not p-values: at n=2 a Wilcoxon floors at p=0.5, so a")
    print("small n reads as a tie no matter how large the effect is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""TraLO vs ALM over the full grid -- the regime map the blind review asked for.

The manuscript reported ALM on six OctMNIST tight-cap cells, which is what the
B3 probe had run. The expansion (300 runs) now covers 3 datasets x 3 backbones x
9 caps x 4 seeds, so the comparison can be reported everywhere instead of only
where we lead -- including wherever ALM leads us.

Pairing is by (dataset, model, cap, seed): the ALM configs were cloned from the
frozen paper_final Fioretto-LDF configs and differ only in the dual rule, so
they share the CE warmup cache with the TraLO runs of the same cell.

Run:  python paper/scripts/analyze_alm.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "corpus", "corpus_final.csv")
ALM = os.path.join(ROOT, "data", "corpus", "alm_results.csv")

TIGHT = ["L30_G30", "L40_G40"]
MAIN_BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]


def sweep_family(s):
    """Collapse dispatch-split sweep names to the campaign they belong to.

    `octmnist_MobileNetV2_s1..s4` are ONE campaign whose four seeds were split
    across four sweep names to run in parallel: identical warmup (50), the same
    five caps and the same six methods, one seed each. Merging them is a naming
    fix, not a pool across campaigns -- without it the four-seed filter discards
    every OctMNIST MobileNetV2 cell.
    """
    return s.str.replace(r"_s[1-4]$", "", regex=True)


def load(metric="cc_f1"):
    tr = pd.read_csv(CORPUS)
    tr = tr[(tr.sweep == "paper_final")][
        ["dataset", "model", "constraint_tag", "seed", "method", metric]]
    alm = pd.read_csv(ALM)
    alm = alm[alm.method == "fioretto_alm"][
        ["dataset", "model", "constraint_tag", "seed", metric]]
    alm = alm.rename(columns={metric: "alm"})

    key = ["dataset", "model", "constraint_tag", "seed"]
    out = {}
    for m in ("tralo", "fioretto_ldf"):
        sub = tr[tr.method == m][key + [metric]].rename(columns={metric: m})
        out[m] = sub
    df = out["tralo"].merge(out["fioretto_ldf"], on=key).merge(alm, on=key)
    # Only cells with all four seeds on BOTH arms are comparable.
    n = df.groupby(key[:3])["seed"].transform("nunique")
    return df[n == 4].copy()


def cells(df, arm="alm"):
    """Per-cell paired mean gap TraLO - arm, plus the seed-level winrate."""
    df = df.copy()
    df["gap"] = df["tralo"] - df[arm]
    g = df.groupby(["dataset", "model", "constraint_tag"]).agg(
        gap=("gap", "mean"), n=("gap", "size"),
        wins=("gap", lambda s: int((s > 0).sum())))
    return g.reset_index()


def main():
    df = load()
    df = df[df.model.isin(MAIN_BB)]
    c = cells(df)
    print("complete 4-seed cells: %d\n" % len(c))

    tight = c[(c.dataset == "octmnist") & c.constraint_tag.isin(TIGHT)]
    rest = c.drop(tight.index)
    print("REPORTED BAND (OctMNIST tight, the six cells in the manuscript)")
    print(tight[["model", "constraint_tag", "gap", "wins", "n"]].to_string(index=False))
    print("  mean %+.4f\n" % tight.gap.mean())

    print("EVERY OTHER COMPLETE CELL: n=%d, mean %+.4f" % (len(rest), rest.gap.mean()))
    losses = rest[rest.gap < -0.005].sort_values("gap")
    print("  cells where ALM leads by more than the 0.005 tie band: %d" % len(losses))
    print(losses[["dataset", "model", "constraint_tag", "gap", "wins"]].to_string(index=False))

    print("\nBY REGION (dataset x backbone, mean over that cell's caps)")
    reg = c.groupby(["dataset", "model"]).agg(gap=("gap", "mean"), cells=("gap", "size"))
    print(reg.round(4).to_string())

    print("\nBY CAP, OctMNIST x MobileNetV3 (the reviewer's loss region)")
    lr = c[(c.dataset == "octmnist") & (c.model == "MobileNetV3")]
    print(lr[["constraint_tag", "gap", "wins", "n"]].to_string(index=False))

    print("\nALM vs Fioretto-LDF (is ALM actually the stronger dual?)")
    cf = cells(df, "fioretto_ldf")
    both = c.merge(cf, on=["dataset", "model", "constraint_tag"],
                   suffixes=("_alm", "_fio"))
    both["alm_minus_fio"] = both.gap_fio - both.gap_alm
    t = both[(both.dataset == "octmnist") & both.constraint_tag.isin(TIGHT)]
    print("  tight band : ALM - Fioretto = %+.4f" % t.alm_minus_fio.mean())
    print("  whole grid : ALM - Fioretto = %+.4f" % both.alm_minus_fio.mean())

    print("\nHEADLINE SENTENCE INPUTS")
    print("  six reported cells      : %+.4f" % tight.gap.mean())
    print("  all %3d complete cells  : %+.4f" % (len(c), c.gap.mean()))
    print("  cells TraLO leads >.005 : %d" % (c.gap > 0.005).sum())
    print("  cells ALM leads   >.005 : %d" % (c.gap < -0.005).sum())
    print("  ties within +/-.005     : %d" % (c.gap.abs() <= 0.005).sum())


if __name__ == "__main__":
    main()

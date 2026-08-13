"""What the paper needs vs. what has actually been run.

Reads paper/data/manifest/experiments.csv (see build_experiment_manifest.py) and
checks it against the grid the paper claims, so gaps surface as a list of
missing cells rather than as a surprise in a table caption.

The target after the professor's second round:

    3 datasets x 4 backbones x 9 symmetric caps x 7 methods x 4 seeds
    at warmup 50 = 3024 runs

which is the frozen 1944-run grid plus a 4th backbone (MobileNetV2) and a 7th
method (ALM). The three cap levels the headline tables print (L30/L50/L70) are
reported separately, because those gate Tables 1-2 while the rest only gate the
nine-cap figures.

Run from the repo root:
    python paper/scripts/coverage_report.py [--csv gaps.csv]
"""

import argparse
import itertools
import os

import pandas as pd

MANIFEST = os.path.join("paper", "data", "manifest", "experiments.csv")

DATASETS = ["octmnist", "dermmnist", "tissuemnist"]
BACKBONES = ["MobileNetV3", "RegNetY400MF", "ViTB16", "MobileNetV2"]
CAPS = ["L%d0_G%d0" % (i, i) for i in range(1, 10)]
TABLE_CAPS = ["L30_G30", "L50_G50", "L70_G70"]
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "fioretto_alm", "heuristic", "danits_lp"]
SEEDS = [1, 2, 3, 4]
WARMUP = 50


def load():
    if not os.path.exists(MANIFEST):
        raise SystemExit("missing %s -- run build_experiment_manifest.py on the "
                         "server first, then copy it here" % MANIFEST)
    df = pd.read_csv(MANIFEST, low_memory=False)
    df = df[(df.status == "completed") & (df.warmup == WARMUP)]
    df = df[df.cc_f1.notna()]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="write the missing cells here")
    args = ap.parse_args()

    df = load()
    have = set(zip(df.dataset, df.model, df.cap, df.method, df.seed))

    missing = []
    for d, b, c, m, s in itertools.product(DATASETS, BACKBONES, CAPS, METHODS, SEEDS):
        if (d, b, c, m, s) not in have:
            missing.append({"dataset": d, "model": b, "cap": c,
                            "method": m, "seed": s,
                            "gates_headline_table": c in TABLE_CAPS})

    total = len(DATASETS) * len(BACKBONES) * len(CAPS) * len(METHODS) * len(SEEDS)
    print("=" * 74)
    print("TARGET GRID  %d datasets x %d backbones x %d caps x %d methods x %d seeds"
          % (len(DATASETS), len(BACKBONES), len(CAPS), len(METHODS), len(SEEDS)))
    print("             = %d runs at warmup %d" % (total, WARMUP))
    print("HAVE         %d" % (total - len(missing)))
    print("MISSING      %d" % len(missing))
    print("=" * 74)

    if not missing:
        print("\nNo gaps. The full expanded grid is complete.")
        return

    md = pd.DataFrame(missing)

    print("\n--- missing by method x backbone (all nine caps) ---")
    print(md.pivot_table(index="method", columns="model", values="seed",
                         aggfunc="count", fill_value=0).to_string())

    head = md[md.gates_headline_table]
    print("\n--- of those, gating Tables 1-2 (caps L30/L50/L70): %d ---" % len(head))
    if len(head):
        print(head.pivot_table(index="method", columns="model", values="seed",
                               aggfunc="count", fill_value=0).to_string())
    else:
        print("none -- Tables 1-2 can be built from what is already on disk")

    print("\n--- missing by cap ---")
    print(md.cap.value_counts().sort_index().to_string())

    if args.csv:
        md.sort_values(["method", "model", "dataset", "cap", "seed"]).to_csv(
            args.csv, index=False)
        print("\nwrote %s" % args.csv)


if __name__ == "__main__":
    main()

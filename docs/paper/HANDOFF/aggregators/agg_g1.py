"""Aggregate G1 — MobileNetV2 non-saturated 2nd backbone.

Emits:
    paper/HANDOFF/tables/g1_mobilenetv2_raw.csv     (per-seed rows)
    paper/HANDOFF/tables/g1_mobilenetv2_summary.csv (cell means with paired diff
                                                     vs MobileNetV3 headline)

This is what slots into a future Table C-prime in main.tex.
"""
import csv
from pathlib import Path

from agg_common import collect_sweep, write_csv, summarize, REPO, COLUMNS

SWEEP = "results/pending_runs/g1_mobilenetv2"
OUT_RAW = REPO / "paper/HANDOFF/tables/g1_mobilenetv2_raw.csv"
OUT_SUM = REPO / "paper/HANDOFF/tables/g1_mobilenetv2_summary.csv"

GROUP_KEYS = ["ds", "model", "cls", "grp", "tight"]


def main():
    rows = collect_sweep(SWEEP, phase_tag="g1_mobilenetv2")
    print(f"Collected {len(rows)} per-seed rows from {SWEEP}")
    write_csv(rows, OUT_RAW)

    summary = summarize(rows, GROUP_KEYS)
    print(f"Summarized {len(summary)} cells")
    fields = (GROUP_KEYS + ["method", "n_seeds"]
              + [f"{c}_{s}" for c in ("f1m", "f1w", "acc", "ece", "brier", "flips", "sat")
                 for s in ("mean", "std")])
    with open(OUT_SUM, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {OUT_SUM}")


if __name__ == "__main__":
    main()

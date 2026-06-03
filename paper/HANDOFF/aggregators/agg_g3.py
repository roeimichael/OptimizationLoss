"""Aggregate G3 — multi-class robustness on TissueMNIST."""
import csv
from pathlib import Path
from agg_common import collect_sweep, write_csv, summarize, REPO

SWEEP = "results/pending_runs/g3_multiclass_tissue"
OUT_RAW = REPO / "paper/HANDOFF/tables/g3_multiclass_tissue_raw.csv"
OUT_SUM = REPO / "paper/HANDOFF/tables/g3_multiclass_tissue_summary.csv"
GROUP_KEYS = ["ds", "model", "cls", "grp", "tight"]


def main():
    rows = collect_sweep(SWEEP, phase_tag="g3_multiclass_tissue")
    print(f"Collected {len(rows)} per-seed rows.")
    write_csv(rows, OUT_RAW)
    summary = summarize(rows, GROUP_KEYS)
    fields = (GROUP_KEYS + ["method", "n_seeds"]
              + [f"{c}_{s}" for c in ("f1m", "f1w", "acc", "ece", "brier", "flips", "sat")
                 for s in ("mean", "std")])
    with open(OUT_SUM, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {OUT_SUM}")


if __name__ == "__main__":
    main()

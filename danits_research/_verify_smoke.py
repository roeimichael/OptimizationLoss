"""Print a summary table of the generated smoke experiments for user review."""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status


def main() -> int:
    by_status = get_experiments_by_status("results/pending_runs")
    pending = by_status["pending"]
    completed = by_status["completed"]

    print(f"  pending   : {len(pending)}")
    print(f"  completed : {len(completed)}")
    print()

    rows = []
    for exp_path, cfg in pending:
        path_parts = Path(exp_path).parts
        # results / pending_runs / scenario / tag / model / method / slice
        idx = path_parts.index("pending_runs")
        scenario = path_parts[idx + 1]
        tag = path_parts[idx + 2]
        model = path_parts[idx + 3]
        method = path_parts[idx + 4]
        slice_name = path_parts[idx + 5]
        ds = cfg["dataset_config"]
        rows.append({
            "scenario": scenario,
            "constrained_class": ds["constrained_class"],
            "constraint_tag": tag,
            "local_pct": cfg["constraint"][0],
            "global_pct": cfg["constraint"][1],
            "model": model,
            "method": method,
            "slice": slice_name,
            "base_model_id": cfg["base_model_id"],
            "warmup_epochs": cfg["hyperparams"]["warmup_epochs"],
            "constraint_epochs": cfg["hyperparams"]["constraint_epochs"],
            "methodology_in_cfg": cfg["methodology"],
        })

    rows.sort(key=lambda r: (r["scenario"], r["constraint_tag"], r["method"]))

    # --- table ---
    header = (
        f"  {'scenario':<17s} | {'tag':<8s} | {'class(es)':<13s} | "
        f"{'method':<13s} | {'L%':>4s} | {'G%':>4s} | {'epochs':>7s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        classes = str(r["constrained_class"])
        epochs = (f"wu={r['warmup_epochs']}+"
                  f"cons={r['constraint_epochs']}"
                  if r["method"] == "our_approach"
                  else f"wu={r['warmup_epochs']} only")
        print(f"  {r['scenario']:<17s} | {r['constraint_tag']:<8s} | "
              f"{classes:<13s} | {r['method']:<13s} | "
              f"{r['local_pct']:>4.2f} | {r['global_pct']:>4.2f} | {epochs:>7s}")

    # --- base_model_id sanity ---
    bids_by_scenario: dict = defaultdict(set)
    for r in rows:
        bids_by_scenario[r["scenario"]].add(r["base_model_id"])
    print()
    print("  base_model_id per scenario (all methods should share one hash):")
    for sc, ids in bids_by_scenario.items():
        marker = "OK" if len(ids) == 1 else "DIVERGED"
        print(f"    {sc:<17s} : {list(ids)[0] if len(ids) == 1 else list(ids)}  [{marker}]")

    # unique warmup trainings needed
    unique_bids = set()
    for r in rows:
        unique_bids.add(r["base_model_id"])
    print()
    print(f"  unique warmup models to train: {len(unique_bids)}")
    for b in sorted(unique_bids):
        print(f"    - {b}")

    # breakdown by method
    print()
    print("  count by methodology:")
    mc: dict = defaultdict(int)
    for r in rows:
        mc[r["method"]] += 1
    for m, c in sorted(mc.items()):
        print(f"    {m:<13s} : {c}")

    # breakdown by scenario
    print()
    print("  count by scenario:")
    sc: dict = defaultdict(int)
    for r in rows:
        sc[r["scenario"]] += 1
    for s, c in sorted(sc.items()):
        print(f"    {s:<17s} : {c}")

    # --- model_cache emptiness sanity ---
    cache_dir = Path("model_cache")
    pts = list(cache_dir.glob("*.pt"))
    print()
    print(f"  model_cache/ .pt files: {len(pts)}  "
          f"{'(empty - fresh training will run)' if not pts else '(NOT empty - warmup will load cached!)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

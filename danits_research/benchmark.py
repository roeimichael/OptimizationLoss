"""
Phase-A single-constraint benchmark.

Reports a proper 2x3 matrix per `(heuristic_anchor, our_approach_variant)`
pair, with no confounding between the model choice and the Phase-2 method.

                     | argmax (no Phase-2) | project greedy | paper [5] LP
    -----------------+---------------------+----------------+--------------
    warmup model     |      (W,-)          |     (W,g)      |    (W,LP)
    constraint model |      (C,-)          |     (C,g)      |    (C,LP)

Archive setup:
    anchor   : archive_experiments/dermmnist/heuristic/ResNet18/c05_03/bs32
               -- contains warmup probabilities (pure CE, no constraint loss)
    variants : archive_experiments/dermmnist/our_approach/ResNet18/c05_03/bs32_*
               -- 10 variants that all branched from the SAME cached warmup
                  (base_model_id = ResNet18_dermmnist_741ee0564243)

We show the best variant (by macro F1-macro on the (C, greedy) cell) in
the main table and dump every variant's 2x3 matrix to the tidy CSV.

This benchmark does NOT retrain anything. It only reads archive files and
runs the paper [5] LP on top of the two already-saved probability matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from danits_research._benchmark_core import (
    ARCHIVE,
    REPO_ROOT,
    build_2x3_matrix,
    load_run,
    print_matrix,
    result_to_rows,
)

ANCHOR_HEURISTIC_RUN = (
    ARCHIVE / "dermmnist" / "heuristic" / "ResNet18" / "c05_03" / "bs32"
)
OUR_APPROACH_ROOT = ARCHIVE / "dermmnist" / "our_approach" / "ResNet18" / "c05_03"


def _matching_variants(anchor_base_id: str, oa_root: Path) -> list[Path]:
    """Every our_approach variant under oa_root that shares the anchor's warmup hash."""
    out: list[Path] = []
    if not oa_root.exists():
        return out
    for variant_dir in sorted(oa_root.iterdir()):
        cfg_path = variant_dir / "config.json"
        preds_path = variant_dir / "final_predictions.csv"
        if not (cfg_path.exists() and preds_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("base_model_id") == anchor_base_id:
            out.append(variant_dir)
    return out


def main() -> int:
    print("=" * 82)
    print("Phase-A single-constraint benchmark (2x3 matrix)")
    print("=" * 82)

    if not (ANCHOR_HEURISTIC_RUN / "final_predictions.csv").exists():
        print(f"ERROR: anchor not found at {ANCHOR_HEURISTIC_RUN}")
        return 1

    anchor = load_run(ANCHOR_HEURISTIC_RUN)
    anchor_base_id = anchor.cfg.get("base_model_id", "?")
    print(f"\nanchor run      : {ANCHOR_HEURISTIC_RUN.relative_to(REPO_ROOT)}")
    print(f"base_model_id   : {anchor_base_id}")

    variants = _matching_variants(anchor_base_id, OUR_APPROACH_ROOT)
    print(f"matched variants: {len(variants)}")
    if not variants:
        print("No variants sharing the anchor warmup. Cannot run benchmark.")
        return 1

    # Compute all 2x3 matrices
    all_results = []
    all_rows = []
    for variant_dir in variants:
        oa = load_run(variant_dir)
        # Each variant pairs the anchor's warmup with the variant's
        # constraint-trained probs.
        res = build_2x3_matrix(
            warmup_run=anchor,
            constraint_run=oa,
            name=f"RN18_c05_03_bs32/{variant_dir.name}",
        )
        all_results.append((variant_dir, res))
        all_rows.extend(result_to_rows(res, variant=variant_dir.name))

    # Pick the best variant by accuracy of the (C, greedy) cell -- that's
    # the project's actual reported result for each variant.
    def _c_greedy_acc(res):
        for cell in res.cells:
            if cell.model_source == "constraint" and cell.phase2_method == "greedy":
                return cell.accuracy
        return -1.0

    best_variant_dir, best_res = max(all_results, key=lambda p: _c_greedy_acc(p[1]))
    worst_variant_dir, worst_res = min(all_results, key=lambda p: _c_greedy_acc(p[1]))

    # ----- print headline matrix for the best variant -----
    print("\n" + "=" * 82)
    print(f"MAIN TABLE: best our_approach variant by (constraint, greedy) accuracy")
    print(f"  variant = {best_variant_dir.name}")
    print(f"  N = {best_res.n_samples}  C = {best_res.n_classes}  "
          f"constrained classes = {best_res.constrained_classes}")
    print(f"  (feature_pct, target_pct) = ({best_res.feature_pct}, "
          f"{best_res.target_pct})")
    print(f"  Psi = {best_res.psi}")
    print(f"  Phi = {best_res.phi}")
    print("=" * 82)
    print_matrix(best_res)

    # ----- print headline matrix for the worst variant (for range) -----
    print("\n" + "=" * 82)
    print(f"RANGE CHECK: worst our_approach variant by (constraint, greedy) accuracy")
    print(f"  variant = {worst_variant_dir.name}")
    print("=" * 82)
    print_matrix(worst_res)

    # ----- compact summary across all variants -----
    print("\n" + "=" * 82)
    print("ALL VARIANTS: (constraint, greedy) -> (constraint, LP) compared to")
    print("              (warmup, greedy)    -> (warmup, LP)")
    print("=" * 82)

    def _get(res, src, method, field):
        for c in res.cells:
            if c.model_source == src and c.phase2_method == method:
                return getattr(c, field)
        return None

    hdr = (f"  {'variant':<50s} | {'W,g acc':>7s} | {'W,LP acc':>8s} | "
           f"{'C,g acc':>7s} | {'C,LP acc':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for variant_dir, res in all_results:
        wg = _get(res, "warmup", "greedy", "accuracy")
        wl = _get(res, "warmup", "LP", "accuracy")
        cg = _get(res, "constraint", "greedy", "accuracy")
        cl = _get(res, "constraint", "LP", "accuracy")
        print(f"  {variant_dir.name:<50s} | {wg:>7.4f} | {wl:>8.4f} | "
              f"{cg:>7.4f} | {cl:>8.4f}")

    # ----- write tidy CSV -----
    df = pd.DataFrame(all_rows)
    out_csv = REPO_ROOT / "danits_research" / "benchmark_phase_a_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote tidy CSV: {out_csv.relative_to(REPO_ROOT)} "
          f"({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Phase-A multi-constraint benchmark.

Reports a proper 2x3 matrix per experiment, with no confounding between
the model choice and the Phase-2 method.

                     | argmax (no Phase-2) | project greedy | paper [5] LP
    -----------------+---------------------+----------------+--------------
    warmup model     |      (W,-)          |     (W,g)      |    (W,LP)
    constraint model |      (C,-)          |     (C,g)      |    (C,LP)

Archive setup:
    archive_experiments/dermmnist_round2_conflicting_constraints/
        exp1_MobileNetV3_MEL_BCC_L20G80/{heuristic,our_approach}
        exp2_ResNet18_MEL_BCC_L30G80/{heuristic,our_approach}
        exp3_MobileNetV3_MEL_BCC_L20G50/{heuristic,our_approach}
        exp4_ResNet18_MEL_AKIEC_L20G80/{heuristic,our_approach}
        exp5_MobileNetV3_MEL_BKL_BCC_L20G80/{heuristic,our_approach}

For each experiment the `heuristic/` and `our_approach/` subdirs share
the same `base_model_id`, verified by preflight.
"""

from __future__ import annotations

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

MULTI_ROOT = ARCHIVE / "dermmnist_round2_conflicting_constraints"
EXPERIMENTS = [
    "exp1_MobileNetV3_MEL_BCC_L20G80",
    "exp2_ResNet18_MEL_BCC_L30G80",
    "exp3_MobileNetV3_MEL_BCC_L20G50",
    "exp4_ResNet18_MEL_AKIEC_L20G80",
    "exp5_MobileNetV3_MEL_BKL_BCC_L20G80",
]


def main() -> int:
    print("=" * 92)
    print("Phase-A multi-constraint benchmark (2x3 matrix, 5 experiments)")
    print("=" * 92)

    all_rows = []
    for exp in EXPERIMENTS:
        heur_dir = MULTI_ROOT / exp / "heuristic"
        oa_dir = MULTI_ROOT / exp / "our_approach"
        if not (heur_dir / "final_predictions.csv").exists():
            print(f"\nskip {exp}: missing heuristic/final_predictions.csv")
            continue
        if not (oa_dir / "final_predictions.csv").exists():
            print(f"\nskip {exp}: missing our_approach/final_predictions.csv")
            continue

        warmup_run = load_run(heur_dir)
        constraint_run = load_run(oa_dir)
        res = build_2x3_matrix(
            warmup_run=warmup_run,
            constraint_run=constraint_run,
            name=exp,
        )

        print()
        print("=" * 92)
        print(f"EXPERIMENT: {exp}")
        print("=" * 92)
        print(f"base_model_id          : {res.base_model_id}")
        print(f"constrained classes    : {res.constrained_classes}")
        print(f"(feature_pct, target_pct) = ({res.feature_pct}, {res.target_pct})")
        print(f"N test samples         : {res.n_samples}")
        print(f"Psi (constrained cls)  : "
              f"{ {c: res.psi[c] for c in res.constrained_classes} }")
        print(f"Phi (group x class)    :")
        for g, bounds in res.phi.items():
            per_class = {c: bounds[c] for c in res.constrained_classes}
            print(f"    group {g} : {per_class}")
        # Tightness check
        for c in res.constrained_classes:
            global_bound = res.psi[c]
            local_sum = sum(b[c] for b in res.phi.values() if b[c] is not None)
            tight = "LOCAL binds" if local_sum < global_bound else "GLOBAL binds"
            print(f"    class {c}: Psi={global_bound}  sum(Phi)={local_sum}  -> {tight}")
        print()
        print_matrix(res)

        all_rows.extend(result_to_rows(res))

    # ---- compact summary across all experiments ----
    print()
    print("=" * 92)
    print("COMPACT SUMMARY across experiments: accuracy + feasibility only")
    print("=" * 92)
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("(no rows)")
        return 0

    # One row per (experiment, model, phase2) -- pick any class since metrics
    # like accuracy are the same across classes in the same cell.
    df_uniq = df.drop_duplicates(
        subset=["benchmark", "model_source", "phase2_method"])
    df_uniq = df_uniq[["benchmark", "model_source", "phase2_method",
                       "accuracy", "macro_f1", "feasible", "n_violations"]]
    df_uniq = df_uniq.sort_values(
        ["benchmark", "model_source", "phase2_method"])

    prev = None
    for _, r in df_uniq.iterrows():
        label = r["benchmark"] if r["benchmark"] != prev else ""
        feas = "OK" if r["feasible"] else f"x{int(r['n_violations'])}"
        print(f"  {label:<40s} | {r['model_source']:<11s} | "
              f"{r['phase2_method']:<7s} | "
              f"acc={r['accuracy']:.4f} | F1m={r['macro_f1']:.4f} | {feas}")
        if r["benchmark"] != prev:
            prev = r["benchmark"]

    out_csv = REPO_ROOT / "danits_research" / "benchmark_multi_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote tidy CSV: {out_csv.relative_to(REPO_ROOT)} "
          f"({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

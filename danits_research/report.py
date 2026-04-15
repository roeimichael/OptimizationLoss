"""
Thesis-style consolidated report over both benchmarks.

Reads the tidy CSVs produced by
    danits_research/benchmark.py         (single-constraint, best variant)
    danits_research/benchmark_multi.py   (5 multi-constraint experiments)

and prints a single headline table with the numbers that actually matter
for the thesis comparison. Also computes the two derived quantities that
tell us most about where gains come from:

    (1) model_gain  = (C,g).acc - (W,g).acc
        -- held at the same Phase-2 greedy method, how much does switching
           from warmup to constraint-trained help?

    (2) phase2_gain = (C,LP).acc - (C,g).acc
        -- held at the same constraint-trained model, how much does
           switching from project greedy to paper [5] LP help?

    (3) projection_cost = (C,argmax).acc - (C,g).acc
        -- how much accuracy does the constraint-trained model give up
           in being projected to feasibility? Large values mean the
           training didn't shape the posterior enough.

Usage:
    python -m danits_research.report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SC_CSV = REPO_ROOT / "danits_research" / "benchmark_phase_a_results.csv"
MC_CSV = REPO_ROOT / "danits_research" / "benchmark_multi_results.csv"


def _one_row(df: pd.DataFrame, benchmark: str, variant: str | None,
             model: str, phase2: str) -> pd.Series | None:
    sel = (
        (df["benchmark"] == benchmark)
        & (df["model_source"] == model)
        & (df["phase2_method"] == phase2)
    )
    if variant is not None:
        sel = sel & (df["variant"] == variant)
    sub = df[sel]
    if len(sub) == 0:
        return None
    # The 2x3 matrix writes one row per (cell, constrained class) in long
    # format; accuracy/F1 are identical across classes for the same cell,
    # so we drop duplicates to get the unique cell row.
    return sub.drop_duplicates(
        subset=["model_source", "phase2_method"]).iloc[0]


def _format_delta(v: float | None) -> str:
    if v is None:
        return "    -   "
    sign = "+" if v >= 0 else "-"
    return f"{sign}{abs(v):.4f}"


def _headline_row(df: pd.DataFrame, benchmark: str,
                  variant: str | None = None) -> dict | None:
    """Build one headline row summarizing a single experiment (or variant)."""
    cells = {
        "W_argmax": _one_row(df, benchmark, variant, "warmup", "argmax"),
        "W_greedy": _one_row(df, benchmark, variant, "warmup", "greedy"),
        "W_LP":     _one_row(df, benchmark, variant, "warmup", "LP"),
        "C_argmax": _one_row(df, benchmark, variant, "constraint", "argmax"),
        "C_greedy": _one_row(df, benchmark, variant, "constraint", "greedy"),
        "C_LP":     _one_row(df, benchmark, variant, "constraint", "LP"),
    }
    if any(v is None for v in cells.values()):
        return None

    wa = cells["W_argmax"]["accuracy"]
    wg = cells["W_greedy"]["accuracy"]
    wl = cells["W_LP"]["accuracy"]
    ca = cells["C_argmax"]["accuracy"]
    cg = cells["C_greedy"]["accuracy"]
    cl = cells["C_LP"]["accuracy"]

    wa_feas = bool(cells["W_argmax"]["feasible"])
    ca_feas = bool(cells["C_argmax"]["feasible"])

    return {
        "benchmark": benchmark,
        "variant":   variant if variant is not None else "",
        "W_argmax":  wa,  "W_argmax_feas": wa_feas,
        "W_greedy":  wg,
        "W_LP":      wl,
        "C_argmax":  ca,  "C_argmax_feas": ca_feas,
        "C_greedy":  cg,
        "C_LP":      cl,
        # derived quantities
        "model_gain_greedy": cg - wg,   # effect of training, Phase-2 held at greedy
        "model_gain_LP":     cl - wl,   # effect of training, Phase-2 held at LP
        "phase2_gain_W":     wl - wg,   # LP vs greedy on warmup model
        "phase2_gain_C":     cl - cg,   # LP vs greedy on constraint model
        "projection_cost_W": wa - wg,   # feasibility cost on warmup
        "projection_cost_C": ca - cg,   # feasibility cost on constraint
    }


def _print_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n{title}: (no rows)")
        return
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    # Main accuracy grid: W,g -> W,LP and C,g -> C,LP
    hdr = (f"  {'benchmark / variant':<50s} | "
           f"{'W,g':>7s} | {'W,LP':>7s} | "
           f"{'C,g':>7s} | {'C,LP':>7s} | "
           f"{'mdl gain':>9s} | {'Ph2 gain':>9s} | {'proj cost':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        label = r["benchmark"]
        if r["variant"]:
            label = f"{label} / {r['variant']}"
        label = label[:48]
        print(f"  {label:<50s} | "
              f"{r['W_greedy']:>7.4f} | {r['W_LP']:>7.4f} | "
              f"{r['C_greedy']:>7.4f} | {r['C_LP']:>7.4f} | "
              f"{_format_delta(r['model_gain_greedy']):>9s} | "
              f"{_format_delta(r['phase2_gain_C']):>9s} | "
              f"{_format_delta(-r['projection_cost_C']):>9s}")

    print()
    print("  legend:")
    print("    W,g  = warmup probs    + project greedy post-hoc")
    print("    W,LP = warmup probs    + paper [5] LP post-hoc")
    print("    C,g  = constraint-trained probs + project greedy post-hoc")
    print("    C,LP = constraint-trained probs + paper [5] LP post-hoc")
    print("    mdl gain  = C,g - W,g    (effect of training, Phase-2=greedy)")
    print("    Ph2 gain  = C,LP - C,g   (effect of LP vs greedy, model=constraint)")
    print("    proj cost = C,g - C,argmax  (accuracy lost by projecting to feasibility)")
    print("                (shown as negative number = accuracy cost)")


def _print_argmax_reference(rows: list[dict], title: str) -> None:
    """Separate table for the (infeasible) argmax reference rows."""
    if not rows:
        return
    print("\n" + "=" * 120)
    print(f"REFERENCE -- {title}: raw argmax rows (usually infeasible, shown for diagnostics)")
    print("=" * 120)
    hdr = (f"  {'benchmark / variant':<50s} | "
           f"{'W,argmax':>9s} | {'feas':>4s} | "
           f"{'C,argmax':>9s} | {'feas':>4s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        label = r["benchmark"]
        if r["variant"]:
            label = f"{label} / {r['variant']}"
        label = label[:48]
        print(f"  {label:<50s} | "
              f"{r['W_argmax']:>9.4f} | "
              f"{'OK' if r['W_argmax_feas'] else 'x':>4s} | "
              f"{r['C_argmax']:>9.4f} | "
              f"{'OK' if r['C_argmax_feas'] else 'x':>4s}")


def main() -> int:
    if not SC_CSV.exists():
        print(f"missing: {SC_CSV} -- run benchmark.py first")
        return 1
    if not MC_CSV.exists():
        print(f"missing: {MC_CSV} -- run benchmark_multi.py first")
        return 1

    sc = pd.read_csv(SC_CSV)
    mc = pd.read_csv(MC_CSV)

    # ==== single-constraint: one row per variant ====
    # Each variant in benchmark.py is written as a distinct benchmark name
    # "RN18_c05_03_bs32/<variant>" plus a variant column. We iterate over
    # the (benchmark, variant) pairs the CSV actually contains.
    sc_pairs = (
        sc[["benchmark", "variant"]].drop_duplicates().itertuples(index=False)
    )
    sc_rows = []
    for benchmark, variant in sc_pairs:
        row = _headline_row(sc, benchmark=benchmark, variant=variant)
        if row is not None:
            # Use the short variant name as the label so the table stays readable.
            row["benchmark"] = variant
            row["variant"] = ""
            sc_rows.append(row)
    sc_rows.sort(key=lambda r: r["C_greedy"], reverse=True)

    # ==== multi-constraint: one row per experiment ====
    mc_benchmarks = mc["benchmark"].unique().tolist()
    mc_rows = []
    for b in mc_benchmarks:
        row = _headline_row(mc, b)
        if row is not None:
            mc_rows.append(row)
    mc_rows.sort(key=lambda r: r["benchmark"])

    _print_table(sc_rows, "SINGLE-CONSTRAINT (ResNet18 / c05_03 / 10 our_approach variants)")
    _print_argmax_reference(sc_rows, "SINGLE-CONSTRAINT")
    _print_table(mc_rows, "MULTI-CONSTRAINT (5 conflicting_constraints experiments)")
    _print_argmax_reference(mc_rows, "MULTI-CONSTRAINT")

    # ---- aggregate takeaways ----
    print("\n" + "=" * 120)
    print("AGGREGATE TAKEAWAYS")
    print("=" * 120)
    for label, rows in [
        ("single-constraint (avg over 10 variants)", sc_rows),
        ("multi-constraint (avg over 5 experiments)", mc_rows),
    ]:
        if not rows:
            continue
        n = len(rows)
        mean_mg = sum(r["model_gain_greedy"] for r in rows) / n
        mean_p2 = sum(r["phase2_gain_C"] for r in rows) / n
        mean_pc = sum(r["projection_cost_C"] for r in rows) / n
        print(f"\n  {label} (n={n})")
        print(f"    mean model_gain  (C,g  -  W,g)     = {_format_delta(mean_mg)}")
        print(f"    mean phase2_gain (C,LP -  C,g)     = {_format_delta(mean_p2)}")
        print(f"    mean proj_cost   (C,argmax - C,g)  = {_format_delta(mean_pc)}  "
              f"(positive = projection drops accuracy)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

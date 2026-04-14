"""
5-experiment smoke test with per-method flip accounting and explicit
feasibility audits for the paper [5] local + global constraints.

Methods computed for every experiment (on the same test set, same Psi/Phi):

    MAIN 5 methods (the user's list):
      (1) W + LP         :: warmup probs -> paper [5] LP (no post-hoc)
      (2) W + LP + ph    :: (1) -> gap-fill post-hoc
      (3) heuristic       :: warmup probs -> project's apply_allocation_heuristic
                             (read from heuristic/Predicted_Label in archive)
      (4) C + argmax + ph :: constraint-trained probs -> argmax -> gap-fill post-hoc
      (5) C + LP          :: constraint-trained probs -> paper [5] LP (no post-hoc)

    DIAGNOSTIC rows:
      W argmax              -- warmup argmax only, usually infeasible
      W + minimal_correction -- project's minimal_correction on warmup
      C argmax              -- constraint-trained argmax only
      C + LP + ph            -- (5) -> gap-fill post-hoc (sanity for LP slack)
      C + targeted_correction -- project's current our_approach saved output
      C + minimal_correction  -- project's minimal_correction on constraint-trained

Every method is scored with:
    - accuracy, macro F1
    - per-constrained-class recall/precision/count
    - feasible / #violations
    - flips vs that model's raw argmax (= how many samples moved away from
      what the model originally thought)
    - flip breakdown when the method has phases (gap-fill post-hoc reports
      its phase-1/2/3/4 counts).

Explicit feasibility audit is printed for every method: per (class, group)
bound vs count, with BINDING / SLACK annotation.

Runs on the 5 multi-constraint experiments already in the archive
(archive_experiments/dermmnist_round2_conflicting_constraints/). No
retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    build_psi_phi_from_percentages,
    solve_lp_assignment,
)
from danits_research._benchmark_core import load_run
from danits_research.fill_to_budget import posthoc_fill_gap

# Project's existing post-hoc variants (for comparison only; we do NOT
# modify them).
from src.utils.posthoc_adjustment import (
    minimal_correction,
    targeted_correction,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTI_ROOT = (
    REPO_ROOT
    / "archive_experiments"
    / "dermmnist_round2_conflicting_constraints"
)
EXPERIMENTS = [
    "exp1_MobileNetV3_MEL_BCC_L20G80",
    "exp2_ResNet18_MEL_BCC_L30G80",
    "exp3_MobileNetV3_MEL_BCC_L20G50",
    "exp4_ResNet18_MEL_AKIEC_L20G80",
    "exp5_MobileNetV3_MEL_BKL_BCC_L20G80",
]

UNLIMITED = 1e10  # matches the project's UNLIMITED sentinel


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def _counts(y, n):
    return {int(i): int((y == i).sum()) for i in range(n)}


def _recall(y_true, y_pred, c):
    m = y_true == c
    if not m.any():
        return float("nan")
    return float((y_pred[m] == c).mean())


def _precision(y_true, y_pred, c):
    m = y_pred == c
    if not m.any():
        return float("nan")
    return float((y_true[m] == c).mean())


def _macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        p = _precision(y_true, y_pred, c)
        r = _recall(y_true, y_pred, c)
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s))


# ----------------------------------------------------------------------
# constraint format bridge: our (psi, phi) <-> project's (global_con, local_con)
# ----------------------------------------------------------------------

def _to_project_constraints(psi: list, phi: dict, n_classes: int):
    """Convert (psi, phi) to the (global_con, local_con) shape expected by
    src.utils.posthoc_adjustment. None -> UNLIMITED, int -> int."""
    global_con = [UNLIMITED] * n_classes
    for i, v in enumerate(psi):
        if v is not None:
            global_con[i] = int(v)
    local_con: dict = {}
    for gid, bounds in phi.items():
        lst = [UNLIMITED] * n_classes
        for i, v in enumerate(bounds):
            if v is not None:
                lst[i] = int(v)
        local_con[int(gid) if hasattr(gid, "__int__") else gid] = lst
    return global_con, local_con


# ----------------------------------------------------------------------
# feasibility audit (the "receipts" printer)
# ----------------------------------------------------------------------

def _feasibility_full(y_pred, groups, psi, phi, constrained_classes,
                      n_classes) -> dict:
    """Return structured data about every Psi and Phi constraint."""
    psi_rows = []
    for c in constrained_classes:
        bound = psi[c] if c < len(psi) else None
        if bound is None:
            continue
        cnt = int((y_pred == c).sum())
        psi_rows.append({
            "class": c,
            "bound": bound,
            "count": cnt,
            "slack": bound - cnt,
            "status": ("BINDING" if cnt == bound
                       else "SLACK" if cnt < bound
                       else "VIOLATION"),
        })
    phi_rows = []
    for gid, bounds in phi.items():
        mask = groups == gid
        for c in constrained_classes:
            if c >= len(bounds) or bounds[c] is None:
                continue
            bound = bounds[c]
            cnt = int(((y_pred == c) & mask).sum())
            phi_rows.append({
                "group": gid,
                "class": c,
                "bound": bound,
                "count": cnt,
                "slack": bound - cnt,
                "status": ("BINDING" if cnt == bound
                           else "SLACK" if cnt < bound
                           else "VIOLATION"),
            })
    n_psi_viol = sum(1 for r in psi_rows if r["status"] == "VIOLATION")
    n_phi_viol = sum(1 for r in phi_rows if r["status"] == "VIOLATION")
    n_psi_bind = sum(1 for r in psi_rows if r["status"] == "BINDING")
    n_phi_bind = sum(1 for r in phi_rows if r["status"] == "BINDING")
    return {
        "psi_rows": psi_rows,
        "phi_rows": phi_rows,
        "feasible": (n_psi_viol == 0 and n_phi_viol == 0),
        "n_psi_bind": n_psi_bind,
        "n_phi_bind": n_phi_bind,
        "n_psi_viol": n_psi_viol,
        "n_phi_viol": n_phi_viol,
    }


def _print_feasibility(audit: dict, indent: str = "      "):
    print(f"{indent}Global (Psi) constraints:")
    for r in audit["psi_rows"]:
        marker = {"BINDING": "*", "SLACK": " ", "VIOLATION": "!"}[r["status"]]
        print(f"{indent}  {marker} class {r['class']:>2d}: "
              f"bound={r['bound']:>4d}  count={r['count']:>4d}  "
              f"slack={r['slack']:>+4d}  [{r['status']}]")
    print(f"{indent}Local (Phi) constraints:")
    for r in audit["phi_rows"]:
        marker = {"BINDING": "*", "SLACK": " ", "VIOLATION": "!"}[r["status"]]
        print(f"{indent}  {marker} group {r['group']:>2d}  class {r['class']:>2d}: "
              f"bound={r['bound']:>3d}  count={r['count']:>3d}  "
              f"slack={r['slack']:>+3d}  [{r['status']}]")
    bind = audit["n_psi_bind"] + audit["n_phi_bind"]
    total = len(audit["psi_rows"]) + len(audit["phi_rows"])
    viol = audit["n_psi_viol"] + audit["n_phi_viol"]
    verdict = "FEASIBLE" if viol == 0 else f"INFEASIBLE ({viol} violations)"
    print(f"{indent}Summary: {verdict}, {bind}/{total} bounds binding")


# ----------------------------------------------------------------------
# method evaluation + row construction
# ----------------------------------------------------------------------

@dataclass
class MethodResult:
    label: str
    y_pred: np.ndarray
    accuracy: float
    macro_f1: float
    recall_by_class: dict
    precision_by_class: dict
    counts_by_class: dict
    flips_vs_source_argmax: int         # how many differ from source-model argmax
    feasibility: dict
    phase_breakdown: Optional[dict] = None
    source_model: str = ""              # "warmup" or "constraint"
    notes: str = ""


def _eval_method(
    label: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    psi: list,
    phi: dict,
    n_classes: int,
    constrained_classes: list,
    source_argmax: np.ndarray,
    source_model: str,
    phase_breakdown: Optional[dict] = None,
    notes: str = "",
) -> MethodResult:
    audit = _feasibility_full(
        y_pred, groups, psi, phi, constrained_classes, n_classes)
    return MethodResult(
        label=label,
        y_pred=y_pred,
        accuracy=float((y_pred == y_true).mean()),
        macro_f1=_macro_f1(y_true, y_pred, n_classes),
        recall_by_class={c: _recall(y_true, y_pred, c) for c in constrained_classes},
        precision_by_class={c: _precision(y_true, y_pred, c) for c in constrained_classes},
        counts_by_class=_counts(y_pred, n_classes),
        flips_vs_source_argmax=int((y_pred != source_argmax).sum()),
        feasibility=audit,
        phase_breakdown=phase_breakdown,
        source_model=source_model,
        notes=notes,
    )


# ----------------------------------------------------------------------
# experiment driver
# ----------------------------------------------------------------------

def _run_experiment(exp_name: str) -> list[dict]:
    print()
    print("#" * 96)
    print(f"# EXPERIMENT: {exp_name}")
    print("#" * 96)

    heur_dir = MULTI_ROOT / exp_name / "heuristic"
    oa_dir = MULTI_ROOT / exp_name / "our_approach"
    if not (heur_dir / "final_predictions.csv").exists():
        print(f"skip: no heuristic predictions")
        return []
    if not (oa_dir / "final_predictions.csv").exists():
        print(f"skip: no our_approach predictions")
        return []

    heur = load_run(heur_dir)
    oa = load_run(oa_dir)

    y_true = heur.y_true
    groups = heur.groups
    n_classes = heur.n_classes

    # Constraint spec from the config
    constrained_raw = heur.cfg["dataset_config"]["constrained_class"]
    if isinstance(constrained_raw, int):
        constrained_classes = [int(constrained_raw)]
    else:
        constrained_classes = [int(c) for c in constrained_raw]
    feature_pct = float(heur.cfg["constraint"][0])
    target_pct = float(heur.cfg["constraint"][1])

    psi, phi = build_psi_phi_from_percentages(
        y_true=y_true, groups=groups, n_classes=n_classes,
        constrained_class=constrained_classes,
        feature_pct=feature_pct, target_pct=target_pct,
    )
    global_con, local_con = _to_project_constraints(psi, phi, n_classes)

    print(f"  base_model_id       : {heur.cfg['base_model_id']}")
    print(f"  constrained classes : {constrained_classes}")
    print(f"  (feature, target)%  : ({feature_pct}, {target_pct})")
    print(f"  groups present      : {sorted(np.unique(groups).tolist())}")
    print(f"  N test samples      : {len(y_true)}")
    print(f"  Psi  :  "
          + " ".join(f"c{c}={psi[c]}" for c in constrained_classes))
    print(f"  Phi  :")
    for gid, bounds in phi.items():
        print("    group " + str(gid) + ": "
              + " ".join(f"c{c}={bounds[c]}" for c in constrained_classes))
    for c in constrained_classes:
        global_bound = psi[c]
        local_sum = sum(b[c] for b in phi.values() if b[c] is not None)
        tight = "LOCAL binds" if local_sum < global_bound else "GLOBAL binds"
        print(f"    class {c}: Psi={global_bound} sum(Phi)={local_sum} -> {tight}")

    # ------ warmup-model methods ------
    w_probs = heur.probs
    w_argmax = w_probs.argmax(axis=1)

    # 1) W + LP
    lp_w = solve_lp_assignment(
        y_proba=w_probs, groups=groups,
        cost_matrix=DERMMNIST_IDENTITY, psi=psi, phi=phi,
    )
    m1 = _eval_method("(1) W + LP", lp_w.y_pred, y_true, groups, psi, phi,
                      n_classes, constrained_classes, w_argmax, "warmup",
                      notes=f"LP runtime={lp_w.runtime_seconds*1000:.1f}ms, "
                            f"status={lp_w.status}")

    # 2) W + LP + ph  (start from LP output, close remaining gaps)
    y2, info2 = posthoc_fill_gap(
        y_pred_in=lp_w.y_pred, y_proba=w_probs, groups=groups,
        psi=psi, phi=phi, constrained_classes=constrained_classes,
    )
    m2 = _eval_method("(2) W + LP + ph", y2, y_true, groups, psi, phi,
                      n_classes, constrained_classes, w_argmax, "warmup",
                      phase_breakdown=info2)

    # 3) heuristic  (read saved Predicted_Label produced by apply_allocation_heuristic)
    m3 = _eval_method("(3) heuristic (W,apply_alloc)",
                      heur.y_pred_saved, y_true, groups, psi, phi,
                      n_classes, constrained_classes, w_argmax, "warmup",
                      notes="from archive/heuristic/Predicted_Label")

    # ------ constraint-trained methods ------
    c_probs = oa.probs
    c_argmax = c_probs.argmax(axis=1)

    # 4) C + argmax + ph  (start from constraint-trained argmax, close gaps)
    y4, info4 = posthoc_fill_gap(
        y_pred_in=c_argmax, y_proba=c_probs, groups=groups,
        psi=psi, phi=phi, constrained_classes=constrained_classes,
    )
    m4 = _eval_method("(4) C + argmax + ph", y4, y_true, groups, psi, phi,
                      n_classes, constrained_classes, c_argmax, "constraint",
                      phase_breakdown=info4)

    # 5) C + LP   (paper [5] LP on constraint-trained probs, no post-hoc)
    lp_c = solve_lp_assignment(
        y_proba=c_probs, groups=groups,
        cost_matrix=DERMMNIST_IDENTITY, psi=psi, phi=phi,
    )
    m5 = _eval_method("(5) C + LP", lp_c.y_pred, y_true, groups, psi, phi,
                      n_classes, constrained_classes, c_argmax, "constraint",
                      notes=f"LP runtime={lp_c.runtime_seconds*1000:.1f}ms, "
                            f"status={lp_c.status}")

    main_methods = [m1, m2, m3, m4, m5]

    # ------ diagnostic rows ------
    diag_methods: list[MethodResult] = []

    # warmup argmax (reference, usually infeasible)
    diag_methods.append(_eval_method(
        "[ref] W argmax", w_argmax, y_true, groups, psi, phi,
        n_classes, constrained_classes, w_argmax, "warmup",
        notes="raw, unconstrained baseline",
    ))

    # constraint argmax (reference; may or may not be feasible)
    diag_methods.append(_eval_method(
        "[ref] C argmax", c_argmax, y_true, groups, psi, phi,
        n_classes, constrained_classes, c_argmax, "constraint",
        notes="raw constraint-trained argmax",
    ))

    # W + minimal_correction (project's only-reduce post-hoc, from argmax)
    wmc, _ = minimal_correction(
        w_probs, groups, global_con, local_con, constrained_classes)
    diag_methods.append(_eval_method(
        "[diag] W + minimal", wmc, y_true, groups, psi, phi,
        n_classes, constrained_classes, w_argmax, "warmup",
        notes="project's minimal_correction (reduce-only)",
    ))

    # C + minimal_correction
    cmc, _ = minimal_correction(
        c_probs, groups, global_con, local_con, constrained_classes)
    diag_methods.append(_eval_method(
        "[diag] C + minimal", cmc, y_true, groups, psi, phi,
        n_classes, constrained_classes, c_argmax, "constraint",
        notes="project's minimal_correction on constraint probs",
    ))

    # C + targeted_correction (= the current our_approach saved output)
    diag_methods.append(_eval_method(
        "[diag] C + targeted", oa.y_pred_saved, y_true, groups, psi, phi,
        n_classes, constrained_classes, c_argmax, "constraint",
        notes="saved our_approach pipeline (targeted_correction from argmax)",
    ))

    # C + LP + ph (sanity check: does the LP output still need filling?)
    y5ph, info5ph = posthoc_fill_gap(
        y_pred_in=lp_c.y_pred, y_proba=c_probs, groups=groups,
        psi=psi, phi=phi, constrained_classes=constrained_classes,
    )
    diag_methods.append(_eval_method(
        "[diag] C + LP + ph", y5ph, y_true, groups, psi, phi,
        n_classes, constrained_classes, c_argmax, "constraint",
        phase_breakdown=info5ph,
        notes="LP output then gap-fill post-hoc",
    ))

    # ------- print main table ---------------------------------------
    print()
    print("  MAIN METHODS (the 5 you asked for):")
    _print_methods_table(main_methods, constrained_classes)

    # ------- print diagnostic table ---------------------------------
    print()
    print("  DIAGNOSTIC ROWS (references and comparisons):")
    _print_methods_table(diag_methods, constrained_classes)

    # ------- feasibility receipts for every method ------------------
    print()
    print("  FEASIBILITY RECEIPTS (paper [5] Eqs. (2) and (3), per method)")
    for m in main_methods + diag_methods:
        print(f"\n    method: {m.label}")
        _print_feasibility(m.feasibility, indent="        ")
        if m.phase_breakdown is not None:
            print(f"        flip breakdown: "
                  f"reduce_global={m.phase_breakdown['phase1_reduce_global_flips']}  "
                  f"reduce_local={m.phase_breakdown['phase2_reduce_local_flips']}  "
                  f"fill_global={m.phase_breakdown['phase3_fill_global_flips']}  "
                  f"fill_local={m.phase_breakdown['phase4_fill_local_flips']}  "
                  f"total={m.phase_breakdown['total_flips']}")

    # ------- flip-count comparison against source argmax -----------
    print()
    print("  FLIPS vs that model's argmax (how much did each method move?):")
    for m in main_methods + diag_methods:
        print(f"    {m.label:<32s}  src={m.source_model:<10s}  "
              f"flips={m.flips_vs_source_argmax:>4d}  ({m.notes})")

    # ------- disagreement between method pairs we care about --------
    def diff(a: MethodResult, b: MethodResult) -> int:
        return int((a.y_pred != b.y_pred).sum())

    print()
    print("  KEY PAIRWISE DISAGREEMENTS (count of samples where predictions differ):")
    pairs = [
        (m1, m2, "(1) W+LP  vs  (2) W+LP+ph   -- does LP leave any slack?"),
        (m1, m3, "(1) W+LP  vs  (3) heuristic -- LP vs project's greedy?"),
        (m4, m5, "(4) C+argmax+ph  vs  (5) C+LP -- does LP help after training?"),
        (m4, diag_methods[4], "(4) C+argmax+ph  vs  [C+targeted] -- new vs old post-hoc?"),
        (m5, diag_methods[5], "(5) C+LP vs [C+LP+ph]         -- does LP need fill?"),
    ]
    for a, b, note in pairs:
        print(f"    {a.label:<20s}  vs  {b.label:<20s}  diff={diff(a, b):>4d}   {note}")

    # ------- return tidy rows for CSV -------------------------------
    all_methods = main_methods + diag_methods
    rows: list[dict] = []
    for m in all_methods:
        for c in constrained_classes:
            row = {
                "experiment": exp_name,
                "method":     m.label,
                "source_model": m.source_model,
                "accuracy":   m.accuracy,
                "macro_f1":   m.macro_f1,
                "class":      c,
                "recall":     m.recall_by_class[c],
                "precision":  m.precision_by_class[c],
                "count":      m.counts_by_class[c],
                "psi_bound":  psi[c],
                "flips_vs_source_argmax": m.flips_vs_source_argmax,
                "feasible":   m.feasibility["feasible"],
                "n_psi_bind": m.feasibility["n_psi_bind"],
                "n_phi_bind": m.feasibility["n_phi_bind"],
                "n_psi_viol": m.feasibility["n_psi_viol"],
                "n_phi_viol": m.feasibility["n_phi_viol"],
            }
            if m.phase_breakdown is not None:
                row.update({
                    f"phase_{k}": v for k, v in m.phase_breakdown.items()
                    if k != "final_violations"
                })
            rows.append(row)
    return rows


def _print_methods_table(methods: list[MethodResult], constrained_classes: list):
    header_parts = [f"{'method':<32s}", f"{'acc':>6s}", f"{'F1m':>6s}"]
    for c in constrained_classes:
        header_parts.append(f"{'c' + str(c) + ' rec':>7s}")
        header_parts.append(f"{'c' + str(c) + ' prec':>8s}")
        header_parts.append(f"{'c' + str(c) + ' n':>6s}")
    header_parts.append(f"{'flips':>5s}")
    header_parts.append(f"{'feas':>5s}")
    hdr = "    " + " | ".join(header_parts)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for m in methods:
        parts = [f"{m.label:<32s}",
                 f"{m.accuracy:>6.4f}",
                 f"{m.macro_f1:>6.4f}"]
        for c in constrained_classes:
            parts.append(f"{m.recall_by_class[c]:>7.4f}")
            parts.append(f"{m.precision_by_class[c]:>8.4f}")
            parts.append(f"{m.counts_by_class[c]:>6d}")
        parts.append(f"{m.flips_vs_source_argmax:>5d}")
        feas = "OK" if m.feasibility["feasible"] else \
               f"x{m.feasibility['n_psi_viol'] + m.feasibility['n_phi_viol']}"
        parts.append(f"{feas:>5s}")
        print("    " + " | ".join(parts))


def main() -> int:
    all_rows: list[dict] = []
    for exp in EXPERIMENTS:
        rows = _run_experiment(exp)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = REPO_ROOT / "danits_research" / "benchmark_smoke_results.csv"
    df.to_csv(out_csv, index=False)
    print()
    print("=" * 96)
    print(f"wrote tidy CSV: {out_csv.relative_to(REPO_ROOT)} "
          f"({len(df)} rows)")

    # ===== cross-experiment headline summary =====
    print()
    print("=" * 96)
    print("HEADLINE across the 5 experiments (accuracy, feasibility only)")
    print("=" * 96)
    # one row per (experiment, method) -- drop class duplicates
    df_u = df.drop_duplicates(subset=["experiment", "method"])
    for exp in EXPERIMENTS:
        sub = df_u[df_u["experiment"] == exp]
        if len(sub) == 0:
            continue
        print(f"\n  {exp}")
        for _, r in sub.iterrows():
            feas = "OK" if r["feasible"] else \
                   f"x{int(r['n_psi_viol'] + r['n_phi_viol'])}"
            print(f"    {r['method']:<32s}  "
                  f"acc={r['accuracy']:.4f}  F1m={r['macro_f1']:.4f}  "
                  f"flips={int(r['flips_vs_source_argmax']):>4d}  {feas}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

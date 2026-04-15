"""
Analyze the freshly-completed smoke grid from `results/pending_runs/`.

For each (scenario, constraint_tag) pair in the tree, we read the three
methodology runs (heuristic / our_approach / danits_lp), extract their
probability matrices and saved predictions, then compute the 5 methods
from the user's list plus diagnostic rows:

    (1) W + LP             -- warmup probs -> paper [5] LP
    (2) W + LP + ph        -- (1) -> gap-closing post-hoc
    (3) heuristic          -- warmup probs -> project's apply_allocation_heuristic
                             (from heuristic/.../final_predictions.csv)
    (4) C + argmax + ph    -- constraint-trained argmax -> gap-closing post-hoc
    (5) C + LP             -- constraint-trained probs -> paper [5] LP

Diagnostic rows:
    [ref] W argmax         -- warmup argmax (usually infeasible)
    [ref] C argmax         -- constraint-trained argmax (usually infeasible)
    [diag] C + targeted    -- our_approach saved Predicted_Label (=
                              constraint-trained + project's targeted_correction,
                              NOT the user's gap-closing post-hoc)

Output: one 9-row x 5-method table per (scenario, constraint_tag), then
a compact headline table spanning all 8 settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    build_psi_phi_from_percentages,
    solve_lp_assignment,
)
from danits_research.fill_to_budget import posthoc_fill_gap

REPO_ROOT = Path(__file__).resolve().parent.parent
PENDING = REPO_ROOT / "results" / "pending_runs"


def _load_run(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "final_predictions.csv")
    n_classes = int(cfg["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "cfg": cfg,
        "probs": probs,
        "y_true": df["True_Label"].to_numpy(dtype=np.int64),
        "y_pred_saved": df["Predicted_Label"].to_numpy(dtype=np.int64),
        "groups": df["Group_ID"].to_numpy(),
        "n_classes": n_classes,
    }


def _counts(y, n):
    return {int(i): int((y == i).sum()) for i in range(n)}


def _recall(yt, yp, c):
    m = yt == c
    return float((yp[m] == c).mean()) if m.any() else float("nan")


def _precision(yt, yp, c):
    m = yp == c
    return float((yt[m] == c).mean()) if m.any() else float("nan")


def _macro_f1(yt, yp, n):
    f1s = []
    for c in range(n):
        p = _precision(yt, yp, c)
        r = _recall(yt, yp, c)
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s))


def _feasibility(y_pred, groups, psi, phi, n_classes) -> tuple[bool, int]:
    viol = 0
    for i, b in enumerate(psi):
        if b is None:
            continue
        if int((y_pred == i).sum()) > b:
            viol += 1
    for g, bounds in phi.items():
        mask = groups == g
        for i, b in enumerate(bounds):
            if b is None:
                continue
            if int(((y_pred == i) & mask).sum()) > b:
                viol += 1
    return (viol == 0), viol


@dataclass
class Row:
    method: str
    model_src: str     # "warmup" or "constraint"
    acc: float
    f1: float
    recall_by_class: dict
    prec_by_class: dict
    count_by_class: dict
    feasible: bool
    n_viol: int
    flips_vs_src: int
    notes: str = ""


def _eval(label, y_pred, model_src, y_true, groups, psi, phi,
          n_classes, constrained_classes, src_argmax, notes=""):
    feas, viol = _feasibility(y_pred, groups, psi, phi, n_classes)
    return Row(
        method=label,
        model_src=model_src,
        acc=float((y_pred == y_true).mean()),
        f1=_macro_f1(y_true, y_pred, n_classes),
        recall_by_class={c: _recall(y_true, y_pred, c) for c in constrained_classes},
        prec_by_class={c: _precision(y_true, y_pred, c) for c in constrained_classes},
        count_by_class=_counts(y_pred, n_classes),
        feasible=feas,
        n_viol=viol,
        flips_vs_src=int((y_pred != src_argmax).sum()),
        notes=notes,
    )


def analyze_one(run_triplet_dir: Path, scenario: str, tag: str):
    """run_triplet_dir points at results/pending_runs/{scenario}/{tag}/{model}/"""
    heur = _load_run(run_triplet_dir / "heuristic" / "slice_1")
    oa   = _load_run(run_triplet_dir / "our_approach" / "slice_1")
    dlp  = _load_run(run_triplet_dir / "danits_lp" / "slice_1")

    y_true = heur["y_true"]
    groups = heur["groups"]
    n_classes = heur["n_classes"]

    # Constraint spec from ANY config (they match by construction)
    cfg = heur["cfg"]
    constrained_raw = cfg["dataset_config"]["constrained_class"]
    if isinstance(constrained_raw, int):
        constrained = [int(constrained_raw)]
    else:
        constrained = [int(c) for c in constrained_raw]
    feature_pct = float(cfg["constraint"][0])
    target_pct  = float(cfg["constraint"][1])
    psi, phi = build_psi_phi_from_percentages(
        y_true, groups, n_classes, constrained, feature_pct, target_pct)

    # Warmup-side: heuristic and danits_lp share the same cached warmup
    # weights, but float non-determinism (cudnn.benchmark=True, BF16 autocast)
    # produces tiny per-sample drift of ~1e-4. We use heuristic's probs as
    # the canonical "warmup probs" reference and check that argmax agrees
    # on at least 99% of samples with the danits_lp probs.
    w_probs = heur["probs"]
    heur_argmax = w_probs.argmax(axis=1)
    dlp_argmax = dlp["probs"].argmax(axis=1)
    argmax_agree = float((heur_argmax == dlp_argmax).mean())
    if argmax_agree < 0.99:
        print(f"WARN: warmup argmax agreement only {argmax_agree:.4f} "
              f"in {run_triplet_dir.name} -- investigate")
    w_argmax = heur_argmax

    c_probs = oa["probs"]
    c_argmax = c_probs.argmax(axis=1)

    rows: list[Row] = []

    # (1) W + LP
    lp_w = solve_lp_assignment(y_probs := w_probs, groups=groups,
                               cost_matrix=DERMMNIST_IDENTITY, psi=psi, phi=phi)
    rows.append(_eval("(1) W + LP", lp_w.y_pred, "warmup", y_true, groups,
                      psi, phi, n_classes, constrained, w_argmax,
                      notes=f"LP {lp_w.runtime_seconds*1000:.0f}ms"))

    # (2) W + LP + ph
    y2, info2 = posthoc_fill_gap(lp_w.y_pred, w_probs, groups, psi, phi, constrained)
    rows.append(_eval("(2) W + LP + ph", y2, "warmup", y_true, groups,
                      psi, phi, n_classes, constrained, w_argmax,
                      notes=f"gap-fill total={info2['total_flips']}"))

    # (3) heuristic  -- saved Predicted_Label from heuristic run
    rows.append(_eval("(3) heuristic", heur["y_pred_saved"], "warmup", y_true,
                      groups, psi, phi, n_classes, constrained, w_argmax,
                      notes="project apply_allocation_heuristic"))

    # (4) C + argmax + ph
    y4, info4 = posthoc_fill_gap(c_argmax, c_probs, groups, psi, phi, constrained)
    rows.append(_eval("(4) C + argmax + ph", y4, "constraint", y_true, groups,
                      psi, phi, n_classes, constrained, c_argmax,
                      notes=f"gap-fill total={info4['total_flips']}"))

    # (5) C + LP
    lp_c = solve_lp_assignment(c_probs, groups=groups,
                               cost_matrix=DERMMNIST_IDENTITY, psi=psi, phi=phi)
    rows.append(_eval("(5) C + LP", lp_c.y_pred, "constraint", y_true, groups,
                      psi, phi, n_classes, constrained, c_argmax,
                      notes=f"LP {lp_c.runtime_seconds*1000:.0f}ms"))

    # reference rows
    rows.append(_eval("[ref] W argmax", w_argmax, "warmup", y_true, groups,
                      psi, phi, n_classes, constrained, w_argmax,
                      notes="raw unconstrained"))
    rows.append(_eval("[ref] C argmax", c_argmax, "constraint", y_true, groups,
                      psi, phi, n_classes, constrained, c_argmax,
                      notes="raw constraint-trained"))
    rows.append(_eval("[diag] C + targeted", oa["y_pred_saved"], "constraint",
                      y_true, groups, psi, phi, n_classes, constrained, c_argmax,
                      notes="project targeted_correction (saved)"))

    return {
        "scenario": scenario,
        "tag": tag,
        "feature_pct": feature_pct,
        "target_pct": target_pct,
        "constrained_classes": constrained,
        "psi": {c: psi[c] for c in constrained},
        "phi_sum": {c: sum(b[c] for b in phi.values() if b[c] is not None) for c in constrained},
        "rows": rows,
    }


def print_setup_table(result):
    scenario = result["scenario"]
    tag = result["tag"]
    c = result["constrained_classes"]
    print("\n" + "=" * 110)
    print(f"  {scenario}  /  {tag}  "
          f"constrained={c}  "
          f"(L={result['feature_pct']}, G={result['target_pct']})  "
          f"Psi={result['psi']}  sum(Phi)={result['phi_sum']}")
    print("=" * 110)
    header_parts = [f"{'method':<22s}", f"{'src':<10s}",
                    f"{'acc':>6s}", f"{'F1m':>6s}"]
    for cls in c:
        header_parts.append(f"{'c'+str(cls)+' rec':>7s}")
        header_parts.append(f"{'c'+str(cls)+' pre':>7s}")
        header_parts.append(f"{'c'+str(cls)+' n':>5s}")
    header_parts.append(f"{'flips':>5s}")
    header_parts.append(f"{'feas':>4s}")
    hdr = "    " + " | ".join(header_parts)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in result["rows"]:
        parts = [f"{r.method:<22s}", f"{r.model_src:<10s}",
                 f"{r.acc:>6.4f}", f"{r.f1:>6.4f}"]
        for cls in c:
            parts.append(f"{r.recall_by_class[cls]:>7.4f}")
            parts.append(f"{r.prec_by_class[cls]:>7.4f}")
            parts.append(f"{r.count_by_class[cls]:>5d}")
        parts.append(f"{r.flips_vs_src:>5d}")
        parts.append(f"{'OK' if r.feasible else 'x'+str(r.n_viol):>4s}")
        print("    " + " | ".join(parts))


def main() -> int:
    if not PENDING.exists():
        print(f"missing: {PENDING}")
        return 1

    # Find all (scenario, tag, model) triplets that have the 3 methodologies
    triplets = []
    for model_dir in PENDING.glob("*/*/*/"):
        # model_dir is .../scenario/tag/model/
        parts = model_dir.parts
        scenario = parts[-3]
        tag = parts[-2]
        # only add if all three methodologies exist
        needed = ["heuristic", "our_approach", "danits_lp"]
        if all((model_dir / m / "slice_1" / "final_predictions.csv").exists() for m in needed):
            triplets.append((scenario, tag, model_dir))

    triplets.sort(key=lambda t: (t[0], t[1]))
    print(f"found {len(triplets)} complete triplets")

    all_rows = []
    for scenario, tag, model_dir in triplets:
        result = analyze_one(model_dir, scenario, tag)
        print_setup_table(result)
        for r in result["rows"]:
            for c in result["constrained_classes"]:
                all_rows.append({
                    "scenario": scenario,
                    "tag": tag,
                    "method": r.method,
                    "model_src": r.model_src,
                    "acc": r.acc,
                    "f1": r.f1,
                    "class": c,
                    "recall": r.recall_by_class[c],
                    "precision": r.prec_by_class[c],
                    "count": r.count_by_class[c],
                    "flips": r.flips_vs_src,
                    "feasible": r.feasible,
                    "n_viol": r.n_viol,
                })

    # --- headline compact table ---
    print("\n" + "=" * 110)
    print("HEADLINE: accuracy and feasibility per (setting x method)")
    print("=" * 110)
    df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["scenario", "tag", "method"])
    pivot = df.pivot_table(
        index=["scenario", "tag"],
        columns="method",
        values="acc",
        aggfunc="first",
    )
    # Keep only main 5 methods in a clean order
    main_methods = ["(1) W + LP", "(2) W + LP + ph", "(3) heuristic",
                    "(4) C + argmax + ph", "(5) C + LP"]
    cols_present = [m for m in main_methods if m in pivot.columns]
    pivot = pivot[cols_present]
    print(pivot.round(4).to_string())

    # and a feasibility-only grid
    print("\n" + "=" * 110)
    print("FEASIBILITY: 'OK' or 'xN' per (setting x method)")
    print("=" * 110)
    df_feas = df.copy()
    df_feas["feas_str"] = df_feas.apply(
        lambda r: "OK" if r["feasible"] else f"x{int(r['n_viol'])}", axis=1)
    feas_pivot = df_feas.pivot_table(
        index=["scenario", "tag"],
        columns="method",
        values="feas_str",
        aggfunc="first",
    )
    feas_pivot = feas_pivot[cols_present + [c for c in feas_pivot.columns if c.startswith("[")]]
    print(feas_pivot.to_string())

    # --- tidy CSV ---
    out = REPO_ROOT / "danits_research" / "smoke_run_results.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

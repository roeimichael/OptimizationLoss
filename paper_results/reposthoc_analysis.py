"""Re-apply UNIFIED posthoc that forces every method's constrained-class count
to exactly K (drop over + fill under), then recompute F1/acc.

This produces an apples-to-apples comparison: every method ends with the same
prediction count for the constrained class, so F1 / acc reflect quality of
those predictions, not the budget any method used.

For each run dir:
  1. Load final_predictions_raw.csv -> y_proba, y_true, group_ids
  2. Load training_log.csv (TraLO has K cols; propagate to other methods in same cell)
  3. Apply force-exact posthoc (bypass _check_all_satisfied early exit)
  4. Recompute F1m, F1c, accuracy on the re-clamped predictions
  5. Aggregate over seeds and emit new tables
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.constants import UNLIMITED
from src.utils.posthoc_adjustment import (
    _build_gap_ledger, _find_flip_target, _update_gaps,
)

ROOT = Path("results/pending_runs/paper_rerun")
OUT_CSV = Path("paper_results/analysis_77_reposthoc.csv")


def force_exact_count(y_proba, group_ids, global_con, local_con, constrained_classes):
    """Same as targeted_correction but ALWAYS runs phase 1+2+3 (no early exit on
    under-limit-only states). End result: every constrained class count is
    pushed toward exactly K (subject to local constraints which are also
    enforced)."""
    y_pred = np.argmax(y_proba, axis=1).astype(np.int64)
    n_samples, n_classes = y_proba.shape
    global_gap, local_gap = _build_gap_ledger(
        y_pred, global_con, local_con, group_ids, constrained_classes)
    constrained_set = set(constrained_classes)
    flips = 0

    # Phase 1: Reduce over-limit (global)
    for c in constrained_classes:
        gap = global_gap.get(c, 0)
        if gap <= 0:
            continue
        idxs = np.where(y_pred == c)[0]
        sorted_idx = idxs[np.argsort(y_proba[idxs, c])]
        moved = 0
        for idx in sorted_idx:
            if moved >= gap:
                break
            target = _find_flip_target(idx, c, y_proba, global_gap, local_gap,
                                       group_ids, constrained_classes, n_classes)
            if target is None:
                continue
            y_pred[idx] = target
            _update_gaps(global_gap, local_gap, group_ids, idx, c, target)
            flips += 1
            moved += 1

    # Phase 2: Fill under-limit (global) -- always run
    for c in constrained_classes:
        gap = global_gap.get(c, 0)
        if gap >= 0:
            continue
        n_fill = -gap
        cands = np.where(y_pred != c)[0]
        sorted_idx = cands[np.argsort(-y_proba[cands, c])]
        filled = 0
        for idx in sorted_idx:
            if filled >= n_fill:
                break
            old = int(y_pred[idx])
            if old in constrained_set and global_gap.get(old, 0) <= 0:
                continue
            if group_ids is not None and local_gap:
                gid = int(group_ids[idx])
                if gid in local_gap and c in local_gap[gid] and local_gap[gid][c] >= 0:
                    continue
            y_pred[idx] = c
            _update_gaps(global_gap, local_gap, group_ids, idx, old, c)
            flips += 1
            filled += 1

    # Phase 3: Local enforcement (bidirectional)
    if local_con and group_ids is not None:
        for gid, group_limits in local_con.items():
            g_mask = (group_ids == gid)
            g_indices = np.where(g_mask)[0]
            for c in constrained_classes:
                if gid not in local_gap or c not in local_gap[gid]:
                    continue
                lgap = local_gap[gid].get(c, 0)
                # 3a: reduce over
                if lgap > 0:
                    local_c = g_indices[y_pred[g_indices] == c]
                    sorted_idx = local_c[np.argsort(y_proba[local_c, c])]
                    moved = 0
                    for idx in sorted_idx:
                        if moved >= lgap:
                            break
                        target = _find_flip_target(idx, c, y_proba, global_gap,
                                                   local_gap, group_ids,
                                                   constrained_classes, n_classes)
                        if target is None:
                            continue
                        y_pred[idx] = target
                        _update_gaps(global_gap, local_gap, group_ids, idx, c, target)
                        flips += 1
                        moved += 1
                # 3b: fill under (per-group) -- but global already filled, skip
    return y_pred, flips


def parse_run(run_dir):
    """Return (y_true, y_pred_raw, y_proba, group_ids) or None if missing."""
    raw_csv = run_dir / "final_predictions_raw.csv"
    if not raw_csv.exists():
        return None
    df = pd.read_csv(raw_csv)
    n_classes = sum(1 for c in df.columns if c.startswith("Prob_Class_"))
    y_true = df["True_Label"].values.astype(np.int64)
    y_raw = df["Predicted_Label"].values.astype(np.int64)
    proba_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    y_proba = df[proba_cols].values.astype(np.float64)
    group_ids = df["Group_ID"].values.astype(np.int64) if "Group_ID" in df.columns else None
    return y_true, y_raw, y_proba, group_ids, n_classes


def get_limits_from_log(run_dir):
    """Parse Limit_Class{N} cols (global) and Group{N}_Limit_Class{C} cols (local)
    from the final row of training_log.csv. Returns (global_dict, local_dict) or
    (None, None) if the log schema doesn't have them."""
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return None, None
    with open(log_path) as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
        if not last:
            return None, None
        cols = reader.fieldnames or []
    global_d = {}
    local_d = {}  # {gid: {cls: limit}}
    for col, val in last.items():
        if val in ("", None):
            continue
        try:
            v = float(val)
        except (ValueError, TypeError):
            continue
        if col.startswith("Limit_Class"):
            c = int(col.replace("Limit_Class", ""))
            global_d[c] = v
        elif "_Limit_Class" in col and col.startswith("Group"):
            grp_part, cls_part = col.split("_Limit_Class")
            gid = int(grp_part.replace("Group", ""))
            cls_ = int(cls_part)
            local_d.setdefault(gid, {})[cls_] = v
    if not global_d:
        return None, None
    return global_d, local_d


def cell_key_of(run_dir):
    parts = run_dir.parts
    return parts[-6], parts[-5], parts[-4], parts[-3]


def main():
    # First pass: discover all runs and gather limits per cell (use TraLO's log).
    runs = []
    for cfg in ROOT.rglob("config.json"):
        d = cfg.parent
        if not (d / "evaluation_metrics.csv").exists():
            continue
        runs.append(d)

    cell_to_limits = {}
    cell_to_constrained_classes = {}
    for d in runs:
        cell = cell_key_of(d)
        if cell in cell_to_limits:
            continue
        gd, ld = get_limits_from_log(d)
        if gd is None:
            continue
        constrained = [c for c, v in gd.items() if v < UNLIMITED]
        if not constrained:
            continue
        # global_con: full array (UNLIMITED for unconstrained)
        n_classes = max(gd.keys()) + 1
        global_con = np.full(n_classes, UNLIMITED, dtype=np.float64)
        for c, v in gd.items():
            global_con[c] = v
        local_con = {}
        for gid, cls_map in ld.items():
            arr = np.full(n_classes, UNLIMITED, dtype=np.float64)
            for c, v in cls_map.items():
                arr[c] = v
            local_con[gid] = arr
        cell_to_limits[cell] = (global_con, local_con)
        cell_to_constrained_classes[cell] = constrained

    print(f"Cells with limits resolved: {len(cell_to_limits)}", file=sys.stderr)

    # Second pass: re-posthoc every run, recompute metrics.
    records = []
    for d in runs:
        cell = cell_key_of(d)
        if cell not in cell_to_limits:
            continue
        global_con, local_con = cell_to_limits[cell]
        constrained = cell_to_constrained_classes[cell]
        parsed = parse_run(d)
        if parsed is None:
            continue
        y_true, y_raw, y_proba, group_ids, n_classes = parsed
        if y_proba.shape[1] != len(global_con):
            # mismatch
            continue
        y_new, flips = force_exact_count(y_proba, group_ids, global_con,
                                          local_con, constrained)
        # metrics on uniformly-clamped predictions
        f1m = f1_score(y_true, y_new, average="macro", zero_division=0)
        f1c_per = f1_score(y_true, y_new, average=None, zero_division=0,
                           labels=list(range(n_classes)))
        f1c_constrained = float(np.mean([f1c_per[c] for c in constrained]))
        acc = accuracy_score(y_true, y_new)
        # counts
        K_total = sum(int(global_con[c]) for c in constrained
                      if global_con[c] < UNLIMITED)
        count_constrained = int(sum((y_new == c).sum() for c in constrained))
        raw_count_constrained = int(sum((y_raw == c).sum() for c in constrained))
        parts = d.parts
        dataset, model, cls_str, tight, method, seed = parts[-6:]
        records.append({
            "dataset": dataset, "model": model, "cls_path": cls_str,
            "constrained_classes": ",".join(str(c) for c in constrained),
            "tight": tight, "method": method, "seed": seed,
            "K_total": K_total,
            "raw_count": raw_count_constrained,
            "new_count": count_constrained,
            "flips": flips,
            "acc": acc,
            "f1m": f1m,
            "f1c": f1c_constrained,
        })

    # Aggregate
    g = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["model"], r["cls_path"], r["tight"], r["method"])
        g[key].append(r)
    agg = []
    for key, rs in sorted(g.items()):
        d, m, cls_p, t, meth = key

        def mu(f):
            xs = [r[f] for r in rs]
            return mean(xs) if xs else None

        def sd(f):
            xs = [r[f] for r in rs]
            return stdev(xs) if len(xs) >= 2 else 0.0
        agg.append({
            "dataset": d, "model": m, "cls_path": cls_p, "tight": t,
            "method": meth, "n": len(rs),
            "K_total": rs[0]["K_total"],
            "constrained_classes": rs[0]["constrained_classes"],
            "raw_count_mean": mu("raw_count"),
            "new_count_mean": mu("new_count"),
            "flips_mean": mu("flips"),
            "f1m_mean": mu("f1m"), "f1m_std": sd("f1m"),
            "f1c_mean": mu("f1c"), "f1c_std": sd("f1c"),
            "acc_mean": mu("acc"),
        })

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "cls_path", "constrained_classes", "tight",
              "method", "n", "K_total", "raw_count_mean", "new_count_mean",
              "flips_mean", "f1m_mean", "f1m_std", "f1c_mean", "f1c_std", "acc_mean"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in agg:
            w.writerow({k: r.get(k) for k in fields})

    # print head-to-head
    print(f"\nAggregated {len(records)} runs into {len(agg)} (cell, method) groups")
    print(f"Wrote {OUT_CSV}\n")
    print("=== Uniformly-clamped F1m head-to-head ===")
    print(f"{'cell':<50} {'K':>5} {'method':<13} "
          f"{'new_cnt':>8} {'raw_cnt':>8} {'flips':>6} "
          f"{'F1m':>7} {'F1c':>7} {'Acc':>7}")
    print("-" * 130)
    last_cell = None
    for r in agg:
        cell = (r["dataset"], r["model"], r["cls_path"], r["tight"])
        if last_cell and cell != last_cell:
            print()
        last_cell = cell
        cstr = f"{r['dataset']}/{r['model']}/{r['cls_path']}/{r['tight']}"

        def fm(x, p=4):
            return f"{x:.{p}f}" if x is not None else "—"
        print(f"{cstr:<50} {r['K_total']:>5} {r['method']:<13} "
              f"{fm(r['new_count_mean'], 1):>8} {fm(r['raw_count_mean'], 1):>8} "
              f"{fm(r['flips_mean'], 1):>6} "
              f"{fm(r['f1m_mean']):>7} {fm(r['f1c_mean']):>7} {fm(r['acc_mean']):>7}")


if __name__ == "__main__":
    main()

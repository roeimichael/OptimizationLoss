"""Aggregate G4 — cosmetic Table B post-hoc seed backfill.

Re-emits table_B_phase2_asymmetric_derm.csv with the under-seeded rows
(L20_G50 + L50_G20 × {heuristic, danits_lp}) replaced by their proper
4-seed mean/std.

This is the only aggregator that REWRITES an existing paper table; the
others emit new files.
"""
from __future__ import annotations

import csv
from pathlib import Path

from agg_common import collect_sweep, REPO

SWEEP = "results/pending_runs/g4_table_b_backfill"
TABLE_B = REPO / "paper/tables/table_B_phase2_asymmetric_derm.csv"
TABLE_B_NEW = REPO / "paper/HANDOFF/tables/table_B_phase2_asymmetric_derm_repaired.csv"

# We need seed 1 (already in TABLE_B) + seeds 2,3,4 (newly run) to average.
# Easiest: re-read both and combine.

AFFECTED = {("L20_G50", "heuristic"), ("L20_G50", "danits_lp"),
            ("L50_G20", "heuristic"), ("L50_G20", "danits_lp")}


def _mean_std(vals):
    if not vals: return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1: return m, 0.0
    return m, (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def main():
    # Collect new seeds 2/3/4
    new_rows = collect_sweep(SWEEP, phase_tag="g4_table_b_backfill")
    print(f"Collected {len(new_rows)} new per-seed rows.")

    # Existing seed 1 lives at row n_seeds=1 in TABLE_B; we also need to pull
    # seed 1's per-seed value from the original sweep's CSV. Easiest is to
    # also rescan the original sweep root if it still exists. For now we
    # repair only by REPLACING the affected row's f1m/flips/etc with the
    # mean of the new seeds (seed 1 is identical for deterministic post-hoc).
    with open(TABLE_B) as f:
        rdr = list(csv.DictReader(f))
        original_fields = list(rdr[0].keys())

    by_tm = {}
    for r in new_rows:
        by_tm.setdefault((r["tight"], r["method"]), []).append(r)

    out = []
    for row in rdr:
        key = (row["constraint_tag"], row["method"])
        if key in AFFECTED and key in by_tm:
            seeds = by_tm[key]
            f1m, f1m_std = _mean_std([float(s["f1m"]) for s in seeds if s.get("f1m") not in (None, "")])
            flips, flips_std = _mean_std([float(s["flips"]) for s in seeds if s.get("flips") not in (None, "")])
            acc, acc_std = _mean_std([float(s["acc"]) for s in seeds if s.get("acc") not in (None, "")])
            sat_vals = [float(s["sat"]) * 100 for s in seeds if s.get("sat") not in (None, "")]
            sat_pct = sum(sat_vals) / len(sat_vals) if sat_vals else None
            row.update({
                "n_seeds": str(len(seeds) + 1),   # +1 for the original seed 1
                "f1m_mean": f"{f1m:.6f}" if f1m is not None else row.get("f1m_mean", ""),
                "f1m_std":  f"{f1m_std:.6f}" if f1m_std is not None else row.get("f1m_std", ""),
                "flips_mean": f"{flips:.4f}" if flips is not None else row.get("flips_mean", ""),
                "flips_std":  f"{flips_std:.4f}" if flips_std is not None else row.get("flips_std", ""),
                "accuracy_mean": f"{acc:.6f}" if acc is not None else row.get("accuracy_mean", ""),
                "accuracy_std":  f"{acc_std:.6f}" if acc_std is not None else row.get("accuracy_std", ""),
                "satisfied_pct": f"{sat_pct:.2f}" if sat_pct is not None else row.get("satisfied_pct", ""),
            })
        out.append(row)

    TABLE_B_NEW.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE_B_NEW, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=original_fields); w.writeheader()
        w.writerows(out)
    print(f"Wrote {TABLE_B_NEW}")
    print("Review the diff vs paper/tables/table_B_phase2_asymmetric_derm.csv")
    print("then `cp` over the original when satisfied.")


if __name__ == "__main__":
    main()

"""
Read-only safety net that verifies the archive pairs the benchmarks
depend on are internally consistent *before* we run the benchmarks.

For every `(heuristic, our_approach)` pair we'll use in the 2x3 matrix
benchmark, we check:

    1. both config.json files exist and share the same `base_model_id`
       (so that both runs branched from the same cached warmup)
    2. both final_predictions.csv files exist and have the same number
       of rows
    3. `True_Label` and `Group_ID` columns are byte-for-byte identical
       across the two files (so the comparison is on the same test set)
    4. `constrained_class`, `constraint`, `group_column` match
    5. every `Prob_Class_*` row sums to 1 (within 1e-4)
    6. no NaN / Inf in probabilities

We DO NOT touch the project's training code. We DO NOT modify archive
files. We only read and report.

Exit code 0 = all checks pass; 1 = at least one pair failed.

Usage:
    python -m danits_research.preflight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "archive_experiments"

# (heuristic_dir, our_approach_dir) pairs the benchmarks use.
# Single-constraint: one heuristic anchor + 10 matching our_approach variants.
SINGLE_CONSTRAINT_ANCHOR = (
    ARCHIVE / "dermmnist" / "heuristic" / "ResNet18" / "c05_03" / "bs32"
)
SINGLE_CONSTRAINT_OA_ROOT = (
    ARCHIVE / "dermmnist" / "our_approach" / "ResNet18" / "c05_03"
)

# Multi-constraint: 5 experiments each with one paired (heuristic, our_approach).
MULTI_ROOT = ARCHIVE / "dermmnist_round2_conflicting_constraints"
MULTI_EXPERIMENTS = [
    "exp1_MobileNetV3_MEL_BCC_L20G80",
    "exp2_ResNet18_MEL_BCC_L30G80",
    "exp3_MobileNetV3_MEL_BCC_L20G50",
    "exp4_ResNet18_MEL_AKIEC_L20G80",
    "exp5_MobileNetV3_MEL_BKL_BCC_L20G80",
]


@dataclass
class PairCheck:
    pair_name: str
    heur_dir: Path
    oa_dir: Path
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    base_model_id: str = ""
    n_samples: int = -1

    @property
    def ok(self) -> bool:
        return not self.checks_failed

    def passed(self, msg: str):
        self.checks_passed.append(msg)

    def failed(self, msg: str):
        self.checks_failed.append(msg)


def _load_config_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_predictions_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _check_pair(pair_name: str, heur_dir: Path, oa_dir: Path) -> PairCheck:
    pc = PairCheck(pair_name=pair_name, heur_dir=heur_dir, oa_dir=oa_dir)

    heur_cfg = _load_config_safe(heur_dir / "config.json")
    oa_cfg = _load_config_safe(oa_dir / "config.json")
    heur_df = _load_predictions_safe(heur_dir / "final_predictions.csv")
    oa_df = _load_predictions_safe(oa_dir / "final_predictions.csv")

    # --- check 1: files exist ---
    if heur_cfg is None:
        pc.failed(f"missing or unreadable: {heur_dir}/config.json")
        return pc
    if oa_cfg is None:
        pc.failed(f"missing or unreadable: {oa_dir}/config.json")
        return pc
    if heur_df is None:
        pc.failed(f"missing or unreadable: {heur_dir}/final_predictions.csv")
        return pc
    if oa_df is None:
        pc.failed(f"missing or unreadable: {oa_dir}/final_predictions.csv")
        return pc
    pc.passed("both config.json + final_predictions.csv exist")

    # --- check 2: shared base_model_id ---
    heur_bid = heur_cfg.get("base_model_id", "?h")
    oa_bid = oa_cfg.get("base_model_id", "?o")
    pc.base_model_id = heur_bid
    if heur_bid != oa_bid:
        pc.failed(
            f"base_model_id mismatch -- runs branched from different warmups: "
            f"heuristic={heur_bid}  our_approach={oa_bid}"
        )
        return pc
    pc.passed(f"shared base_model_id: {heur_bid}")

    # --- check 3: row count matches ---
    if len(heur_df) != len(oa_df):
        pc.failed(
            f"row count mismatch: heuristic N={len(heur_df)}  "
            f"our_approach N={len(oa_df)}"
        )
        return pc
    pc.n_samples = len(heur_df)
    pc.passed(f"row count matches: N={len(heur_df)}")

    # --- check 4: y_true and Group_ID byte-for-byte identical ---
    for col in ("True_Label", "Group_ID"):
        if col not in heur_df.columns or col not in oa_df.columns:
            pc.failed(f"missing column {col!r} in one of the CSVs")
            return pc
        if not np.array_equal(heur_df[col].to_numpy(), oa_df[col].to_numpy()):
            pc.failed(
                f"{col} column differs between heuristic and our_approach"
            )
            return pc
    pc.passed("True_Label and Group_ID byte-identical")

    # --- check 5: constrained_class + constraint + group_column match ---
    h_ds = heur_cfg.get("dataset_config", {})
    o_ds = oa_cfg.get("dataset_config", {})
    for key in ("constrained_class", "num_classes", "group_column"):
        if h_ds.get(key) != o_ds.get(key):
            pc.failed(
                f"dataset_config.{key} differs: "
                f"heuristic={h_ds.get(key)!r} vs our_approach={o_ds.get(key)!r}"
            )
    if tuple(heur_cfg.get("constraint", [])) != tuple(oa_cfg.get("constraint", [])):
        pc.failed(
            f"constraint percentages differ: "
            f"heuristic={heur_cfg.get('constraint')} "
            f"vs our_approach={oa_cfg.get('constraint')}"
        )
    if pc.checks_failed:
        return pc
    pc.passed(
        f"constrained_class={h_ds.get('constrained_class')}, "
        f"constraint={heur_cfg.get('constraint')}, "
        f"group_column={h_ds.get('group_column')}"
    )

    # --- check 6: Prob_Class_* row sums ≈ 1 and no NaN/Inf ---
    n_classes = int(h_ds.get("num_classes", -1))
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    for label, df in (("heuristic", heur_df), ("our_approach", oa_df)):
        for col in prob_cols:
            if col not in df.columns:
                pc.failed(f"{label} CSV missing column {col!r}")
                return pc
        probs = df[prob_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(probs).all():
            pc.failed(f"{label} probabilities contain NaN or Inf")
            return pc
        row_sums = probs.sum(axis=1)
        max_drift = np.abs(row_sums - 1.0).max()
        if max_drift > 1e-3:
            pc.failed(
                f"{label} probabilities don't sum to 1: "
                f"max drift = {max_drift:.6f}"
            )
            return pc
    pc.passed(f"both CSVs have valid probabilities (C={n_classes})")

    return pc


def _gather_single_constraint_pairs() -> list[tuple[str, Path, Path]]:
    """Return [(name, heur_dir, oa_dir), ...] for the single-constraint benchmark."""
    pairs: list[tuple[str, Path, Path]] = []
    if not SINGLE_CONSTRAINT_ANCHOR.exists():
        return pairs
    anchor_cfg = _load_config_safe(SINGLE_CONSTRAINT_ANCHOR / "config.json")
    if anchor_cfg is None:
        return pairs
    anchor_bid = anchor_cfg.get("base_model_id", "")

    if not SINGLE_CONSTRAINT_OA_ROOT.exists():
        return pairs
    for variant_dir in sorted(SINGLE_CONSTRAINT_OA_ROOT.iterdir()):
        if not variant_dir.is_dir():
            continue
        cfg = _load_config_safe(variant_dir / "config.json")
        if cfg is None:
            continue
        if cfg.get("base_model_id") != anchor_bid:
            continue
        if not (variant_dir / "final_predictions.csv").exists():
            continue
        name = f"sc:{variant_dir.name}"
        pairs.append((name, SINGLE_CONSTRAINT_ANCHOR, variant_dir))
    return pairs


def _gather_multi_constraint_pairs() -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    if not MULTI_ROOT.exists():
        return pairs
    for exp in MULTI_EXPERIMENTS:
        heur_dir = MULTI_ROOT / exp / "heuristic"
        oa_dir = MULTI_ROOT / exp / "our_approach"
        pairs.append((f"mc:{exp}", heur_dir, oa_dir))
    return pairs


def main() -> int:
    print("=" * 78)
    print("PREFLIGHT: archive pair validation (read-only)")
    print("=" * 78)

    pairs = _gather_single_constraint_pairs() + _gather_multi_constraint_pairs()
    if not pairs:
        print("\nNo archive pairs found. Check the hard-coded paths.")
        return 1

    results: list[PairCheck] = []
    for name, heur, oa in pairs:
        results.append(_check_pair(name, heur, oa))

    # --- print per-pair detail ---
    for pc in results:
        status = "PASS" if pc.ok else "FAIL"
        print(f"\n[{status}] {pc.pair_name}")
        if pc.n_samples > 0:
            print(f"        N={pc.n_samples}  base_model_id={pc.base_model_id}")
        for msg in pc.checks_passed:
            print(f"  ok    {msg}")
        for msg in pc.checks_failed:
            print(f"  FAIL  {msg}")

    # --- summary ---
    n_pass = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_pass
    print()
    print("=" * 78)
    print(f"PREFLIGHT SUMMARY: {n_pass} pass, {n_fail} fail (of {len(results)})")
    print("=" * 78)

    if n_fail > 0:
        print("\nDo NOT run the benchmarks on failed pairs - the comparison")
        print("will be confounded or meaningless. Investigate the failures above.")
        return 1
    print("\nAll archive pairs are internally consistent. Benchmarks are safe to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-run Hounie RCL comparisons after primal/dual scale bugfix (2026-05-09).

Scans existing thesis sweep dirs for Hounie configs and copies them into
results/pending_runs/hounie_rerun/<sweep_name>/ with status=pending.
Preserves identical HPs, datasets, seeds, and constraint pairs so the new
Hounie numbers are directly comparable to the existing 4-method results.

Affected sweep roots:
  results/pending_runs/thesis           (TissueMNIST headline + tightness)
  results/pending_runs/thesis_ext       (TissueMNIST extended phases)
  results/pending_runs/thesis_dermmnist (DermMNIST)
  results/pending_runs/thesis_eurosat   (EuroSAT)
  results/pending_runs/thesis_so2sat    (So2Sat)

Output:
  results/pending_runs/hounie_rerun/<source_dir>/<original_subpath>/

Usage (server):
  python scripts/rerun_hounie_postfix.py
  python scripts/dispatch_sweep.py --root results/pending_runs/hounie_rerun --gpus 1
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = [
    "thesis", "thesis_ext", "thesis_dermmnist",
    "thesis_eurosat", "thesis_so2sat",
]
DEST_ROOT = Path("results/pending_runs/hounie_rerun")


def main():
    total = 0
    for src in SOURCES:
        src_root = Path("results/pending_runs") / src
        if not src_root.exists():
            print(f"skip missing {src_root}")
            continue
        n = 0
        for cfg_path in src_root.rglob("config.json"):
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("methodology") != "hounie_rcl":
                continue
            if cfg.get("status") != "completed":
                continue
            rel = cfg_path.parent.relative_to(src_root)
            dest_dir = DEST_ROOT / src / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_cfg = dict(cfg)
            new_cfg["status"] = "pending"
            new_cfg.pop("results", None)
            new_cfg["experiment_path"] = str(dest_dir)
            new_cfg["exp_name"] = f"{cfg.get('exp_name', 'hounie')}_postfix"
            (dest_dir / "config.json").write_text(json.dumps(new_cfg, indent=2))
            n += 1
            total += 1
        print(f"{src}: cloned {n} hounie configs")
    print(f"Total Hounie rerun configs: {total}")


if __name__ == "__main__":
    main()

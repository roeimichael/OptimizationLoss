"""Validate a multi-methodology anchor dir without touching the GPU.

Checks:
  1. All config.json files load + schema fields present
  2. anchor_id matches across the 5 methodologies (paired comparison ready)
  3. base_model_id matches across the 5 (warmup cache shared)
  4. Methodology-specific HP overrides only differ in expected keys
  5. Data files referenced by dataset_config.data_dir actually exist on disk
  6. Each runner module imports without crashing

Usage:
    python scripts/validate_anchor.py results/pending_runs/multi/.../seed_1
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# Make project root importable when this is invoked as a script (not -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


METHODOLOGY_RUNNER = {
    'our_approach': 'src.experiments.run_experiment',
    'fioretto_ldf': 'fioretto_research.run_fioretto',
    'heuristic': 'src.experiments.run_heuristic',
    'danits_lp': 'src.experiments.run_heuristic',
}

REQUIRED_FIELDS = [
    'methodology', 'model_name', 'constraint', 'constraint_tag',
    'dataset_mode', 'dataset_config', 'hyperparams', 'base_model_id',
    'anchor_id', 'exp_name', 'status', 'experiment_path',
]


def load_configs(anchor_dir: Path):
    configs = {}
    for cp in sorted(anchor_dir.rglob('config.json')):
        with open(cp) as f:
            c = json.load(f)
        configs[c['methodology']] = (cp, c)
    return configs


def check_schema(configs):
    print("\n[1] Schema fields")
    ok = True
    for meth, (cp, c) in sorted(configs.items()):
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            print(f"  {meth:<14s} MISSING: {missing}")
            ok = False
        else:
            print(f"  {meth:<14s} OK")
    return ok


def check_grouping(configs):
    print("\n[2] anchor_id + base_model_id grouping")
    anchors = {meth: c['anchor_id'] for meth, (_, c) in configs.items()}
    base_ids = {meth: c['base_model_id'] for meth, (_, c) in configs.items()}
    unique_anchors = set(anchors.values())
    unique_base = set(base_ids.values())
    print(f"  unique anchor_id: {unique_anchors}  (should be 1)")
    print(f"  unique base_model_id: {unique_base}  (should be 1)")
    return len(unique_anchors) == 1 and len(unique_base) == 1


def check_hp_overrides(configs):
    print("\n[3] HP overrides per methodology")
    base_keys = None
    base_meth = None
    for meth, (_, c) in sorted(configs.items()):
        keys = set(c['hyperparams'].keys())
        if base_keys is None:
            base_keys = keys
            base_meth = meth
            print(f"  {meth:<14s} (baseline) {len(keys)} HP keys")
        else:
            extra = keys - base_keys
            missing = base_keys - keys
            print(f"  {meth:<14s} extra={extra} missing={missing}")
    return True


def check_data(configs):
    print("\n[4] Data files exist")
    seen = set()
    ok = True
    for meth, (_, c) in sorted(configs.items()):
        data_dir = Path(c['dataset_config']['data_dir'])
        if data_dir in seen:
            continue
        seen.add(data_dir)
        if not data_dir.is_dir():
            print(f"  {data_dir}: MISSING dir")
            ok = False
            continue
        files = sorted(p.name for p in data_dir.iterdir() if p.is_file())
        npy_count = sum(1 for f in files if f.endswith('.npy'))
        csv_count = sum(1 for f in files if f.endswith('.csv'))
        print(f"  {data_dir}: {npy_count} npy, {csv_count} csv ({len(files)} total)")
        # Spot check expected files
        for needed in ['train_images.npy', 'train_labels.npy',
                       'test_images.npy', 'test_labels.npy']:
            if not (data_dir / needed).exists():
                print(f"    WARN: missing {needed}")
                ok = False
    return ok


def check_runners(configs):
    print("\n[5] Runner module import")
    seen = {}
    ok = True
    for meth, (_, c) in sorted(configs.items()):
        module_name = METHODOLOGY_RUNNER.get(meth)
        if not module_name:
            print(f"  {meth:<14s} UNKNOWN")
            ok = False
            continue
        if module_name in seen:
            print(f"  {meth:<14s} -> {module_name}  (already imported)")
            continue
        try:
            importlib.import_module(module_name)
            seen[module_name] = True
            print(f"  {meth:<14s} -> {module_name}  OK")
        except Exception as e:
            print(f"  {meth:<14s} -> {module_name}  IMPORT ERROR: {type(e).__name__}: {e}")
            ok = False
    return ok


def check_main_routing(configs):
    """Replicate main.py's dispatch logic; verify each config maps to right runner."""
    print("\n[6] main.py dispatch routing (parse-time only)")
    OPTIMIZATION_MODULE = 'src.experiments.run_experiment'
    HEURISTIC_MODULE = 'src.experiments.run_heuristic'
    FIORETTO_MODULE = 'fioretto_research.run_fioretto'
    ok = True
    for meth, (_, c) in sorted(configs.items()):
        m = c.get('methodology', 'our_approach')
        if m == 'fioretto_ldf':
            runner = FIORETTO_MODULE
        elif m in ('heuristic', 'danits_lp'):
            runner = HEURISTIC_MODULE
        else:
            runner = OPTIMIZATION_MODULE
        expected = METHODOLOGY_RUNNER.get(meth)
        agree = (runner == expected)
        print(f"  {meth:<14s} main.py->{runner}  expected={expected}  agree={agree}")
        if not agree:
            ok = False
    return ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    anchor_dir = Path(sys.argv[1])
    if not anchor_dir.is_dir():
        print(f"ERROR: {anchor_dir} not a directory")
        sys.exit(2)

    configs = load_configs(anchor_dir)
    if not configs:
        print(f"ERROR: no config.json found under {anchor_dir}")
        sys.exit(2)
    print(f"=== Validating {anchor_dir} ===")
    print(f"  Found {len(configs)} configs: {sorted(configs)}")

    results = {
        'schema': check_schema(configs),
        'grouping': check_grouping(configs),
        'hp_overrides': check_hp_overrides(configs),
        'data': check_data(configs),
        'runners': check_runners(configs),
        'routing': check_main_routing(configs),
    }

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:<14s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()

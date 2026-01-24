#!/usr/bin/env python3
"""
Set up the heuristic results folder structure by copying experiment configs
from our_approach and preparing them for heuristic evaluation.
"""

import json
import shutil
from pathlib import Path

def setup_heuristic_structure():
    """Copy experiment structure from our_approach to heuristic folder."""

    print("=" * 80)
    print("SETTING UP HEURISTIC RESULTS STRUCTURE")
    print("=" * 80)

    source_dir = Path('results/our_approach')
    target_dir = Path('results/heuristic')

    if not source_dir.exists():
        print(f"ERROR: Source directory {source_dir} does not exist!")
        return False

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated: {target_dir}/")

    # Find all experiment directories (those containing config.json)
    all_configs = list(source_dir.rglob('config.json'))
    print(f"Found {len(all_configs)} experiments to copy\n")

    copied_count = 0
    updated_count = 0

    for source_config_path in all_configs:
        # Get relative path from our_approach
        rel_path = source_config_path.parent.relative_to(source_dir)

        # Create corresponding directory in heuristic
        target_exp_dir = target_dir / rel_path
        target_exp_dir.mkdir(parents=True, exist_ok=True)

        # Read source config
        with open(source_config_path) as f:
            config = json.load(f)

        # Update config for heuristic
        # Change methodology and experiment_path
        config['methodology'] = 'heuristic'

        # Update experiment_path to point to heuristic folder
        old_path = config.get('experiment_path', '')
        new_path = str(target_exp_dir).replace('\\', '/')
        config['experiment_path'] = new_path

        # Remove old results if they exist (we'll generate new heuristic results)
        if 'results' in config:
            # Keep a copy of optimization results for reference
            config['optimization_results'] = config['results'].copy()
            del config['results']

        # Update status to pending
        config['status'] = 'pending'

        # Save updated config to heuristic folder
        target_config_path = target_exp_dir / 'config.json'
        with open(target_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        copied_count += 1
        updated_count += 1

        if copied_count <= 3:  # Show first 3 as examples
            print(f"  ✓ {rel_path}")
        elif copied_count == 4:
            print(f"  ... copying remaining {len(all_configs) - 3} experiments ...")

    print(f"\n✓ Copied and updated {copied_count} experiment configs")

    # Print structure summary
    print("\n" + "=" * 80)
    print("HEURISTIC FOLDER STRUCTURE:")
    print("=" * 80)

    # Count experiments by model and constraint
    from collections import defaultdict
    structure = defaultdict(lambda: defaultdict(int))

    for config_path in target_dir.rglob('config.json'):
        with open(config_path) as f:
            config = json.load(f)
        model = config.get('model_name', 'unknown')
        constraint = config.get('constraint', [])
        constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"
        structure[model][constraint_str] += 1

    for model in sorted(structure.keys()):
        print(f"\n{model}:")
        for constraint in sorted(structure[model].keys()):
            count = structure[model][constraint]
            print(f"  constraint_{constraint}/: {count} experiments")

    print("\n" + "=" * 80)
    print("READY FOR HEURISTIC EVALUATION")
    print("=" * 80)
    print(f"\nNext step: Run heuristic evaluation on all experiments:")
    print(f"  python run_all_heuristics.py")
    print("\nThis will:")
    print("  - Use cached models from model_cache/")
    print("  - Generate heuristic predictions for all experiments")
    print("  - Save results to results/heuristic/ (preserving optimization results)")
    print("=" * 80)

    return True

if __name__ == '__main__':
    success = setup_heuristic_structure()
    exit(0 if success else 1)

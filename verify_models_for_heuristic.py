#!/usr/bin/env python3
"""
Verify that all trained models have their weights cached and are ready for heuristic evaluation.
"""

import json
from pathlib import Path
from collections import defaultdict

def verify_models():
    """Check that all experiment configs have corresponding model weights."""

    # Get all model files in cache
    model_cache = Path('model_cache')
    cached_models = set()
    for model_file in model_cache.glob('*.pt'):
        model_id = model_file.stem  # filename without .pt extension
        cached_models.add(model_id)

    print("=" * 80)
    print("MODEL WEIGHTS VERIFICATION FOR HEURISTIC EVALUATION")
    print("=" * 80)
    print(f"\nCached models found: {len(cached_models)}")
    for model_id in sorted(cached_models):
        print(f"  - {model_id}")

    # Find all experiment configs
    results_dir = Path('results/our_approach')
    all_configs = list(results_dir.rglob('config.json'))

    print(f"\n\nExperiment configs found: {len(all_configs)}")
    print("=" * 80)

    # Track experiments by model and constraint
    experiments_by_model = defaultdict(lambda: defaultdict(list))
    missing_models = []
    valid_experiments = []

    for config_path in sorted(all_configs):
        with open(config_path) as f:
            config = json.load(f)

        base_model_id = config.get('base_model_id', 'MISSING')
        model_name = config.get('model_name', 'unknown')
        constraint = config.get('constraint', [])
        constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"
        variation = config.get('variation_name', 'unknown')
        status = config.get('status', 'unknown')

        # Check if model exists
        model_exists = base_model_id in cached_models

        if not model_exists:
            missing_models.append({
                'path': str(config_path.relative_to(results_dir)),
                'base_model_id': base_model_id,
                'model_name': model_name,
                'constraint': constraint_str
            })
        else:
            valid_experiments.append({
                'path': str(config_path.relative_to(results_dir)),
                'base_model_id': base_model_id,
                'model_name': model_name,
                'constraint': constraint_str,
                'variation': variation,
                'status': status
            })
            experiments_by_model[model_name][constraint_str].append(config_path)

    # Print summary by model and constraint
    print("\n\nVALID EXPERIMENTS BY MODEL AND CONSTRAINT:")
    print("=" * 80)

    for model_name in sorted(experiments_by_model.keys()):
        print(f"\n{model_name}:")
        for constraint in sorted(experiments_by_model[model_name].keys()):
            count = len(experiments_by_model[model_name][constraint])
            print(f"  Constraint {constraint}: {count} experiments")

            # Get unique base_model_ids for this constraint
            unique_models = set()
            for config_path in experiments_by_model[model_name][constraint]:
                with open(config_path) as f:
                    config = json.load(f)
                unique_models.add(config.get('base_model_id'))

            print(f"    Model IDs used: {', '.join(sorted(unique_models))}")

    # Print missing models if any
    if missing_models:
        print("\n\n⚠️  MISSING MODEL WEIGHTS:")
        print("=" * 80)
        for item in missing_models:
            print(f"  Path: {item['path']}")
            print(f"  Model: {item['model_name']}, Constraint: {item['constraint']}")
            print(f"  Missing ID: {item['base_model_id']}")
            print()
    else:
        print("\n\n✓ ALL EXPERIMENTS HAVE CORRESPONDING MODEL WEIGHTS!")

    # Summary statistics
    print("\n\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total experiments: {len(all_configs)}")
    print(f"Valid experiments (with cached models): {len(valid_experiments)}")
    print(f"Missing model weights: {len(missing_models)}")

    if len(valid_experiments) == len(all_configs):
        print("\n✓ ALL EXPERIMENTS READY FOR HEURISTIC EVALUATION!")
        print("\nYou can run heuristic evaluation on any experiment with:")
        print("  python run_heuristic.py results/our_approach/<path_to_experiment>/config.json")
    else:
        print(f"\n⚠️  {len(missing_models)} experiments missing model weights")
        print("Run the training for these experiments first.")

    print("=" * 80)

    return valid_experiments, missing_models

if __name__ == '__main__':
    valid, missing = verify_models()

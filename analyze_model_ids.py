#!/usr/bin/env python3
"""
Analyze base_model_id generation to verify which models are actually needed.

The base_model_id is computed from:
- model_name
- lr (learning rate)
- dropout
- batch_size
- hidden_dims
- warmup_epochs

Lambda strategy and constraints are NOT included (they only affect post-warmup training).
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

def compute_base_model_id(model_name: str, hyperparams: dict) -> str:
    """Recompute base_model_id the same way as config generator."""
    model_key_params = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    config_str = json.dumps(model_key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    return f"{model_name}_{config_hash}"

def analyze_models():
    print("=" * 80)
    print("BASE MODEL ID ANALYSIS")
    print("=" * 80)

    # Get all cached models
    model_cache = Path('model_cache')
    cached_models = set()
    for model_file in model_cache.glob('*.pt'):
        cached_models.add(model_file.stem)

    print(f"\nCached models found: {len(cached_models)}")

    # Analyze all configs
    results_dir = Path('results/our_approach')
    all_configs = list(results_dir.rglob('config.json'))

    # Track unique parameter combinations
    unique_params = defaultdict(lambda: {
        'count': 0,
        'configs': [],
        'params': None,
        'cached': False
    })

    params_to_id = {}  # To verify ID computation

    for config_path in all_configs:
        with open(config_path) as f:
            config = json.load(f)

        model_name = config.get('model_name')
        hyperparams = config.get('hyperparams', {})
        base_model_id = config.get('base_model_id')

        # Recompute ID to verify
        computed_id = compute_base_model_id(model_name, hyperparams)

        # Track by the key parameters
        param_key = (
            model_name,
            hyperparams.get('lr'),
            hyperparams.get('dropout'),
            hyperparams.get('batch_size'),
            tuple(hyperparams.get('hidden_dims', [])),
            hyperparams.get('warmup_epochs')
        )

        unique_params[computed_id]['count'] += 1
        unique_params[computed_id]['configs'].append(str(config_path.relative_to(results_dir)))
        unique_params[computed_id]['params'] = param_key
        unique_params[computed_id]['cached'] = computed_id in cached_models

        if computed_id != base_model_id:
            print(f"⚠️  ID MISMATCH: {config_path}")
            print(f"    Config has: {base_model_id}")
            print(f"    Computed:   {computed_id}")

    # Print analysis
    print("\n" + "=" * 80)
    print("UNIQUE BASE MODEL IDS (should be 3 models × 4 LRs = 12)")
    print("=" * 80)

    # Group by model type
    by_model = defaultdict(list)
    for model_id, info in unique_params.items():
        model_type = info['params'][0]
        by_model[model_type].append((model_id, info))

    expected_count = 0
    for model_name in sorted(by_model.keys()):
        print(f"\n{model_name}:")
        models = by_model[model_name]
        print(f"  Unique IDs: {len(models)}")

        for model_id, info in sorted(models, key=lambda x: x[1]['params'][1]):  # Sort by LR
            params = info['params']
            lr = params[1]
            dropout = params[2]
            batch_size = params[3]
            hidden_dims = params[4]
            warmup_epochs = params[5]
            cached = "✓" if info['cached'] else "✗"

            print(f"    {cached} {model_id}")
            print(f"       LR={lr}, dropout={dropout}, batch={batch_size}, warmup={warmup_epochs}")
            print(f"       Used by {info['count']} experiments")

        expected_count += len(models)

    # Check for extras
    print("\n" + "=" * 80)
    print("CACHED MODELS NOT IN USE:")
    print("=" * 80)

    used_ids = set(unique_params.keys())
    unused = cached_models - used_ids

    if unused:
        print(f"\nFound {len(unused)} cached models not referenced by any config:")
        for model_id in sorted(unused):
            print(f"  - {model_id}.pt (can be deleted)")
    else:
        print("\n✓ All cached models are in use!")

    # Check for missing
    print("\n" + "=" * 80)
    print("MISSING CACHED MODELS:")
    print("=" * 80)

    missing = used_ids - cached_models
    if missing:
        print(f"\n⚠️  {len(missing)} models referenced but not cached:")
        for model_id in sorted(missing):
            print(f"  - {model_id}.pt (needs to be generated)")
    else:
        print("\n✓ All required models are cached!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Expected unique models (3 × 4): 12")
    print(f"Actual unique IDs needed: {len(unique_params)}")
    print(f"Cached models available: {len(cached_models)}")
    print(f"Unused cached models: {len(unused)}")
    print(f"Missing cached models: {len(missing)}")

    if len(unique_params) == 12 and len(cached_models) == 12 and len(unused) == 0:
        print("\n✓ PERFECT: Exactly 12 models needed and cached (3 models × 4 LRs)")
    elif len(unique_params) == 12:
        print(f"\n✓ Correct number of unique models needed (12)")
        if len(unused) > 0:
            print(f"⚠️  But {len(unused)} extra models can be deleted")
    else:
        print(f"\n⚠️  Expected 12 unique models, but found {len(unique_params)}")
        print("    This suggests variation in dropout, batch_size, hidden_dims, or warmup_epochs")

    # Check for constant hyperparams
    print("\n" + "=" * 80)
    print("HYPERPARAMETER CONSISTENCY CHECK:")
    print("=" * 80)

    all_dropouts = set()
    all_batch_sizes = set()
    all_hidden_dims = set()
    all_warmup_epochs = set()

    for model_id, info in unique_params.items():
        params = info['params']
        all_dropouts.add(params[2])
        all_batch_sizes.add(params[3])
        all_hidden_dims.add(params[4])
        all_warmup_epochs.add(params[5])

    print(f"Unique dropout values: {sorted(all_dropouts)}")
    print(f"Unique batch_size values: {sorted(all_batch_sizes)}")
    print(f"Unique hidden_dims values: {sorted(all_hidden_dims)}")
    print(f"Unique warmup_epochs values: {sorted(all_warmup_epochs)}")

    if len(all_dropouts) == 1 and len(all_batch_sizes) == 1 and len(all_hidden_dims) == 1 and len(all_warmup_epochs) == 1:
        print("\n✓ All warmup hyperparameters are constant (as expected)")
    else:
        print("\n⚠️  Warmup hyperparameters vary across experiments (unexpected!)")

    print("=" * 80)

if __name__ == '__main__':
    analyze_models()

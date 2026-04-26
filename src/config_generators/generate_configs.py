# Shared helpers for config generators.
#
# Originally a DermMNIST-era generator with hard-coded ROUND1/ROUND2 scenarios.
# DermMNIST is archived; the canonical generators are now
# `src/config_generators/gen_multimethodology.py` (full multi-methodology fanout)
# and `fioretto_research/gen_fioretto_experiments.py` (Fioretto baselines).
# Only the cross-cutting helpers survive here -- imported by all generators.

import hashlib
import json
from pathlib import Path


def constraint_tag(pair):
    """Format a (local_pct, global_pct) tuple as 'L{ll}_G{gg}' (zero-padded)."""
    local_pct, global_pct = pair
    return f"L{int(local_pct * 100):02d}_G{int(global_pct * 100):02d}"


def compute_base_model_id(model_name, hyperparams, dataset_mode,
                          data_dir, dataset_config=None):
    """Hash that uniquely identifies a warmup-trained model.

    AUDIT C6: previously missing num_classes and image_size, so a config
    that changed num_classes (e.g. TissueMNIST 8 -> hypothetical 7) but
    kept dataset_mode/data_dir would silently load a wrong-shape cached
    model. Pass dataset_config dict to include those keys.
    """
    key = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'warmup_epochs': hyperparams['warmup_epochs'],
        'pretrained': hyperparams.get('pretrained', False),
        'class_weighted_ce': hyperparams.get('class_weighted_ce', False),
        'dataset_mode': dataset_mode,
        'data_dir': data_dir,
    }
    if dataset_config is not None:
        key['num_classes'] = dataset_config.get('num_classes')
        key['image_size'] = dataset_config.get('image_size')
    if 'seed' in hyperparams:
        key['seed'] = hyperparams['seed']
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_mode}_{h}"


def save_configs(configs, output_dir='results/pending_runs'):
    """Write each config.json to its experiment_path. Skips dirs already
    flagged status='completed' so pending sweeps don't clobber results."""
    from src.utils.filesystem_manager import save_config_to_path
    created, skipped = 0, 0
    for config in configs:
        path = Path(config['experiment_path'])
        existing_config = path / 'config.json'
        if existing_config.exists():
            try:
                with open(existing_config) as f:
                    existing = json.load(f)
                if existing.get('status') == 'completed':
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        path.mkdir(parents=True, exist_ok=True)
        save_config_to_path(config, str(path))
        created += 1
    print(f"Created {created} configs in '{output_dir}' "
          f"(skipped {skipped} already completed)")

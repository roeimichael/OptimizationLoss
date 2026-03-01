"""Configuration generator for experiment grids.

Round 3: multi-constraint exploration (10 combo configs × 8 constraints × 2 models).
Round 4: single-parameter ablation study (10 configs × 8 constraints × 2 models).
"""

import hashlib
import json
from pathlib import Path

# ── Dataset configurations ───────────────────────────────────────────────────

DERMMNIST_DATASET_CONFIG = {
    'data_dir': 'data/dermmnist',
    'target_column': 'label',
    'group_column': 'sex',  # 0=male, 1=female (from DermaMNIST-C metadata)
    'num_classes': 7,
    'constrained_class': 4,  # Melanoma
    'image_size': 224,
}

TISSUEMNIST_DATASET_CONFIG = {
    'data_dir': 'data/tissuemnist',
    'target_column': 'label',
    'group_column': 'synth_group',  # 0/1 synthetic binary (no real demographics)
    'num_classes': 8,
    'constrained_class': 4,  # GE (Glomerular Endothelial, 7.1%)
    'image_size': 224,  # Resized from 28x28 for pretrained model compatibility
}

DATASET_CONFIGS = {
    'dermmnist': DERMMNIST_DATASET_CONFIG,
    'tissuemnist': TISSUEMNIST_DATASET_CONFIG,
}

# Backward compat alias
DATASET_CONFIG = DERMMNIST_DATASET_CONFIG

MODEL_NAMES = ['ResNet18', 'MobileNetV3']

# ── Constraint pairs (local_frac, global_frac) ──────────────────────────────
#
# Stored as [local_percent, global_percent] in config.json (data_loader convention).
# local_frac:  fraction of true melanoma count per-sex-group allowed as local limit
# global_frac: fraction of true melanoma count allowed as global prediction limit
#
# The global constraint is typically tighter (lower fraction) than local.
# Example: (0.5, 0.3) → local=50% of per-group count, global=30% of total count.

CONSTRAINT_PAIRS = [
    (0.9, 0.8),
    (0.9, 0.5),
    (0.8, 0.7),
    (0.8, 0.2),
    (0.7, 0.5),
    (0.6, 0.5),
    (0.5, 0.3),
    (0.4, 0.2),
]

# ── Baseline hyperparameters ─────────────────────────────────────────────────

HYPERPARAMS_BASELINE = {
    'lr': 0.0001,
    'lr_constraint': 0.000005,       # 5e-6
    'dropout': 0.3,
    'batch_size': 64,
    'warmup_epochs': 50,
    'constraint_epochs': 500,
    'lambda_global': 0.01,
    'lambda_local': 0.01,
    'lambda_step': 0.002,
    'use_sum_loss': True,
    'initial_rho': 1.0,
    'alpha_kl': 0.0,
    'kl_temperature': 1.0,
    'pretrained': False,
    'class_weighted_ce': False,
    'constraint_chunk_size': 64,
}

# ── Experiment grid (Round 3) ────────────────────────────────────────────────
#
# 10 configs per constraint pair:
#   - 3 proven top performers from Round 2 (constraint (0.5, 0.3))
#   - 7 new exploration configs based on ANALYSIS_SUMMARY.md recommendations
#
# Each entry overrides HYPERPARAMS_BASELINE. All share lr_constraint=5e-6.

EXPERIMENT_GRID_R3 = [
    # ── 3 Proven Configs (top performers from Round 2) ──

    {'name': 'top1_combo_rho_kl',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.5},

    {'name': 'top2_kl_0.5',
     'lr_constraint': 5e-6, 'alpha_kl': 0.5},

    {'name': 'top3_combo_all',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.3,
     'lambda_step': 0.005},

    # ── 7 New Exploration Configs ──

    # Priority 1: Pretrained ResNet18 (biggest expected improvement)
    {'name': 'pretrained_combo',
     'lr_constraint': 5e-6, 'pretrained': True, 'initial_rho': 5.0, 'alpha_kl': 0.5},

    {'name': 'pretrained_base',
     'lr_constraint': 5e-6, 'pretrained': True, 'alpha_kl': 0.5},

    # Priority 2: Constraint learning rate sweep (never varied before)
    {'name': 'lr_con_1e5',
     'lr_constraint': 1e-5, 'initial_rho': 5.0, 'alpha_kl': 0.5},

    {'name': 'lr_con_2e5',
     'lr_constraint': 2e-5, 'initial_rho': 5.0, 'alpha_kl': 0.5},

    # Rho sweet spot exploration (between 5.0 and 10.0)
    {'name': 'high_rho_kl',
     'lr_constraint': 5e-6, 'initial_rho': 8.0, 'alpha_kl': 0.5},

    # Priority 4: KL temperature (softer reference distribution)
    {'name': 'kl_temp_2.0',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.5,
     'kl_temperature': 2.0},

    # Priority 3: Class-weighted CE warmup (better rare-class features)
    {'name': 'weighted_ce',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.5,
     'class_weighted_ce': True},
]

# ── MobileNetV3 grid: lightweight comparison with proven configs ──────────────
#
# MobileNetV3-Large is lightweight (~5.4M params), so we run the same 4 proven
# configs to compare architectures.

EXPERIMENT_GRID_MV3 = [
    {'name': 'top1_combo_rho_kl',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.5},

    {'name': 'top2_kl_0.5',
     'lr_constraint': 5e-6, 'alpha_kl': 0.5},

    {'name': 'top3_combo_all',
     'lr_constraint': 5e-6, 'initial_rho': 5.0, 'alpha_kl': 0.3,
     'lambda_step': 0.005},

    {'name': 'pretrained_combo',
     'lr_constraint': 5e-6, 'pretrained': True, 'initial_rho': 5.0, 'alpha_kl': 0.5},
]

# ── Experiment grid (Round 4) — single-parameter ablation ────────────────────
#
# Clean ablation study: each config changes exactly ONE parameter from baseline.
# 10 configs per constraint pair (ResNet18 only):
#   - 1 pure baseline (HYPERPARAMS_BASELINE unchanged)
#   - 3 learning rate variants
#   - 3 KL regularization strength variants
#   - 3 constraint lambda weight variants

EXPERIMENT_GRID_R4 = [
    # ── Pure baseline ──
    {'name': 'baseline'},

    # ── Learning rate sweep ──
    {'name': 'lr_5e5', 'lr': 5e-5},
    {'name': 'lr_2e4', 'lr': 2e-4},
    {'name': 'lr_5e4', 'lr': 5e-4},

    # ── KL regularization strength ──
    {'name': 'kl_0.1', 'alpha_kl': 0.1},
    {'name': 'kl_0.5', 'alpha_kl': 0.5},
    {'name': 'kl_1.0', 'alpha_kl': 1.0},

    # ── Constraint lambda weight (global + local set together) ──
    {'name': 'lambda_0.005', 'lambda_global': 0.005, 'lambda_local': 0.005},
    {'name': 'lambda_0.05', 'lambda_global': 0.05, 'lambda_local': 0.05},
    {'name': 'lambda_0.1', 'lambda_global': 0.1, 'lambda_local': 0.1},
]

# ── Model -> grid mapping ─────────────────────────────────────────────────────

MODEL_GRIDS_R3 = {
    'ResNet18': EXPERIMENT_GRID_R3,    # 10 configs (full exploration)
    'MobileNetV3': EXPERIMENT_GRID_MV3,  # 4 configs (focused comparison)
}

MODEL_GRIDS_R4 = {
    'ResNet18': EXPERIMENT_GRID_R4,    # 10 configs (single-param ablation)
    'MobileNetV3': EXPERIMENT_GRID_R4,   # 10 configs (single-param ablation)
}

MODEL_GRIDS = MODEL_GRIDS_R3  # Default (backward compat)

ROUND_GRIDS = {
    'round3': MODEL_GRIDS_R3,
    'round4': MODEL_GRIDS_R4,
}

# ── Legacy grid (Round 2) — kept for reference/heuristic dedup ───────────────

EXPERIMENT_GRID = EXPERIMENT_GRID_R3  # Alias for backwards compatibility


# ── Helpers ──────────────────────────────────────────────────────────────────

def constraint_tag(pair):
    """Generate a short directory-safe tag from a constraint pair.

    (0.9, 0.8) -> 'c09_08', (0.5, 0.3) -> 'c05_03'
    """
    g, l = pair
    return f"c{int(g*10):02d}_{int(l*10):02d}"


def model_tag(model_name):
    """Generate a short directory-safe tag from a model name.

    'ResNet18' -> 'r18', 'MobileNetV3' -> 'mv3'
    """
    tags = {'ResNet18': 'r18', 'MobileNetV3': 'mv3'}
    return tags.get(model_name, model_name.lower())


def compute_base_model_id(model_name, hyperparams, dataset_mode='dermmnist'):
    """Generate unique ID for model caching based on warmup-relevant params.

    Includes: lr, dropout, batch_size, warmup_epochs, pretrained, class_weighted_ce.
    Excludes: constraint-phase params (lr_constraint, lambda, rho, kl, constraint_epochs).
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
    }
    if 'hidden_dims' in hyperparams:
        key['hidden_dims'] = tuple(hyperparams['hidden_dims'])
    if 'seed' in hyperparams:
        key['seed'] = hyperparams['seed']
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_mode}_{h}"


def _build_config(methodology, exp_name, hyperparams, constraint_pair,
                   model_name='ResNet18', dataset_mode='dermmnist'):
    """Build a single experiment config dict."""
    ds_config = DATASET_CONFIGS.get(dataset_mode, DERMMNIST_DATASET_CONFIG)
    return {
        'methodology': methodology,
        'model_name': model_name,
        'constraint': list(constraint_pair),
        'dataset_mode': dataset_mode,
        'dataset_config': ds_config.copy(),
        'hyperparams': hyperparams,
        'base_model_id': compute_base_model_id(model_name, hyperparams,
                                                dataset_mode=dataset_mode),
        'exp_name': exp_name,
        'status': 'pending',
    }


def generate_configs(methodology='our_approach', constraint_pairs=None,
                     model_names=None, round='round3', dataset_mode='dermmnist'):
    """Generate experiment configs for given methodology across all constraint pairs.

    For 'our_approach': generates configs per model per constraint pair.
      - round3: ResNet18 (10 combo configs) + MobileNetV3 (4 focused configs) [completed]
      - round4: ResNet18 + MobileNetV3 (10 single-parameter ablation configs each)
    For 'heuristic': generates 1 baseline per model per constraint pair.

    Directory structure: {constraint_tag}/{model_name}/{short_name}/
    exp_name is a globally unique identifier for display/logging.
    """
    if constraint_pairs is None:
        constraint_pairs = CONSTRAINT_PAIRS

    round_model_grids = ROUND_GRIDS.get(round, MODEL_GRIDS_R3)

    if model_names is None:
        model_names = list(round_model_grids.keys()) if methodology != 'heuristic' else MODEL_NAMES

    configs = []

    for mn in model_names:
        mtag = model_tag(mn)
        grid = round_model_grids.get(mn, EXPERIMENT_GRID_R3)

        if methodology == 'heuristic':
            for pair in constraint_pairs:
                ctag = constraint_tag(pair)
                hp = HYPERPARAMS_BASELINE.copy()
                short_name = 'baseline'
                exp_name = f"{mtag}_{ctag}_baseline"  # unique display name
                configs.append(_build_config('heuristic', exp_name, hp, pair,
                                             model_name=mn,
                                             dataset_mode=dataset_mode))
        else:
            for pair in constraint_pairs:
                ctag = constraint_tag(pair)
                for exp in grid:
                    hp = HYPERPARAMS_BASELINE.copy()
                    hp.update({k: v for k, v in exp.items() if k != 'name'})
                    short_name = exp['name']
                    exp_name = f"{mtag}_{ctag}_{short_name}"  # unique display name
                    configs.append(_build_config(methodology, exp_name, hp, pair,
                                                 model_name=mn,
                                                 dataset_mode=dataset_mode))

    return configs


def save_configs(configs, output_dir='results/pending_runs'):
    """Create directory structure and save configs.

    Path: {output_dir}/{constraint_tag}/{model_name}/{short_name}/config.json
    For heuristic: {output_dir}/{constraint_tag}/{model_name}/heuristic/{short_name}/config.json
    Skips directories that already have a completed experiment.
    """
    from src.utils.filesystem_manager import save_config_to_path

    created, skipped = 0, 0

    for config in configs:
        methodology = config['methodology']
        model_name = config['model_name']
        ctag = constraint_tag(tuple(config['constraint']))
        # Short name: strip model tag and constraint tag prefix from exp_name
        mtag = model_tag(model_name)
        full_name = config['exp_name']
        short_name = full_name.replace(f'{mtag}_{ctag}_', '', 1)

        if methodology == 'heuristic':
            path = Path(output_dir) / ctag / model_name / 'heuristic' / short_name
        else:
            path = Path(output_dir) / ctag / model_name / short_name

        # Don't overwrite completed experiments
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
        config['experiment_path'] = str(path)
        save_config_to_path(config, str(path))
        created += 1

    methodology_label = configs[0]['methodology'] if configs else 'unknown'
    print(f"Created {created} {methodology_label} configs in '{output_dir}' "
          f"(skipped {skipped} already completed)")


def reset_all_to_pending(results_dir='results'):
    """Reset all experiment statuses to pending."""
    from src.utils.filesystem_manager import get_all_experiment_configs, save_config_to_path

    experiments = get_all_experiment_configs(results_dir)
    count = 0
    for path, config in experiments:
        if config.get('status') != 'pending':
            config['status'] = 'pending'
            save_config_to_path(config, path)
            count += 1
    print(f"Reset {count} experiments to pending")


def main():
    n_pairs = len(CONSTRAINT_PAIRS)
    print(f"=== Experiment Config Generator ===")
    print(f"Constraint pairs: {n_pairs} -> {[constraint_tag(p) for p in CONSTRAINT_PAIRS]}")
    print()

    # Ablation study summary
    r4_models = list(MODEL_GRIDS_R4.keys())
    print("Ablation study (single-parameter):")
    print(f"  Models: {r4_models}")
    for mn in r4_models:
        grid = MODEL_GRIDS_R4[mn]
        n_per = len(grid)
        print(f"    {mn}: {n_per} configs/pair x {n_pairs} pairs = {n_per * n_pairs}")
    total_r4 = sum(len(MODEL_GRIDS_R4[mn]) * n_pairs for mn in r4_models)
    print(f"  Total: {total_r4} optimization")

    print()
    print("1. Generate ablation our_approach configs (ResNet18 + MobileNetV3)")
    print("2. Generate heuristic baseline configs")
    print("9. Reset all to pending")
    print("0. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        save_configs(generate_configs('our_approach', round='round4'),
                     output_dir='results/pending_runs')
    elif choice == '2':
        save_configs(generate_configs('heuristic', round='round4'),
                     output_dir='results/pending_runs')
    elif choice == '9':
        reset_all_to_pending()


if __name__ == "__main__":
    main()

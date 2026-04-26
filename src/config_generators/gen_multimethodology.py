"""Multi-methodology config fanout.

Given a single training anchor (dataset, model, scenario, constraint_pair,
seeds, hyperparams), emit one config per (seed, methodology) for all
methodologies under test. All configs for the same seed share
`base_model_id` so warmup runs once and the other methodologies hit cache.

Output:
    results/pending_runs/multi/{dataset}/{scenario}/{ctag}/{model}/seed_{N}/{methodology}/
        config.json

Each config carries an `anchor_id` field = hash of (dataset, model,
scenario, ctag, seed, warmup-affecting HP). Same `anchor_id` across all
methodologies for a seed -> master CSV groups for paired comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id,
    constraint_tag,
)

METHODOLOGIES = ['our_approach', 'fioretto_ldf', 'heuristic', 'po_lp', 'danits_lp']

DATASET_DEFAULTS = {
    'tissuemnist': {
        'target_column': 'label',
        'group_column': 'synth_group',
        'num_classes': 8,
        'image_size': 224,
        'data_dir_template': 'data/tissuemnist/slice_{slice_idx}',
    },
    'cifar100': {
        'target_column': 'label',
        'group_column': 'coarse_label',
        'num_classes': 100,
        'image_size': 224,
        'data_dir_template': 'data/cifar100/slice_{slice_idx}',
    },
}

# Baseline HP shared across all methodologies. Methodology-specific keys are
# overlaid in METHODOLOGY_HP below.
DEFAULT_HP = {
    'lr': 1e-4,
    'lr_constraint': 5e-6,
    'dropout': 0.3,
    'batch_size': 64,
    'warmup_epochs': 50,
    'constraint_epochs': 300,
    'lambda_global': 0.01,
    'lambda_local': 0.01,
    'lambda_step': 0.002,
    'use_sum_loss': True,
    'initial_rho': 5.0,
    'rho_target': 100.0,
    'alpha_kl': 0.0,
    'kl_temperature': 1.0,
    'pretrained': True,
    'class_weighted_ce': False,
    'constraint_chunk_size': 256,
}

METHODOLOGY_HP = {
    'fioretto_ldf': {'fioretto_step_size': 0.005},
    'danits_lp': {'danits_cost_preset': 'identity'},
}


def compute_anchor_id(base_model_id, scenario, ctag, hp):
    """Hash for an experimental anchor. Strict superset of base_model_id:
    two configs with different base_model_id (= different warmup) MUST
    have different anchor_id, otherwise the "paired comparison" assumption
    breaks (different methodologies should compare on the SAME warmup).
    AUDIT B10.

    base_model_id already encodes model, dataset, lr, dropout, batch_size,
    warmup_epochs, pretrained, class_weighted_ce, num_classes, image_size,
    seed. anchor_id additionally encodes scenario, constraint pair, and
    constraint-phase HP that affect downstream training but not warmup.
    """
    key = {
        'base_model_id': base_model_id,
        'scenario': scenario,
        'ctag': ctag,
        'lr_constraint': hp['lr_constraint'],
        'constraint_epochs': hp['constraint_epochs'],
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def build_config(dataset, model_name, scenario_name, constrained_class,
                 constraint_pair, slice_idx, seed, hp, methodology):
    ds_template = DATASET_DEFAULTS[dataset]
    ds_config = {
        'target_column': ds_template['target_column'],
        'group_column': ds_template['group_column'],
        'num_classes': ds_template['num_classes'],
        'image_size': ds_template['image_size'],
        'data_dir': ds_template['data_dir_template'].format(slice_idx=slice_idx),
        'constrained_class': constrained_class,
    }
    ctag = constraint_tag(constraint_pair)

    full_hp = dict(hp)
    full_hp['seed'] = seed
    full_hp.update(METHODOLOGY_HP.get(methodology, {}))

    base_model_id = compute_base_model_id(
        model_name, full_hp, dataset_mode=dataset,
        data_dir=ds_config['data_dir'], dataset_config=ds_config)
    anchor_id = compute_anchor_id(base_model_id, scenario_name, ctag, full_hp)

    path = (Path('results/pending_runs/multi') / dataset / scenario_name / ctag
            / model_name / f'seed_{seed}' / methodology)

    return {
        'methodology': methodology,
        'model_name': model_name,
        'constraint': list(constraint_pair),
        'constraint_tag': ctag,
        'dataset_mode': dataset,
        'dataset_config': ds_config,
        'hyperparams': full_hp,
        'base_model_id': base_model_id,
        'anchor_id': anchor_id,
        'exp_name': f'{dataset}_{scenario_name}_{ctag}_{model_name}_s{seed}_{methodology}',
        'status': 'pending',
        'experiment_path': str(path),
    }


def generate_anchor_configs(dataset, model_name, scenario, constrained_class,
                             constraint_pair, slice_idx=1, seeds=(1, 2, 3),
                             hp=None, methodologies=None):
    hp = hp if hp is not None else DEFAULT_HP
    methodologies = methodologies if methodologies is not None else METHODOLOGIES
    configs = []
    for seed in seeds:
        for meth in methodologies:
            configs.append(build_config(
                dataset, model_name, scenario, constrained_class,
                constraint_pair, slice_idx, seed, hp, meth))
    return configs


def save(configs, force=False):
    """Write configs. By default skips already-completed dirs.

    force=True -> overwrite anything not currently 'running'.
    """
    from src.utils.filesystem_manager import save_config_to_path
    created, skipped, overwritten = 0, 0, 0
    for c in configs:
        path = Path(c['experiment_path'])
        existing = path / 'config.json'
        if existing.exists() and not force:
            try:
                with open(existing) as f:
                    e = json.load(f)
                if e.get('status') == 'completed':
                    skipped += 1
                    continue
            except Exception:
                pass
            overwritten += 1
        path.mkdir(parents=True, exist_ok=True)
        save_config_to_path(c, str(path))
        created += 1
    print(f"Created {created}, skipped {skipped} (already completed), "
          f"overwrote {overwritten}")


def summarize(configs):
    by_anchor = defaultdict(list)
    by_warmup = defaultdict(list)
    for c in configs:
        by_anchor[c['anchor_id']].append(c['methodology'])
        by_warmup[c['base_model_id']].append(c['methodology'])
    print(f"\nTotal configs: {len(configs)}")
    print(f"Unique anchors: {len(by_anchor)} (each = one paired comparison row)")
    print(f"Unique warmup hashes: {len(by_warmup)} "
          f"(each warmup trains once, reused {len(configs) // max(len(by_warmup), 1)}x)")
    print()
    for aid in sorted(by_anchor):
        print(f"  anchor {aid}: {sorted(by_anchor[aid])}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-methodology configs for one anchor')
    parser.add_argument('--dataset', choices=['tissuemnist', 'cifar100'],
                        default='tissuemnist')
    parser.add_argument('--model', default='MobileNetV3')
    parser.add_argument('--scenario', default='single_GE')
    parser.add_argument('--constrained_class', type=int, default=4,
                        help='int for single, comma list for multi (e.g. 4,2)')
    parser.add_argument('--local_pct', type=float, required=True)
    parser.add_argument('--global_pct', type=float, required=True)
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--methodologies', nargs='+', default=METHODOLOGIES,
                        choices=METHODOLOGIES)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print summary but do not write configs')
    args = parser.parse_args()

    cc = args.constrained_class

    configs = generate_anchor_configs(
        dataset=args.dataset, model_name=args.model,
        scenario=args.scenario, constrained_class=cc,
        constraint_pair=(args.local_pct, args.global_pct),
        slice_idx=args.slice, seeds=tuple(args.seeds),
        methodologies=args.methodologies,
    )

    ctag = constraint_tag((args.local_pct, args.global_pct))
    print(f"Anchor: {args.dataset}/{args.scenario}/{args.model}/{ctag}")
    print(f"  seeds={args.seeds}  methodologies={args.methodologies}")
    summarize(configs)

    if not args.dry_run:
        save(configs)


if __name__ == '__main__':
    main()

"""Staged inventories, frozen releases and receipt-bound campaign admission.

Data hashes prove bytes at freeze. Admission checks resolved paths/size/mtime
for ordinary changes; it does not rehash multi-GB pixels or prove immutability.
This module establishes provenance, not GPU ownership or launch authorization.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import uuid

from src.pipeline.config import validate_hyperparams
from src.utils.constants import INFERENCE_CHUNK_SIZE

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PLAN = 'campaign_plan.json'
MANIFEST = 'campaign_manifest.json'
RECEIPT = 'completion_receipt.json'
SCHEMA = 'fresh-campaign-v1'
DEPLOYMENT = 'joint-greedy-prob%d-v1' % INFERENCE_CHUNK_SIZE
METRICS = 'fixed-classes-zero-division0-ccf1-v1'
DATA_FILES = tuple(f'{split}_{kind}' for split in ('train', 'test')
                   for kind in ('images.npy', 'labels.npy', 'meta.csv'))
IMMUTABLE = {'methodology', 'model_name', 'constraint', 'constraint_tag', 'dataset_mode',
             'dataset_config', 'hyperparams', 'base_model_id', 'arm', 'exp_name', 'code_version'}
MUTABLE = {'status', 'failures', 'run_code_version', 'data_fingerprint', 'results',
           'reordering', 'results_comparison', 'divergence', 'divergence_reason',
           'campaign_id', 'release_id', 'cache_identity', 'warmup_checkpoint'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _is_link(path):
    try:
        stat = Path(path).lstat()
    except FileNotFoundError:
        return False
    return Path(path).is_symlink() or bool(getattr(stat, 'st_file_attributes', 0) & 0x400)


def safe_path(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if _is_link(part):
            raise ValueError('campaign/config/output symlink refused: %s' % part)
        if part.name.lower() in ('archive', 'archives') or part.name.lower().startswith('archive_'):
            raise ValueError('archive root refused: %s' % part)
    return path


def config_identity(config):
    unknown = set(config) - IMMUTABLE - MUTABLE
    if unknown:
        raise ValueError('unknown config fields: %s' % sorted(unknown))
    if not IMMUTABLE.issubset(config):
        raise ValueError('missing immutable config fields: %s' % sorted(IMMUTABLE-set(config)))
    validate_hyperparams(config['methodology'], config['hyperparams'])
    return digest({k: config[k] for k in sorted(IMMUTABLE)})


def _inventory(root, entries):
    found = set()
    for path in root.rglob('*'):
        if _is_link(path):
            raise ValueError('campaign/config/output symlink refused: %s' % path)
        if path.is_dir() and (path.name.lower() in ('archive', 'archives') or path.name.lower().startswith('archive_')):
            raise ValueError('archive entry in campaign: %s' % path)
        if path.name == 'config.json':
            found.add(path.relative_to(root).as_posix())
    if found != set(entries):
        raise ValueError('config inventory mismatch: missing=%s extra=%s' %
                         (sorted(set(entries)-found), sorted(found-set(entries))))
    for rel, expected in entries.items():
        path = safe_path(root / rel)
        if not path.is_relative_to(root) or path.name != 'config.json':
            raise ValueError('config path escapes root: %s' % rel)
        if config_identity(_read(path)) != expected:
            raise ValueError('immutable config changed: %s' % rel)


def stage_campaign(root, configs, protocol):
    root = safe_path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('campaign root must be empty: %s' % root)
    if not configs:
        raise ValueError('empty campaign inventory')
    entries = {rel: config_identity(cfg) for rel, cfg in configs.items()}
    for rel in entries:
        if not safe_path(root/rel).is_relative_to(root) or Path(rel).name != 'config.json':
            raise ValueError('config path escapes root: %s' % rel)
    plan = dict(schema=SCHEMA, campaign_id=uuid.uuid4().hex, configs=entries,
                protocol=protocol, deployment=DEPLOYMENT, metrics=METRICS)
    root.mkdir(parents=True, exist_ok=True)
    for rel, cfg in configs.items():
        path = root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, cfg)
    _atomic_json(root/PLAN, plan)  # Marker last: partial generation cannot launch.
    return plan


def source_inventory(source_root=SOURCE_ROOT):
    source_root = Path(source_root)
    paths = {p for directory in ('src', 'configs', 'scripts')
             for p in (source_root/directory).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.sh', '.yml', '.yaml')}
    paths.update(source_root/p for p in ('main.py', 'requirements.txt'))
    return {p.relative_to(source_root).as_posix(): file_digest(p) for p in sorted(paths)}


def release_runtime():
    import torch
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    return dict(host=platform.node(), python=platform.python_version(),
                packages={p: importlib.metadata.version(p) for p in
                          ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'PyYAML')},
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if cuda else None,
                precision='bf16' if bf16 else ('fp16' if cuda else 'fp32'),
                grad_scaler=bool(cuda and not bf16))


def _data_record(config, source_root, frozen_files=None):
    import numpy as np
    import pandas as pd
    from src.utils.data_loader import _encode_groups, _check_group_leakage
    from src.training.constraints import compute_global_constraints, compute_local_constraints
    dc = config['dataset_config']
    directory = Path(dc['data_dir'])
    if not directory.is_absolute():
        directory = source_root/directory
    files = frozen_files if frozen_files is not None else {}
    frames = {}
    for name in DATA_FILES:
        if name in files:
            continue
        path = directory/name
        resolved = str(path.resolve())
        before = path.stat()
        hashed = file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, resolved) != (after.st_size, after.st_mtime_ns, str(path.resolve())):
            raise ValueError('data changed during freeze: %s' % path)
        files[name] = dict(logical=str(path.absolute()), resolved=str(path.resolve()),
                           size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=hashed)
    for split in ('train', 'test'):
        labels = np.load(directory/f'{split}_labels.npy')
        images = np.load(directory/f'{split}_images.npy', mmap_mode='r')
        frame = pd.read_csv(directory/f'{split}_meta.csv')
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('data labels must be one-dimensional integers')
        if len(labels) != len(images) or len(labels) != len(frame) or not len(labels):
            raise ValueError('data row count mismatch')
        if not np.array_equal(frame['label'].to_numpy(), labels):
            raise ValueError('data metadata/label mismatch')
        if labels.min() < 0 or labels.max() >= dc['num_classes']:
            raise ValueError('data labels outside declared classes')
        frames[split] = frame
    _check_group_leakage(str(directory), dc['group_column'], dc.get('disjoint_groups', False))
    frame = frames['test'].copy()
    frame[dc['group_column']] = _encode_groups(frame[dc['group_column']], dc['group_column'])
    lp, gp = config['constraint']
    kwargs = dict(constrained_class=dc['constrained_class'], num_classes=dc['num_classes'])
    global_con = compute_global_constraints(frame, 'label', gp, **kwargs)
    local_con = compute_local_constraints(frame, 'label', lp, dc['group_column'],
                                          group_budget_shares=dc.get('group_budget_shares'),
                                          permute_group_budgets=dc.get('permute_group_budgets'),
                                          **kwargs)
    quotas = {'global': [int(v) for v in global_con],
              'local': {str(k): [int(v) for v in values] for k, values in local_con.items()}}
    return dict(files=files, quotas=quotas, classes=list(range(dc['num_classes'])),
                constrained_classes=dc['constrained_class'], test_rows=len(frame))


def freeze_campaign(root, source_root=SOURCE_ROOT):
    root, source_root = safe_path(root), Path(source_root).resolve()
    if (root/MANIFEST).exists():
        raise ValueError('campaign already frozen')
    if not (root/PLAN).is_file():
        raise ValueError('unmarked campaign; staged plan required')
    plan = _read(root/PLAN)
    _inventory(root, plan['configs'])
    data, runs, file_inventories = {}, {}, {}
    for rel in plan['configs']:
        cfg = _read(root/rel)
        if cfg['dataset_mode'] == 'bcn':
            raise ValueError('BCN operational hold: cross-split conflicting duplicates')
        key = digest([cfg['dataset_mode'], cfg['dataset_config'], cfg['constraint']])
        if key not in data:
            data_key = digest(cfg['dataset_config'])
            files = file_inventories.get(data_key)
            data[key] = _data_record(cfg, source_root, files)
            file_inventories[data_key] = data[key]['files']
        runs[rel] = {'data_id': key, 'config_id': plan['configs'][rel]}
    manifest = dict(plan, plan_id=digest(plan), source=source_inventory(source_root),
                    source_root=str(source_root), data=data, runs=runs,
                    runtime=release_runtime())
    manifest['release_id'] = digest(manifest)
    _atomic_json(root/MANIFEST, manifest)
    return manifest


def validate_campaign(root, config_path=None, check_data=True, check_runtime=True):
    root = safe_path(root)
    if not (root/MANIFEST).is_file() or not (root/PLAN).is_file():
        raise ValueError('unmarked or not frozen campaign: %s' % root)
    manifest = _read(root/MANIFEST)
    if manifest.get('schema') != SCHEMA or manifest.get('deployment') != DEPLOYMENT or manifest.get('metrics') != METRICS:
        raise ValueError('unsupported campaign protocol')
    if digest({k:v for k,v in manifest.items() if k != 'release_id'}) != manifest['release_id']:
        raise ValueError('manifest identity mismatch')
    if digest(_read(root/PLAN)) != manifest['plan_id']:
        raise ValueError('staged plan mismatch')
    _inventory(root, manifest['configs'])
    if config_path is not None:
        path = safe_path(config_path)
        if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in manifest['configs']:
            raise ValueError('foreign config is not a campaign member')
    if source_inventory() != manifest['source']:
        raise ValueError('source bytes differ from frozen release')
    if check_runtime and release_runtime() != manifest['runtime']:
        raise ValueError('runtime/host/precision differs from frozen release')
    if check_data:
        for data in manifest['data'].values():
            for record in data['files'].values():
                path = Path(record['logical'])
                stat = path.stat()
                if (str(path.resolve()), stat.st_size, stat.st_mtime_ns) != (record['resolved'], record['size'], record['mtime_ns']):
                    raise ValueError('frozen data stat/resolved path changed: %s' % path)
    return manifest


def campaign_for_config(config_path):
    path = safe_path(config_path)
    for root in path.parents:
        if (root/PLAN).exists() or (root/MANIFEST).exists():
            return root, validate_campaign(root, path)
    raise ValueError('unmarked config: no frozen campaign membership')


def run_identity(root, manifest, config_path):
    rel = safe_path(config_path).relative_to(safe_path(root)).as_posix()
    run = manifest['runs'][rel]
    data = manifest['data'][run['data_id']]
    return dict(campaign_id=manifest['campaign_id'], release_id=manifest['release_id'],
                config_id=run['config_id'], data_id=digest(data['files']),
                quota_id=digest(data['quotas']))


def write_receipt(root, config_path, checkpoint):
    manifest = validate_campaign(root, config_path)
    run = Path(config_path).parent
    receipt = dict(run_identity(root, manifest, config_path), runtime=manifest['runtime'],
                   checkpoint=checkpoint,
                   outputs={name: file_digest(run/name) for name in
                            ('final_predictions.csv', 'final_predictions_raw.csv', 'evaluation_metrics.csv')})
    receipt['receipt_id'] = digest(receipt)
    _atomic_json(run/RECEIPT, receipt)
    return receipt


def validate_receipts(root, *, complete=True):
    manifest = validate_campaign(root, check_data=False, check_runtime=False)
    root = safe_path(root)
    completed, pending, missing = [], [], []
    for rel in manifest['configs']:
        config = _read(root/rel)
        receipt_path = (root/rel).with_name(RECEIPT)
        if config.get('status') != 'completed':
            pending.append('%s (%s)' % (rel, config.get('status')))
            continue
        if not receipt_path.exists():
            missing.append(rel)
            continue
        receipt = _read(receipt_path)
        if receipt.get('receipt_id') != digest({k:v for k,v in receipt.items() if k != 'receipt_id'}):
            raise ValueError('completion receipt altered: %s' % rel)
        expected = run_identity(root, manifest, root/rel)
        if any(receipt.get(k) != v for k, v in expected.items()) or receipt.get('runtime') != manifest['runtime']:
            raise ValueError('foreign/mixed completion receipt: %s' % rel)
        if set(receipt.get('outputs', {})) != {'final_predictions.csv', 'final_predictions_raw.csv', 'evaluation_metrics.csv'}:
            raise ValueError('incomplete receipt output inventory: %s' % rel)
        for name, hashed in receipt['outputs'].items():
            path = safe_path((root/rel).parent/name)
            if path.parent != (root/rel).parent or file_digest(path) != hashed:
                raise ValueError('receipt output mismatch: %s' % path)
        completed.append(rel)
    inventory = dict(completed=completed, pending=pending, missing=missing)
    if complete and (pending or missing):
        raise ValueError('incomplete campaign: ' + json.dumps(inventory))
    return manifest, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'validate'))
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    try:
        manifest = freeze_campaign(args.root) if args.action == 'freeze' else validate_campaign(args.root)
        print(json.dumps(dict(root=str(safe_path(args.root)), campaign_id=manifest['campaign_id'],
                              configs=list(manifest['configs']))))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('REFUSED: %s' % exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

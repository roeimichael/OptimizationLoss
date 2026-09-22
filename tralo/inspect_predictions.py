"""Audit supplied predictions. This command does not train or allocate them."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

from .events import EventLog
from .metrics import classification_metrics
from .quotas import audit_quotas


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _report(data):
    required = {'sample_ids', 'groups', 'probabilities', 'labels',
                'allocated_predictions', 'global_caps', 'local_caps',
                'constrained_classes', 'allocation_policy'}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(f'input must contain exactly: {sorted(required)}')
    ids, probs = data['sample_ids'], data['probabilities']
    if not isinstance(ids, list) or not ids or any(type(s) is not str or not s for s in ids):
        raise ValueError('sample_ids must be nonempty strings')
    if len(set(ids)) != len(ids):
        raise ValueError('sample_ids must be unique')
    if not isinstance(probs, list) or len(probs) != len(ids):
        raise ValueError('probabilities must contain one row per sample')
    if not isinstance(probs[0], list) or not probs[0]:
        raise ValueError('probabilities need at least one class')
    n_classes = len(probs[0])
    for row in probs:
        if not isinstance(row, list) or len(row) != n_classes:
            raise ValueError('probability rows must have equal length')
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in row):
            raise ValueError('probabilities must be finite numbers in [0,1]')
        if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError('each probability row must sum to one')
    for name in ('groups', 'labels', 'allocated_predictions', 'constrained_classes', 'global_caps'):
        if not isinstance(data[name], list):
            raise ValueError(f'{name} must be a list')
    if any(len(data[name]) != len(ids) for name in ('groups', 'labels', 'allocated_predictions')):
        raise ValueError('sample fields must have matching lengths')
    if type(data['allocation_policy']) is not str or not data['allocation_policy'].strip():
        raise ValueError('name the policy that produced these supplied predictions')
    # Lowest class index wins a raw-probability tie, stated explicitly.
    raw = [max(range(n_classes), key=row.__getitem__) for row in probs]
    arguments = (data['groups'], n_classes, data['global_caps'], data['local_caps'])
    raw_quotas = audit_quotas(raw, *arguments)
    allocated_quotas = audit_quotas(data['allocated_predictions'], *arguments)
    capped = {c for caps in [data['global_caps'], *data['local_caps'].values()]
              for c, cap in enumerate(caps) if cap is not None}
    raw_metrics = classification_metrics(data['labels'], raw, n_classes, data['constrained_classes'])
    allocated_metrics = classification_metrics(data['labels'], data['allocated_predictions'], n_classes, data['constrained_classes'])
    if set(data['constrained_classes']) != capped:
        raise ValueError('constrained_classes must match classes with declared caps')
    if not allocated_quotas['feasible']:
        raise ValueError('supplied allocated predictions violate the declared quotas')
    return {
        'schema_version': 1, 'prediction_origin': 'supplied_not_generated',
        'allocation_policy': data['allocation_policy'], 'allocation_policy_verified': False,
        'raw_metrics': raw_metrics, 'allocated_metrics': allocated_metrics,
        'raw_quotas': raw_quotas, 'allocated_quotas': allocated_quotas,
        'changed_predictions': sum(a != b for a, b in zip(raw, data['allocated_predictions'])),
        'samples': [{'id': sid, 'raw': a, 'allocated': b}
                    for sid, a, b in zip(ids, raw, data['allocated_predictions'])],
    }


def inspect_file(input_path, output_dir):
    """Validate completely before reserving an exclusive output directory."""
    content = Path(input_path).read_bytes()
    data = json.loads(content, object_pairs_hook=_unique_object)
    report = _report(data)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob('*.py'))}
    config = {k: data[k] for k in ('global_caps', 'local_caps', 'constrained_classes', 'allocation_policy')}
    config['n_classes'] = len(data['probabilities'][0])
    provenance = {
        'input_sha256': hashlib.sha256(content).hexdigest(),
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest(),
        'source_sha256': source_hashes,
        'python': platform.python_version(), 'platform': platform.platform(),
    }
    report['provenance'] = provenance
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with EventLog(output / 'events.jsonl') as log:
        log.emit('started', **provenance, operation='inspect_supplied_predictions')
        try:
            (output / 'input.json').write_bytes(content)
            encoded = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')
            (output / 'report.json').write_bytes(encoded)
            log.emit('completed', report_sha256=hashlib.sha256(encoded).hexdigest())
        except Exception as exc:
            log.emit('failed', error_type=type(exc).__name__, message=str(exc))
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = inspect_file(args.input, args.output)
    print(json.dumps({'raw_cc_f1': report['raw_metrics']['cc_f1'],
                      'allocated_cc_f1': report['allocated_metrics']['cc_f1']}))


if __name__ == '__main__':
    main()

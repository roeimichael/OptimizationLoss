"""Compare two explicitly named global-only allocation diagnostics, not winners."""

import argparse
import hashlib
import json
from pathlib import Path

from .events import EventLog
from .global_clipper import allocate
from .metrics import classification_metrics


def evaluate_global(probabilities, labels, caps, sample_ids):
    predictions = {policy: allocate(probabilities, caps, sample_ids, policy)
                   for policy in ('upper_bound_correction', 'capped_first')}
    raw = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    predictions = {'raw': raw, **predictions}
    constrained = [c for c, cap in enumerate(caps) if cap is not None]
    result = {}
    for policy, values in predictions.items():
        metrics = classification_metrics(labels, values, len(caps), constrained)
        counts = [row['predicted_count'] for row in metrics['per_class']]
        feasible = all(cap is None or counts[c] <= cap for c, cap in enumerate(caps))
        if policy != 'raw' and not feasible:
            raise RuntimeError('allocator output violates the declared caps')
        result[policy] = {'metrics': metrics, 'counts': counts, 'feasible': feasible,
                          'changed': sum(a != b for a, b in zip(raw, values)),
                          'predictions': values}
    return result


def run(predictions_path, caps_path, output):
    content, cap_bytes = Path(predictions_path).read_bytes(), Path(caps_path).read_bytes()
    inputs, config = json.loads(content), json.loads(cap_bytes)
    if set(config) != {'global_caps', 'rationale'} or not isinstance(config['rationale'], str):
        raise ValueError('caps config requires global_caps and rationale')
    report = evaluate_global(inputs['probabilities'], inputs['labels'], config['global_caps'], inputs['sample_ids'])
    provenance = {'input_sha256': hashlib.sha256(content).hexdigest(),
                  'caps_sha256': hashlib.sha256(cap_bytes).hexdigest(),
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob('*.py'))}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with EventLog(output / 'events.jsonl') as log:
        log.emit('started', **provenance, scope='global_only_allocation_diagnostics')
        try:
            (output / 'input_probabilities.json').write_bytes(content)
            (output / 'caps.json').write_bytes(cap_bytes)
            data = {'provenance': provenance, 'global_caps': config['global_caps'],
                    'rationale': config['rationale'], 'sample_ids': inputs['sample_ids'],
                    'results': report}
            encoded = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()
            (output / 'report.json').write_bytes(encoded)
            log.emit('completed', report_sha256=hashlib.sha256(encoded).hexdigest())
        except Exception as exc:
            log.emit('failed', error_type=type(exc).__name__, message=str(exc))
            raise
    print(json.dumps({name: {key: row['metrics'][key] for key in ('accuracy', 'macro_f1', 'cc_f1')}
                      for name, row in report.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions')
    parser.add_argument('caps')
    parser.add_argument('output')
    args = parser.parse_args()
    run(args.predictions, args.caps, args.output)

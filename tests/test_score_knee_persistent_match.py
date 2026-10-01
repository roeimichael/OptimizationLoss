import json

import pytest
import torch

from analysis.score_knee_persistent_match import _arm, _check_dose, _check_label_firewall, _events, _quality
from tralo.events import EventLog
from tralo.knee_experiment import digest


def test_complete_arm_gate_recomputes_epoch_and_rejects_corrupt_count(tmp_path):
    name = 'pto'
    root = tmp_path
    path = root / name
    path.mkdir()
    probabilities = torch.full((826, 5), 0.2)
    torch.save(probabilities, path / 'epoch01.pt')
    torch.save(probabilities, path / 'final_probabilities.pt')
    (path / 'model.pt').write_bytes(b'model-artifact')
    training = dict(best_epoch=1, epochs_run=1, best_stop_loss=0.5, task_updates=2,
                    first_order_sha256='order', first_batch_sha256='batch')
    (path / 'training.json').write_text(json.dumps(training))
    (path / 'snapshots.json').write_text(json.dumps([
        dict(epoch=1, file='epoch01.pt', sha256=digest(path / 'epoch01.pt'))]))
    epoch = dict(epoch=1, training_loss=0.6, stop_loss=0.5, task_updates=2,
                 first_task_gradient_norm=1.0, first_task_displacement_norm=0.1,
                 hard_counts=[826, 0, 0, 0, 0],
                 soft_count_capped=float(probabilities[:, 3].sum()),
                 epoch_order_sha256='order', epoch_first_batch_sha256='batch')
    with EventLog(path / 'events.jsonl') as log:
        log.emit('epoch', **epoch)
        log.emit('training_completed', **training,
                 probability_sha256=digest(path / 'final_probabilities.pt'),
                 model_sha256=digest(path / 'model.pt'))
    item = dict(file='pto/final_probabilities.pt',
                probability_sha256=digest(path / 'final_probabilities.pt'),
                model_sha256=digest(path / 'model.pt'),
                training_file='pto/training.json', seconds=1.0, peak_gpu_bytes=100, **training)
    observed, rows = _arm(root, name, item, 2, 1)
    assert torch.equal(observed, probabilities) and len(rows) == 1
    events = _events(path / 'events.jsonl')
    events[0]['hard_counts'][0] -= 1
    (path / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
    with pytest.raises(RuntimeError, match='epoch log differs from snapshot'):
        _arm(root, name, item, 2, 1)


def test_scoring_fixture_reports_perfect_grade3_and_weighted_f1():
    probabilities = torch.eye(5)
    labels = list(range(5))
    results = _quality(probabilities, labels, [f'k{i}' for i in range(5)], 1)
    for policy in ('raw', 'capped_first', 'upper_bound_correction'):
        assert results[policy]['cc_f1'] == 1.0
        assert results[policy]['weighted_f1'] == 1.0
        assert results[policy]['accuracy'] == 1.0


def test_gate_helpers_reject_development_label_and_sham_radius_mutations():
    manifest = dict(rows=[dict(split='train', label=3), dict(split='val')])
    _check_label_firewall(manifest)
    manifest['rows'][1]['label'] = 3
    with pytest.raises(RuntimeError, match='label entered'):
        _check_label_firewall(manifest)
    _check_dose('cap54_sham', dict(tensor_displacement_norms=[0.1, 0.2]), [0.1, 0.2], 54)
    with pytest.raises(RuntimeError, match='sham tensor dose'):
        _check_dose('cap54_sham', dict(tensor_displacement_norms=[0.1, 0.25]), [0.1, 0.2], 54)

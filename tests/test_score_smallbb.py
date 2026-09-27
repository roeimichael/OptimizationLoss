import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import score_smallbb as sb  # noqa: E402
from score_yuval import paired  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

N, CAPS = 826, [None, None, None, 76, None]
CLASS = {'mn3': ('mobilenet_v3_large', 'MobileNetV3'), 'rgy': ('regnet_y_400mf', 'RegNet')}


def tree(root, seeds, signal=0.1, cls=None, landed=76):
    """Write knee_yuval-shaped seed directories; on mn3 seeds tralo_final carries extra grade-3 signal."""
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 5, (N,), generator=g).tolist()
    rows = [dict(split='val', label=y, sample_id=f'v{i}') for i, y in enumerate(labels)]
    onehot = torch.nn.functional.one_hot(torch.tensor(labels), 5).float()
    for seed in seeds:
        block = 'mn3' if 4300 <= seed < 4400 else 'rgy'
        d = Path(root) / f'seed{seed}'
        d.mkdir(parents=True)
        pto = (torch.randn(N, 5, generator=g) + 1.2 * onehot).softmax(1)
        push = torch.zeros(N, 5)
        push[:, 3] = (signal if block == 'mn3' else 0.0) * (onehot[:, 3] - 0.5)
        stepped = (pto + push).clamp_min(1e-6)
        for arm, p in dict(pto=pto, tralo_final=stepped / stepped.sum(1, keepdim=True), sham_final=pto).items():
            (d / arm).mkdir()
            (d / arm / 'report.json').write_text(json.dumps(evaluate_global(p.tolist(), labels, CAPS, [r['sample_id'] for r in rows])))
        hard = int((pto.argmax(1) == 3).sum())
        (d / 'summary.json').write_text(json.dumps(dict(
            seed=seed, cap=76, converged=False, retrains=[dict(retrain=1, hard_count=hard, best_epoch=4, epochs_run=9)],
            steps=dict(tralo_final=dict(applied=True, radius=0.01, hard_before=hard, hard_after=landed),
                       sham_final=dict(applied=True, radius=0.01, hard_before=hard, hard_after=hard)))))
        (d / 'manifest.json').write_text(json.dumps(dict(rows=rows)))
        arch, name = CLASS[block]
        (d / 'events.jsonl').write_text(json.dumps(dict(event='model_initialized', architecture=arch,
                                                        model_class=cls or name)) + '\n')


def test_block_recovers_the_built_in_contrast_and_rejects_foreign_seeds(tmp_path):
    tree(tmp_path / 'mn3', (4300, 4301, 4302))
    tree(tmp_path / 'rgy', (4400, 4401, 4402))
    mn3 = sb.block(tmp_path / 'mn3', *sb.BLOCKS[0][1:])
    rgy = sb.block(tmp_path / 'rgy', *sb.BLOCKS[1][1:])
    p2 = lambda kept: paired([s['arms']['tralo_final']['cc_f1'] - s['arms']['sham_final']['cc_f1'] for s in kept])
    assert len(mn3) == len(rgy) == 3 and p2(mn3)['mean'] > 0 and p2(rgy)['mean'] == 0
    with pytest.raises(RuntimeError, match='not a study seed'):
        sb.block(tmp_path / 'rgy', *sb.BLOCKS[0][1:])            # RegNetY seeds scored as the MobileNetV3 block
    tree(tmp_path / 'wrong', (4303,), cls='ResNet')
    with pytest.raises(RuntimeError, match='was not trained on'):
        sb.block(tmp_path / 'wrong', *sb.BLOCKS[0][1:])
    with pytest.raises(RuntimeError, match='not a study seed'):
        tree(tmp_path / 'pilot', (4399,))
        sb.block(tmp_path / 'pilot', *sb.BLOCKS[0][1:])            # a pilot is never a study seed


def test_gate_passes_clean_pilots_and_fails_every_defect(tmp_path, capsys):
    tree(tmp_path / 'ok', (4399, 4499))
    sb.gate(tmp_path / 'ok')
    assert 'PILOT GATE PASSED' in capsys.readouterr().out
    for name, make in (('class', lambda r: tree(r, (4399, 4499), cls='ResNet')),
                       ('landed', lambda r: tree(r, (4399, 4499), landed=80)),
                       ('missing', lambda r: tree(r, (4399,))),
                       ('study seed', lambda r: tree(r, (4399, 4499, 4300)))):
        make(tmp_path / name)
        with pytest.raises(SystemExit):
            sb.gate(tmp_path / name)
    tree(tmp_path / 'radius', (4399, 4499))
    path = tmp_path / 'radius' / 'seed4499' / 'summary.json'
    s = json.loads(path.read_text())
    s['steps']['sham_final']['radius'] = 0.02
    path.write_text(json.dumps(s))
    with pytest.raises(SystemExit):
        sb.gate(tmp_path / 'radius')


def test_v3_reference_must_be_the_same_backbones_24_seeds(tmp_path):
    report = evaluate_global(torch.full((N, 5), 0.2).tolist(), [0] * N, CAPS, [f'v{i}' for i in range(N)])
    rows = [dict(split='val', label=0, sample_id=f'v{i}') for i in range(N)]

    def v3(root, seeds, backbone):
        for seed in seeds:
            d = Path(root) / f'seed{seed}'
            (d / 'clipper').mkdir(parents=True)
            (d / 'clipper' / 'report.json').write_text(json.dumps(report))
            (d / 'manifest.json').write_text(json.dumps(dict(rows=rows)))
            (d / 'config.json').write_text(json.dumps(dict(seed=seed, backbone=backbone)))
    v3(tmp_path / 'good', range(3001, 3025), 'mobilenet_v3_large')
    assert len(sb.v3_clipper(tmp_path / 'good', 'mobilenet_v3_large')) == 24
    with pytest.raises(RuntimeError):
        sb.v3_clipper(tmp_path / 'good', 'regnet_y_400mf')          # the roots swapped
    v3(tmp_path / 'relabelled', range(3101, 3125), 'mobilenet_v3_large')   # RegNetY seed range, wrong backbone
    with pytest.raises(RuntimeError, match='expected regnet_y_400mf'):
        sb.v3_clipper(tmp_path / 'relabelled', 'regnet_y_400mf')
    v3(tmp_path / 'short', range(3001, 3020), 'mobilenet_v3_large')
    with pytest.raises(RuntimeError, match='expected 24'):
        sb.v3_clipper(tmp_path / 'short', 'mobilenet_v3_large')

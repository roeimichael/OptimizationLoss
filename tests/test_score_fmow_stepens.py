import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import score_fmow_stepens as sf  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

N, CAP, K = 300, 30, 1
CAPS = [None, CAP, None, None, None, None, None, None]


def tree(root, seeds, signal=0.1, best=6, run=9, edit=None, arch=('mobilenet_v3_large', 'MobileNetV3'), cap=CAP):
    """Write fmow_yuval-shaped seed directories with snapshot steps; the tralo snapshots carry extra class-1 signal."""
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 8, (N,), generator=g).tolist()
    rows = [dict(split='val', label=y, sample_id=f'test{i}', location='IRQ') for i, y in enumerate(labels)]
    onehot = torch.nn.functional.one_hot(torch.tensor(labels), 8).float()
    caps = [None] * 8
    caps[K] = cap
    for seed in seeds:
        d = Path(root) / f'seed{seed}'
        (d / 'retrain1').mkdir(parents=True)
        base = torch.randn(N, 8, generator=g) + 1.2 * onehot
        steps, snaps = {}, {}
        for e in range(1, run + 1):
            p = (base + 0.3 * torch.randn(N, 8, generator=g)).softmax(1)
            push = torch.zeros(N, 8)
            push[:, K] = signal * (onehot[:, K] - 0.5)
            tralo = (p + push).clamp_min(1e-6)
            snaps[e] = p
            for suffix, v in (('', p), ('_tralo', tralo / tralo.sum(1, keepdim=True)), ('_sham', p)):
                torch.save(v, d / 'retrain1' / f'epoch{e:02d}{suffix}.pt')
            hard = int((p.argmax(1) == K).sum())
            record = dict(applied=True, radius=0.01, hard_before=hard, hard_after=cap)
            steps[str(e)] = dict(tralo=dict(record), sham=dict(record, hard_after=hard))
        torch.save(snaps[best], d / 'retrain1' / 'final_probabilities.pt')
        for arm in ('pto', 'tralo_final', 'sham_final'):
            (d / arm).mkdir()
            (d / arm / 'report.json').write_text(json.dumps(evaluate_global(snaps[best].tolist(), labels, caps,
                                                                            [r['sample_id'] for r in rows])))
        summary = dict(seed=seed, cap=cap, capped_class=K, steps={},
                       retrains=[dict(retrain=1, best_epoch=best, epochs_run=run, hard_count=40, snapshot_steps=steps)])
        if edit:
            edit(summary)
        (d / 'summary.json').write_text(json.dumps(summary))
        (d / 'manifest.json').write_text(json.dumps(dict(cap=cap, capped_class=K, rows=rows)))
        event = dict(event='model_initialized', architecture=arch[0], model_class=arch[1], initial_sha256=f'init{seed}')
        (d / 'events.jsonl').write_text(json.dumps(dict(event='started')) + '\n' + json.dumps(event) + '\n')


def test_ensembles_recover_the_built_in_signal_on_class_one(tmp_path, capsys):
    tree(tmp_path / 'study', (5000, 5001, 5002, 5047))
    s = sf.load(tmp_path / 'study' / 'seed5000')
    assert s['window'] == 6 and s['applied'] == 6 and s['cap'] == CAP
    assert s['arms']['ens_tralo']['cc_f1'] > s['arms']['ens_sham']['cc_f1'] == s['arms']['ens_pto']['cc_f1']
    sf.main(tmp_path / 'study')
    out = capsys.readouterr().out
    assert 'fmow2 mobilenet_v3_large: 4 complete seeds of 48' in out and 'E2 ens_tralo - ens_pto' in out
    assert f'class 1 cap {CAP} on a pool with {s["n_true"]} true class-1 items' in out
    for name, seeds in (('beyond', (5048,)), ('pilot', (5099,)), ('knee', (4700,))):
        tree(tmp_path / name, seeds)
        with pytest.raises(RuntimeError, match='not a study seed'):
            sf.main(tmp_path / name)
    tree(tmp_path / 'wrongnet', (5000,), arch=('resnet18', 'ResNet'))
    with pytest.raises(RuntimeError, match='not trained on mobilenet_v3_large'):
        sf.main(tmp_path / 'wrongnet')
    tree(tmp_path / 'mixed', (5000,))
    tree(tmp_path / 'mixed2', (5001,), cap=CAP + 1)
    (tmp_path / 'mixed2' / 'seed5001').rename(tmp_path / 'mixed' / 'seed5001')
    with pytest.raises(RuntimeError, match='differ in the cap'):
        sf.main(tmp_path / 'mixed')


def test_load_rejects_out_of_spec_runs(tmp_path):
    def drop(s):
        del s['retrains'][0]['snapshot_steps']['3']

    def radius(s):
        s['retrains'][0]['snapshot_steps']['5']['sham']['radius'] = 0.02

    def overshoot(s):
        s['retrains'][0]['snapshot_steps']['7']['tralo']['hard_after'] = CAP + 1

    def cap(s):
        s['cap'] = CAP + 1

    def klass(s):
        s['capped_class'] = 3
    for name, edit in (('drop', drop), ('radius', radius), ('overshoot', overshoot), ('cap', cap), ('klass', klass)):
        tree(tmp_path / name, (5000,), edit=edit)
        with pytest.raises(RuntimeError):
            sf.load(tmp_path / name / 'seed5000')


def test_gate_passes_only_a_byte_identical_pilot_against_its_reference(tmp_path, capsys):
    tree(tmp_path / 'pilot', (5099,))
    (tmp_path / 'pilot' / 'seed5099').rename(tmp_path / 'pilot' / 'seed5099_stepens')
    tree(tmp_path / 'ref', (5099,))
    with pytest.raises(SystemExit, match='no reference run'):
        sf.gate(tmp_path / 'pilot', tmp_path / 'ref')
    (tmp_path / 'ref' / 'seed5099').rename(tmp_path / 'ref' / 'seed5099_ref')
    sf.gate(tmp_path / 'pilot', tmp_path / 'ref')
    out = capsys.readouterr().out
    assert 'PILOT GATE PASSED for 5099' in out and '9 of 9 epochs stepped' in out and 'predicts 40 class-1 items' in out
    epoch = tmp_path / 'ref' / 'seed5099_ref' / 'retrain1' / 'epoch04.pt'
    torch.save(torch.load(epoch, weights_only=True) * (1 + 1e-6), epoch)
    with pytest.raises(SystemExit):
        sf.gate(tmp_path / 'pilot', tmp_path / 'ref')
    tree(tmp_path / 'ref2', (5099,), best=5)
    (tmp_path / 'ref2' / 'seed5099').rename(tmp_path / 'ref2' / 'seed5099_ref')
    with pytest.raises(SystemExit):
        sf.gate(tmp_path / 'pilot', tmp_path / 'ref2')
    tree(tmp_path / 'ref4', (5099,))
    (tmp_path / 'ref4' / 'seed5099').rename(tmp_path / 'ref4' / 'seed5099_ref')
    final = tmp_path / 'ref4' / 'seed5099_ref' / 'retrain1' / 'final_probabilities.pt'
    torch.save(torch.load(final, weights_only=True) * (1 + 1e-6), final)
    with pytest.raises(SystemExit):
        sf.gate(tmp_path / 'pilot', tmp_path / 'ref4')
    tree(tmp_path / 'lone', (5099,))
    with pytest.raises(SystemExit, match='unexpected seed'):
        sf.gate(tmp_path / 'lone', tmp_path / 'ref2')
    tree(tmp_path / 'other', (5099,), arch=('resnet18', 'ResNet'))
    (tmp_path / 'other' / 'seed5099').rename(tmp_path / 'other' / 'seed5099_stepens')
    tree(tmp_path / 'ref3', (5099,))
    (tmp_path / 'ref3' / 'seed5099').rename(tmp_path / 'ref3' / 'seed5099_ref')
    with pytest.raises(SystemExit, match='initial weights'):
        sf.gate(tmp_path / 'other', tmp_path / 'ref3')

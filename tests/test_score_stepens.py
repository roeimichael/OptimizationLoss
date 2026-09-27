import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import score_stepens as se  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

N, CAPS = 826, [None, None, None, 76, None]


def tree(root, seeds, signal=0.1, best=6, run=9, edit=None, arch=None):
    """Write knee_yuval-shaped seed directories with snapshot steps; the tralo snapshots carry extra grade-3 signal."""
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 5, (N,), generator=g).tolist()
    rows = [dict(split='val', label=y, sample_id=f'v{i}') for i, y in enumerate(labels)]
    onehot = torch.nn.functional.one_hot(torch.tensor(labels), 5).float()
    for seed in seeds:
        d = Path(root) / f'seed{seed}'
        (d / 'retrain1').mkdir(parents=True)
        base = torch.randn(N, 5, generator=g) + 1.2 * onehot
        steps, snaps = {}, {}
        for e in range(1, run + 1):
            p = (base + 0.3 * torch.randn(N, 5, generator=g)).softmax(1)
            push = torch.zeros(N, 5)
            push[:, 3] = signal * (onehot[:, 3] - 0.5)
            tralo = (p + push).clamp_min(1e-6)
            snaps[e] = p
            for suffix, v in (('', p), ('_tralo', tralo / tralo.sum(1, keepdim=True)), ('_sham', p)):
                torch.save(v, d / 'retrain1' / f'epoch{e:02d}{suffix}.pt')
            hard = int((p.argmax(1) == 3).sum())
            record = dict(applied=True, radius=0.01, hard_before=hard, hard_after=76)
            steps[str(e)] = dict(tralo=dict(record), sham=dict(record, hard_after=hard))
        torch.save(snaps[best], d / 'retrain1' / 'final_probabilities.pt')
        for arm, p in dict(pto=snaps[best], tralo_final=snaps[best], sham_final=snaps[best]).items():
            (d / arm).mkdir()
            (d / arm / 'report.json').write_text(json.dumps(evaluate_global(p.tolist(), labels, CAPS, [r['sample_id'] for r in rows])))
        summary = dict(seed=seed, cap=76, converged=False, steps={},
                       retrains=[dict(retrain=1, best_epoch=best, epochs_run=run, hard_count=0, snapshot_steps=steps)])
        if edit:
            edit(summary)
        (d / 'summary.json').write_text(json.dumps(summary))
        (d / 'manifest.json').write_text(json.dumps(dict(rows=rows)))
        regnet = seed == 4400 or 4600 <= seed < 4700
        architecture, cls = arch or (('regnet_y_400mf', 'RegNet') if regnet else ('resnet18', 'ResNet'))
        event = dict(event='model_initialized', architecture=architecture, model_class=cls, initial_sha256=f'init{seed}')
        (d / 'events.jsonl').write_text(json.dumps(dict(event='run_started')) + '\n' + json.dumps(event) + '\n')


def test_ensembles_average_the_window_and_recover_the_built_in_signal(tmp_path, capsys):
    tree(tmp_path / 'study', (4500, 4501, 4502, 4503))
    d = tmp_path / 'study' / 'seed4500'
    s = se.load(d)
    assert s['window'] == 9 - 4 + 1 and s['applied'] == s['window']       # epochs max(1, 6 - 2)..9
    manual = torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}.pt', weights_only=True) for e in range(4, 10)]).mean(0)
    rows = json.loads((d / 'manifest.json').read_text())['rows']
    report = evaluate_global(manual.tolist(), [r['label'] for r in rows], CAPS, [r['sample_id'] for r in rows])
    assert s['arms']['ens_pto']['cc_f1'] == se.metrics([r['label'] for r in rows], report['capped_first']['predictions'])['cc_f1']
    assert s['arms']['ens_tralo']['cc_f1'] > s['arms']['ens_sham']['cc_f1'] == s['arms']['ens_pto']['cc_f1']
    se.main(tmp_path / 'study')
    out = capsys.readouterr().out
    assert 'resnet18: 4 complete seeds of 72' in out and 'E1 ens_tralo - ens_sham' in out and 'slots' in out
    tree(tmp_path / 'rgy', (4600, 4601, 4671))
    se.main(tmp_path / 'rgy')
    assert 'regnet_y_400mf: 3 complete seeds of 72' in capsys.readouterr().out
    for name, seeds, arch in (('foreign', (4572,), None), ('mixed', (4500, 4600), None), ('pilot', (4400,), None)):
        tree(tmp_path / name, seeds, arch=arch)
        with pytest.raises(RuntimeError, match='not a study seed'):
            se.main(tmp_path / name)
    tree(tmp_path / 'wrongnet', (4600,), arch=('resnet18', 'ResNet'))
    with pytest.raises(RuntimeError, match='not trained on regnet_y_400mf'):
        se.main(tmp_path / 'wrongnet')


def test_load_rejects_missing_or_out_of_spec_snapshot_steps(tmp_path):
    def drop(s):
        del s['retrains'][0]['snapshot_steps']['3']

    def radius(s):
        s['retrains'][0]['snapshot_steps']['5']['sham']['radius'] = 0.02

    def applied(s):
        s['retrains'][0]['snapshot_steps']['2']['sham']['applied'] = False

    def overshoot(s):
        s['retrains'][0]['snapshot_steps']['7']['tralo']['hard_after'] = 77

    def retrained(s):
        s['retrains'].append(dict(s['retrains'][0], retrain=2))
    for name, edit in (('drop', drop), ('radius', radius), ('applied', applied), ('overshoot', overshoot),
                       ('retrained', retrained)):
        tree(tmp_path / name, (4500,), edit=edit)
        with pytest.raises(RuntimeError):
            se.load(tmp_path / name / 'seed4500')


def pilot(root, seed, **kw):
    tree(root, (seed,), **kw)
    (root / f'seed{seed}').rename(root / f'seed{seed}_stepens')


def test_gate_passes_only_a_byte_identical_pilot(tmp_path, capsys):
    for seed, backbone in ((4000, 'resnet18'), (4400, 'regnet_y_400mf')):
        base = tmp_path / str(seed)
        pilot(base / 'pilot', seed)
        tree(base / 'ref', (seed,))                   # the same generator: the stored run and the pilot agree
        se.gate(base / 'pilot', base / 'ref')
        assert f'PILOT GATE PASSED for {seed} ({backbone})' in capsys.readouterr().out
        epoch = base / 'ref' / f'seed{seed}' / 'retrain1' / 'epoch05.pt'
        torch.save(torch.load(epoch, weights_only=True) + 1e-7, epoch)
        with pytest.raises(SystemExit):
            se.gate(base / 'pilot', base / 'ref')
        tree(base / 'ref2', (seed,), best=5)
        with pytest.raises(SystemExit):
            se.gate(base / 'pilot', base / 'ref2')
    tree(tmp_path / 'crowded', (4000, 4500))
    (tmp_path / 'crowded' / 'seed4000').rename(tmp_path / 'crowded' / 'seed4000_stepens')
    tree(tmp_path / 'clean', (4000,))                 # an unperturbed reference: only the extra seed can fail it
    with pytest.raises(SystemExit, match='unexpected seed'):
        se.gate(tmp_path / 'crowded', tmp_path / 'clean')
    tree(tmp_path / 'lone', (4000,))                  # a study-shaped directory, not the pilot job
    with pytest.raises(SystemExit, match='unexpected seed'):
        se.gate(tmp_path / 'lone', tmp_path / 'clean')
    pilot(tmp_path / 'wrongnet', 4400, arch=('resnet18', 'ResNet'))
    tree(tmp_path / 'rgyref', (4400,))
    with pytest.raises(SystemExit, match='initial weights'):
        se.gate(tmp_path / 'wrongnet', tmp_path / 'rgyref')
    pilot(tmp_path / 'otherinit', 4400)
    tree(tmp_path / 'otherref', (4400,))
    (tmp_path / 'otherref' / 'seed4400' / 'events.jsonl').write_text(json.dumps(
        dict(event='model_initialized', architecture='regnet_y_400mf', initial_sha256='another')) + '\n')
    with pytest.raises(SystemExit, match='initial weights'):
        se.gate(tmp_path / 'otherinit', tmp_path / 'otherref')

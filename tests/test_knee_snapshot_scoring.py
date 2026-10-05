import json
from pathlib import Path

import pytest

from analysis.score_knee_snapshot_local import CONTRASTS, holm, metrics, paired, score_panel
from tralo.knee_snapshot_data import encode, sha256
from test_knee_snapshot_run import cpu_threads, fit, pack


@pytest.fixture
def panel(tmp_path, pack):
    runs = []
    for i in range(4):
        output = tmp_path/f'run{i}'
        runs.append((output,fit(pack,output,61+i)))
    return runs


def test_fictitious_panel_scores_all_arms_policies_and_primary_family(pack, panel):
    scored = score_panel(panel, pack[2], sha256(pack[2].read_bytes()), allow_simulation=True)
    assert scored['status'] == 'fictitious_component_check'
    assert scored['scientific_superiority_certified'] is False
    assert len(scored['contrasts']) == 10
    assert len(CONTRASTS) == 5 and len(scored['scores']) == 4
    for arms in scored['scores'].values():
        assert len(arms) == 6
        for policies in arms.values():
            assert set(policies) == {'global_only','global_local'}
            for item in policies.values():
                assert item['pooled_feasible']
                assert 0 <= item['metrics']['weighted_f1'] <= 1
                assert set(item['groups']) == {'H0','H1'}
    for row in scored['contrasts'].values():
        if row['inference_status'] == 'unavailable_zero_empirical_variance':
            assert row['interval95'] is None and row['p_two_sided'] is None and row['p_holm'] is None
        else:
            assert row['inference_status'] == 'available_t3'
            assert row['p_holm'] >= row['p_two_sided']


def test_weighted_metrics_match_fixed_golden_confusion_fixture():
    # Supports 3/1/1, selected counts 2/2/1, true positives 2/1/1.
    row = metrics([0,0,0,1,2],[0,0,1,1,2])
    assert row['weighted_f1'] == pytest.approx((.8*3+2/3+1)/5)
    assert row['macro_f1'] == pytest.approx((.8+2/3+1)/5)
    assert row['accuracy'] == .8 and row['cc_f1'] == 0


@pytest.mark.parametrize('problem', ['missing', 'tampered_run', 'duplicate_seed', 'simulation'])
def test_failed_panel_gate_never_opens_private_labels(pack, panel, monkeypatch, problem):
    if problem == 'missing': panel = panel[:3]
    if problem == 'tampered_run': (panel[3][0]/'initial.pt').write_bytes(b'tampered')
    if problem == 'duplicate_seed': panel[3] = panel[0]
    original = Path.open
    def guarded(path,*args,**kwargs):
        if path.resolve() == pack[2].resolve():
            raise AssertionError('private labels opened before complete panel gate')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(ValueError):
        score_panel(panel,pack[2],'0'*64,allow_simulation=problem != 'simulation')


@pytest.mark.parametrize('problem', ['hash', 'label', 'id', 'order', 'public_pin', 'source_ids'])
def test_private_join_refuses_rehashed_bad_targets(pack,panel,problem):
    targets = json.loads(pack[2].read_bytes())
    if problem == 'label': targets['rows'][0]['label'] = True
    if problem == 'id': targets['rows'][0]['sample_id'] = targets['rows'][1]['sample_id']
    if problem == 'order': targets['rows'].reverse()
    if problem == 'public_pin': targets['public_manifest_sha256'] = '0'*64
    if problem == 'source_ids': targets['source_sha256_by_id'].pop(next(iter(targets['source_sha256_by_id'])))
    pack[2].write_bytes(encode(targets))
    pin = '0'*64 if problem == 'hash' else sha256(pack[2].read_bytes())
    with pytest.raises(ValueError):
        score_panel(panel,pack[2],pin,allow_simulation=True)


def test_four_seed_interval_and_holm_match_independent_arithmetic():
    from scipy.stats import t
    row = paired([0., .1, .2, .3])
    error = (sum((v-.15)**2 for v in [0.,.1,.2,.3])/3)**.5 / 2
    assert row['mean'] == pytest.approx(.15)
    assert row['interval95'] == pytest.approx([.15-t.ppf(.975,3)*error, .15+t.ppf(.975,3)*error])
    assert row['p_two_sided'] == pytest.approx(2*t.sf(.15/error,3))
    rows = {str(i):dict(p_two_sided=p) for i,p in enumerate([.01,.04,.03,.8])}
    holm(rows)
    assert [rows[str(i)]['p_holm'] for i in range(4)] == pytest.approx([.04,.09,.09,.8])
    assert paired([0.]*4)['p_two_sided'] is None
    assert paired([.1]*4)['p_two_sided'] is None
    with pytest.raises(ValueError): paired([0.]*3)

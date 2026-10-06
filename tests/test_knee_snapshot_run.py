import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from torchvision.transforms import ToTensor

from tralo.knee_snapshot_data import encode, prepare, sha256
from tralo.knee_snapshot_local import RECIPE
from tralo.knee_snapshot_run import fit_public, verify_run


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def pack(tmp_path):
    source, public, private = tmp_path / 'source', tmp_path / 'public', tmp_path / 'private.json'
    for split, count, offset in (('train', 40, 1000000), ('val', 100, 2000000)):
        for i in range(count):
            path = source / split / str(i % 5) / f'{offset+i:07d}L.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (7, 9), (i, offset//1000000, i//2)).save(path)
    pin = prepare(source, public, private, {'train':40, 'development':100})['public_manifest_sha256']
    return public, pin, private, source


def model():
    net = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 5))
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.zero_()
        net[-1].bias[0] = 1.
    return net


def fit(pack, output, seed=61):
    public, pin, _, _ = pack
    return fit_public(model(), public, pin, dict(RECIPE, seed=seed, max_epochs=2, patience=2), output,
                      source_commit='0'*40, simulation=True, transforms=(ToTensor(), ToTensor()))


def repin(directory, relative, value):
    (directory / relative).write_bytes(encode(value))
    completion = json.loads((directory / 'complete.json').read_bytes())
    completion['files'][relative] = sha256((directory / relative).read_bytes())
    (directory / 'complete.json').write_bytes(encode(completion))
    return sha256((directory / 'complete.json').read_bytes())


def test_self_resource_observations_bind_start_result_and_completion(tmp_path, pack):
    # Prospectively declared fictitious logging fixture; never a science seed.
    pin = fit(pack, tmp_path/'run', seed=9482001)
    checked = verify_run(tmp_path/'run', pin, allow_simulation=True)
    result, events = checked['result'], checked['events']
    usage = result['process_usage']
    assert events[0]['process_usage'] == usage['start']
    assert events[-1]['process_usage'] == usage
    assert usage['start']['scope'] == 'process_self_not_children_or_gpu'
    assert usage['start']['pid'] == usage['end']['pid']
    assert usage['delta']['status'] in {'observed', 'unavailable'}


def test_rehashed_self_resource_tampering_is_semantically_refused(tmp_path, pack):
    pin = fit(pack, tmp_path/'run', seed=9482002)
    result = json.loads((tmp_path/'run/result.json').read_bytes())
    result['process_usage']['end']['pid'] += 1
    events = [json.loads(line) for line in (tmp_path/'run/events.jsonl').read_bytes().splitlines()]
    events[-1]['process_usage'] = result['process_usage']
    (tmp_path/'run/events.jsonl').write_bytes(b''.join(encode(e) for e in events))
    completion = json.loads((tmp_path/'run/complete.json').read_bytes())
    completion['files']['events.jsonl'] = sha256((tmp_path/'run/events.jsonl').read_bytes())
    (tmp_path/'run/complete.json').write_bytes(encode(completion))
    pin = repin(tmp_path/'run', 'result.json', result)
    with pytest.raises(ValueError, match='process|usage'):
        verify_run(tmp_path/'run', pin, allow_simulation=True)


def test_usage_observer_failure_preserves_original_fit_failure(tmp_path, pack, monkeypatch):
    from tralo.process_usage import self_usage
    first = self_usage()
    calls = []
    def usage():
        calls.append(True)
        if len(calls) == 1: return first
        raise OSError('fictitious resource observation failure')
    def failed(*args, **kwargs):
        raise RuntimeError('fictitious original correction failure')
    monkeypatch.setattr('tralo.process_usage.self_usage', usage)
    monkeypatch.setattr('tralo.knee_snapshot_local.local_targeted_step', failed)
    output = tmp_path/'run'
    with pytest.raises(RuntimeError, match='original correction failure'):
        fit(pack, output, seed=9482003)
    event = json.loads((output/'events.jsonl').read_bytes().splitlines()[-1])
    assert event['reason'] == 'fictitious original correction failure'
    assert event['process_usage']['status'] == 'observation_failed'
    assert event['process_usage']['error'] == 'OSError'
    assert not (output/'complete.json').exists()


def test_complete_real_reader_fit_logs_and_verifies_without_private_reads(tmp_path, pack, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        candidate = path.resolve()
        if candidate == pack[2] or pack[3] in candidate.parents:
            raise AssertionError('runtime opened private/original data')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    output = tmp_path / 'run'
    pin = fit(pack, output)
    checked = verify_run(output, pin, allow_simulation=True)
    assert checked['result']['fit']['epochs_run'] == 2
    assert checked['result']['window'] == [1, 2]
    assert len(checked['averages']) == 6
    assert checked['result']['execution']['cuda_initialized'] is False
    events = checked['events']
    assert events[0]['event'] == 'run_started' and events[-1]['event'] == 'run_completed'
    assert len([e for e in events if e['event'] == 'snapshot_arm_completed']) == 12
    assert not torch.cuda.is_initialized()
    with pytest.raises(FileExistsError):
        fit(pack, output)
    with pytest.raises(ValueError, match='simulation'):
        verify_run(output, pin)


def test_final_best_checkpoint_and_original_pto_trajectory_match_reference(tmp_path, pack):
    from tralo.global_comparison import _state_hash
    from tralo.knee_snapshot_data import load_public
    from tralo.knee_end_to_end import development_images
    from tralo.knee_yuval import Images, train_run
    initial = model()
    config = dict(RECIPE, seed=61, max_epochs=2, patience=2)
    _, rows = load_public(pack[0], pack[1], allow_synthetic=True)
    training, stopping = Images(pack[0], rows['train']), Images(pack[0], rows['stop'])
    eval_tf = ToTensor()
    stop = [stopping.batch(list(range(len(stopping.labels))), eval_tf)]
    pool = development_images(pack[0], rows['development'], eval_tf, 16)
    reference_events = []
    reference = copy.deepcopy(initial)
    expected = train_run(reference, training, stop, pool, config, torch.ones(5), reference_events.append,
                         lambda *args: None, transforms=(eval_tf, eval_tf))
    pin = fit(pack, tmp_path/'run')
    checked = verify_run(tmp_path/'run', pin, allow_simulation=True)
    assert checked['result']['fit'] == expected
    assert checked['result']['restored_model_sha256'] == _state_hash(reference)
    logged = [{k:v for k,v in e.items() if k not in {'schema_version','sequence','timestamp'}}
              for e in checked['events'] if e['event'] == 'epoch']
    assert logged == reference_events


def test_driver_verifier_accepts_active_joint_and_native_with_matched_controls(tmp_path, pack):
    net = model()
    with torch.no_grad():
        net[-1].bias.zero_()
        net[-1].bias[3] = .01
    output = tmp_path/'active'
    pin = fit_public(net,pack[0],pack[1],dict(RECIPE,seed=71,max_epochs=2,patience=2),output,
                     source_commit='0'*40,simulation=True,transforms=(ToTensor(),ToTensor()))
    checked = verify_run(output,pin,allow_simulation=True)
    assert checked['result']['snapshots']['1']['arms']['joint_local']['applied']
    assert checked['result']['snapshots']['1']['arms']['global_native']['applied']


@pytest.mark.parametrize('bad', ['config', 'window', 'best', 'cost', 'artifact_path', 'file_set'])
def test_rehashed_semantic_tampering_refused(tmp_path, pack, bad):
    output = tmp_path/'run'
    pin = fit(pack, output)
    result = json.loads((output/'result.json').read_bytes())
    if bad == 'config':
        config = json.loads((output/'config.json').read_bytes())
        config['weight_decay'] = 1e-4
        pin = repin(output, 'config.json', config)
    elif bad == 'file_set':
        completion = json.loads((output/'complete.json').read_bytes())
        completion['files'].pop('initial.pt')
        (output/'complete.json').write_bytes(encode(completion))
        pin = sha256((output/'complete.json').read_bytes())
    else:
        if bad == 'window': result['window'] = [2]
        if bad == 'best': result['fit']['best_epoch'] = 2 if result['fit']['best_epoch'] == 1 else 1
        if bad == 'cost': result['elapsed_wall_seconds'] = -1
        if bad == 'artifact_path': result['ensemble']['pto']['file'] = '../private.json'
        # Rebind all altered result fields in the closing event so this fixture
        # reaches semantic validation rather than stopping at the event hash.
        events = [json.loads(line) for line in (output/'events.jsonl').read_bytes().splitlines()]
        for key in ('fit','window','ensemble','elapsed_wall_seconds'):
            events[-1][key] = result[key]
        (output/'events.jsonl').write_bytes(b''.join(encode(e) for e in events))
        completion = json.loads((output/'complete.json').read_bytes())
        completion['files']['events.jsonl'] = sha256((output/'events.jsonl').read_bytes())
        (output/'complete.json').write_bytes(encode(completion))
        pin = repin(output, 'result.json', result)
    with pytest.raises(ValueError):
        verify_run(output, pin, allow_simulation=True)


def test_ensemble_tensor_tamper_rejected_even_when_artifact_hash_updated(tmp_path, pack):
    output = tmp_path/'run'
    fit(pack, output)
    result = json.loads((output/'result.json').read_bytes())
    path = output/result['ensemble']['pto']['file']
    values = torch.load(path, weights_only=True).roll(1, 1)
    torch.save(values, path)
    result['ensemble']['pto']['sha256'] = sha256(path.read_bytes())
    events = [json.loads(line) for line in (output/'events.jsonl').read_bytes().splitlines()]
    events[-1]['ensemble'] = result['ensemble']
    (output/'events.jsonl').write_bytes(b''.join(encode(e) for e in events))
    completion = json.loads((output/'complete.json').read_bytes())
    completion['files'][str(path.relative_to(output)).replace(os.sep, '/')] = sha256(path.read_bytes())
    completion['files']['events.jsonl'] = sha256((output/'events.jsonl').read_bytes())
    (output/'complete.json').write_bytes(encode(completion))
    pin = repin(output, 'result.json', result)
    with pytest.raises(ValueError, match='average'):
        verify_run(output, pin, allow_simulation=True)


def test_failed_fit_preserves_log_and_partial_snapshot_without_completion(tmp_path, pack, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('fictitious bounded-search failure')
    monkeypatch.setattr('tralo.knee_snapshot_local.local_targeted_step', failed)
    output = tmp_path/'run'
    with pytest.raises(RuntimeError, match='bounded-search failure'):
        fit(pack, output)
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'run_failed'
    assert events[-1]['process_usage']['start'] == events[0]['process_usage']
    assert events[-1]['process_usage']['delta']['status'] in {'observed', 'unavailable'}
    assert (output/'snapshots/epoch01/global_native.pt').exists()
    assert not (output/'complete.json').exists()


@pytest.mark.parametrize('selection', ['', '0', 'GPU-abcd', 'GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002',
                                      'GPU-00000000-0000-0000-0000-000000000001'])
def test_launch_cli_refuses_before_any_artifacts_or_tensor_imports(tmp_path, selection):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=selection)
    result = subprocess.run([sys.executable, '-m', 'tralo.knee_snapshot_run', str(tmp_path/'unused')],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert ('one complete physical' if not selection.startswith('GPU-00000000-0000-0000-0000-000000000001')
            or ',' in selection else 'campaign gates') in result.stderr
    assert not (tmp_path/'unused').exists()
    # A fresh interpreter proves the refusal path never imports torch.
    code = "import sys; from tralo.knee_snapshot_run import main;\ntry: main()\nexcept RuntimeError: pass\nassert 'torch' not in sys.modules"
    guard = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert guard.returncode == 0, guard.stderr

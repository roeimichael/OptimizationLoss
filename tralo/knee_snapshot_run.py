"""Training/artifact driver for the approved isolated knee snapshot comparison.

The CLI refuses until an authenticated real campaign-gate issuer exists. The
callable driver accepts a constructed model; it does not obtain devices, claims,
pretrained weights or budget permission. CPU fixture mode is explicitly marked.
No tensor library is imported by the CLI refusal path.
"""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import time

from .knee_snapshot_data import encode, group_for, load_public, sha256

FORMAT = 'knee_snapshot_run_20261005'


def _write(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def _simulation_config(config):
    from .knee_snapshot_local import RECIPE, SEEDS
    if set(config) != set(RECIPE) | {'seed'}:
        raise ValueError('fixture configuration keys differ')
    overrides = {'seed', 'max_epochs', 'patience', 'batch_size', 'development_batch_size'}
    if (any(type(config[k]) is not int or config[k] <= 0 for k in overrides)
            or config['seed'] in SEEDS or config['max_epochs'] > 3):
        raise ValueError('fixture mode cannot use scientific seeds or horizons')
    for key in set(RECIPE) - overrides:
        if type(config[key]) is not type(RECIPE[key]) or config[key] != RECIPE[key]:
            raise ValueError('fixture changed a fixed recipe field: ' + key)


def _source_pins():
    root = Path(__file__).resolve().parent
    return {'tralo/' + path.name: sha256(path.read_bytes()) for path in sorted(root.glob('*.py'))}


def fit_public(model, public, manifest_sha256, config, output, *, source_commit,
               simulation=False, transforms=None):
    """One PTO fit, six isolated corrections per epoch and identical ensembles.

    This is not an authorization boundary: a future campaign launcher must first
    certify actual data/derivative/logging/source/cost/ownership/seed gates. The
    disabled CLI prevents this preparation stage from dispatching a campaign.
    """
    import torch
    from .events import EventLog
    from .global_comparison import _state_hash
    from .knee_end_to_end import development_images
    from .knee_snapshot_local import ARMS, average_snapshots, snapshot, validate
    from .knee_yuval import Images, train_run, transforms_for

    if type(simulation) is not bool or re.fullmatch(r'[0-9a-f]{40}', source_commit or '') is None:
        raise ValueError('invalid execution/source identity')
    if simulation:
        _simulation_config(config)
        if torch.cuda.is_initialized() or any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('fictitious readiness driver is CPU only')
    else:
        validate(config)
        if transforms is not None:
            raise ValueError('campaign transforms cannot be replaced')
    public, output = Path(public).resolve(), Path(output).resolve()
    if output == public or public in output.parents or output in public.parents:
        raise ValueError('run output must be separate from public pack')
    manifest, rows = load_public(public, manifest_sha256, allow_synthetic=simulation)
    if simulation and manifest['synthetic_or_nonstandard_counts'] is not True:
        raise ValueError('fixture mode requires a fictitious/nonstandard pack')
    manifest_data, row_data = (public/'manifest.json').read_bytes(), (public/'rows.json').read_bytes()
    if sha256(manifest_data) != manifest_sha256 or sha256(row_data) != manifest['rows_sha256']:
        raise ValueError('public metadata changed between validation and capture')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('approved driver requires float32 parameters')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    execution = dict(mode='cpu_fictitious' if simulation else 'campaign_driver',
                     host=socket.gethostname(), device=str(next(model.parameters()).device),
                     cuda_initialized=torch.cuda.is_initialized(), torch_version=str(torch.__version__),
                     precision='float32', model_class=type(model).__name__)
    source_pins = _source_pins()
    with EventLog(output / 'events.jsonl') as log:
        def emit(event):
            log.emit(event['event'], **{k:v for k,v in event.items() if k != 'event'})
        def tensor_file(path, value):
            with path.open('xb') as stream:
                torch.save(value, stream)
            return dict(file=path.relative_to(output).as_posix(), sha256=sha256(path.read_bytes()))
        try:
            log.emit('run_started', seed=config['seed'], config=config, public_manifest_sha256=manifest_sha256,
                     source_commit=source_commit, source_sha256=source_pins, execution=execution)
            _write(output/'config.json', encode(config))
            _write(output/'public_manifest.json', manifest_data)
            _write(output/'public_rows.json', row_data)
            initial_hash = _state_hash(model)
            tensor_file(output/'initial.pt', {k:v.detach().cpu() for k,v in model.state_dict().items()})
            train_tf, eval_tf = transforms or transforms_for()
            training, stopping = Images(public, rows['train']), Images(public, rows['stop'])
            stop = [stopping.batch(list(range(start, min(start+config['batch_size'], len(stopping.labels)))), eval_tf)
                    for start in range(0, len(stopping.labels), config['batch_size'])]
            pool = development_images(public, rows['development'], eval_tf, config['development_batch_size'])
            groups = [r['group'] for r in rows['development']]
            quota = dict(global_cap=manifest['global_cap'], local_caps=manifest['local_caps'])
            records, epoch_hashes = {}, {}
            def save_epoch(epoch, probabilities):
                epoch_hashes[str(epoch)] = _state_hash(model)
                records[str(epoch)] = snapshot(model, pool, groups, quota, config['seed'], epoch,
                                              output/'snapshots'/f'epoch{epoch:02d}', probabilities, emit)
            fitted = train_run(model, training, stop, pool, config, torch.ones(5), emit, save_epoch,
                               transforms=(train_tf, eval_tf))
            window, averages = average_snapshots(output/'snapshots', records,
                                                 fitted['best_epoch'], fitted['epochs_run'])
            (output/'ensemble').mkdir()
            ensemble = {arm:tensor_file(output/'ensemble'/(arm+'.pt'), averages[arm]) for arm in ARMS}
            final = tensor_file(output/'final.pt', {k:v.detach().cpu() for k,v in model.state_dict().items()})
            restored = _state_hash(model)
            if restored != epoch_hashes[str(fitted['best_epoch'])]:
                raise RuntimeError('training-only best state was not restored')
            if simulation and torch.cuda.is_initialized():
                raise RuntimeError('CPU driver initialized CUDA')
            result = dict(format=FORMAT, seed=config['seed'], source_commit=source_commit,
                          source_sha256=source_pins, execution=execution, fit=fitted, window=window,
                          public_manifest_sha256=manifest_sha256, snapshots=records,
                          epoch_state_sha256=epoch_hashes, initial_model_sha256=initial_hash,
                          restored_model_sha256=restored, final=final, ensemble=ensemble,
                          elapsed_wall_seconds=time.monotonic()-started,
                          limitation='Driver wall time is not GPU runtime/ownership/budget certification; model provenance is a separate gate.')
            _write(output/'result.json', encode(result))
            log.emit('run_completed', seed=config['seed'], fit=fitted, window=window,
                     ensemble=ensemble, restored_model_sha256=restored,
                     elapsed_wall_seconds=result['elapsed_wall_seconds'])
        except BaseException as exc:
            log.emit('run_failed', seed=config['seed'], exception=type(exc).__name__, reason=str(exc),
                     elapsed_wall_seconds=time.monotonic()-started)
            raise
    files = {p.relative_to(output).as_posix():sha256(p.read_bytes())
             for p in sorted(output.rglob('*')) if p.is_file()}
    completion = dict(format=FORMAT, status='driver_complete_not_campaign_certification',
                      mode=execution['mode'], seed=config['seed'], files=files)
    _write(output/'complete.json', encode(completion))
    return sha256((output/'complete.json').read_bytes())


def verify_run(directory, completion_sha256, *, allow_simulation=False):
    """Check a complete run before a scorer may open private targets.

    This validates stored identity, events, counts, doses, early stopping and
    averaging. It does not replay models or prove historical device ownership.
    """
    from .knee_snapshot_local import ARMS, validate
    from .local_policy import size_share_caps
    from .knee_snapshot_data import FORMAT as INPUT_FORMAT, NAMESPACE, ROW_KEYS, stopping_subject
    directory = Path(directory).resolve()
    def read(relative):
        path = directory / relative
        if path.is_symlink() or directory not in path.resolve().parents:
            raise ValueError('run artifact escapes directory')
        return path.read_bytes()
    completion_data = read('complete.json')
    if sha256(completion_data) != completion_sha256:
        raise ValueError('completion pin mismatch')
    completion = json.loads(completion_data)
    if (set(completion) != {'format','status','mode','seed','files'} or completion['format'] != FORMAT
            or completion['status'] != 'driver_complete_not_campaign_certification'):
        raise ValueError('invalid completion contract')
    simulation = completion['mode'] == 'cpu_fictitious'
    if completion['mode'] not in {'cpu_fictitious','campaign_driver'} or simulation and not allow_simulation:
        raise ValueError('simulation is not a scientific campaign')
    for name, pin in completion['files'].items():
        if not isinstance(name, str) or sha256(read(name)) != pin:
            raise ValueError('changed run artifact: ' + str(name))
    result, config = json.loads(read('result.json')), json.loads(read('config.json'))
    (_simulation_config if simulation else validate)(config)
    manifest_data, row_data = read('public_manifest.json'), read('public_rows.json')
    manifest, rows = json.loads(manifest_data), json.loads(row_data)
    if (manifest['format'] != INPUT_FORMAT or manifest['namespace'] != NAMESPACE
            or manifest['status'] != 'prepared_not_campaign_ready' or manifest['global_cap'] != 76
            or manifest['capped_class'] != 3 or manifest['local_total'] != 95
            or manifest['rows_sha256'] != sha256(row_data) or set(rows) != {'train','stop','development'}):
        raise ValueError('copied public input contract mismatch')
    identities, subjects = set(), {}
    for role, items in rows.items():
        subjects[role] = set()
        for row in items:
            sid = row['sample_id']
            image_role = 'development' if role == 'development' else 'train'
            if (set(row) != (ROW_KEYS if role == 'development' else ROW_KEYS | {'label'})
                    or not re.fullmatch(r'\d{7}[LR]', sid) or sid[:7] != row['subject'] or sid in identities
                    or row['path'] != f'images/{image_role}/{sid}.png' or row['group'] != group_for(row['subject'])
                    or row['split'] != ('val' if role == 'development' else 'train')):
                raise ValueError('copied public row identity/label boundary mismatch')
            if role != 'development' and (type(row['label']) is not int or not 0 <= row['label'] < 5
                                         or stopping_subject(row['subject']) != (role == 'stop')):
                raise ValueError('copied training label/carve mismatch')
            identities.add(sid)
            subjects[role].add(row['subject'])
        if not items or [r['sample_id'] for r in items] != sorted(r['sample_id'] for r in items):
            raise ValueError('empty or nonneutral copied row order')
    if any(subjects[a] & subjects[b] for a,b in [('train','stop'),('train','development'),('stop','development')]):
        raise ValueError('copied subject roles overlap')
    groups = [r['group'] for r in rows['development']]
    if (manifest['local_caps'] != size_share_caps(groups, 95) or set(groups) != {'H0','H1'}
            or manifest['counts'] != {role:len(items) for role,items in rows.items()}
            or not simulation and (len(rows['train'])+len(rows['stop']) != 5778 or len(groups) != 826)
            or manifest['synthetic_or_nonstandard_counts'] != simulation):
        raise ValueError('copied input counts/quotas/mode mismatch')
    if (result['format'] != FORMAT or result['seed'] != config['seed'] or completion['seed'] != config['seed']
            or result['public_manifest_sha256'] != sha256(manifest_data)
            or result['execution']['mode'] != completion['mode']
            or not math.isfinite(result['elapsed_wall_seconds']) or result['elapsed_wall_seconds'] <= 0):
        raise ValueError('result identity/time mismatch')
    events = [json.loads(line) for line in read('events.jsonl').splitlines()]
    if (not events or [e['sequence'] for e in events] != list(range(len(events)))
            or any(e['schema_version'] != 1 for e in events)
            or events[0]['event'] != 'run_started' or events[-1]['event'] != 'run_completed'
            or any(e['event'].endswith('failed') for e in events)):
        raise ValueError('incomplete/nonsequential event log')
    start, end = events[0], events[-1]
    if (any(start[k] != result[k] for k in ['seed','source_commit','source_sha256','execution','public_manifest_sha256'])
            or start['config'] != config or any(end[k] != result[k] for k in
                ['seed','fit','window','ensemble','restored_model_sha256','elapsed_wall_seconds'])):
        raise ValueError('events differ from result identity')
    fit = result['fit']
    epoch_events = [e for e in events if e['event'] == 'epoch']
    last, best = fit['epochs_run'], fit['best_epoch']
    if (type(last) is not int or not 1 <= last <= config['max_epochs']
            or [e['epoch'] for e in epoch_events] != list(range(1,last+1))):
        raise ValueError('incomplete training epochs')
    minimum, chosen, waited = math.inf, 0, 0
    updates_per_epoch = math.ceil(len(rows['train'])/config['batch_size'])
    for e in epoch_events:
        improved = e['stop_loss'] < minimum
        if improved: minimum, chosen, waited = e['stop_loss'], e['epoch'], 0
        else: waited += 1
        if (any(not math.isfinite(e[k]) for k in ['stop_loss','training_loss','first_task_gradient_norm',
                                                'first_task_displacement_norm','soft_count_capped'])
                or e['improved'] != improved
                or e['task_updates'] != updates_per_epoch*e['epoch']
                or e['epoch'] < last and waited >= config['patience']):
            raise ValueError('training-only stopping/update contract mismatch')
    window = list(range(max(1, chosen-2), last+1))
    if (best != chosen or fit['best_stop_loss'] != minimum or result['window'] != window
            or last < config['max_epochs'] and waited < config['patience']
            or fit['task_updates'] != updates_per_epoch*last
            or fit['first_order_sha256'] != epoch_events[0]['epoch_order_sha256']
            or fit['first_batch_sha256'] != epoch_events[0]['epoch_first_batch_sha256']
            or set(result['snapshots']) != {str(i) for i in range(1,last+1)}
            or set(result['epoch_state_sha256']) != set(result['snapshots'])
            or result['restored_model_sha256'] != result['epoch_state_sha256'][str(best)]):
        raise ValueError('best/window/task trajectory contract mismatch')
    import torch
    def tensor(relative, pin, shape):
        data = read(relative)
        if sha256(data) != pin: raise ValueError('tensor pin mismatch')
        values = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if (values.shape != shape or values.dtype != torch.float32 or not torch.isfinite(values).all()
                or bool((values < 0).any())
                or not torch.allclose(values.sum(1), torch.ones(shape[0]), atol=1e-6, rtol=0)):
            raise ValueError('invalid probability tensor')
        return values
    expected_files = {'config.json','public_manifest.json','public_rows.json','initial.pt','final.pt',
                      'result.json','events.jsonl'}
    for checkpoint, identity in [('initial.pt','initial_model_sha256'),('final.pt','restored_model_sha256')]:
        state = torch.load(io.BytesIO(read(checkpoint)), map_location='cpu', weights_only=True)
        digest = hashlib.sha256()
        for name, value in state.items():
            digest.update(name.encode()); digest.update(value.contiguous().numpy().tobytes())
        if digest.hexdigest() != result[identity]: raise ValueError('checkpoint state identity mismatch')
    snapshot_values = {arm:[] for arm in ARMS}
    completed = [e for e in events if e['event'] == 'snapshot_completed']
    if [e['epoch'] for e in completed] != list(range(1,last+1)):
        raise ValueError('snapshot completion epochs mismatch')
    for epoch in range(1,last+1):
        saved = result['snapshots'][str(epoch)]
        arm_events = [e for e in events if e['event'] == 'snapshot_arm_completed' and e['epoch'] == epoch]
        if ([e['arm'] for e in arm_events] != list(ARMS) or set(saved['arms']) != set(ARMS)
                or set(saved['files']) != set(ARMS) or completed[epoch-1]['files'] != saved['files']
                or completed[epoch-1]['original_state_sha256'] != result['epoch_state_sha256'][str(epoch)]):
            raise ValueError('snapshot arm/state completeness mismatch')
        begun = [e for e in events if e['event'] == 'snapshot_arm_started' and e['epoch'] == epoch]
        if ([e['arm'] for e in begun] != list(ARMS[1:]) or any(e['seed'] != config['seed']
                or e['original_state_sha256'] != result['epoch_state_sha256'][str(epoch)] for e in begun)):
            raise ValueError('snapshot start/state identity mismatch')
        pto_values = None
        for arm, event in zip(ARMS, arm_events):
            artifact, record = saved['files'][arm], saved['arms'][arm]
            relative = f'snapshots/epoch{epoch:02d}/{arm}.pt'
            if artifact['file'] != arm+'.pt' or event['artifact'] != artifact or any(event[k] != v for k,v in record.items()):
                raise ValueError('snapshot event/artifact binding mismatch')
            values = tensor(relative, artifact['sha256'], (len(groups),5))
            if arm == 'pto': pto_values = values
            calls = values.argmax(1).tolist()
            counts = record['after_counts']
            if (counts['hard_global'] != sum(c==3 for c in calls)
                    or counts['hard_local'] != {g:sum(c==3 for c,name in zip(calls,groups) if name==g) for g in {'H0','H1'}}
                    or not math.isclose(counts['soft_global'], float(values[:,3].sum()), abs_tol=1e-5)
                    or any(not math.isclose(counts['soft_local'][g],float(values[[i for i,name in enumerate(groups)
                                                                                 if name==g],3].sum()),abs_tol=1e-5)
                           for g in {'H0','H1'})):
                raise ValueError('snapshot saved counts mismatch')
            if (event['seed'] != config['seed'] or record['before_counts'] != saved['arms']['pto']['after_counts']
                    or record['global_residual_after'] != counts['hard_global'] - 76
                    or record['local_residual_after'] != {g:counts['hard_local'][g]-cap for g,cap in manifest['local_caps'].items()}
                    or record['global_residual_before'] != record['before_counts']['hard_global'] - 76
                    or record['local_residual_before'] != {g:record['before_counts']['hard_local'][g]-cap
                                                          for g,cap in manifest['local_caps'].items()}
                    or type(record['applied']) is not bool
                    or any(not math.isfinite(record.get(k,0.)) or record.get(k,0.) < 0
                           for k in ['radius','displacement','gradient_norm'])
                    or any(not math.isfinite(n) or n < 0 for n in record['tensor_displacement_norms'])
                    or not math.isclose(record['displacement'],math.sqrt(sum(n*n for n in record['tensor_displacement_norms'])),
                                        rel_tol=1e-6,abs_tol=1e-8)
                    or record['planned_checks'] != int(arm != 'pto')
                    or record['applied_updates'] != int(record['applied'])
                    or record['skipped_updates'] != int(arm != 'pto' and not record['applied'])
                    or record['attempted_updates'] != int(record.get('gradient_norm',0.)>0)
                    or record['multiplier'] != 'not_applicable' or record['augmentation_coefficient'] != 'not_applicable'):
                raise ValueError('snapshot residual/dose/update accounting mismatch')
            if not record['applied'] and (record['displacement'] != 0 or not torch.equal(values,pto_values)):
                raise ValueError('inactive correction changed the PTO snapshot')
            if record['applied'] and arm in {'global_native','joint_local'} and (
                    counts['hard_global'] > 76 or arm == 'joint_local' and
                    any(counts['hard_local'][g]>cap for g,cap in manifest['local_caps'].items())):
                raise ValueError('targeted snapshot does not meet its hard constraints')
            if arm == 'pto' and (epoch_events[epoch-1]['hard_counts'] != torch.bincount(values.argmax(1), minlength=5).tolist()
                                or epoch_events[epoch-1]['soft_count_capped'] != counts['soft_global']):
                raise ValueError('PTO epoch/snapshot counts mismatch')
            expected_files.add(relative)
            if epoch in window: snapshot_values[arm].append(values)
        for real, control, per_tensor in [('global_native','global_native_sham',True),
                                          ('joint_local','global_at_joint_radius',False),('joint_local','joint_sham',True)]:
            a, b = saved['arms'][real], saved['arms'][control]
            if (a['applied'] != b['applied'] or any(not math.isclose(a[k],b[k],rel_tol=1e-4,abs_tol=1e-5)
                                                  for k in ['radius','displacement'])
                    or per_tensor and (len(a['tensor_displacement_norms']) != len(b['tensor_displacement_norms'])
                                       or any(not math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-5)
                                              for x,y in zip(a['tensor_displacement_norms'],b['tensor_displacement_norms'])))):
                raise ValueError('snapshot matched-control dose mismatch')
    if set(result['ensemble']) != set(ARMS) or result['final']['file'] != 'final.pt' or result['final']['sha256'] != sha256(read('final.pt')):
        raise ValueError('final/ensemble artifact contract mismatch')
    averages = {}
    for arm in ARMS:
        artifact = result['ensemble'][arm]
        relative = f'ensemble/{arm}.pt'
        if artifact['file'] != relative: raise ValueError('ensemble artifact path mismatch')
        values = tensor(relative, artifact['sha256'], (len(groups),5))
        if not torch.equal(values, torch.stack(snapshot_values[arm]).mean(0)):
            raise ValueError('ensemble differs from independent snapshot average')
        averages[arm] = values
        expected_files.add(relative)
    if set(completion['files']) != expected_files:
        raise ValueError('completion file set differs from complete frozen run')
    return dict(result=result, config=config, manifest=manifest, rows=rows, events=events,
                averages=averages, completion_sha256=completion_sha256)


def main():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', selected) is None:
        raise RuntimeError('run requires one complete physical GPU UUID before any artifacts or CUDA')
    raise RuntimeError('authenticated real campaign gates, finite budget and exclusive seed/device claims are not implemented; dispatch refused')


if __name__ == '__main__':
    main()

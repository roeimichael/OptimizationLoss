"""Independent, complete-block knee scorer with a label-blind integrity gate.

python -m analysis.score_knee_persistent_match --gate RUN_ROOT
python -m analysis.score_knee_persistent_match --score FULL_ROOT DATA_ROOT NEW_JSON

The gate does not import development labels. --score first gates all 36 study
runs, then reads the development labels once for the fixed complete block.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from tralo.global_report import evaluate_global
from tralo.knee_experiment import digest, source
from tralo.knee_persistent_match import (BACKBONES, CAPS_STUDY, SEEDS_STUDY,
                                         VIT_RECOVERY_PILOT, WEIGHT_SHA256, validate)
from tralo.knee_yuval import CAPPED, carve, f_and_derivative

CAPS_CURVE = (43, 54, 65, 76, 86, 97)


def _load(path):
    return json.loads(Path(path).read_text())


def _events(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or [r['sequence'] for r in rows] != list(range(len(rows))):
        raise RuntimeError('missing or nonsequential events')
    return rows


def _probabilities(path, expected_hash, size=826):
    import torch
    path = Path(path)
    if digest(path) != expected_hash:
        raise RuntimeError('probability artifact hash mismatch: ' + str(path))
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value.shape != (size, 5) or not bool(torch.isfinite(value).all()) or
            bool((value < 0).any()) or
            not bool(torch.allclose(value.sum(1), torch.ones(size), atol=1e-6))):
        raise RuntimeError('invalid probability matrix: ' + str(path))
    return value


def _arm(root, name, item, train_count, batch_size):
    """Recompute a saved arm's predictions and complete epoch/accounting trail."""
    import torch
    path = Path(root) / name
    probabilities = _probabilities(Path(root) / item['file'], item['probability_sha256'])
    if item['file'] != f'{name}/final_probabilities.pt' or item['training_file'] != f'{name}/training.json':
        raise RuntimeError(name + ' artifact paths differ from the fixed layout')
    if digest(path / 'model.pt') != item['model_sha256']:
        raise RuntimeError(name + ' model artifact hash mismatch')
    if (not math.isfinite(item['seconds']) or item['seconds'] <= 0 or
            type(item['peak_gpu_bytes']) is not int or item['peak_gpu_bytes'] <= 0):
        raise RuntimeError(name + ' cost log is invalid')
    training = _load(path / 'training.json')
    for key in ('best_epoch', 'epochs_run', 'best_stop_loss', 'task_updates',
                'first_order_sha256', 'first_batch_sha256'):
        if training[key] != item[key]:
            raise RuntimeError(name + ' training summary differs from saved artifact')
    rows = _events(path / 'events.jsonl')
    epochs = [row for row in rows if row['event'] == 'epoch']
    if (len(epochs) != item['epochs_run'] or rows[-1]['event'] != 'training_completed' or
            not 1 <= item['best_epoch'] <= item['epochs_run']):
        raise RuntimeError(name + ' epoch or completion log is incomplete')
    completed = rows[-1]
    for key, value in training.items():
        if completed[key] != value:
            raise RuntimeError(name + ' completion log differs from training artifact')
    if (completed['probability_sha256'] != item['probability_sha256'] or
            completed['model_sha256'] != item['model_sha256']):
        raise RuntimeError(name + ' completion hashes differ from artifacts')
    if item['task_updates'] != item['epochs_run'] * math.ceil(train_count / batch_size):
        raise RuntimeError(name + ' task update count is incomplete')
    snapshots = _load(path / 'snapshots.json')
    if len(snapshots) != len(epochs):
        raise RuntimeError(name + ' snapshot count differs from epoch count')
    for epoch, (row, snapshot) in enumerate(zip(epochs, snapshots), 1):
        if (row['epoch'] != epoch or snapshot['epoch'] != epoch or
                snapshot['file'] != f'epoch{epoch:02d}.pt' or
                digest(path / snapshot['file']) != snapshot['sha256']):
            raise RuntimeError(name + ' epoch/snapshot provenance mismatch')
        saved = torch.load(path / snapshot['file'], map_location='cpu', weights_only=True)
        if (saved.shape != probabilities.shape or
                torch.bincount(saved.argmax(1), minlength=5).tolist() != row['hard_counts'] or
                not math.isclose(float(saved[:, CAPPED].sum()), row['soft_count_capped'],
                                 rel_tol=0, abs_tol=1e-6) or
                row['task_updates'] != epoch * math.ceil(train_count / batch_size)):
            raise RuntimeError(name + ' epoch log differs from snapshot')
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in
               ('training_loss', 'stop_loss', 'first_task_gradient_norm',
                'first_task_displacement_norm')):
            raise RuntimeError(name + ' task-loss or gradient diagnostics are invalid')
        if epoch == item['best_epoch'] and not torch.equal(saved, probabilities):
            raise RuntimeError(name + ' restored checkpoint differs from reported output')
    if epochs[0]['epoch_order_sha256'] != item['first_order_sha256'] or epochs[0]['epoch_first_batch_sha256'] != item['first_batch_sha256']:
        raise RuntimeError(name + ' first-batch provenance mismatch')
    if item['best_stop_loss'] != min(row['stop_loss'] for row in epochs):
        raise RuntimeError(name + ' best checkpoint was not chosen by stop loss')
    return probabilities, epochs


def _check_dose(name, row, expected, cap):
    norms = row['tensor_displacement_norms']
    if len(norms) != len(expected) or any(not math.isfinite(n) or n < 0 for n in norms):
        raise RuntimeError(name + ' tensor-dose shape or value differs')
    if name.endswith('_sham') and any(abs(a - b) > max(2e-6, 2e-3 * b)
                                       for a, b in zip(norms, expected)):
        raise RuntimeError('sham tensor dose differs from target')
    if name.endswith('_tralo'):
        if norms != expected:
            raise RuntimeError('target tensor-dose provenance differs')
        step = row['target_step']
        if step['applied'] and step['hard_after'] > cap:
            raise RuntimeError('targeted step missed the cap')
        if (step['hard_before'] <= cap) == step['applied']:
            raise RuntimeError('targeted step trigger is inconsistent with hard count')
        if (not math.isfinite(step['soft_before']) or
                not math.isfinite(step['soft_after']) or
                not math.isfinite(step['gradient_norm']) or
                step['gradient_norm'] < 0):
            raise RuntimeError('targeted step diagnostics are invalid')
        if step['applied'] and step['gradient_norm'] <= 0:
            raise RuntimeError('applied target step has no gradient')


def _check_label_firewall(manifest):
    if any('label' in row for row in manifest['rows'] if row['split'] != 'train'):
        raise RuntimeError('development or sealed-test label entered the run manifest')


def _check_vit_replay(backbone, started, rows):
    replay = [row for row in rows if row['event'] == 'vit_attention_replay']
    if backbone != 'vit_b_16':
        if replay:
            raise RuntimeError('unexpected ViT attention replay in a non-ViT run')
        return
    if (started.get('mha_fastpath_enabled') is not False or len(replay) != 1):
        raise RuntimeError('ViT attention replay receipt missing or fastpath enabled')
    row = replay[0]
    if (row.get('passed') is not True or row.get('mha_fastpath_enabled') is not False or
            type(row.get('images_count')) is not int or row['images_count'] <= 0 or
            any(type(row.get(key)) not in (int, float) or
                not math.isfinite(row[key]) or row[key] < 0
                for key in ('max_absolute_difference', 'max_tolerance_ratio')) or
            row['max_tolerance_ratio'] > 1):
        raise RuntimeError('ViT attention replay failed or diagnostics differ')


def gate(run_root):
    """Verify a completed run without reading its development labels or scoring."""
    import torch
    root = Path(run_root)
    config = _load(root / 'config.json')
    validate(config)
    seed, backbone = config['seed'], config['backbone']
    expected_input = (Path(__file__).resolve().parents[1] / 'experiments' / 'configs' /
                      f'knee_persistent_{backbone}_{seed}.json')
    if _load(expected_input) != config:
        raise RuntimeError('run config values differ from the immutable input')
    rows = _events(root / 'events.jsonl')
    if rows[0]['event'] != 'started' or rows[-1]['event'] != 'completed':
        raise RuntimeError('run has not completed successfully')
    started, completed = rows[0], rows[-1]
    _check_vit_replay(backbone, started, rows)
    if (started['config_sha256'] != digest(expected_input) or
            started['source_sha256'] != source() or
            started['manifest_sha256'] != digest(root / 'manifest.json') or
            started['architecture'] != backbone or
            started['weight_sha256'] != WEIGHT_SHA256[backbone] or
            digest(started['weight_path']) != started['weight_sha256'] or
            started['precision'] != 'fp32'):
        raise RuntimeError('run source/config/data identity differs from release')
    manifest = _load(root / 'manifest.json')
    _check_label_firewall(manifest)
    if (manifest['counts'] != {'train': 5778, 'val': 826, 'test': 1656} or
            manifest['cross_split_subject_overlap'] != 0 or
            manifest['cross_split_pixel_overlap'] != 0):
        raise RuntimeError('run manifest violates the split or label firewall')
    if Counter(r['split'] for r in manifest['rows']) != manifest['counts']:
        raise RuntimeError('manifest row count differs from declared split counts')
    if len({r['sample_id'] for r in manifest['rows']}) != len(manifest['rows']):
        raise RuntimeError('manifest sample IDs are not unique')
    subjects = {split: {r['subject'] for r in manifest['rows'] if r['split'] == split}
                for split in manifest['counts']}
    pixels = {split: {r['pixel_sha256'] for r in manifest['rows'] if r['split'] == split}
              for split in manifest['counts']}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if subjects[left] & subjects[right] or pixels[left] & pixels[right]:
            raise RuntimeError('manifest split identities overlap')
    if Counter(r['label'] for r in manifest['rows'] if r['split'] == 'train')[CAPPED] != 757:
        raise RuntimeError('training-only grade-3 quota reference changed')
    train_rows, stop_rows = carve(manifest['rows'])
    if len(train_rows) != started['train_carve'] or len(stop_rows) != started['stop_carve']:
        raise RuntimeError('subject-stable stop carve differs from runner')
    if hashlib.sha256('\n'.join(r['sample_id'] for r in manifest['rows']
                                if r['split'] == 'val').encode()).hexdigest() != started['dev_ids_sha256']:
        raise RuntimeError('development order differs from the runner')
    if not isinstance(completed['seconds'], (int, float)) or not math.isfinite(completed['seconds']):
        raise RuntimeError('invalid run time')
    if digest(root / 'summary.json') != completed['summary_sha256']:
        raise RuntimeError('summary artifact hash mismatch')
    summary = _load(root / 'summary.json')
    if (summary['seed'] != seed or summary['backbone'] != backbone or
            summary['caps'] != config['caps'] or
            summary['source_sha256'] != started['source_sha256'] or
            summary['config_sha256'] != started['config_sha256'] or
            summary['manifest_sha256'] != started['manifest_sha256'] or
            summary['weights_sha256'] != started['weight_sha256'] or
            summary['initial_sha256'] != started['initial_sha256']):
        raise RuntimeError('summary provenance mismatch')
    pilot = seed in (6700, VIT_RECOVERY_PILOT)
    expected_arms = {'pto'} | ({'null'} if pilot else set()) | {f'cap{cap}' for cap in config['caps']}
    if set(summary['arms']) != expected_arms:
        raise RuntimeError('arm set differs from fixed protocol')
    pto = summary['arms']['pto']
    pto_probs, pto_epochs = _arm(root, 'pto', pto, started['train_carve'], config['batch_size'])
    horizon = pto['epochs_run']
    if not 1 <= horizon <= config['max_epochs']:
        raise RuntimeError('invalid PTO stopping horizon')
    if pilot:
        null = summary['arms']['null']
        null_probs, null_epochs = _arm(root, 'null', null, started['train_carve'], config['batch_size'])
        if not torch.equal(pto_probs, null_probs) or pto['model_sha256'] != null['model_sha256']:
            raise RuntimeError('null does not reproduce PTO')
        if [(r['epoch_order_sha256'], r['epoch_first_batch_sha256']) for r in null_epochs] != [
                (r['epoch_order_sha256'], r['epoch_first_batch_sha256']) for r in pto_epochs]:
            raise RuntimeError('null task stream differs from PTO')
        for row in null_epochs:
            if (not row['null_hook'] or row['pre_hook_hard_counts'] != row['hard_counts'] or
                    not math.isclose(row['pre_hook_soft_count_capped'], row['soft_count_capped'],
                                     rel_tol=0, abs_tol=1e-6)):
                raise RuntimeError('null hook changed the predictions')
    cost_seconds = pto['seconds'] + (summary['arms']['null']['seconds'] if pilot else 0)
    for cap in config['caps']:
        item = summary['arms'][f'cap{cap}']
        if len(item['target_tensor_doses']) != horizon:
            raise RuntimeError('target dose count differs from the task horizon')
        target, sham, pao = item['tralo'], item['sham'], item['pao']
        for name, arm in ((f'cap{cap}_tralo', target), (f'cap{cap}_sham', sham)):
            _, epochs = _arm(root, name, arm, started['train_carve'], config['batch_size'])
            cost_seconds += arm['seconds']
            if (arm['epochs_run'] != horizon or arm['task_updates'] != pto['task_updates'] or
                    arm['first_order_sha256'] != pto['first_order_sha256'] or
                    arm['first_batch_sha256'] != pto['first_batch_sha256']):
                raise RuntimeError(name + ' task dose or batch identity differs')
            for baseline, row in zip(pto_epochs, epochs):
                if (row['epoch_order_sha256'] != baseline['epoch_order_sha256'] or
                        row['epoch_first_batch_sha256'] != baseline['epoch_first_batch_sha256']):
                    raise RuntimeError(name + ' task stream differs from PTO')
            for epoch, row in enumerate(epochs):
                expected = item['target_tensor_doses'][epoch]
                _check_dose(name, row, expected, cap)
        retrains = item['pao_retrains']
        if (not retrains or len(retrains) > config['max_retrains'] or
                retrains[0]['file'] != pto['file']):
            raise RuntimeError('PAO does not start at the shared PTO')
        previous_C = torch.ones(5)
        previous_count = int((pto_probs.argmax(1) == CAPPED).sum())
        for number, retrain in enumerate(retrains, 1):
            if retrain['retrain'] != number:
                raise RuntimeError('PAO retrain numbering changed')
            if number == 1:
                if retrain['C'] != previous_C.tolist():
                    raise RuntimeError('PAO initial cost differs from PTO')
                if retrain['arm'] != pto:
                    raise RuntimeError('PAO initial artifact differs from PTO')
                prob = pto_probs
            else:
                f, derivative = f_and_derivative(cap, previous_count, config['b'])
                if f < 1e-5:
                    raise RuntimeError('PAO retrained after its declared stopping point')
                previous_C += config['mu'] * derivative
                previous_C[CAPPED] = 1
                if any(abs(a - b) > 1e-6 for a, b in zip(retrain['C'], previous_C.tolist())):
                    raise RuntimeError('PAO dual recurrence differs from the author baseline')
                arm_name = f'cap{cap}_pao{number}'
                arm = retrain['arm']
                prob, _ = _arm(root, arm_name, arm, started['train_carve'], config['batch_size'])
                cost_seconds += arm['seconds']
            if (retrain['file'] != retrain['arm']['file'] or
                    retrain['probability_sha256'] != retrain['arm']['probability_sha256']):
                raise RuntimeError('PAO retrain artifact pointer differs from saved arm')
            if int((prob.argmax(1) == 3).sum()) != retrain['hard_count']:
                raise RuntimeError('PAO logged count differs from output')
            previous_count = retrain['hard_count']
        if item['pao_converged'] != (retrains[-1]['hard_count'] <= cap):
            raise RuntimeError('PAO convergence record differs from count')
        if not item['pao_converged'] and len(retrains) < config['max_retrains']:
            raise RuntimeError('PAO stopped early while the hard cap remained violated')
        if pao != retrains[-1]['arm']:
            raise RuntimeError('PAO final output differs from last retrain')
    if cost_seconds > completed['seconds'] + 1:
        raise RuntimeError('arm GPU time exceeds whole-run wall time')
    cap_events = [row for row in rows if row['event'] == 'cap_completed']
    if [row['cap'] for row in cap_events] != config['caps']:
        raise RuntimeError('completed cap events differ from fixed training caps')
    for row in cap_events:
        item = summary['arms'][f"cap{row['cap']}"]
        if (row['pao_retrains'] != len(item['pao_retrains']) or
                row['pao_converged'] != item['pao_converged'] or
                row['tralo_sha256'] != item['tralo']['probability_sha256'] or
                row['sham_sha256'] != item['sham']['probability_sha256']):
            raise RuntimeError('cap completion receipt differs from saved arms')
    return dict(seed=seed, backbone=backbone, caps=config['caps'],
                seconds=completed['seconds'], source_sha256=started['source_sha256'],
                manifest_sha256=started['manifest_sha256'],
                summary_sha256=completed['summary_sha256'])


def _quality(probabilities, labels, ids, cap):
    reports = evaluate_global(probabilities.tolist(), labels, [None, None, None, cap, None], ids)
    result = {}
    for policy, detail in reports.items():
        metrics = detail['metrics']
        rows = metrics['per_class']
        total = sum(r['support'] for r in rows)
        result[policy] = dict(cc_f1=metrics['cc_f1'], accuracy=metrics['accuracy'],
                              macro_f1=metrics['macro_f1'],
                              weighted_f1=sum(r['f1'] * r['support'] for r in rows) / total,
                              precision_3=rows[3]['precision'], recall_3=rows[3]['recall'],
                              per_class=rows, confusion=metrics['confusion'],
                              counts=detail['counts'], feasible=detail['feasible'],
                              changed=detail['changed'])
    return result


def score(full_root, data_root, output_json):
    import torch
    from scipy.stats import t as student_t
    from tralo.knee_data import audit

    root = Path(full_root)
    output = Path(output_json)
    if output.exists():
        raise FileExistsError(output)
    expected = [(backbone, seed, root / backbone / f'seed{seed}')
                for backbone in BACKBONES for seed in SEEDS_STUDY]
    if any(not path.is_dir() for _, _, path in expected):
        raise RuntimeError('complete 36-run block is required before scoring')
    unexpected = [p for p in root.glob('*/seed*') if p.is_dir() and
                  p not in {path for _, _, path in expected}]
    if unexpected:
        raise RuntimeError('unexpected run directory in the full block')
    gates = [gate(path) for _, _, path in expected]
    if len({g['manifest_sha256'] for g in gates}) != 1:
        raise RuntimeError('run block differs in data manifest')
    # Development labels are first read only after every run has passed gate().
    manifest = audit(data_root)
    public = dict(manifest, rows=[{k: v for k, v in row.items()
                                   if row['split'] == 'train' or k != 'label'}
                                  for row in manifest['rows']])
    if public != _load(expected[0][2] / 'manifest.json'):
        raise RuntimeError('scoring data differ from the gated training cohort')
    val = [r for r in manifest['rows'] if r['split'] == 'val']
    labels, ids = [r['label'] for r in val], [r['sample_id'] for r in val]
    scores = {}
    deployment_curves = {}
    for backbone, seed, path in expected:
        summary = _load(path / 'summary.json')
        rows = {}
        for cap in CAPS_STUDY:
            block = summary['arms'][f'cap{cap}']
            arm_map = dict(pto=summary['arms']['pto'], pao=block['pao'],
                           tralo=block['tralo'], sham=block['sham'])
            rows[str(cap)] = {}
            for name, arm in arm_map.items():
                probabilities = _probabilities(path / arm['file'], arm['probability_sha256'])
                rows[str(cap)][name] = _quality(probabilities, labels, ids, cap)
                deployment_curves.setdefault(backbone, {}).setdefault(str(seed), {}).setdefault(str(cap), {})[name] = {
                    str(deploy_cap): _quality(probabilities, labels, ids, deploy_cap)['capped_first']
                    for deploy_cap in CAPS_CURVE}
        scores.setdefault(backbone, {})[str(seed)] = rows
    grouped = {}
    contrasts = []
    for backbone in BACKBONES:
        grouped[backbone] = {}
        for cap in CAPS_STUDY:
            arm_values = {}
            for name in ('pto', 'pao', 'tralo', 'sham'):
                arm_values[name] = {}
                for metric in ('cc_f1', 'accuracy', 'macro_f1', 'weighted_f1',
                               'precision_3', 'recall_3'):
                    values = [scores[backbone][str(seed)][str(cap)][name]['capped_first'][metric]
                              for seed in SEEDS_STUDY]
                    arm_values[name][metric] = dict(mean=statistics.mean(values),
                                                    seed_sd=statistics.stdev(values), values=values)
            grouped[backbone][str(cap)] = arm_values
            for left, right in (('tralo', 'sham'), ('tralo', 'pto'), ('tralo', 'pao')):
                deltas = [scores[backbone][str(seed)][str(cap)][left]['capped_first']['cc_f1'] -
                          scores[backbone][str(seed)][str(cap)][right]['capped_first']['cc_f1']
                          for seed in SEEDS_STUDY]
                mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
                sem = sd / math.sqrt(len(deltas))
                radius = student_t.ppf(0.975, len(deltas) - 1) * sem if sem > 0 else None
                p = float(2 * student_t.sf(abs(mean / sem), len(deltas) - 1)) if sem > 0 else None
                contrasts.append(dict(backbone=backbone, cap=cap, contrast=f'{left}-{right}',
                                      n=len(deltas), mean=mean, seed_sd=sd,
                                      ci95=([mean - radius, mean + radius] if radius is not None else None),
                                      deltas=deltas, p_unadjusted=p))
    indexed = sorted(range(len(contrasts)), key=lambda i: (1.0 if contrasts[i]['p_unadjusted'] is None
                                                           else contrasts[i]['p_unadjusted']))
    adjusted = 0.0
    for rank, i in enumerate(indexed):
        raw = 1.0 if contrasts[i]['p_unadjusted'] is None else contrasts[i]['p_unadjusted']
        adjusted = max(adjusted, min(1.0, raw * (len(contrasts) - rank)))
        contrasts[i]['p_holm'] = adjusted
    result = dict(scope='repeatedly_viewed_knee_development_only', n_seeds=12,
                  scorer_sha256=digest(__file__),
                  train_caps=list(CAPS_STUDY), backbones=list(BACKBONES),
                  run_gates=gates, per_seed=scores, grouped=grouped,
                  deployment_curve_caps=list(CAPS_CURVE), deployment_curves=deployment_curves,
                  paired_contrasts=contrasts,
                  note='Six-cap curves change only the deployment allocator on fixed saved outputs; models were trained only at caps 54 and 86. Curves are secondary and must not select a training setting.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gate', metavar='RUN_ROOT')
    group.add_argument('--score', nargs=3, metavar=('FULL_ROOT', 'DATA_ROOT', 'OUTPUT_JSON'))
    args = parser.parse_args()
    if args.gate:
        print(json.dumps(gate(args.gate), indent=2))
    else:
        score(*args.score)

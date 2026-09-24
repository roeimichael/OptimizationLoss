"""One-seed, three-arm end-to-end Chen knee mechanism experiment.

Usage: python -m tralo.knee_end_to_end DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY
The validation image loader returns images only. Labels enter only the offline
scorer after all training and probability snapshots are complete.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .global_comparison import _state_hash, audited_arm_log
from .knee_experiment import cuda_setup, digest, save, source
from .knee_data import audit


def validate(config):
    expected = {'seed', 'epochs', 'warmup_epochs', 'batch_size', 'task_lr',
                'constraint_lr', 'caps', 'lambda_initial', 'lambda_step',
                'rho_initial', 'rho_target', 'development_batch_size'}
    if set(config) != expected:
        raise ValueError('end-to-end config keys differ from the declared experiment')
    for key in ('seed', 'epochs', 'warmup_epochs', 'batch_size', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['seed'] not in (1301, 1302, 1303, 1304):
        raise ValueError('seed is outside the preregistered four-seed set')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('warmup must precede the constraint phase')
    if config['caps'] != [None, None, None, 76, None]:
        raise ValueError('this preregistered study has only the grade-3 cap of 76')
    for key in ('task_lr', 'constraint_lr', 'lambda_initial', 'lambda_step',
                'rho_initial', 'rho_target'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)
    if config['task_lr'] == 0 or config['constraint_lr'] == 0:
        raise ValueError('optimizer learning rates must be positive')
    if config['rho_target'] < config['rho_initial']:
        raise ValueError('rho must be nondecreasing')


def image_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


def development_images(data_root, rows, transform, batch_size):
    """An image-only replayable cohort, with no label output or label lookup."""
    from PIL import Image
    cohort = [dict(path=r['path'], sha256=r['sha256']) for r in rows if r['split'] == 'val']
    if not cohort:
        raise ValueError('empty development cohort')
    chunks = []
    for start in range(0, len(cohort), batch_size):
        tensors = []
        for row in cohort[start:start + batch_size]:
            path = Path(data_root) / row['path']
            if digest(path) != row['sha256']:
                raise RuntimeError('development image changed after audit')
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
        import torch
        chunks.append(torch.stack(tensors))
    return chunks


def infer(model, chunks):
    import torch
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            values = torch.cat([model(x.to(device)).softmax(1).cpu() for x in chunks])
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError('nonfinite development probabilities')
        return values
    finally:
        model.train(was_training)


def train_one(model, train_images, val_chunks, config, arm, emit, snapshot):
    """Train without any validation-label argument; save every intervention boundary."""
    import torch
    from .global_constraint import advance_controller
    from .streamed_constraint import streamed_step
    if arm not in ('clipper', 'tralo_null', 'tralo'):
        raise ValueError('unknown arm')
    seed = config['seed']
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 1)
    task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
    constraint = torch.optim.Adam(model.parameters(), lr=config['constraint_lr']) if arm == 'tralo' else None
    multipliers = [float(config['lambda_initial'])] * 5
    rho = float(config['rho_initial'])
    rho_step = (config['rho_target'] - rho) / (config['epochs'] - config['warmup_epochs'])
    frozen = False
    task_updates = constraint_updates = constraint_checks = 0
    planned_task = config['epochs'] * math.ceil(len(train_images) / config['batch_size'])
    batch_hash = hashlib.sha256()
    warmup = None
    previous_after = None
    emit(dict(event='started', arm=arm, seed=seed, planned_task_updates=planned_task,
              constraint_opportunities=config['epochs'] - config['warmup_epochs'] if arm == 'tralo' else 0))
    for epoch in range(config['epochs']):
        if epoch == config['warmup_epochs'] and arm != 'clipper':
            task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
            emit(dict(event='task_optimizer_reset', epoch=epoch + 1))
        order = torch.randperm(len(train_images), generator=generator)
        batch_hash.update(order.numpy().tobytes())
        model.train()
        ce_total = 0.
        for start in range(0, len(order), config['batch_size']):
            examples = [train_images[int(i)] for i in order[start:start + config['batch_size']]]
            images = torch.stack([row[0] for row in examples]).to(device)
            labels = torch.tensor([row[1] for row in examples], dtype=torch.long, device=device)
            task.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite supervised loss')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters() if p.requires_grad):
                raise RuntimeError('invalid supervised gradient')
            task.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError('nonfinite supervised parameters')
            task_updates += 1
            ce_total += float(loss.detach()) * len(examples)
        if epoch + 1 == config['warmup_epochs']:
            warmup = _state_hash(model)
        row = dict(event='epoch', epoch=epoch + 1, training_ce=ce_total / len(train_images),
                   task_updates_cumulative=task_updates, constraint_updates_cumulative=constraint_updates)
        if epoch >= config['warmup_epochs']:
            before = infer(model, val_chunks)
            if previous_after is not None:
                snapshot(epoch + 1, 'after_next_supervised_epoch', before)
            snapshot(epoch + 1, 'before_constraint', before)
            hard = torch.bincount(before.argmax(1), minlength=5).tolist()
            row.update(hard_counts_before=hard, soft_counts_before=before.sum(0).tolist(),
                       multipliers=list(multipliers), rho=rho)
            applied = False
            if arm == 'tralo':
                penalty_multipliers = torch.tensor(multipliers, device=device, dtype=before.dtype)
                # streamed_step will recalculate the probabilities at identical
                # weights, then backpropagate their analytic full-cohort derivative.
                constraint_checks += 1
                result = streamed_step(model, val_chunks, config['caps'],
                                       penalty_multipliers, rho, constraint)
                if not torch.allclose(result['probabilities'].cpu(), before, atol=1e-7, rtol=1e-6):
                    raise RuntimeError('before snapshot differs from streamed count pass')
                applied = result['applied']
                constraint_updates += int(applied)
                multipliers, rho, frozen = advance_controller(
                    hard, config['caps'], multipliers, rho, rho_step,
                    config['lambda_step'], frozen)
                row.update(multipliers_after=list(multipliers), rho_after=rho,
                           controller_frozen=frozen, constraint_checks_cumulative=constraint_checks)
            after = infer(model, val_chunks)
            snapshot(epoch + 1, 'after_constraint', after)
            previous_after = after
            row.update(constraint_attempted=applied, constraint_applied=applied,
                       hard_counts_after=torch.bincount(after.argmax(1), minlength=5).tolist(),
                       soft_counts_after=after.sum(0).tolist(),
                       constraint_updates_cumulative=constraint_updates)
        emit(row)
    if task_updates != planned_task:
        raise RuntimeError('supervised update dose mismatch')
    return dict(warmup_sha256=warmup, batch_sha256=batch_hash.hexdigest(),
                task_updates_planned=planned_task, task_updates_attempted=task_updates,
                task_updates_applied=task_updates, task_updates_skipped=0,
                constraint_opportunities=config['epochs'] - config['warmup_epochs'] if arm == 'tralo' else 0,
                constraint_checks=constraint_checks,
                constraint_updates_inactive=constraint_checks - constraint_updates,
                constraint_updates_attempted=constraint_updates,
                constraint_updates_applied=constraint_updates,
                constraint_updates_skipped=0, final_probabilities=infer(model, val_chunks))


def score_snapshots(directory, snapshots, labels, ids, caps):
    """Offline audit only: labels never flow back to the training function."""
    import torch
    from .global_report import evaluate_global
    result = []
    predictions = {}
    for item in snapshots:
        path = directory / item['file']
        if digest(path) != item['sha256']:
            raise RuntimeError('snapshot hash mismatch')
        probabilities = torch.load(path, map_location='cpu', weights_only=True).tolist()
        report = evaluate_global(probabilities, labels, caps, ids)
        key = (item['epoch'], item['phase'])
        predictions[key] = report
        result.append(dict(epoch=item['epoch'], phase=item['phase'],
                           scores={name:dict(metrics=row['metrics'], counts=row['counts'],
                                             changed=row['changed'], feasible=row['feasible'])
                                   for name, row in report.items()}))
    transitions = []
    for epoch in sorted({item['epoch'] for item in snapshots}):
        before = predictions[(epoch, 'before_constraint')]
        after = predictions[(epoch, 'after_constraint')]
        policies = {}
        for policy in ('raw', 'upper_bound_correction', 'capped_first'):
            prior = before[policy]['predictions']
            later = after[policy]['predictions']
            entering = [i for i, (a, b) in enumerate(zip(prior, later)) if a != 3 and b == 3]
            leaving = [i for i, (a, b) in enumerate(zip(prior, later)) if a == 3 and b != 3]
            policies[policy] = dict(entries=len(entering), entries_correct=sum(labels[i] == 3 for i in entering),
                                    exits=len(leaving), exits_correct=sum(labels[i] == 3 for i in leaving),
                                    entering_ids=[ids[i] for i in entering], leaving_ids=[ids[i] for i in leaving])
        transitions.append(dict(epoch=epoch, policies=policies))
    return dict(snapshots=result, transitions=transitions)


def run(data_root, config_path, output):
    import torch
    from torchvision import models
    from .global_report import evaluate_global
    from .supervised_adaptation import TrainingImages
    config = json.loads(Path(config_path).read_text())
    validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / 'events.jsonl') as log:
        cuda_setup()
        manifest = audit(data_root)
        if manifest['counts']['train'] != 5778 or manifest['counts']['val'] != 826:
            raise RuntimeError('unexpected Chen split sizes')
        save(output / 'manifest.json', manifest)
        save(output / 'config.json', config)
        log.emit('started', source_sha256=source(), config_sha256=digest(config_path),
                 data_counts=manifest['counts'], device=str(torch.cuda.get_device_name()), precision='fp32')
        transform = image_transform()
        train_images = TrainingImages(data_root, manifest['rows'], transform)
        val_chunks = development_images(data_root, manifest['rows'], transform,
                                        config['development_batch_size'])
        val_rows = [r for r in manifest['rows'] if r['split'] == 'val']
        val_ids = [r['sample_id'] for r in val_rows]
        torch.manual_seed(config['seed'])
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
        base.fc = torch.nn.Linear(base.fc.in_features, 5)
        initial_sha = _state_hash(base)
        log.emit('model_initialized', initial_sha256=initial_sha,
                 imagenet_weights_sha256=digest(weight_path), architecture='ResNet18',
                 transform='RGB resize224 ImageNet normalization no augmentation')
        identities = []
        summary = []
        started = time.monotonic()
        for arm in ('clipper', 'tralo_null', 'tralo'):
            directory = output / arm
            directory.mkdir()
            with audited_arm_log(directory / 'events.jsonl') as arm_log:
                model = copy.deepcopy(base).cuda()
                if _state_hash(model) != initial_sha:
                    raise RuntimeError('arm initialization differs')
                snapshots = []
                def snapshot(epoch, phase, values):
                    path = directory / f'epoch{epoch:02d}_{phase}.pt'
                    torch.save(values.cpu(), path)
                    snapshots.append(dict(epoch=epoch, phase=phase, file=path.name,
                                          sha256=digest(path)))
                result = train_one(model, train_images, val_chunks, config, arm,
                    lambda row: arm_log.emit(row['event'], **{k:v for k,v in row.items() if k!='event'}),
                    snapshot)
                torch.save(model.state_dict(), directory / 'model.pt')
                torch.save(result.pop('final_probabilities'), directory / 'final_probabilities.pt')
                save(directory / 'snapshots.json', snapshots)
                identities.append((result['warmup_sha256'], result['batch_sha256']))
                arm_log.emit('training_completed', **result,
                             artifacts={p.name:digest(p) for p in directory.iterdir()
                                        if p.name != 'events.jsonl'})
                summary.append(dict(arm=arm, **result))
                del model
                torch.cuda.empty_cache()
        if len(set(identities)) != 1:
            raise RuntimeError('warmup or batch identity differs across matched arms')
        # The separate scorer receives labels only after all model updates finish.
        labels = [r['label'] for r in val_rows]
        for row in summary:
            directory = output / row['arm']
            probs = torch.load(directory / 'final_probabilities.pt', map_location='cpu', weights_only=True)
            report = evaluate_global(probs.tolist(), labels, config['caps'], val_ids)
            save(directory / 'report.json', report)
            snapshots = json.loads((directory / 'snapshots.json').read_text())
            save(directory / 'snapshot_report.json',
                 score_snapshots(directory, snapshots, labels, val_ids, config['caps']))
            row['scores'] = {policy:{key:details['metrics'][key]
                                    for key in ('accuracy', 'macro_f1', 'cc_f1')}
                             for policy,details in report.items()}
        save(output / 'summary.json', summary)
        artifacts = {p.relative_to(output).as_posix():digest(p)
                     for p in output.rglob('*') if p.is_file() and p != output / 'events.jsonl'}
        log.emit('completed', seconds=time.monotonic() - started,
                 matched_warmup_sha256=identities[0][0], matched_batch_sha256=identities[0][1],
                 summary_sha256=digest(output / 'summary.json'), artifacts=artifacts)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])

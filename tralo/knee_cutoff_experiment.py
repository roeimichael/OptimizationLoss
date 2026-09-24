"""Five matched end-to-end knee arms for training-only cutoff ranking.

Usage: python -m tralo.knee_cutoff_experiment DATA_ROOT CONFIG_JSON OUTPUT_DIR
Development labels enter only score_snapshots, after every fit has ended.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .global_comparison import _state_hash, audited_arm_log
from .knee_data import audit
from .knee_end_to_end import image_transform, development_images, infer, score_snapshots
from .knee_experiment import cuda_setup, digest, save, source
from .supervised_adaptation import TrainingImages

ARMS = ('clipper', 'tralo_null', 'count_only', 'rank_only', 'rank_count')


def validate(config):
    required = {'seed', 'epochs', 'warmup_epochs', 'batch_size',
                'development_batch_size', 'auxiliary_batch_size', 'task_lr',
                'auxiliary_lr', 'caps', 'train_capacity', 'lambda_initial',
                'lambda_step', 'rho_initial', 'rho_target'}
    if set(config) != required:
        raise ValueError('cutoff config keys differ from fixed protocol')
    if config['seed'] not in (1401, 1402, 1403, 1404):
        raise ValueError('seed outside fixed four-seed protocol')
    if (config['epochs'] != 10 or config['warmup_epochs'] != 5 or
            config['caps'] != [None, None, None, 76, None] or
            config['train_capacity'] != 532):
        raise ValueError('fixed epochs/capacity/caps differ from protocol')
    for name in ('batch_size', 'development_batch_size', 'auxiliary_batch_size'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be positive integer')
    if config['batch_size'] != 32 or config['development_batch_size'] != 16:
        raise ValueError('fixed batch sizes differ from protocol')
    for name in ('task_lr', 'auxiliary_lr', 'lambda_initial', 'lambda_step',
                 'rho_initial', 'rho_target'):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid ' + name)
    if config['task_lr'] != 0.0001 or config['auxiliary_lr'] != 0.00003:
        raise ValueError('fixed optimizer rates differ from protocol')
    if config['rho_target'] < config['rho_initial']:
        raise ValueError('rho must be nondecreasing')


class TrainingChunks:
    """Re-read the immutable training images on each fixed-weight pass.

    No validation labels or images can enter this iterator. Chunk order is the
    manifest order, matching the separately supplied training IDs and labels.
    """

    def __init__(self, train_images, batch_size):
        self.images = train_images
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        import torch
        for start in range(0, len(self.images), self.batch_size):
            yield torch.stack([self.images[i][0]
                               for i in range(start, min(start + self.batch_size,
                                                         len(self.images)))])


def train_one(model, train_images, train_ids, train_labels, val_chunks,
              config, arm, emit, snapshot):
    """One arm; validation labels are absent from its API by construction."""
    import torch
    from .global_constraint import advance_controller
    from .streamed_constraint import streamed_step
    from .streamed_rank import streamed_rank_count_step

    if arm not in ARMS or len(train_ids) != len(train_images) or len(train_labels) != len(train_images):
        raise ValueError('invalid arm or training identity alignment')
    use_rank = arm in ('rank_only', 'rank_count')
    use_count = arm in ('count_only', 'rank_count')
    auxiliary_arm = use_rank or use_count
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(config['seed'] + 1)
    task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
    auxiliary = (torch.optim.Adam(model.parameters(), lr=config['auxiliary_lr'])
                 if auxiliary_arm else None)
    train_chunks = TrainingChunks(train_images, config['auxiliary_batch_size']) if use_rank else None
    multipliers = [float(config['lambda_initial'])] * 5
    rho = float(config['rho_initial'])
    opportunities = config['epochs'] - config['warmup_epochs']
    rho_step = (config['rho_target'] - rho) / opportunities
    frozen = False
    task_updates = auxiliary_checks = auxiliary_updates = 0
    planned_task = config['epochs'] * math.ceil(len(train_images) / config['batch_size'])
    batch_hash = hashlib.sha256()
    warmup = None
    previous_after = None
    emit(dict(event='started', arm=arm, seed=config['seed'],
              planned_task_updates=planned_task,
              auxiliary_opportunities=opportunities if auxiliary_arm else 0))
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
                raise RuntimeError('nonfinite supervised parameter')
            task_updates += 1
            ce_total += float(loss.detach()) * len(examples)
        if epoch + 1 == config['warmup_epochs']:
            warmup = _state_hash(model)
        row = dict(event='epoch', epoch=epoch + 1,
                   training_ce=ce_total / len(train_images),
                   task_updates_cumulative=task_updates,
                   auxiliary_updates_cumulative=auxiliary_updates)
        if epoch >= config['warmup_epochs']:
            before = infer(model, val_chunks)
            if previous_after is not None:
                snapshot(epoch + 1, 'after_next_supervised_epoch', before)
            snapshot(epoch + 1, 'before_auxiliary', before)
            hard = torch.bincount(before.argmax(1), minlength=5).tolist()
            row.update(hard_counts_before=hard, soft_counts_before=before.sum(0).tolist())
            if auxiliary_arm:
                auxiliary_checks += 1
                multipliers_tensor = torch.tensor(multipliers, device=device, dtype=before.dtype)
                if use_rank:
                    result = streamed_rank_count_step(
                        model, train_chunks, train_labels, train_ids,
                        config['train_capacity'], val_chunks, config['caps'],
                        multipliers_tensor, rho, auxiliary,
                        use_rank=True, use_count=use_count)
                    rank = result['ranking']
                    # Full ID lists are large; the source cohort and selected
                    # correct count suffice to recompute the training quota.
                    row.update(rank_loss=result['rank_loss'],
                               count_loss=result['count_loss'],
                               active_pairs=rank['active_pairs'],
                               training_selected_true=rank['selected_true_count'],
                               training_missed_true=len(rank['missed_true_ids']),
                               training_wrong_occupants=len(rank['wrong_occupant_ids']),
                               rank_logit_gradient_norm=result['rank_logit_gradient_norm'],
                               count_logit_gradient_norm=result['count_logit_gradient_norm'])
                    replayed = result['development_probabilities'].cpu()
                else:
                    result = streamed_step(model, val_chunks, config['caps'],
                                           multipliers_tensor, rho, auxiliary)
                    replayed = result['probabilities'].cpu()
                if not torch.allclose(replayed, before, atol=1e-7, rtol=1e-6):
                    raise RuntimeError('before snapshot differs from auxiliary replay')
                auxiliary_updates += int(result['applied'])
                row['auxiliary_applied'] = result['applied']
                if use_count:
                    multipliers, rho, frozen = advance_controller(
                        hard, config['caps'], multipliers, rho, rho_step,
                        config['lambda_step'], frozen)
                    row.update(multipliers_after=list(multipliers), rho_after=rho,
                               controller_frozen=frozen)
            else:
                row['auxiliary_applied'] = False
            after = infer(model, val_chunks)
            snapshot(epoch + 1, 'after_auxiliary', after)
            previous_after = after
            row.update(hard_counts_after=torch.bincount(after.argmax(1), minlength=5).tolist(),
                       soft_counts_after=after.sum(0).tolist(),
                       auxiliary_checks_cumulative=auxiliary_checks,
                       auxiliary_updates_cumulative=auxiliary_updates)
        emit(row)
    if task_updates != planned_task:
        raise RuntimeError('supervised update dose mismatch')
    return dict(warmup_sha256=warmup, batch_sha256=batch_hash.hexdigest(),
                task_updates_planned=planned_task, task_updates_attempted=task_updates,
                task_updates_applied=task_updates, task_updates_skipped=0,
                auxiliary_opportunities=opportunities if auxiliary_arm else 0,
                auxiliary_checks=auxiliary_checks,
                auxiliary_updates_inactive=auxiliary_checks - auxiliary_updates,
                auxiliary_updates_attempted=auxiliary_updates,
                auxiliary_updates_applied=auxiliary_updates,
                auxiliary_updates_skipped=0,
                final_probabilities=infer(model, val_chunks))


def run(data_root, config_path, output):
    import torch
    from torchvision import models
    from .global_report import evaluate_global

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
                 data_counts=manifest['counts'], device=str(torch.cuda.get_device_name()),
                 precision='fp32')
        transform = image_transform()
        train_images = TrainingImages(data_root, manifest['rows'], transform)
        train_ids = [r['sample_id'] for r in train_images.rows]
        train_labels = [r['label'] for r in train_images.rows]
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
        for arm in ARMS:
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
                result = train_one(model, train_images, train_ids, train_labels,
                    val_chunks, config, arm,
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
        labels = [r['label'] for r in val_rows]
        for row in summary:
            directory = output / row['arm']
            probs = torch.load(directory / 'final_probabilities.pt', map_location='cpu', weights_only=True)
            report = evaluate_global(probs.tolist(), labels, config['caps'], val_ids)
            save(directory / 'report.json', report)
            snapshots = json.loads((directory / 'snapshots.json').read_text())
            # The existing scorer uses the neutral word "constraint" for the
            # boundary, so translate phase names only for offline scoring.
            scoring_snapshots = [dict(item, phase=item['phase'].replace('auxiliary', 'constraint'))
                                 for item in snapshots]
            save(directory / 'snapshot_report.json',
                 score_snapshots(directory, scoring_snapshots, labels, val_ids, config['caps']))
            row['scores'] = {policy:{key:details['metrics'][key]
                                    for key in ('accuracy', 'macro_f1', 'cc_f1')}
                             for policy,details in report.items()}
        save(output / 'summary.json', summary)
        artifacts = {p.relative_to(output).as_posix():digest(p)
                     for p in output.rglob('*') if p.is_file() and p != output / 'events.jsonl'}
        log.emit('completed', seconds=time.monotonic() - started,
                 matched_warmup_sha256=identities[0][0],
                 matched_batch_sha256=identities[0][1],
                 summary_sha256=digest(output / 'summary.json'), artifacts=artifacts)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])

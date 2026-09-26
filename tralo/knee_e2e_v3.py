"""End-to-end knee study v3: TraLO's direction at the dose that lands on the hard cap.

Usage: python -m tralo.knee_e2e_v3 DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY [ARM ...]

Protocol: experiments/claude_targeted_step_protocol_20260925.md (fixed before launch);
seeds 1901-1924 at cap 50: experiments/claude_targeted_step_cap50_protocol_20260926.md;
seeds 3000-3024 (MobileNetV3-Large) and 3100-3124 (RegNet-Y-400MF) at cap 76: backbone replication.
Why: the v1 pilot (seed 1701) showed the published step overshoots ~10x (grade-3 soft count
82.6 -> 8.2 in one step against a cap of 76) and its controller froze at the first check.

Five arms share initialisation, warm-up and minibatch order (hash-checked):
  clipper       CE throughout, no optimizer reset
  tralo_null    CE, task-Adam reset at the phase boundary, no constraint step
  tralo_adam    the published arm: separate Adam constraint step (replication)
  tralo_target  tralo.targeted_step: steepest descent of the grade-3 soft count, smallest
                radius that brings the HARD count to the cap; triggered by the hard count
  sham_target   the same radius and per-tensor norms, seeded random direction
Development labels are read only by the offline scorer after every arm has trained.
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
from .knee_end_to_end import development_images, image_transform, infer, score_snapshots
from .knee_experiment import cuda_setup, digest, save, source
from .targeted_step import targeted_step

ARMS = ('clipper', 'tralo_null', 'tralo_adam', 'tralo_target', 'sham_target')
CONSTRAINED = ('tralo_adam', 'tralo_target', 'sham_target')
TARGETED = ('tralo_target', 'sham_target')
# preregistered seed block -> grade-3 cap (v3: 76; v3b deeper cut: 50)
STUDY_CAPS = {**{s: 76 for s in range(1801, 1825)}, **{s: 50 for s in range(1901, 1925)},
              **{s: 76 for s in range(3000, 3025)}, **{s: 76 for s in range(3100, 3125)}}
# preregistered seed block -> backbone (every other seed: resnet18, the default when the key is absent)
STUDY_BACKBONES = {**{s: 'mobilenet_v3_large' for s in range(3000, 3025)},
                   **{s: 'regnet_y_400mf' for s in range(3100, 3125)}}
# backbone -> (logged architecture, torchvision weights name). The server is offline: these are the
# enums whose files are cached there (resnet18-f37072fd, mobilenet_v3_large-5c1a4163, regnet_y_400mf-e6988f5f).
BACKBONES = {'resnet18': ('ResNet18', 'ResNet18_Weights.IMAGENET1K_V1'),
             'mobilenet_v3_large': ('MobileNetV3-Large', 'MobileNet_V3_Large_Weights.IMAGENET1K_V2'),
             'regnet_y_400mf': ('RegNetY-400MF', 'RegNet_Y_400MF_Weights.IMAGENET1K_V2')}


def validate(config):
    expected = {'seed', 'epochs', 'warmup_epochs', 'batch_size', 'task_lr',
                'constraint_lr', 'caps', 'lambda_initial', 'lambda_step',
                'rho_initial', 'rho_target', 'development_batch_size'}
    if set(config) - {'backbone'} != expected:
        raise ValueError('config keys differ from the declared experiment')
    if config['seed'] not in STUDY_CAPS:
        raise ValueError('seed is outside the preregistered sets 1801-1824, 1901-1924, 3000-3024 and 3100-3124')
    if config['caps'] != [None, None, None, STUDY_CAPS[config['seed']], None]:
        raise ValueError('the grade-3 cap does not match the seed block (1801-1824: 76, 1901-1924: 50, 3000-3024: 76, 3100-3124: 76)')
    if config.get('backbone', 'resnet18') not in BACKBONES:
        raise ValueError('backbone must be one of ' + ', '.join(BACKBONES))
    if config.get('backbone', 'resnet18') != STUDY_BACKBONES.get(config['seed'], 'resnet18'):
        raise ValueError('the backbone does not match the seed block (3000-3024: mobilenet_v3_large, '
                         '3100-3124: regnet_y_400mf, otherwise resnet18)')
    for key in ('epochs', 'warmup_epochs', 'batch_size', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('warmup must precede the constraint phase')
    for key in ('task_lr', 'constraint_lr', 'lambda_initial', 'lambda_step', 'rho_initial', 'rho_target'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)
    if config['task_lr'] == 0 or config['constraint_lr'] == 0 or config['rho_target'] < config['rho_initial']:
        raise ValueError('invalid learning rate or rho schedule')


def build_model(backbone='resnet18', pretrained=True):
    """ImageNet backbone with a fresh 5-way head. The resnet18 path is the original v3 construction
    (same calls, same RNG draws); pretrained=False exists only for tests."""
    import torch
    from torchvision import models
    if backbone not in BACKBONES:
        raise ValueError('backbone must be one of ' + ', '.join(BACKBONES))
    weights = models.get_weight(BACKBONES[backbone][1]) if pretrained else None
    if backbone == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
    elif backbone == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
    else:
        model = models.regnet_y_400mf(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
    return model


def constraint_optimizer(arm, model, config):
    params = list(model.parameters())
    if arm == 'tralo_adam':
        import torch
        return torch.optim.Adam(params, lr=config['constraint_lr'])
    return None


def _flat(model):
    import torch
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def train_one(model, train_images, val_chunks, config, arm, emit, snapshot):
    """Train without any development-label argument; record dose and binding."""
    import torch
    from .global_constraint import advance_controller
    from .streamed_constraint import streamed_step
    if arm not in ARMS:
        raise ValueError('unknown arm')
    seed = config['seed']
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 1)
    task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
    constraint = constraint_optimizer(arm, model, config)
    sham_generator = torch.Generator().manual_seed(seed + 7) if arm == 'sham_target' else None
    targeted = []
    multipliers = [float(config['lambda_initial'])] * 5
    rho = float(config['rho_initial'])
    rho_step = (config['rho_target'] - rho) / (config['epochs'] - config['warmup_epochs'])
    frozen = False
    task_updates = constraint_updates = constraint_checks = 0
    planned_task = config['epochs'] * math.ceil(len(train_images) / config['batch_size'])
    batch_hash = hashlib.sha256()
    warmup = None
    first_hard = None
    displacements = []
    emit(dict(event='started', arm=arm, seed=seed, planned_task_updates=planned_task))
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
            task.step()
            task_updates += 1
            ce_total += float(loss.detach()) * len(examples)
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('nonfinite supervised parameters')
        if epoch + 1 == config['warmup_epochs']:
            warmup = _state_hash(model)
        row = dict(event='epoch', epoch=epoch + 1, training_ce=ce_total / len(train_images),
                   task_updates_cumulative=task_updates)
        if epoch >= config['warmup_epochs']:
            before = infer(model, val_chunks)
            snapshot(epoch + 1, 'before_constraint', before)
            hard = torch.bincount(before.argmax(1), minlength=5).tolist()
            if first_hard is None:
                first_hard = hard
            row.update(hard_counts_before=hard, soft_counts_before=before.sum(0).tolist(),
                       multipliers=list(multipliers), rho=rho)
            applied = False
            if arm in CONSTRAINED:
                constraint_checks += 1
                start_params = _flat(model)
                if arm in TARGETED:
                    result = targeted_step(model, val_chunks, config['caps'], sham_generator=sham_generator)
                    if result['hard_before'] != hard[3]:
                        raise RuntimeError('before snapshot differs from the targeted step count')
                    targeted.append(result)
                    row.update(targeted=result)
                else:
                    result = streamed_step(model, val_chunks, config['caps'],
                                           torch.tensor(multipliers, device=device, dtype=before.dtype),
                                           rho, constraint)
                    if not torch.allclose(result['probabilities'].cpu(), before, atol=1e-7, rtol=1e-6):
                        raise RuntimeError('before snapshot differs from streamed count pass')
                applied = result['applied']
                moved = float((_flat(model) - start_params).double().norm())
                displacements.append(moved)
                constraint_updates += int(applied)
                multipliers, rho, frozen = advance_controller(
                    hard, config['caps'], multipliers, rho, rho_step, config['lambda_step'], frozen)
                row.update(multipliers_after=list(multipliers), rho_after=rho, controller_frozen=frozen,
                           parameter_displacement=moved,
                           constraint_lr_effective=getattr(constraint, 'lr', config['constraint_lr']))
            after = infer(model, val_chunks)
            snapshot(epoch + 1, 'after_constraint', after)
            row.update(constraint_applied=applied,
                       hard_counts_after=torch.bincount(after.argmax(1), minlength=5).tolist(),
                       soft_counts_after=after.sum(0).tolist(), constraint_updates_cumulative=constraint_updates)
        emit(row)
    if task_updates != planned_task:
        raise RuntimeError('supervised update dose mismatch')
    cap = config['caps'][3]
    return dict(warmup_sha256=warmup, batch_sha256=batch_hash.hexdigest(),
                task_updates_planned=planned_task, task_updates_applied=task_updates,
                constraint_checks=constraint_checks, constraint_updates_applied=constraint_updates,
                parameter_displacements=displacements, targeted_steps=targeted,
                first_constraint_hard_counts=first_hard,
                cap_binds_on_hard_count=bool(first_hard is not None and first_hard[3] > cap),
                final_probabilities=infer(model, val_chunks))


def run(data_root, config_path, output, arms=ARMS):
    import torch
    from .global_report import evaluate_global
    from .supervised_adaptation import TrainingImages
    config = json.loads(Path(config_path).read_text())
    validate(config)
    arms = tuple(arms)
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError('arms must be a nonempty subset of ' + ', '.join(ARMS))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / 'events.jsonl') as log:
        cuda_setup()
        manifest = audit(data_root)
        if manifest['counts']['train'] != 5778 or manifest['counts']['val'] != 826:
            raise RuntimeError('unexpected Chen split sizes')
        save(output / 'manifest.json', manifest)
        save(output / 'config.json', config)
        log.emit('started', source_sha256=source(), config_sha256=digest(config_path), arms=list(arms),
                 data_counts=manifest['counts'], device=str(torch.cuda.get_device_name()), precision='fp32')
        transform = image_transform()
        train_images = TrainingImages(data_root, manifest['rows'], transform)
        val_chunks = development_images(data_root, manifest['rows'], transform, config['development_batch_size'])
        val_rows = [r for r in manifest['rows'] if r['split'] == 'val']
        val_ids = [r['sample_id'] for r in val_rows]
        torch.manual_seed(config['seed'])
        backbone = config.get('backbone', 'resnet18')
        base = build_model(backbone)
        initial_sha = _state_hash(base)
        log.emit('model_initialized', initial_sha256=initial_sha, architecture=BACKBONES[backbone][0],
                 backbone=backbone, weights=BACKBONES[backbone][1],
                 transform='RGB resize224 ImageNet normalization no augmentation')
        identities, summary = [], []
        started = time.monotonic()
        for arm in arms:
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
                    snapshots.append(dict(epoch=epoch, phase=phase, file=path.name, sha256=digest(path)))
                result = train_one(model, train_images, val_chunks, config, arm,
                                   lambda row: arm_log.emit(row['event'], **{k: v for k, v in row.items() if k != 'event'}),
                                   snapshot)
                torch.save(result.pop('final_probabilities'), directory / 'final_probabilities.pt')
                save(directory / 'snapshots.json', snapshots)
                identities.append((result['warmup_sha256'], result['batch_sha256']))
                arm_log.emit('training_completed', **result)
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
            save(directory / 'snapshot_report.json', score_snapshots(directory, snapshots, labels, val_ids, config['caps']))
            row['scores'] = {policy: {key: details['metrics'][key] for key in ('accuracy', 'macro_f1', 'cc_f1')}
                             for policy, details in report.items()}
        save(output / 'summary.json', summary)
        log.emit('completed', seconds=time.monotonic() - started,
                 matched_warmup_sha256=identities[0][0], matched_batch_sha256=identities[0][1])


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or ARMS)

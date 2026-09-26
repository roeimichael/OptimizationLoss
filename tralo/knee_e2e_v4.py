"""End-to-end knee study v4 (BANDCONS): augmentation-view consistency near the cut.

Usage: python -m tralo.knee_e2e_v4 DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY [ARM ...]

Why: v3 (seeds 1801-1824, 1901-1924) showed a count constraint on development probabilities
only reproduces the post-hoc allocator: at the right dose it knows how many, not who.
A training-time win needs information the allocator lacks; here it is the disagreement
between a weak and a strong view of unlabeled development images ranked near the cut.

Six arms share initialisation, warm-up and minibatch order (hash-checked):
  clipper        CE throughout, no optimizer reset
  tralo_null     CE, task-Adam reset at the phase boundary
  aug_clip       tralo_null with strong augmentation of the TRAIN images after warm-up
  bandcons       + dosed consistency on U items per batch from the 2w-item band centred on
                 rank `cap` (the capped_first boundary)
  bandcons_unc   the same band centred on the natural argmax count of grade 3
  bandcons_rand  2w items drawn uniformly outside the cap band, redrawn every epoch
Bands are recomputed at the START of every post-warm-up epoch from the probabilities of
the model at that moment. Every arm takes exactly one task-optimizer step per labeled batch,
so the task dose is equal in every arm.
Dose rule (amendment after pilot 2000, where beta = 1 let the consistency term dominate CE:
band disagreement 2.19 log-odds against CE ~0.1, bandcons_rand's disagreement reaching 91.1
and bandcons_unc's natural grade-3 count swinging to 805/826): in bandcons* the consistency
gradient is rescaled per batch to DOSE_RATIO x the CE gradient's global L2 norm, so it
can steer but never outweigh the supervised signal. clipper, tralo_null and aug_clip keep
the single-backward path, bit-identical to the pilot.
Snapshots: epochNN_before_constraint = start-of-epoch probabilities (the band source),
epochNN_after_constraint = end-of-epoch; no arm takes a separate constraint step.
Development labels are read only by the offline scorer after every arm has trained.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .band_consistency import (band_indices, consistency_loss, gather, natural_center, random_band,
                               strong_view, tta_probabilities, view_disagreement)
from .global_comparison import _state_hash, audited_arm_log
from .knee_data import audit
from .knee_end_to_end import development_images, image_transform, infer, score_snapshots
from .knee_experiment import cuda_setup, digest, save, source

ARMS = ('clipper', 'tralo_null', 'aug_clip', 'bandcons', 'bandcons_unc', 'bandcons_rand')
BANDCONS = ('bandcons', 'bandcons_unc', 'bandcons_rand')
DOSE_RATIO = 0.1        # ||consistency step|| / ||CE step|| per batch, fixed before launch
BAND_FRACTION = 0.25    # w = round(BAND_FRACTION * cap) items each side of the centre
U = 8                   # band items per labeled batch
TTA_DRAWS = 8
# own generators, never the batch generator (seed + 1) nor the global RNG
AUGMENT, CYCLE, RANDOM_BAND, LOG_VIEWS, TTA = 11, 13, 17, 19, 23
# preregistered seed block -> grade-3 cap (2000 is the integrity pilot, not a study seed)
# (2000-2124: the beta = 1 design, kept so pilot 2000 stays reproducible; 2400-2524: the dose rule)
STUDY_CAPS = {2000: 50, **{s: 50 for s in range(2001, 2025)}, **{s: 76 for s in range(2101, 2125)},
              2400: 50, **{s: 50 for s in range(2401, 2425)}, **{s: 76 for s in range(2501, 2525)}}


def validate(config):
    expected = {'seed', 'epochs', 'warmup_epochs', 'batch_size', 'task_lr',
                'constraint_lr', 'caps', 'lambda_initial', 'lambda_step',
                'rho_initial', 'rho_target', 'development_batch_size'}
    if set(config) != expected:
        raise ValueError('config keys differ from the declared experiment')
    if config['seed'] not in STUDY_CAPS:
        raise ValueError('seed is outside the preregistered sets 2000-2024, 2101-2124, 2400-2424, 2501-2524')
    if config['caps'] != [None, None, None, STUDY_CAPS[config['seed']], None]:
        raise ValueError('the grade-3 cap does not match the seed block (2000-2024, 2400-2424: 50; '
                         '2101-2124, 2501-2524: 76)')
    for key in ('epochs', 'warmup_epochs', 'batch_size', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('warmup must precede the consistency phase')
    for key in ('task_lr', 'constraint_lr', 'lambda_initial', 'lambda_step', 'rho_initial', 'rho_target'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)
    if config['task_lr'] == 0:
        raise ValueError('invalid learning rate')


def half_width(cap):
    return round(BAND_FRACTION * cap)


def dual_backward(model, ce, consistency, ratio):
    """p.grad = g_ce + ratio * (||g_ce|| / ||g_cons||) * g_cons, global float64 L2 norms.

    Two backwards, grads cleared between them so g_cons carries no CE part. `consistency`
    is called after the CE backward and must run its forward in eval mode, so BatchNorm
    running statistics move only once, in the caller's train-mode CE forward. A zero
    consistency gradient falls back to CE alone.
    """
    import torch
    params = [p for p in model.parameters() if p.requires_grad]

    def grads():
        return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]

    def norm(values):
        return math.sqrt(sum(float(g.double().square().sum()) for g in values))
    ce.backward()
    g_ce = grads()
    for p in params:
        p.grad = None
    term = consistency()
    if not bool(torch.isfinite(term)):
        raise RuntimeError('nonfinite consistency loss')
    term.backward()
    g_cons = grads()
    n_ce, n_cons = norm(g_ce), norm(g_cons)
    scale = ratio * n_ce / n_cons if n_cons > 0 else 0.0
    added = [scale * g for g in g_cons]
    for p, a, b in zip(params, g_ce, added):
        p.grad = a + b
    return term, dict(ce_grad_norm=n_ce, consistency_grad_norm=n_cons,
                      realised_ratio=norm(added) / n_ce if n_ce > 0 else 0.0)


def select_band(arm, probabilities, cap, w, generator):
    """The arm's development band from start-of-epoch probabilities; returns (indices, centre)."""
    p3 = probabilities[:, 3]
    if arm == 'bandcons':
        return band_indices(p3, cap, w), cap
    if arm == 'bandcons_unc':
        centre = natural_center(probabilities, w)
        return band_indices(p3, centre, w), centre
    if arm == 'bandcons_rand':
        return random_band(p3, cap, w, generator), None
    raise ValueError('arm has no band')


def train_one(model, train_images, val_chunks, config, arm, emit, snapshot):
    """Train without any development-label argument; record dose, bands and consistency."""
    import torch
    if arm not in ARMS:
        raise ValueError('unknown arm')
    seed = config['seed']
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 1)
    augment = torch.Generator().manual_seed(seed + AUGMENT)
    cycle = torch.Generator().manual_seed(seed + CYCLE)
    rand_band = torch.Generator().manual_seed(seed + RANDOM_BAND)
    task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
    cap = config['caps'][3]
    w = half_width(cap)
    task_updates = 0
    planned_task = config['epochs'] * math.ceil(len(train_images) / config['batch_size'])
    batch_hash = hashlib.sha256()
    warmup = None
    first_hard = None
    current = None
    band_logs = []
    natural_counts = []

    def next_items():
        while len(queue) < U:
            queue.extend(band[torch.randperm(len(band), generator=cycle)].tolist())
        picked = queue[:U]
        del queue[:U]
        return gather(val_chunks, picked)
    emit(dict(event='started', arm=arm, seed=seed, planned_task_updates=planned_task,
              band_half_width=w if arm in BANDCONS else None))
    for epoch in range(config['epochs']):
        if epoch == config['warmup_epochs'] and arm != 'clipper':
            task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
            emit(dict(event='task_optimizer_reset', epoch=epoch + 1))
        row = dict(event='epoch', epoch=epoch + 1)
        band, queue, consistency, doses = None, [], [], []
        if epoch >= config['warmup_epochs']:
            snapshot(epoch + 1, 'before_constraint', current)
            hard = torch.bincount(current.argmax(1), minlength=5).tolist()
            if first_hard is None:
                first_hard = hard
            natural_counts.append(hard[3])
            row.update(hard_counts_before=hard, soft_counts_before=current.sum(0).tolist(), natural_count=hard[3])
            if arm in BANDCONS:
                band, centre = select_band(arm, current, cap, w, rand_band)
                disagreement = view_disagreement(model, gather(val_chunks, band),
                                                 torch.Generator().manual_seed(seed + LOG_VIEWS))
                row.update(band=band.tolist(), band_size=len(band), band_center=centre,
                           band_disagreement_start=disagreement)
        order = torch.randperm(len(train_images), generator=generator)
        batch_hash.update(order.numpy().tobytes())
        model.train()
        ce_total = 0.
        for start in range(0, len(order), config['batch_size']):
            examples = [train_images[int(i)] for i in order[start:start + config['batch_size']]]
            images = torch.stack([row_[0] for row_ in examples])
            if arm == 'aug_clip' and epoch >= config['warmup_epochs']:
                images = strong_view(images, augment)
            labels = torch.tensor([row_[1] for row_ in examples], dtype=torch.long, device=device)
            task.zero_grad(set_to_none=True)
            ce = torch.nn.functional.cross_entropy(model(images.to(device)), labels)
            if not bool(torch.isfinite(ce)):
                raise RuntimeError('nonfinite supervised loss')
            if arm in BANDCONS and epoch >= config['warmup_epochs']:
                term, dose = dual_backward(model, ce, lambda: consistency_loss(model, next_items(), augment),
                                           DOSE_RATIO)
                consistency.append(float(term.detach()))
                doses.append(dose)
            else:
                ce.backward()
            task.step()
            task_updates += 1
            ce_total += float(ce.detach()) * len(examples)
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('nonfinite supervised parameters')
        if epoch + 1 == config['warmup_epochs']:
            warmup = _state_hash(model)
        row.update(training_ce=ce_total / len(train_images), task_updates_cumulative=task_updates)
        if epoch + 1 >= config['warmup_epochs']:
            # every arm runs this same pass; the next epoch's band is computed from it
            current = infer(model, val_chunks)
        if epoch >= config['warmup_epochs']:
            snapshot(epoch + 1, 'after_constraint', current)
            row.update(hard_counts_after=torch.bincount(current.argmax(1), minlength=5).tolist(),
                       soft_counts_after=current.sum(0).tolist())
            if band is not None:
                row.update(consistency_loss_mean=sum(consistency) / len(consistency),
                           consistency_terms=len(consistency), dose_ratio=DOSE_RATIO)
                for key in ('ce_grad_norm', 'consistency_grad_norm', 'realised_ratio'):
                    values = [d[key] for d in doses]
                    row.update({key + '_mean': sum(values) / len(values), key + '_max': max(values)})
                band_logs.append({k: row[k] for k in (
                    'epoch', 'band_size', 'band_center', 'band_disagreement_start', 'consistency_loss_mean',
                    'consistency_terms', 'dose_ratio', 'ce_grad_norm_mean', 'ce_grad_norm_max',
                    'consistency_grad_norm_mean', 'consistency_grad_norm_max', 'realised_ratio_mean',
                    'realised_ratio_max')})
        emit(row)
    if task_updates != planned_task:
        raise RuntimeError('supervised update dose mismatch')
    tta, tta_sha = tta_probabilities(model, val_chunks, seed + TTA, TTA_DRAWS)
    return dict(warmup_sha256=warmup, batch_sha256=batch_hash.hexdigest(),
                task_updates_planned=planned_task, task_updates_applied=task_updates,
                band_half_width=w if arm in BANDCONS else None, band_logs=band_logs,
                natural_counts_start=natural_counts, dose_ratio=DOSE_RATIO if arm in BANDCONS else None,
                first_constraint_hard_counts=first_hard,
                cap_binds_on_hard_count=bool(first_hard is not None and first_hard[3] > cap),
                tta_draws_sha256=tta_sha, final_probabilities=current, tta_probabilities=tta)


def run(data_root, config_path, output, arms=ARMS):
    import torch
    from torchvision import models
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
                 data_counts=manifest['counts'], device=str(torch.cuda.get_device_name()), precision='fp32',
                 dose_ratio=DOSE_RATIO, band_fraction=BAND_FRACTION, band_items_per_batch=U, tta_draws=TTA_DRAWS)
        transform = image_transform()
        train_images = TrainingImages(data_root, manifest['rows'], transform)
        val_chunks = development_images(data_root, manifest['rows'], transform, config['development_batch_size'])
        val_rows = [r for r in manifest['rows'] if r['split'] == 'val']
        val_ids = [r['sample_id'] for r in val_rows]
        torch.manual_seed(config['seed'])
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = torch.nn.Linear(base.fc.in_features, 5)
        initial_sha = _state_hash(base)
        log.emit('model_initialized', initial_sha256=initial_sha, architecture='ResNet18',
                 transform='RGB resize224 ImageNet normalization; strong augmentation only where the arm says')
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
                torch.save(result.pop('tta_probabilities'), directory / 'tta_probabilities.pt')
                save(directory / 'snapshots.json', snapshots)
                identities.append((result['warmup_sha256'], result['batch_sha256'], result['tta_draws_sha256']))
                arm_log.emit('training_completed', **result)
                summary.append(dict(arm=arm, **result))
                del model
                torch.cuda.empty_cache()
        if len(set(identities)) != 1:
            raise RuntimeError('warmup, batch or TTA-draw identity differs across matched arms')
        labels = [r['label'] for r in val_rows]
        for row in summary:
            directory = output / row['arm']
            row['scores'] = {}
            for name, file, report_file in (('clean', 'final_probabilities.pt', 'report.json'),
                                            ('tta', 'tta_probabilities.pt', 'report_tta.json')):
                probs = torch.load(directory / file, map_location='cpu', weights_only=True)
                report = evaluate_global(probs.tolist(), labels, config['caps'], val_ids)
                save(directory / report_file, report)
                row['scores'][name] = {policy: {key: details['metrics'][key] for key in ('accuracy', 'macro_f1', 'cc_f1')}
                                       for policy, details in report.items()}
            snapshots = json.loads((directory / 'snapshots.json').read_text())
            save(directory / 'snapshot_report.json', score_snapshots(directory, snapshots, labels, val_ids, config['caps']))
        save(output / 'summary.json', summary)
        log.emit('completed', seconds=time.monotonic() - started, matched_warmup_sha256=identities[0][0],
                 matched_batch_sha256=identities[0][1], matched_tta_draws_sha256=identities[0][2])


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or ARMS)

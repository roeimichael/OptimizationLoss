"""End-to-end knee study v5 (CUTPAIR): a supervised hinge anchored at the capped_first cut.

Usage: python -m tralo.knee_e2e_v5 DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY [ARM ...]

Why: the CUTPAIR liveness gate (experiments/claude_cutpair_gate_20260926.md) was dead without
augmentation (the training set is memorised, so no training item sits near the cut) but
alive at cap 76 under the v4 aug_clip schedule (mean N_act 73.8, P_act 225 over epochs 6-10).
So CUTPAIR runs on top of aug_clip, and aug_clip is its null. The WHO information comes from
TRAIN labels; the unlabeled development pool only places the anchor tau, the logit of the
cap-th largest development p3, which is exactly where capped_first cuts.

Five arms share initialisation, warm-up and minibatch order (hash-checked):
  clipper, tralo_null, aug_clip   exactly as in tralo.knee_e2e_v4 (bit-identical, tested)
  cutpair_aug        aug_clip + the cut hinge at tau_e (rank cap)
  cutpair_aug_shift  the same hinge at a deliberately wrong cut: rank round(cap/3) or 3*cap,
                     chosen per epoch by its own generator (two-sided shift) -- is the effect
                     the ANCHOR, or just any extra margin on grade 3?
At the start of each post-warm-up epoch the cutpair arms fix tau_e from the start-of-epoch
development probabilities and their active sets from a clean eval-mode, no-grad pass over the
training images (tralo.cutpair_gate.training_log_odds):
  N_act = {y != 3, s > tau - m},  P_act = {y == 3, tau - W < s < tau + m},  m = 1, W = 3.
Per labeled batch, the hinge averages relu(s - tau + m) over active negatives and
relu(tau + m - s) over active positives, on the SAME train-mode logits as CE (one forward, so
BatchNorm statistics move once). Its gradient is rescaled to DOSE_RATIO x ||g_CE|| (the v4
BANDCONS dose rule), and every arm takes exactly one task-optimizer step per batch.
Snapshots: epochNN_before_constraint = start of epoch, epochNN_after_constraint = end of epoch.
Development labels are read only by the offline scorer after every arm has trained.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .band_consistency import log_odds, strong_view, tta_probabilities
from .cutpair_gate import training_log_odds
from .global_comparison import _state_hash, audited_arm_log
from .knee_data import audit
from .knee_end_to_end import development_images, image_transform, infer, score_snapshots
from .knee_experiment import cuda_setup, digest, save, source

ARMS = ('clipper', 'tralo_null', 'aug_clip', 'cutpair_aug', 'cutpair_aug_shift')
CUTPAIR = ('cutpair_aug', 'cutpair_aug_shift')
AUGMENTED = ('aug_clip',) + CUTPAIR
DOSE_RATIO = 0.1        # ||hinge step|| / ||CE step|| per batch, fixed before launch
MARGIN, WINDOW = 1.0, 3.0
TTA_DRAWS = 8
# own generators, never the batch generator (seed + 1) nor the global RNG
AUGMENT, TTA, SHIFT = 11, 23, 29
# preregistered seed block -> grade-3 cap (2700 is the integrity pilot, not a study seed)
STUDY_CAPS = {2700: 76, **{s: 76 for s in range(2701, 2725)}}


def validate(config):
    expected = {'seed', 'epochs', 'warmup_epochs', 'batch_size', 'task_lr',
                'constraint_lr', 'caps', 'lambda_initial', 'lambda_step',
                'rho_initial', 'rho_target', 'development_batch_size'}
    if set(config) != expected:
        raise ValueError('config keys differ from the declared experiment')
    if config['seed'] not in STUDY_CAPS:
        raise ValueError('seed is outside the preregistered sets 2700-2724')
    if config['caps'] != [None, None, None, STUDY_CAPS[config['seed']], None]:
        raise ValueError('the grade-3 cap does not match the seed block (2700-2724: 76)')
    for key in ('epochs', 'warmup_epochs', 'batch_size', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('warmup must precede the hinge phase')
    for key in ('task_lr', 'constraint_lr', 'lambda_initial', 'lambda_step', 'rho_initial', 'rho_target'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)
    if config['task_lr'] == 0:
        raise ValueError('invalid learning rate')


def rank_p3(p3, rank):
    """The rank-th largest p3 over ALL development items (capped_first's boundary at rank=cap)."""
    import torch
    if not 1 <= rank <= len(p3):
        raise ValueError('anchor rank outside the cohort')
    return float(torch.sort(p3.double(), descending=True).values[rank - 1])


def cut_logit(p3, rank):
    p = min(max(rank_p3(p3, rank), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def shift_ranks(cap):
    return (round(cap / 3), 3 * cap)


def active_sets(s, y, tau):
    neg = (y != 3) & (s > tau - MARGIN)
    pos = (y == 3) & (s > tau - WINDOW) & (s < tau + MARGIN)
    return neg, pos


def cut_hinge(s, neg, pos, tau):
    """Mean hinge over the batch's active items, or None when none is active."""
    import torch
    active = neg | pos
    if not bool(active.any()):
        return None
    terms = torch.where(neg, torch.relu(s - tau + MARGIN), torch.relu(tau + MARGIN - s))
    return terms[active].mean()


def dosed_gradient(model, ce, hinge, ratio):
    """p.grad = g_ce + ratio * (||g_ce|| / ||g_cut||) * g_cut, global float64 L2 norms.

    Both gradients come from the SAME train-mode forward (autograd.grad, graph retained for the
    second), so BatchNorm statistics moved once. No hinge or a zero hinge gradient -> CE alone.
    """
    import torch
    params = [p for p in model.parameters() if p.requires_grad]

    def grads(loss, retain):
        values = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, values)]

    def norm(values):
        return math.sqrt(sum(float(g.double().square().sum()) for g in values))
    g_ce = grads(ce, hinge is not None)
    g_cut = grads(hinge, False) if hinge is not None else [torch.zeros_like(p) for p in params]
    n_ce, n_cut = norm(g_ce), norm(g_cut)
    scale = ratio * n_ce / n_cut if n_cut > 0 else 0.0
    added = [scale * g for g in g_cut]
    for p, a, b in zip(params, g_ce, added):
        p.grad = a + b
    return dict(ce_grad_norm=n_ce, cut_grad_norm=n_cut, realised_ratio=norm(added) / n_ce if n_ce > 0 else 0.0)


def _summary(values):
    return (sum(values) / len(values), max(values)) if values else (None, None)


def train_one(model, train_images, val_chunks, config, arm, emit, snapshot):
    """Train without any development-label argument; record dose, anchors and active sets."""
    import torch
    if arm not in ARMS:
        raise ValueError('unknown arm')
    seed = config['seed']
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 1)
    augment = torch.Generator().manual_seed(seed + AUGMENT)
    shift = torch.Generator().manual_seed(seed + SHIFT)
    task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
    cap = config['caps'][3]
    task_updates = 0
    planned_task = config['epochs'] * math.ceil(len(train_images) / config['batch_size'])
    batch_hash = hashlib.sha256()
    warmup = None
    first_hard = None
    current = None
    cut_logs = []
    natural_counts = []
    emit(dict(event='started', arm=arm, seed=seed, planned_task_updates=planned_task))
    for epoch in range(config['epochs']):
        if epoch == config['warmup_epochs'] and arm != 'clipper':
            task = torch.optim.Adam(model.parameters(), lr=config['task_lr'])
            emit(dict(event='task_optimizer_reset', epoch=epoch + 1))
        row = dict(event='epoch', epoch=epoch + 1)
        hinge_on, hinges, doses = False, [], []
        if epoch >= config['warmup_epochs']:
            snapshot(epoch + 1, 'before_constraint', current)
            hard = torch.bincount(current.argmax(1), minlength=5).tolist()
            if first_hard is None:
                first_hard = hard
            natural_counts.append(hard[3])
            row.update(hard_counts_before=hard, soft_counts_before=current.sum(0).tolist(), natural_count=hard[3])
            if arm in CUTPAIR:
                hinge_on = True
                rank = cap if arm == 'cutpair_aug' else shift_ranks(cap)[int(torch.randint(0, 2, (1,), generator=shift))]
                tau = cut_logit(current[:, 3], rank)
                bank_s, bank_y = training_log_odds(model, train_images)
                neg_act, pos_act = active_sets(bank_s, bank_y, tau)
                row.update(anchor_rank=rank, anchor_p3=rank_p3(current[:, 3], rank), tau=tau,
                           n_act=int(neg_act.sum()), p_act=int(pos_act.sum()))
        order = torch.randperm(len(train_images), generator=generator)
        batch_hash.update(order.numpy().tobytes())
        model.train()
        ce_total = 0.
        for start in range(0, len(order), config['batch_size']):
            index = order[start:start + config['batch_size']]
            examples = [train_images[int(i)] for i in index]
            images = torch.stack([row_[0] for row_ in examples])
            if arm in AUGMENTED and epoch >= config['warmup_epochs']:
                images = strong_view(images, augment)
            labels = torch.tensor([row_[1] for row_ in examples], dtype=torch.long, device=device)
            task.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            ce = torch.nn.functional.cross_entropy(logits, labels)
            if not bool(torch.isfinite(ce)):
                raise RuntimeError('nonfinite supervised loss')
            if hinge_on:
                hinge = cut_hinge(log_odds(logits), neg_act[index].to(device), pos_act[index].to(device), tau)
                if hinge is not None:
                    if not bool(torch.isfinite(hinge)):
                        raise RuntimeError('nonfinite hinge loss')
                    hinges.append(float(hinge.detach()))
                doses.append(dosed_gradient(model, ce, hinge, DOSE_RATIO))
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
            # every arm runs this same pass; the next epoch's anchor is computed from it
            current = infer(model, val_chunks)
        if epoch >= config['warmup_epochs']:
            snapshot(epoch + 1, 'after_constraint', current)
            row.update(hard_counts_after=torch.bincount(current.argmax(1), minlength=5).tolist(),
                       soft_counts_after=current.sum(0).tolist())
            if hinge_on:
                active = [d for d in doses if d['cut_grad_norm'] > 0]
                # active: >= 1 active item in the batch; dosed: the hinge gradient was nonzero and rescaled
                row.update(active_batches=len(hinges), dosed_batches=len(active), batches=len(doses),
                           hinge_mean=_summary(hinges)[0], dose_ratio=DOSE_RATIO)
                for key in ('ce_grad_norm', 'cut_grad_norm', 'realised_ratio'):
                    row[key + '_mean'], row[key + '_max'] = _summary([d[key] for d in active])
                cut_logs.append({k: row[k] for k in (
                    'epoch', 'anchor_rank', 'anchor_p3', 'tau', 'n_act', 'p_act', 'active_batches', 'dosed_batches',
                    'batches', 'hinge_mean', 'dose_ratio',
                    'ce_grad_norm_mean', 'ce_grad_norm_max', 'cut_grad_norm_mean', 'cut_grad_norm_max',
                    'realised_ratio_mean', 'realised_ratio_max')})
        emit(row)
    if task_updates != planned_task:
        raise RuntimeError('supervised update dose mismatch')
    tta, tta_sha = tta_probabilities(model, val_chunks, seed + TTA, TTA_DRAWS)
    return dict(warmup_sha256=warmup, batch_sha256=batch_hash.hexdigest(),
                task_updates_planned=planned_task, task_updates_applied=task_updates,
                cut_logs=cut_logs, natural_counts_start=natural_counts,
                dose_ratio=DOSE_RATIO if arm in CUTPAIR else None,
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
                 dose_ratio=DOSE_RATIO, margin=MARGIN, window=WINDOW, tta_draws=TTA_DRAWS)
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

"""Knee study under Yuval Kassif's training pipeline, with his outer loop (PAO) as a rival arm.

Usage: python -m tralo.knee_yuval DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY

Pipeline source: github.com/YuvalKassif/ConstrainedClassification @ 413d96c (2026-09-25):
load_data._transforms_for (RGB branch) and get_weighted_sampler; config.py (Adam 1e-4, weight
decay 1e-4, batch 32, x0.8 every 5 epochs, patience 5, 75 epochs); train.train_model (per-batch
dynamic LR, early stopping on the loss, best weights restored); losses.CustomLoss; main.py
(reseed before every retrain, C += mu*dF, C[k] = 1, stop when F < 1e-5);
update_weights.calculate_F_and_derivative (b = 100).
Deviations, each forced by our evaluation rules or the offline server:
  * early stopping reads a split carved from TRAIN (subjects whose ID hash is 0 mod 10, both
    knees together), never the development pool, whose labels enter only the offline scorer;
  * the outer loop counts argmax grade-3 predictions on development IMAGES against our
    label-free cap, not on the test set against a count of test labels;
  * at most MAX_RETRAINS retrains; a loop still above the cap is logged as unconverged;
  * seeds 4000-4023 use ResNet18 (cached ImageNet weights); seeds 4100-4123 use Yuval's own
    backbone, timm efficientnet_b5 (sw_in12k_ft_in1k, uploaded because the server is offline).
Arms, all deployed with capped_first at the same cap:
  pto          Yuval's PTO: the first model of the loop (CustomLoss with C = 1, i.e. CE)
  tralo_final  pto + one tralo.targeted_step (TraLO's count direction, radius meeting the cap)
  sham_final   pto + the same radius in a seeded random direction
  pao          Yuval's PAO: the model of the last retrain
Protocol: experiments/claude_yuval_pipeline_prereg_20260927.md (+ amendment 1: the B5 block).
Recipe factorial (seeds 4200-4223, pilot 4299, ResNet18, experiments/claude_recipe_factorial_prereg_20260927.md):
PTO only (max_retrains 1, no pao arm), with three switches that turn one part of the pipeline off:
augment (off: the evaluation transform), balanced (off: a uniform permutation per epoch) and
early_stop (off: exactly max_epochs epochs, the last one kept).
Small backbones (experiments/claude_yuval_smallbb_prereg_20260927.md): the full pipeline, PTO only
(max_retrains 1), on torchvision mobilenet_v3_large (seeds 4300-4323, pilot 4399) and regnet_y_400mf
(4400-4423, pilot 4499), built as in the v3 studies.
Step ensemble (seeds 4500-4571, ResNet18, experiments/claude_stepens_prereg_20260927.md): the full
pipeline, PTO only, with snapshot_steps: at every epoch TraLO's targeted step and its sham act on side
copies of the model (epochNN_tralo.pt, epochNN_sham.pt), so each arm can be snapshot-ensembled. The pilot
reruns stored seed 4000, whose PTO trajectory must stay byte-identical. Its replication on regnet_y_400mf
(seeds 4600-4671, experiments/claude_stepens_rgy_prereg_20260927.md) is the same design, and its pilot reruns
stored seed 4400. The dsisco02 blocks (experiments/claude_stepens_d2_prereg_20260928.md) are the same design on
mobilenet_v3_large (seeds 4700-4771, pilot 4300) and efficientnet_b5 (seeds 4800-4847, pilot 4100). Their pilots are
gated against a reference run of the same seed on the same host, a pilot config with snapshot_steps false.
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
from .knee_e2e_v3 import build_model
from .knee_end_to_end import development_images, infer
from .knee_experiment import cuda_setup, digest, save, source
from .targeted_step import targeted_step

CAPPED = 3
MAX_RETRAINS = 8
AUGMENT_OFFSET = 11     # torchvision draws augmentation from the global CPU RNG, reseeded per retrain
SAMPLER_OFFSET = 1
SHAM_OFFSET = 7
MEAN, STD = [0.66133188] * 3, [0.21229856] * 3
STUDY_SEEDS = tuple(range(4000, 4024)) + tuple(range(4100, 4124))
PILOT_SEEDS = (4099, 4199)
FACTORIAL_SEEDS = tuple(range(4200, 4224)) + (4299,)
SMALL_SEEDS = tuple(range(4300, 4324)) + (4399,) + tuple(range(4400, 4424)) + (4499,)
STEPENS_SEEDS = (tuple(range(4500, 4572)) + tuple(range(4600, 4672))    # ResNet18, then its RegNetY replication,
                 + tuple(range(4700, 4772)) + tuple(range(4800, 4848)))   # then MobileNetV3 and B5 on dsisco02
STEPENS_PILOTS = (4000, 4400, 4300, 4100)
KEYS = {'seed', 'cap', 'max_epochs', 'patience', 'batch_size', 'lr', 'weight_decay', 'decay_epoch',
        'decay_factor', 'mu', 'b', 'development_batch_size'}
RECIPE = ('augment', 'balanced', 'early_stop')
B5_WEIGHTS = Path.home() / 'tralo-rebuild/data/weights/timm-efficientnet_b5.sw_in12k_ft_in1k/model.safetensors'
B5_SHA256 = '0e5c09ad618a28d977acf8b7846105443c553a4b425dddb54598a0ac6088aca7'


def backbone_for(seed):
    if 4300 <= seed < 4400 or 4700 <= seed < 4800:
        return 'mobilenet_v3_large'
    if 4400 <= seed < 4500 or 4600 <= seed < 4700:
        return 'regnet_y_400mf'
    return 'efficientnet_b5' if 4100 <= seed < 4200 or 4800 <= seed < 4900 else 'resnet18'


def validate(config):
    factorial = config.get('seed') in FACTORIAL_SEEDS
    stepens = 'snapshot_steps' in config
    small = config.get('seed') in SMALL_SEEDS
    extra =({'max_retrains', *RECIPE} if factorial else {'max_retrains', 'snapshot_steps'} if stepens
             else {'max_retrains'} if small else set())
    if set(config) - {'backbone'} != KEYS | extra:
        raise ValueError('config keys differ from the declared experiment')
    if config['seed'] not in STUDY_SEEDS + PILOT_SEEDS + FACTORIAL_SEEDS + SMALL_SEEDS + STEPENS_SEEDS:
        raise ValueError('seed is outside the preregistered blocks 4000-4847 and pilots')
    if (stepens or config['seed'] in STEPENS_SEEDS) and not (
            (config.get('snapshot_steps') is True and config['seed'] in STEPENS_SEEDS + STEPENS_PILOTS
             or config.get('snapshot_steps') is False and config['seed'] in STEPENS_PILOTS)
            and type(config['max_retrains']) is int and config['max_retrains'] == 1):
        raise ValueError('the step-ensemble studies (seeds 4500-4847, pilots 4000, 4400, 4300 and 4100) step every '
                         'snapshot and train PTO once; only a pilot seed may run a reference with the steps off')
    if factorial and (config['max_retrains'] != 1 or any(type(config[k]) is not bool for k in RECIPE)):
        raise ValueError('the recipe factorial trains PTO once, with boolean switches')
    if small and (type(config['max_retrains']) is not int or config['max_retrains'] != 1):
        raise ValueError('the small-backbone blocks train PTO once')
    if config.get('backbone', 'resnet18') != backbone_for(config['seed']):
        raise ValueError('the backbone does not match the seed block (4100s and 4800s: efficientnet_b5, '
                         '4300s and 4700s: mobilenet_v3_large, 4400s and 4600s: regnet_y_400mf)')
    if config['cap'] != 76:
        raise ValueError('the preregistered grade-3 cap is 76')
    for key in ('max_epochs', 'patience', 'batch_size', 'decay_epoch', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    for key in ('lr', 'weight_decay', 'decay_factor', 'mu', 'b'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)


def transforms_for():
    """load_data._transforms_for, RGB branch, with Yuval's knee mean and std."""
    from torchvision import transforms as T
    normalize = T.Normalize(mean=MEAN, std=STD)
    train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=0.5), T.RandomRotation(3),
                       T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                       T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), T.ToTensor(), normalize])
    evaluation = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])
    return train, evaluation


def efficientnet_b5(pretrained=True):
    """utils.get_model('EfficientNetB5'): timm efficientnet_b5 with a fresh 5-way classifier."""
    import timm
    import torch
    if pretrained and digest(B5_WEIGHTS) != B5_SHA256:
        raise RuntimeError('efficientnet_b5 weights differ from the uploaded file')
    model = timm.create_model('efficientnet_b5', pretrained=pretrained,
                              **(dict(pretrained_cfg_overlay=dict(file=str(B5_WEIGHTS))) if pretrained else {}))
    model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
    return model


def make_backbone(backbone, pretrained=True):
    return efficientnet_b5(pretrained) if backbone == 'efficientnet_b5' else build_model(backbone, pretrained)


def carve(rows):
    """Train rows split by subject: sha256(subject) % 10 == 0 -> early-stopping split."""
    held = lambda r: int(hashlib.sha256(r['subject'].encode()).hexdigest(), 16) % 10 == 0
    train = [r for r in rows if r['split'] == 'train' and not held(r)]
    stop = [r for r in rows if r['split'] == 'train' and held(r)]
    if {r['subject'] for r in train} & {r['subject'] for r in stop}:
        raise RuntimeError('a subject straddles the early-stopping carve')
    return train, stop


class Images:
    """Train-split images decoded once after a hash check; the transform runs at every access."""

    def __init__(self, root, rows):
        from PIL import Image
        self.images, self.labels = [], []
        for row in rows:
            if row['split'] != 'train':
                raise ValueError('only train rows may carry labels into training')
            path = Path(root) / row['path']
            if digest(path) != row['sha256']:
                raise RuntimeError('training image changed after audit')
            with Image.open(path) as image:
                self.images.append(image.convert('RGB'))
            self.labels.append(row['label'])

    def batch(self, indices, transform):
        import torch
        return (torch.stack([transform(self.images[i]) for i in indices]),
                torch.tensor([self.labels[i] for i in indices], dtype=torch.long))

    def weights(self):
        """get_weighted_sampler: 1 / class count per sample."""
        import torch
        counts = {c: self.labels.count(c) for c in set(self.labels)}
        return torch.tensor([1.0 / counts[c] for c in self.labels], dtype=torch.double)


def custom_loss(logits, labels, C, k=CAPPED):
    """losses.CustomLoss, same arithmetic. Returns the loss and the clamped mean gate (last_tanh_mean)."""
    import torch
    y_pred = logits.softmax(1)
    y_true = torch.nn.functional.one_hot(labels, y_pred.shape[1]).float()
    gate = torch.tanh(50000000 * (y_pred.max(1)[0] - y_pred[:, k])).unsqueeze(1)
    term1 = C.unsqueeze(0).detach() * torch.log(1e-7 + torch.relu(1 + (y_pred - 1) * (1 - gate)))
    loss1 = -torch.sum(y_true * term1, dim=1)
    loss2 = -torch.sum(y_true * torch.log(1e-7 + torch.relu(1 + (y_pred - 1) * gate)), dim=1)
    return torch.mean(loss1 + loss2), float(torch.clamp(gate.mean(), 0.0, 1.0))


def f_and_derivative(cap, count, b):
    """update_weights.calculate_F_and_derivative in closed form: F = d^2 (tanh(b d) + 1), d = count - cap."""
    d = count - cap
    th = math.tanh(b * d)
    return d * d * (th + 1), 2 * d * (th + 1) + d * d * b * (1 - th * th)


def stop_loss(model, batches, C):
    import torch
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        total = n = 0
        with torch.no_grad():
            for images, labels in batches:
                loss, _ = custom_loss(model(images.to(device)), labels.to(device), C)
                total += float(loss) * len(labels)
                n += len(labels)
        return total / n
    finally:
        model.train(was_training)


def snapshot_steps(model, pool, caps, seed, epoch, directory):
    """TraLO's targeted step and its sham on side copies of one epoch's model (the step-ensemble study).

    Training is untouched: both steps act on deep copies and the global RNG states are restored, so the
    PTO trajectory stays byte-identical to a run without them. Writes epochNN_tralo.pt and epochNN_sham.pt."""
    import torch
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    steps = {}
    for arm, sham in (('tralo', None), ('sham', torch.Generator().manual_seed(seed + SHAM_OFFSET + 1000 * epoch))):
        side = copy.deepcopy(model)
        steps[arm] = targeted_step(side, pool, caps, sham_generator=sham)
        torch.save(infer(side, pool), Path(directory) / f'epoch{epoch:02d}_{arm}.pt')
        del side
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    if steps['tralo'].get('radius') != steps['sham'].get('radius'):
        raise RuntimeError(f'epoch {epoch}: sham radius differs from the targeted radius')
    return steps


def train_run(model, data, stop, pool, config, C, emit, snapshot):
    """train.train_model on one retrain, with common random numbers across retrains."""
    import torch
    seed = config['seed']
    torch.manual_seed(seed + AUGMENT_OFFSET)
    sampler = torch.Generator().manual_seed(seed + SAMPLER_OFFSET)
    train_tf, eval_tf = transforms_for()
    if not config.get('augment', True):
        train_tf = eval_tf
    early_stop = config.get('early_stop', True)
    device = next(model.parameters()).device
    C = C.to(device)
    mean_c = float(C.mean())
    weights = data.weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    best, best_state, best_epoch, waited = math.inf, None, 0, 0
    first_order = first_batch = None
    updates = 0
    for epoch in range(config['max_epochs']):
        base = config['lr'] * config['decay_factor'] ** (epoch // config['decay_epoch'])
        order = (torch.multinomial(weights, len(weights), replacement=True, generator=sampler)
                 if config.get('balanced', True) else torch.randperm(len(weights), generator=sampler))
        if first_order is None:
            first_order = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        model.train()
        total = gates = live = 0.
        for start in range(0, len(order), config['batch_size']):
            images, labels = data.batch(order[start:start + config['batch_size']].tolist(), train_tf)
            if first_batch is None:
                first_batch = hashlib.sha256(images.numpy().tobytes()).hexdigest()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, t = custom_loss(logits, labels, C)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite training loss')
            lr = max((1.0 - t) * base / max(mean_c, 1e-12) + t * base, 1e-12)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss.backward()
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(labels)
            gates += t * len(labels)
            live += float(((logits.detach().argmax(1) == CAPPED) & (labels != CAPPED)).sum())
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('nonfinite parameters')
        held = stop_loss(model, stop, C)
        probabilities = infer(model, pool)
        snapshot(epoch + 1, probabilities)
        hard = torch.bincount(probabilities.argmax(1), minlength=5).tolist()
        improved = held < best
        if improved:
            best, best_state, best_epoch, waited = held, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            waited += 1
        emit(dict(event='epoch', epoch=epoch + 1, training_loss=total / len(order), stop_loss=held,
                  base_lr=base, last_lr=lr, mean_gate=gates / len(order), live_false_positives=live,
                  hard_counts=hard, soft_count_capped=float(probabilities[:, CAPPED].sum()), improved=improved))
        if early_stop and waited >= config['patience']:
            break
    if early_stop:
        model.load_state_dict(best_state)
    else:                                         # a fixed schedule keeps the last epoch
        best, best_epoch = held, epoch + 1
    return dict(best_epoch=best_epoch, epochs_run=epoch + 1, best_stop_loss=best, task_updates=updates,
                first_order_sha256=first_order, first_batch_sha256=first_batch)


def run(data_root, config_path, output):
    import torch
    from .global_report import evaluate_global
    config = json.loads(Path(config_path).read_text())
    validate(config)
    seed, cap = config['seed'], config['cap']
    caps = [None] * 5
    caps[CAPPED] = cap
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / 'events.jsonl') as log:
        cuda_setup()
        manifest = audit(data_root)
        if manifest['counts']['train'] != 5778 or manifest['counts']['val'] != 826:
            raise RuntimeError('unexpected Chen split sizes')
        train_rows, stop_rows = carve(manifest['rows'])
        save(output / 'manifest.json', manifest)
        save(output / 'config.json', config)
        log.emit('started', source_sha256=source(), config_sha256=digest(config_path),
                 data_counts=manifest['counts'], carve=dict(train=len(train_rows), stop=len(stop_rows)),
                 device=str(torch.cuda.get_device_name()), precision='fp32')
        _, eval_tf = transforms_for()
        data = Images(data_root, train_rows)
        held = Images(data_root, stop_rows)
        stop = [held.batch(list(range(i, min(i + config['batch_size'], len(held.labels)))), eval_tf)
                for i in range(0, len(held.labels), config['batch_size'])]
        pool = development_images(data_root, manifest['rows'], eval_tf, config['development_batch_size'])
        val_rows = [r for r in manifest['rows'] if r['split'] == 'val']
        val_ids = [r['sample_id'] for r in val_rows]
        torch.manual_seed(seed)
        backbone = config.get('backbone', 'resnet18')
        base = make_backbone(backbone)
        initial_sha = _state_hash(base)
        log.emit('model_initialized', initial_sha256=initial_sha, architecture=backbone, model_class=type(base).__name__,
                 transform=('Yuval RGB: hflip, rotate 3, affine t0.1 s0.9-1.1, jitter 0.2, mean .6613 std .2123'
                            if config.get('augment', True) else 'Yuval RGB, no augmentation: resize 224, mean .6613 std .2123'),
                 sampler=('class-balanced with replacement' if config.get('balanced', True) else 'uniform permutation'),
                 early_stop=config.get('early_stop', True), train_label_counts=[data.labels.count(c) for c in range(5)])
        started = time.monotonic()
        C = torch.ones(5)
        retrains, final = [], {}
        max_retrains = config.get('max_retrains', MAX_RETRAINS)
        for r in range(1, max_retrains + 1):
            directory = output / f'retrain{r}'
            directory.mkdir()
            with audited_arm_log(directory / 'events.jsonl') as rlog:
                model = copy.deepcopy(base).cuda()
                if _state_hash(model) != initial_sha:
                    raise RuntimeError('retrain initialization differs')
                snaps, side_steps = {}, {}

                def snapshot(epoch, values):
                    path = directory / f'epoch{epoch:02d}.pt'
                    torch.save(values.cpu(), path)
                    snaps[epoch] = values
                    if config.get('snapshot_steps'):
                        side_steps[str(epoch)] = snapshot_steps(model, pool, caps, seed, epoch, directory)

                result = train_run(model, data, stop, pool, config, C,
                                   lambda row: rlog.emit(row['event'], **{k: v for k, v in row.items() if k != 'event'}),
                                   snapshot)
                if side_steps:
                    result['snapshot_steps'] = side_steps
                probabilities = infer(model, pool)
                if not torch.equal(probabilities, snaps[result['best_epoch']]):
                    raise RuntimeError('restored best weights do not reproduce the best-epoch pool output')
                torch.save(probabilities, directory / 'final_probabilities.pt')
                count = int((probabilities.argmax(1) == CAPPED).sum())
                f, df = f_and_derivative(cap, count, config['b'])
                row = dict(retrain=r, C=C.tolist(), hard_count=count, F=f, dF=df, **result)
                rlog.emit('training_completed', **row)
            retrains.append(row)
            final['pao'] = directory / 'final_probabilities.pt'
            if r == 1:
                final['pto'] = directory / 'final_probabilities.pt'
                pto_state, pto_sha = copy.deepcopy(model.state_dict()), _state_hash(model)
            del model
            torch.cuda.empty_cache()
            if f < 1e-5:
                break
            C = C + config['mu'] * df
            C[CAPPED] = 1
        if len({(x['first_order_sha256'], x['first_batch_sha256']) for x in retrains}) != 1:
            raise RuntimeError('retrains differ in sampler order or augmentation draws')
        steps = {}
        for arm, sham in (('tralo_final', None), ('sham_final', torch.Generator().manual_seed(seed + SHAM_OFFSET))):
            directory = output / arm
            directory.mkdir()
            model = copy.deepcopy(base).cuda()
            model.load_state_dict(pto_state)
            if _state_hash(model) != pto_sha:
                raise RuntimeError(arm + ' does not start from the PTO weights')
            steps[arm] = targeted_step(model, pool, caps, sham_generator=sham)
            torch.save(infer(model, pool), directory / 'final_probabilities.pt')
            final[arm] = directory / 'final_probabilities.pt'
            del model
            torch.cuda.empty_cache()
        if steps['tralo_final'].get('radius') != steps['sham_final'].get('radius'):
            raise RuntimeError('sham radius differs from the targeted radius')
        converged = retrains[-1]['F'] < 1e-5
        log.emit('training_completed', retrains=len(retrains), converged=converged, steps=steps,
                 seconds=time.monotonic() - started)
        labels = [r['label'] for r in val_rows]
        summary = dict(seed=seed, cap=cap, converged=converged, retrains=retrains, steps=steps, arms={})
        for arm in ('pto', 'tralo_final', 'sham_final') + (('pao',) if max_retrains > 1 else ()):
            probabilities = torch.load(final[arm], map_location='cpu', weights_only=True)
            report = evaluate_global(probabilities.tolist(), labels, caps, val_ids)
            (output / arm).mkdir(exist_ok=True)
            save(output / arm / 'report.json', report)
            summary['arms'][arm] = dict(file=str(final[arm].relative_to(output)), sha256=digest(final[arm]),
                                        scores={p: {k: d['metrics'][k] for k in ('accuracy', 'macro_f1', 'cc_f1')}
                                                for p, d in report.items()})
        for row in retrains:
            probabilities = torch.load(output / f"retrain{row['retrain']}" / 'final_probabilities.pt',
                                       map_location='cpu', weights_only=True)
            report = evaluate_global(probabilities.tolist(), labels, caps, val_ids)
            row['scores'] = {k: report['capped_first']['metrics'][k] for k in ('accuracy', 'macro_f1', 'cc_f1')}
        save(output / 'summary.json', summary)
        log.emit('completed', seconds=time.monotonic() - started)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])

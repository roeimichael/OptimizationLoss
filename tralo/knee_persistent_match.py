"""Matched, label-blind knee training under the Kassif recipe.

python -m tralo.knee_persistent_match DATA_ROOT CONFIG_JSON NEW_OUTPUT_ROOT
python -m tralo.knee_persistent_match --preflight DATA_ROOT CONFIG_JSON NEW_JSON

This runner saves predictions and diagnostics only. Development labels and all
quality metrics belong to an independent scorer after the block is complete.
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
from .knee_end_to_end import development_images, infer
from .knee_experiment import cuda_setup, digest, save, source
from .knee_yuval import (B5_SHA256, B5_WEIGHTS, CAPPED, Images, carve,
                         f_and_derivative, make_backbone as make_existing_backbone,
                         train_run, transforms_for)
from .targeted_step import targeted_step

CAPS_PILOT = (76,)
CAPS_STUDY = (54, 86)
SEEDS_STUDY = tuple(range(6701, 6713))
VIT_RECOVERY_PILOT = 6713
BACKBONES = ('efficientnet_b5', 'mobilenet_v3_large', 'vit_b_16')
WEIGHT_SHA256 = {
    'efficientnet_b5': B5_SHA256,
    'mobilenet_v3_large': '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997',
    'vit_b_16': 'c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9',
}
WEIGHT_FILE = {
    'efficientnet_b5': B5_WEIGHTS,
    'mobilenet_v3_large': 'mobilenet_v3_large-5c1a4163.pth',
    'vit_b_16': 'vit_b_16-c867db91.pth',
}
CONFIG_KEYS = {'seed', 'backbone', 'caps', 'max_epochs', 'patience', 'batch_size',
               'lr', 'weight_decay', 'decay_epoch', 'decay_factor', 'mu', 'b',
               'development_batch_size', 'max_retrains'}


def validate(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError('configuration keys differ from the fixed knee protocol')
    if type(config['seed']) is not int or config['seed'] not in (6700, VIT_RECOVERY_PILOT) + SEEDS_STUDY:
        raise ValueError('seed is outside the declared pilot and study blocks')
    if config['backbone'] not in BACKBONES:
        raise ValueError('backbone is outside the declared study')
    if config['seed'] == VIT_RECOVERY_PILOT and config['backbone'] != 'vit_b_16':
        raise ValueError('the recovery pilot is ViT only')
    expected = CAPS_PILOT if config['seed'] in (6700, VIT_RECOVERY_PILOT) else CAPS_STUDY
    if type(config['caps']) is not list or tuple(config['caps']) != expected:
        raise ValueError('training caps differ from the declared block')
    for key in ('max_epochs', 'patience', 'batch_size', 'decay_epoch',
                'development_batch_size', 'max_retrains'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    for key in ('lr', 'weight_decay', 'decay_factor', 'mu', 'b'):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid ' + key)
    if (config['max_epochs'] != 75 or config['patience'] != 5 or
            config['batch_size'] != 32 or config['lr'] != 1e-4 or
            config['weight_decay'] != 0 or config['decay_epoch'] != 5 or
            config['decay_factor'] != 0.8 or config['mu'] != 8 / 600 or
            config['b'] != 100.0 or config['development_batch_size'] != 16 or
            config['max_retrains'] != 8):
        raise ValueError('training recipe differs from the fixed comparison')


def _weights(backbone):
    import torch
    if backbone == 'efficientnet_b5':
        path = B5_WEIGHTS
    else:
        path = Path(torch.hub.get_dir()) / 'checkpoints' / WEIGHT_FILE[backbone]
    if digest(path) != WEIGHT_SHA256[backbone]:
        raise RuntimeError(backbone + ' pretrained weights differ from the pinned bytes')
    return str(path), WEIGHT_SHA256[backbone]


def make_model(backbone, pretrained=True):
    if backbone != 'vit_b_16':
        return make_existing_backbone(backbone, pretrained)
    import torch
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 5)
    return model


def disable_vit_mha_fastpath():
    """Keep ViT attention on one path for no-grad inference and grad replay."""
    import torch

    torch.backends.mha.set_fastpath_enabled(False)
    if torch.backends.mha.get_fastpath_enabled():
        raise RuntimeError('ViT MHA fastpath remained enabled')
    return False


def vit_attention_replay(model, images):
    """Check fixed-weight probability identity before the first ViT training arm."""
    import torch

    if torch.backends.mha.get_fastpath_enabled():
        raise RuntimeError('ViT MHA fastpath enabled during replay')
    no_grad = infer(model, [images])
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            grad = model(images.to(next(model.parameters()).device)).softmax(1).detach().cpu()
    finally:
        model.train(was_training)
    if grad.shape != no_grad.shape or not bool(torch.isfinite(grad).all()):
        return dict(passed=False, images_count=len(images),
                    max_absolute_difference=None, max_tolerance_ratio=None)
    difference = (grad - no_grad).abs()
    tolerance = 1e-7 + 1e-6 * no_grad.abs()
    return dict(passed=bool(torch.all(difference <= tolerance)),
                images_count=len(images),
                max_absolute_difference=float(difference.max()),
                max_tolerance_ratio=float((difference / tolerance).max()))


def _train_arm(root, name, base, data, stop, pool, config, C, after_epoch=None,
               fixed_horizon=False):
    import torch
    path = root / name
    path.mkdir()
    arm_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = copy.deepcopy(base).cuda()
    if _state_hash(model) != _state_hash(base):
        raise RuntimeError(name + ' changed the matched initialization')
    snaps = []

    def snapshot(epoch, values):
        filename = f'epoch{epoch:02d}.pt'
        torch.save(values.cpu(), path / filename)
        snaps.append(dict(epoch=epoch, file=filename, sha256=digest(path / filename)))

    with audited_arm_log(path / 'events.jsonl') as log:
        result = train_run(model, data, stop, pool, config, C,
                           lambda row: log.emit(row['event'],
                                                **{k: v for k, v in row.items() if k != 'event'}),
                           snapshot, after_epoch=after_epoch, fixed_horizon=fixed_horizon)
        probabilities = infer(model, pool)
        if not torch.equal(probabilities, torch.load(path / f"epoch{result['best_epoch']:02d}.pt",
                                                    map_location='cpu', weights_only=True)):
            raise RuntimeError(name + ' restored checkpoint does not reproduce its epoch output')
        torch.save(probabilities, path / 'final_probabilities.pt')
        torch.save(model.state_dict(), path / 'model.pt')
        save(path / 'training.json', result)
        save(path / 'snapshots.json', snaps)
        log.emit('training_completed', **result,
                 probability_sha256=digest(path / 'final_probabilities.pt'),
                 model_sha256=digest(path / 'model.pt'))
    summary = dict(file=f'{name}/final_probabilities.pt',
                   probability_sha256=digest(path / 'final_probabilities.pt'),
                   model_sha256=digest(path / 'model.pt'),
                   training_file=f'{name}/training.json',
                   seconds=time.monotonic() - arm_started,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(), **result)
    del model
    torch.cuda.empty_cache()
    return summary, probabilities


def _target_hook(pool, cap, doses):
    import torch
    caps = [None] * 5
    caps[CAPPED] = cap

    def hook(epoch, model):
        params = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        step = targeted_step(model, pool, caps)
        norms = [float((p.detach() - old).double().norm()) for p, old in zip(params, before)]
        doses.append(norms)
        if step['applied'] and step['hard_after'] > cap:
            raise RuntimeError('targeted step did not meet the hard cap')
        return dict(intervention='tralo', target_step=step,
                    tensor_displacement_norms=norms,
                    displacement_l2=math.sqrt(sum(x * x for x in norms)))

    return hook


def _null_hook(pool):
    import torch

    def hook(epoch, model):
        probabilities = infer(model, pool)
        return dict(intervention='null', null_hook=True,
                    pre_hook_hard_counts=torch.bincount(probabilities.argmax(1), minlength=5).tolist(),
                    pre_hook_soft_count_capped=float(probabilities[:, CAPPED].sum()))

    return hook


def _sham_hook(doses, seed):
    import torch
    generator = None

    def hook(epoch, model):
        nonlocal generator
        norms = doses[epoch - 1]
        params = [p for p in model.parameters() if p.requires_grad]
        if len(norms) != len(params):
            raise RuntimeError('target and sham parameter tensors differ')
        if generator is None:
            generator = torch.Generator(device=params[0].device).manual_seed(seed + 17000)
        actual = []
        with torch.no_grad():
            for p, radius in zip(params, norms):
                if radius == 0:
                    actual.append(0.0)
                    continue
                noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
                noise *= radius / float(noise.double().norm())
                old = p.detach().clone()
                p.add_(noise)
                actual.append(float((p - old).double().norm()))
        if any(not math.isfinite(x) for x in actual):
            raise RuntimeError('nonfinite sham displacement')
        if any(abs(a - b) > max(2e-6, 2e-3 * b) for a, b in zip(actual, norms)):
            raise RuntimeError('sham tensor displacement differs from target')
        return dict(intervention='sham', tensor_displacement_norms=actual,
                    target_tensor_displacement_norms=norms,
                    displacement_l2=math.sqrt(sum(x * x for x in actual)))

    return hook


def run(data_root, config_path, output):
    import torch
    config = json.loads(Path(config_path).read_text())
    validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / 'events.jsonl') as log:
        cuda_setup()
        manifest = audit(data_root, include_val_labels=False)
        if manifest['counts'] != {'train': 5778, 'val': 826, 'test': 1656}:
            raise RuntimeError('Chen split counts changed')
        if (manifest['cross_split_subject_overlap'] != 0 or
                manifest['cross_split_pixel_overlap'] != 0):
            raise RuntimeError('Chen split overlap changed')
        train_rows, stop_rows = carve(manifest['rows'])
        if not train_rows or not stop_rows:
            raise RuntimeError('empty training or stop carve')
        public = dict(manifest, rows=[{k: v for k, v in row.items()
                                      if row['split'] == 'train' or k != 'label'}
                                     for row in manifest['rows']])
        save(output / 'manifest.json', public)
        save(output / 'config.json', config)
        train_tf, eval_tf = transforms_for()
        data = Images(data_root, train_rows)
        held = Images(data_root, stop_rows)
        stop = [held.batch(list(range(i, min(i + config['batch_size'], len(held.labels)))), eval_tf)
                for i in range(0, len(held.labels), config['batch_size'])]
        pool = development_images(data_root, public['rows'], eval_tf,
                                  config['development_batch_size'])
        val_ids = [r['sample_id'] for r in public['rows'] if r['split'] == 'val']
        backbone = config['backbone']
        mha_fastpath_enabled = (disable_vit_mha_fastpath()
                                if backbone == 'vit_b_16' else None)
        weight_path, weight_sha = _weights(backbone)
        torch.manual_seed(config['seed'])
        base = make_model(backbone)
        if backbone == 'vit_b_16':
            replay = vit_attention_replay(base, pool[0][:2])
        initial_sha = _state_hash(base)
        log.emit('started', source_sha256=source(), config_sha256=digest(config_path),
                 manifest_sha256=digest(output / 'manifest.json'),
                 counts=manifest['counts'], train_carve=len(train_rows), stop_carve=len(stop_rows),
                 dev_ids_sha256=__import__('hashlib').sha256('\n'.join(val_ids).encode()).hexdigest(),
                 architecture=backbone, weight_path=weight_path, weight_sha256=weight_sha,
                 initial_sha256=initial_sha, device=torch.cuda.get_device_name(), precision='fp32',
                 mha_fastpath_enabled=mha_fastpath_enabled)
        if backbone == 'vit_b_16':
            log.emit('vit_attention_replay', **replay,
                     mha_fastpath_enabled=mha_fastpath_enabled)
            if not replay['passed']:
                raise RuntimeError('ViT attention differs between no-grad and grad replay')
        started = time.monotonic()
        ones = torch.ones(5)
        summary = dict(seed=config['seed'], backbone=backbone, caps=config['caps'],
                       source_sha256=source(), config_sha256=digest(config_path),
                       manifest_sha256=digest(output / 'manifest.json'),
                       initial_sha256=initial_sha, weights_sha256=weight_sha, arms={})
        pto, pto_prob = _train_arm(output, 'pto', base, data, stop, pool, config, ones)
        summary['arms']['pto'] = pto
        horizon = pto['epochs_run']
        fixed_config = dict(config, max_epochs=horizon)
        if config['seed'] in (6700, VIT_RECOVERY_PILOT):
            null, null_prob = _train_arm(output, 'null', base, data, stop, pool,
                                         fixed_config, ones, after_epoch=_null_hook(pool),
                                         fixed_horizon=True)
            if not torch.equal(pto_prob, null_prob):
                raise RuntimeError('PTO and phase-matched null differ')
            summary['arms']['null'] = null
        for cap in config['caps']:
            caps = [None] * 5
            caps[CAPPED] = cap
            count = int((pto_prob.argmax(1) == CAPPED).sum())
            C = torch.ones(5)
            retrains = [dict(retrain=1, hard_count=count, C=C.tolist(), arm=pto,
                             file=pto['file'], probability_sha256=pto['probability_sha256'])]
            pao = pto
            for number in range(2, config['max_retrains'] + 1):
                f, derivative = f_and_derivative(cap, count, config['b'])
                if f < 1e-5:
                    break
                C = C + config['mu'] * derivative
                C[CAPPED] = 1
                pao, probability = _train_arm(output, f'cap{cap}_pao{number}', base,
                                              data, stop, pool, config, C)
                count = int((probability.argmax(1) == CAPPED).sum())
                retrains.append(dict(retrain=number, hard_count=count, C=C.tolist(), arm=pao,
                                     file=pao['file'], probability_sha256=pao['probability_sha256']))
            doses = []
            target, _ = _train_arm(output, f'cap{cap}_tralo', base, data, stop, pool,
                                   fixed_config, ones, after_epoch=_target_hook(pool, cap, doses),
                                   fixed_horizon=True)
            if len(doses) != horizon:
                raise RuntimeError('target radius history differs from fixed task horizon')
            sham, _ = _train_arm(output, f'cap{cap}_sham', base, data, stop, pool,
                                 fixed_config, ones, after_epoch=_sham_hook(doses, config['seed'] + cap),
                                 fixed_horizon=True)
            if (target['first_order_sha256'] != pto['first_order_sha256'] or
                    target['first_batch_sha256'] != pto['first_batch_sha256'] or
                    sham['first_order_sha256'] != pto['first_order_sha256'] or
                    sham['first_batch_sha256'] != pto['first_batch_sha256']):
                raise RuntimeError('task sampler or augmentation identity differs')
            summary['arms'][f'cap{cap}'] = dict(pao=pao, pao_retrains=retrains,
                                               pao_converged=count <= cap,
                                               tralo=target, sham=sham,
                                               target_tensor_doses=doses)
            log.emit('cap_completed', cap=cap, pao_retrains=len(retrains),
                     pao_converged=count <= cap,
                     tralo_sha256=target['probability_sha256'],
                     sham_sha256=sham['probability_sha256'])
        save(output / 'summary.json', summary)
        log.emit('completed', seconds=time.monotonic() - started,
                 summary_sha256=digest(output / 'summary.json'))


def preflight(data_root, config_path, output_json):
    """Read-only data/model/source check; no GPU training or development labels."""
    import torch
    import torchvision
    config = json.loads(Path(config_path).read_text())
    validate(config)
    manifest = audit(data_root, include_val_labels=False)
    if (manifest['counts'] != {'train': 5778, 'val': 826, 'test': 1656} or
            manifest['cross_split_subject_overlap'] or manifest['cross_split_pixel_overlap'] or
            any('label' in row for row in manifest['rows'] if row['split'] != 'train')):
        raise RuntimeError('Chen data boundary differs from the fixed protocol')
    train_rows, stop_rows = carve(manifest['rows'])
    if not train_rows or not stop_rows:
        raise RuntimeError('empty subject-stable stopping carve')
    weight_path, weight_sha = _weights(config['backbone'])
    mha_fastpath_enabled = (disable_vit_mha_fastpath()
                            if config['backbone'] == 'vit_b_16' else None)
    torch.manual_seed(config['seed'])
    model = make_model(config['backbone'])
    initial_sha = _state_hash(model)
    del model
    identity = hashlib.sha256(json.dumps(
        [(r['split'], r['sample_id'], r['sha256'], r['pixel_sha256']) for r in manifest['rows']],
        separators=(',', ':')).encode()).hexdigest()
    receipt = dict(scope='knee_persistent_match_label_blind_preflight',
                   config_sha256=digest(config_path), source_sha256=source(),
                   data_identity_sha256=identity, counts=manifest['counts'],
                   train_carve=len(train_rows), stop_carve=len(stop_rows),
                   cross_split_subject_overlap=0, cross_split_pixel_overlap=0,
                   backbone=config['backbone'], seed=config['seed'],
                   pretrained_weight_path=weight_path, pretrained_weight_sha256=weight_sha,
                   initialized_model_sha256=initial_sha,
                   torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                   precision='fp32', mha_fastpath_enabled=mha_fastpath_enabled)
    with Path(output_json).open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--preflight':
        preflight(*sys.argv[2:])
        raise SystemExit(0)
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])

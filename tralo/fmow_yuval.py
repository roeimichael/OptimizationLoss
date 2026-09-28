"""fmow2 satellite images under Yuval Kassif's training pipeline: the step-ensemble study on a second dataset.

Usage: python -m tralo.fmow_yuval DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY

Protocol: experiments/claude_fmow_stepens_prereg_20260928.md.

DATA_ROOT holds fmow2's arrays, checked against the sha256 values below: 224x224 RGB uint8 images in 8 classes, with
17670 train items from 139 countries and 3442 test items from 10 other countries. Every role is decided by country and
never by label:
  * early stopping reads the train items of the 14 countries whose sha256 is 0 mod 10, the way tralo.knee_yuval
    carves subjects;
  * the development pool is the test items of the 5 test countries with the smallest sha256 (IRQ, NLD, DZA, PHL, TUR:
    1673 items). The step reads their images only; their labels are written to manifest.json for the offline scorer
    and the per-arm reports, as in the knee runner;
  * the other 5 test countries (EGY, CAN, IND, MEX, JPN) are reserved: no image or label of theirs enters training, a
    step, a checkpoint choice or a score. Their files are only hashed.
Capped class 1 (crop_field, the corpus's first capped class) at cap = pool size // cap_divisor, deployed with
capped_first. The pipeline is tralo.knee_yuval.train_run: augmentation, the class-balanced sampler, Adam 1e-4 with
weight decay 1e-4, x0.8 every 5 epochs, patience 5 and the best weights restored. ImageNet normalisation replaces the
knee's mean and std. PTO is trained once. With snapshot_steps, TraLO's targeted step and its sham act on side copies at
every epoch, as in the knee step-ensemble studies. The backbone is torchvision mobilenet_v3_large, ImageNet weights,
with a fresh 8-way head.
Arms: pto (the restored best); tralo_final and sham_final, one step or sham from pto; and the epoch snapshots
epochNN.pt, epochNN_tralo.pt and epochNN_sham.pt that analysis/score_fmow_stepens.py ensembles.
The pilot seed may run with snapshot_steps false: that reference run gates the pilot on the same host.
"""

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .global_comparison import _state_hash, audited_arm_log
from .knee_e2e_v3 import build_model
from .knee_end_to_end import infer
from .knee_experiment import cuda_setup, digest, save, source
from .knee_yuval import SHAM_OFFSET, Images, snapshot_steps, train_run
from .targeted_step import targeted_step

CLASSES = 8
CAPPED = 1
FILES = {'train_images.npy': 'f62e48c2c763d52dd9b63b179938377ebc0062ff052a747f44538f56ff868db8',
         'train_labels.npy': '61aa4855ad7b05dc05167a7fc5779fc13e59048f262be51ee4154df10ebda93d',
         'train_meta.csv': 'c2c9462630dc174e7984fc4fe04a0b41bf1b7ff8970fa678c542f2a272715e4a',
         'test_images.npy': 'b363f1529663efc212854762ec1b9d9244414ce2ff5929f56d98d3ec22281ac5',
         'test_labels.npy': '75eaccbebe2b192d71dae9ed5207ded47fc8a8f23bc3766f1b9465ca83d788a6',
         'test_meta.csv': '20d97b086cde8e85d8dad79c93fa06cd8c2781f892ab6055d9193704a3ad2b63'}
COUNTS = dict(train=17670, test=3442)
SEEDS = tuple(range(5000, 5048))
PILOT = 5099
KEYS = {'seed', 'backbone', 'capped_class', 'cap_divisor', 'max_epochs', 'patience', 'batch_size', 'lr',
        'weight_decay', 'decay_epoch', 'decay_factor', 'development_batch_size', 'snapshot_steps'}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def validate(config):
    if set(config) != KEYS:
        raise ValueError('config keys differ from the declared experiment')
    if config['seed'] not in SEEDS + (PILOT,):
        raise ValueError('seed is outside the preregistered block 5000-5047 and pilot 5099')
    if not (config['snapshot_steps'] is True or config['snapshot_steps'] is False and config['seed'] == PILOT):
        raise ValueError('the study steps every snapshot; only the pilot seed may run a reference with the steps off')
    if config['backbone'] != 'mobilenet_v3_large' or config['capped_class'] != CAPPED:
        raise ValueError('the preregistered design is mobilenet_v3_large with class 1 capped')
    if type(config['cap_divisor']) is not int or config['cap_divisor'] not in (10, 20):
        raise ValueError('the cap is the pool size // 10, or // 20 under the preregistered fallback')
    for key in ('max_epochs', 'patience', 'batch_size', 'decay_epoch', 'development_batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    for key in ('lr', 'weight_decay', 'decay_factor'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('invalid ' + key)


def transforms_for():
    """tralo.knee_yuval.transforms_for with ImageNet normalisation for RGB satellite images."""
    from torchvision import transforms as T
    normalize = T.Normalize(mean=MEAN, std=STD)
    train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=0.5), T.RandomRotation(3),
                       T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                       T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), T.ToTensor(), normalize])
    evaluation = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])
    return train, evaluation


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def roles(train_meta, test_meta):
    """Indices by country: train / early-stopping carve of the train split, development pool / reserved of the test split."""
    held = lambda c: int(_hash(c), 16) % 10 == 0
    train = [i for i, r in enumerate(train_meta) if not held(r['location'])]
    stop = [i for i, r in enumerate(train_meta) if held(r['location'])]
    countries = sorted({r['location'] for r in test_meta}, key=_hash)
    dev_countries, reserved = countries[:len(countries) // 2], countries[len(countries) // 2:]
    dev = [i for i, r in enumerate(test_meta) if r['location'] in dev_countries]
    if {train_meta[i]['location'] for i in train} & {train_meta[i]['location'] for i in stop}:
        raise RuntimeError('a country straddles the early-stopping carve')
    if {r['location'] for r in train_meta} & set(countries):
        raise RuntimeError('a country is in both train and test')
    return dict(train=train, stop=stop, dev=dev, stop_countries=sorted({train_meta[i]['location'] for i in stop}),
                dev_countries=dev_countries, reserved_countries=reserved)


def load(root, include_pool_labels=True):
    """Hash-checked arrays and roles; optionally keep pool labels for offline scoring."""
    import numpy as np
    root = Path(root)
    for name, sha in FILES.items():
        if digest(root / name) != sha:
            raise RuntimeError(name + ' differs from the preregistered file')
    with (root / 'train_meta.csv').open(newline='') as handle:
        train_meta = list(csv.DictReader(handle))
    if include_pool_labels:
        with (root / 'test_meta.csv').open(newline='') as handle:
            test_meta = list(csv.DictReader(handle))
    else:
        # Keep only country IDs in the training runner. The label-bearing CSV
        # is hash-checked above but no label field is retained or indexed.
        with (root / 'test_meta.csv').open(newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            location_col = header.index('location')
            test_meta = [{'location': row[location_col]} for row in reader]
    meta = {'train': train_meta, 'test': test_meta}
    images = {s: np.load(root / f'{s}_images.npy', mmap_mode='r') for s in ('train', 'test')}
    train_labels = np.load(root / 'train_labels.npy')
    for s in ('train', 'test'):
        if images[s].shape != (COUNTS[s], 224, 224, 3) or images[s].dtype != np.uint8 or len(meta[s]) != COUNTS[s]:
            raise RuntimeError(f'unexpected {s} arrays')
    if [int(r['label']) for r in meta['train']] != train_labels.tolist() or set(train_labels.tolist()) != set(range(CLASSES)):
        raise RuntimeError('train labels disagree with the metadata or miss a class')
    r = roles(meta['train'], meta['test'])
    pool_rows = [dict(split='val', sample_id=f'test{i}', location=meta['test'][i]['location'])
                 for i in r['dev']]
    if include_pool_labels:
        test_labels = np.load(root / 'test_labels.npy', mmap_mode='r')
        for row, index in zip(pool_rows, r['dev']):
            row['label'] = int(test_labels[index])
        del test_labels
    return images, train_labels, pool_rows, r


class ArrayImages(Images):
    """Rows of a memory-mapped uint8 image array with their training labels; the transform runs at every access."""

    def __init__(self, array, indices, labels):
        self.array, self.indices = array, list(indices)
        self.labels = [int(labels[i]) for i in self.indices]

    def batch(self, positions, transform):
        import torch
        from PIL import Image
        return (torch.stack([transform(Image.fromarray(self.array[self.indices[p]])) for p in positions]),
                torch.tensor([self.labels[p] for p in positions], dtype=torch.long))


def pool_chunks(array, indices, transform, batch_size):
    """The development pool as image-only chunks, in pool order."""
    import torch
    from PIL import Image
    return [torch.stack([transform(Image.fromarray(array[i])) for i in indices[s:s + batch_size]])
            for s in range(0, len(indices), batch_size)]


def make_model(pretrained=True):
    import torch
    model = build_model('mobilenet_v3_large', pretrained)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, CLASSES)
    return model


def run(data_root, config_path, output):
    import torch
    from .global_report import evaluate_global
    config = json.loads(Path(config_path).read_text())
    validate(config)
    seed = config['seed']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / 'events.jsonl') as log:
        cuda_setup()
        images, train_labels, pool_rows, r = load(data_root)
        cap = len(r['dev']) // config['cap_divisor']
        caps = [None] * CLASSES
        caps[CAPPED] = cap
        counts = dict(train=len(r['train']), stop=len(r['stop']), dev=len(r['dev']))
        save(output / 'manifest.json', dict(files=FILES, counts=counts, cap=cap, capped_class=CAPPED,
                                            stop_countries=r['stop_countries'], dev_countries=r['dev_countries'],
                                            reserved_countries=r['reserved_countries'], rows=pool_rows))
        save(output / 'config.json', config)
        log.emit('started', source_sha256=source(), config_sha256=digest(config_path), counts=counts, cap=cap,
                 device=str(torch.cuda.get_device_name()), precision='fp32')
        train_tf, eval_tf = transforms_for()
        data = ArrayImages(images['train'], r['train'], train_labels)
        held = ArrayImages(images['train'], r['stop'], train_labels)
        bs = config['batch_size']
        stop = [held.batch(list(range(i, min(i + bs, len(held.labels)))), eval_tf) for i in range(0, len(held.labels), bs)]
        pool = pool_chunks(images['test'], r['dev'], eval_tf, config['development_batch_size'])
        torch.manual_seed(seed)
        base = make_model()
        initial_sha = _state_hash(base)
        log.emit('model_initialized', initial_sha256=initial_sha, architecture='mobilenet_v3_large',
                 model_class=type(base).__name__, classes=CLASSES, transform='Yuval RGB with ImageNet mean and std',
                 sampler='class-balanced with replacement', early_stop=True,
                 train_label_counts=[data.labels.count(c) for c in range(CLASSES)])
        started = time.monotonic()
        C = torch.ones(CLASSES)
        directory = output / 'retrain1'
        directory.mkdir()
        with audited_arm_log(directory / 'events.jsonl') as rlog:
            model = copy.deepcopy(base).cuda()
            snaps, side_steps = {}, {}

            def snapshot(epoch, values):
                torch.save(values.cpu(), directory / f'epoch{epoch:02d}.pt')
                snaps[epoch] = values
                if config['snapshot_steps']:
                    side_steps[str(epoch)] = snapshot_steps(model, pool, caps, seed, epoch, directory)

            result = train_run(model, data, stop, pool, config, C,
                               lambda row: rlog.emit(row['event'], **{k: v for k, v in row.items() if k != 'event'}),
                               snapshot, capped=CAPPED, transforms=(train_tf, eval_tf))
            if side_steps:
                result['snapshot_steps'] = side_steps
            probabilities = infer(model, pool)
            if not torch.equal(probabilities, snaps[result['best_epoch']]):
                raise RuntimeError('restored best weights do not reproduce the best-epoch pool output')
            torch.save(probabilities, directory / 'final_probabilities.pt')
            row = dict(retrain=1, hard_count=int((probabilities.argmax(1) == CAPPED).sum()), **result)
            rlog.emit('training_completed', **row)
        final = dict(pto=directory / 'final_probabilities.pt')
        pto_state, pto_sha = copy.deepcopy(model.state_dict()), _state_hash(model)
        del model
        torch.cuda.empty_cache()
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
        log.emit('training_completed', retrains=1, steps=steps, seconds=time.monotonic() - started)
        labels, ids = [p['label'] for p in pool_rows], [p['sample_id'] for p in pool_rows]
        summary = dict(seed=seed, cap=cap, capped_class=CAPPED, retrains=[row], steps=steps, arms={})
        for arm in ('pto', 'tralo_final', 'sham_final'):
            values = torch.load(final[arm], map_location='cpu', weights_only=True)
            report = evaluate_global(values.tolist(), labels, caps, ids)
            (output / arm).mkdir(exist_ok=True)
            save(output / arm / 'report.json', report)
            summary['arms'][arm] = dict(file=str(final[arm].relative_to(output)), sha256=digest(final[arm]),
                                        scores={p: {k: d['metrics'][k] for k in ('accuracy', 'macro_f1', 'cc_f1')}
                                                for p, d in report.items()})
        save(output / 'summary.json', summary)
        log.emit('completed', seconds=time.monotonic() - started)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])

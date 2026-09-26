"""Dose-response probe: how much WHO information does the constraint direction carry, by push depth?

Usage: python -m tralo.step_probe DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY

Motivation (analysis/washout.py, exploratory): at cap 50 the FIRST targeted step, from a state
identical to the null's, adds +1.12 correct capped slots [+0.62, +1.63] (n = 24), and the next
CE epoch erases it. This probe removes the wash-out and measures the step alone.

Per seed: train the tralo_null schedule for warm-up + 1 epoch (state S = the epoch-6 pre-step
state of every arm). From copies of S, apply targeted_step to hard grade-3 targets
t = round(f x cap) for f in FRACTIONS, along TraLO's direction and along a same-radius sham
(seeded, per f). Save the development probabilities of S and of every stepped copy. No training
follows. Scoring (offline, development labels) is in analysis/score_step_probe.py.
Development labels are not read here.
"""

import copy
import json
from pathlib import Path
import sys

from .knee_data import audit
from .knee_e2e_v3 import train_one
from .knee_end_to_end import development_images, image_transform, infer
from .knee_experiment import cuda_setup, save
from .targeted_step import targeted_step

FRACTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)
SEEDS = tuple(range(2601, 2625))


def run(data_root, config_path, output):
    import torch
    from torchvision import models
    from .supervised_adaptation import TrainingImages
    config = json.loads(Path(config_path).read_text())
    if config['seed'] not in SEEDS:
        raise ValueError('the probe runs on its preregistered seeds 2601-2624 only')
    cap = config['caps'][3]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cuda_setup()
    manifest = audit(data_root)
    save(output / 'manifest.json', manifest)
    save(output / 'config.json', config)
    transform = image_transform()
    train_images = TrainingImages(data_root, manifest['rows'], transform)
    val_chunks = development_images(data_root, manifest['rows'], transform, config['development_batch_size'])
    torch.manual_seed(config['seed'])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model = model.cuda()
    short = dict(config, epochs=config['warmup_epochs'] + 1)
    train_one(model, train_images, val_chunks, short, 'tralo_null', lambda row: None, lambda *a: None)
    base = infer(model, val_chunks)
    torch.save(base, output / 'state_S.pt')
    rows = []
    for k, f in enumerate(FRACTIONS):
        target = max(1, round(f * cap))
        caps = [None, None, None, target, None]
        for kind in ('tralo', 'sham'):
            m = copy.deepcopy(model)
            gen = torch.Generator().manual_seed(config['seed'] * 100 + k) if kind == 'sham' else None
            out = targeted_step(m, val_chunks, caps, sham_generator=gen)
            probs = infer(m, val_chunks)
            name = '%s_f%.1f.pt' % (kind, f)
            torch.save(probs, output / name)
            rows.append(dict(kind=kind, fraction=f, target=target, file=name,
                             **{key: out.get(key) for key in ('applied', 'hard_before', 'hard_after', 'radius', 'displacement')}))
            print(json.dumps(rows[-1]), flush=True)
            del m
            torch.cuda.empty_cache()
    save(output / 'probe.json', dict(cap=cap, rows=rows))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])

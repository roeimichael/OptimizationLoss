"""Liveness gate for CUTPAIR, run BEFORE any CUTPAIR code is trusted with a study seed.

Usage: python -m tralo.cutpair_gate DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY

CUTPAIR (a supervised hinge on the grade-3 log-odds of TRAIN items, anchored at the score
where capped_first cuts the unlabeled development pool) can only act on training items
whose log-odds sit near that cut. The knee training set is memorised by epoch 5 (training
CE ~0.15, then ~0.06), so this may be nearly empty; the earlier train-top-K ranking loss
went silent for exactly that reason (experiments/knee_hard_pair_protocol_20260924.md).

This trains the tralo_null schedule (CE, task-Adam reset at the boundary) on a NON-study
seed and, after each post-warm-up epoch, measures on an eval-mode training bank:
  tau    = logit of the cap-th largest development p3 (the capped_first boundary)
  N_act  = training non-grade-3 items with s > tau - m          (m = 1)
  P_act  = training grade-3 items with tau - W < s < tau + m     (W = 3)
for caps 50 and 76. Preregistered kill rule (claude_cutpair_gate_20260926.md): CUTPAIR is
dead on arrival at a cap if the mean over epochs 6-10 of N_act or of P_act is below 20.
Development labels are not read.
"""

import json
import math
from pathlib import Path
import sys

from .knee_data import audit
from .knee_e2e_v3 import train_one
from .knee_end_to_end import development_images, image_transform, infer
from .knee_experiment import cuda_setup, save

CAPS, MARGIN, WINDOW = (50, 76), 1.0, 3.0


def training_log_odds(model, train_images, batch_size=128):
    import torch
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    scores, labels = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(train_images), batch_size):
                rows = [train_images[i] for i in range(start, min(start + batch_size, len(train_images)))]
                z = model(torch.stack([r[0] for r in rows]).to(device)).double()
                other = torch.cat((z[:, :3], z[:, 4:]), 1)
                scores.append((z[:, 3] - torch.logsumexp(other, 1)).cpu())
                labels.extend(r[1] for r in rows)
    finally:
        model.train(was_training)
    return torch.cat(scores), torch.tensor(labels)


def gate_counts(dev_p3, train_s, train_y, cap):
    import torch
    p = float(torch.sort(dev_p3.double(), descending=True).values[cap - 1])
    p = min(max(p, 1e-12), 1 - 1e-12)
    tau = math.log(p / (1 - p))
    pos, neg = train_y == 3, train_y != 3
    return dict(cap=cap, tau=tau,
                n_act=int((neg & (train_s > tau - MARGIN)).sum()),
                p_act=int((pos & (train_s > tau - WINDOW) & (train_s < tau + MARGIN)).sum()),
                positives_above=int((pos & (train_s >= tau + MARGIN)).sum()),
                positives_far_below=int((pos & (train_s <= tau - WINDOW)).sum()),
                negatives_far_below=int((neg & (train_s <= tau - MARGIN)).sum()))


def run(data_root, config_path, output):
    import torch
    from torchvision import models
    from .supervised_adaptation import TrainingImages
    config = json.loads(Path(config_path).read_text())
    if config['seed'] != 2000:
        raise ValueError('the gate runs on the non-study seed 2000 only')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cuda_setup()
    manifest = audit(data_root)
    transform = image_transform()
    train_images = TrainingImages(data_root, manifest['rows'], transform)
    val_chunks = development_images(data_root, manifest['rows'], transform, config['development_batch_size'])
    torch.manual_seed(config['seed'])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model = model.cuda()
    rows = []

    def snapshot(epoch, phase, values):
        if phase != 'before_constraint':
            return
        s, y = training_log_odds(model, train_images)
        row = dict(epoch=epoch, train_accuracy_proxy=float(((s > 0) == (y == 3)).double().mean()),
                   counts=[gate_counts(values[:, 3], s, y, cap) for cap in CAPS])
        rows.append(row)
        print(json.dumps(row), flush=True)

    train_one(model, train_images, val_chunks, config, 'tralo_null', lambda row: None, snapshot)
    verdict = {}
    for k, cap in enumerate(CAPS):
        n = sum(r['counts'][k]['n_act'] for r in rows) / len(rows)
        p = sum(r['counts'][k]['p_act'] for r in rows) / len(rows)
        verdict[str(cap)] = dict(mean_n_act=n, mean_p_act=p, alive=bool(n >= 20 and p >= 20))
    save(output / 'gate.json', dict(epochs=rows, verdict=verdict))
    print(json.dumps(verdict), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])

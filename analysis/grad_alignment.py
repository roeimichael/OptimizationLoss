"""Mechanism diagnostic on a REAL knee model: are the grade-3 gradients of right and wrong items aligned?

Usage (server, from a lab directory): PYTHONPATH=<release> python grad_alignment.py DATA_ROOT CONFIG_JSON STUDY_SEED_DIR OUT_JSON

Re-trains retrain 1 (Yuval's PTO) of one tralo.knee_yuval seed with the release's own code, checks the
development probabilities are byte-identical to the study's (training is bit-deterministic), then takes
parameter gradients of the grade-3 soft count over item groups of the development pool, in eval mode.
Labels are read AFTER training, for this offline diagnostic only.
The synthetic lab (analysis/lab_pao/fp_lab_out.txt) says a count-sized step cannot separate right from
wrong items because these gradients share one dominant component; this measures that on a real model.
"""

import json
import math
from pathlib import Path
import sys

import torch

from tralo.knee_data import audit
from tralo.knee_e2e_v3 import build_model
from tralo.knee_end_to_end import development_images, infer
from tralo.knee_experiment import cuda_setup
from tralo.knee_yuval import CAPPED, Images, carve, train_run, transforms_for


def group_gradient(model, pool, mask):
    """Flat gradient of the sum of p3 over the masked pool items (eval mode, chunked)."""
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    model.zero_grad(set_to_none=True)
    start = 0
    device = params[0].device
    for chunk in pool:
        m = mask[start:start + len(chunk)]
        start += len(chunk)
        if m.any():
            model(chunk[m].to(device)).softmax(1)[:, CAPPED].sum().backward()
    return torch.cat([p.grad.flatten() for p in params]).double(), [p.grad.numel() for p in params]


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def main(data_root, config_path, seed_dir, out_path):
    cuda_setup()
    config = json.loads(Path(config_path).read_text())
    manifest = audit(data_root)
    train_rows, stop_rows = carve(manifest['rows'])
    _, eval_tf = transforms_for()
    data, held = Images(data_root, train_rows), Images(data_root, stop_rows)
    stop = [held.batch(list(range(i, min(i + config['batch_size'], len(held.labels)))), eval_tf)
            for i in range(0, len(held.labels), config['batch_size'])]
    pool = development_images(data_root, manifest['rows'], eval_tf, config['development_batch_size'])
    torch.manual_seed(config['seed'])
    model = build_model('resnet18').cuda()
    result = train_run(model, data, stop, pool, config, torch.ones(5), lambda row: None, lambda e, v: None)
    probs = infer(model, pool)
    study = torch.load(Path(seed_dir) / 'retrain1' / 'final_probabilities.pt', weights_only=True)
    identical = bool(torch.equal(probs, study))
    labels = torch.tensor([r['label'] for r in manifest['rows'] if r['split'] == 'val'])
    cap = config['cap']
    top = torch.zeros(len(labels), dtype=torch.bool)
    top[torch.argsort(-probs[:, CAPPED], stable=True)[:cap]] = True
    groups = {'all': torch.ones(len(labels), dtype=torch.bool), 'true3': labels == CAPPED, 'not3': labels != CAPPED,
              'top_tp': top & (labels == CAPPED), 'top_fp': top & (labels != CAPPED)}
    grads, sizes = {}, None
    for name, mask in groups.items():
        grads[name], sizes = group_gradient(model, pool, mask)
    head = sum(sizes[-2:])                  # fc.weight + fc.bias are the last two parameter tensors
    n = {k: int(v.sum()) for k, v in groups.items()}
    per_item = {k: grads[k] / n[k] for k in ('true3', 'not3', 'top_tp', 'top_fp')}
    out = dict(seed=config['seed'], identical_to_study=identical, best_epoch=result['best_epoch'], counts=n,
               cos_true3_not3=cos(grads['true3'], grads['not3']),
               cos_top_tp_top_fp=cos(grads['top_tp'], grads['top_fp']),
               cos_all_true3=cos(grads['all'], grads['true3']), cos_all_not3=cos(grads['all'], grads['not3']),
               discriminative_share_top=float((per_item['top_fp'] - per_item['top_tp']).norm()
                                              / (0.5 * (per_item['top_fp'] + per_item['top_tp'])).norm()),
               head_share_of_count_gradient=float(grads['all'][-head:].norm() / grads['all'].norm()),
               cos_head_true3_not3=cos(grads['true3'][-head:], grads['not3'][-head:]),
               cos_backbone_true3_not3=cos(grads['true3'][:-head], grads['not3'][:-head]))
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

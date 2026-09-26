"""Augmentation-view consistency on unlabeled development images near the cut (BANDCONS).

Why: a count constraint on development probabilities only reproduces the post-hoc
allocator -- capped_first already gives the capped class to the top-`cap` items by p3.
A training-time effect needs information the allocator lacks. Disagreement between a weak
and a strong view of the SAME unlabeled image is such information, and it matters only for
items whose slot can change: those ranked near the cut. The consistency target is the
weak view's grade-3 log-odds, detached, so the term pulls the strong view onto it and never
moves the target itself.

Every random draw takes an explicit CPU torch.Generator and augmentation runs on CPU, so a
view is a pure function of (images, generator state). Parameters are sampled first and then
applied, so the exact draws can be hashed (the TTA draws must be the same in every arm).
No development label is read here.
"""

import hashlib
import json
import math

import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def log_odds(logits, c=3):
    """z_c - logsumexp(z_k, k != c): the class-c log-odds, invariant to a shared logit shift."""
    others = torch.cat([logits[:, :c], logits[:, c + 1:]], 1)
    return logits[:, c] - torch.logsumexp(others, 1)


def band_indices(p3, center_rank, w):
    """Items at 1-based ranks center_rank-w+1 .. center_rank+w of p3 descending, ties by index."""
    n = len(p3)
    if w <= 0 or not 1 <= center_rank <= n:
        raise ValueError('band needs w > 0 and a centre rank inside the cohort')
    order = torch.sort(-p3.detach().cpu().double(), stable=True).indices
    return order[max(center_rank - w, 0):min(center_rank + w, n)]


def natural_center(probabilities, w, c=3):
    """The argmax count of class c on the same probabilities, clipped to [w, n-w]."""
    n = len(probabilities)
    count = int((probabilities.argmax(1) == c).sum())
    return min(max(count, w), n - w)


def random_band(p3, cap, w, generator):
    """2w items drawn uniformly from those NOT in the cap band (the location-free control)."""
    excluded = torch.zeros(len(p3), dtype=torch.bool)
    excluded[band_indices(p3, cap, w)] = True
    candidates = torch.nonzero(~excluded).flatten()
    if len(candidates) < 2 * w:
        raise ValueError('too few items outside the cap band')
    return candidates[torch.randperm(len(candidates), generator=generator)[:2 * w]]


def gather(chunks, indices):
    """Rows `indices` of the concatenated chunks, without materialising the whole cohort."""
    offsets, total = [], 0
    for chunk in chunks:
        offsets.append(total)
        total += len(chunk)
    rows = []
    for i in (int(i) for i in indices):
        if not 0 <= i < total:
            raise ValueError('development index out of range')
        k = max(j for j, o in enumerate(offsets) if o <= i)
        rows.append(chunks[k][i - offsets[k]])
    return torch.stack(rows)


def _uniform(generator, low, high):
    return low + (high - low) * float(torch.rand(1, generator=generator, dtype=torch.float64))


def _flip(generator):
    return bool(torch.rand(1, generator=generator) < 0.5)


def weak_parameters(n, generator):
    return [dict(flip=_flip(generator)) for _ in range(n)]


def _crop(height, width, generator, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)):
    """RandomResizedCrop.get_params, drawing from the explicit generator."""
    area = height * width
    for _ in range(10):
        target = area * _uniform(generator, *scale)
        aspect = math.exp(_uniform(generator, math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(target * aspect)))
        h = int(round(math.sqrt(target / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(torch.randint(0, height - h + 1, (1,), generator=generator))
            left = int(torch.randint(0, width - w + 1, (1,), generator=generator))
            return [top, left, h, w]
    return [0, 0, height, width]


def strong_parameters(n, height, width, generator):
    out = []
    for _ in range(n):
        flip = _flip(generator)
        crop = _crop(height, width, generator)
        out.append(dict(flip=flip, crop=crop, angle=_uniform(generator, -10.0, 10.0),
                        brightness=_uniform(generator, 0.8, 1.2), contrast=_uniform(generator, 0.8, 1.2)))
    return out


def parameters_sha256(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def apply_weak(images, parameters):
    """A horizontal flip commutes with per-channel normalisation, so it runs in normalised space."""
    return torch.stack([x.flip(-1) if p['flip'] else x for x, p in zip(images, parameters)])


def apply_strong(images, parameters):
    """Geometry and photometric jitter in [0, 1] pixel space, resized back to the input size."""
    from torchvision.transforms.v2 import InterpolationMode, functional as F
    mean = torch.tensor(MEAN, dtype=images.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=images.dtype).view(3, 1, 1)
    size = list(images.shape[-2:])
    out = []
    for x, p in zip(images.cpu(), parameters):
        x = x * std + mean
        if p['flip']:
            x = x.flip(-1)
        x = F.resized_crop(x, *p['crop'], size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = F.rotate(x, p['angle'], interpolation=InterpolationMode.BILINEAR)
        x = F.adjust_brightness(x, p['brightness'])
        x = F.adjust_contrast(x, p['contrast'])
        out.append((x.clamp(0.0, 1.0) - mean) / std)
    return torch.stack(out)


def weak_view(images, generator):
    return apply_weak(images.cpu(), weak_parameters(len(images), generator))


def strong_view(images, generator):
    return apply_strong(images.cpu(), strong_parameters(len(images), *images.shape[-2:], generator))


def _view_scores(model, images, generator):
    """Weak and strong log-odds from ONE eval-mode forward; BatchNorm statistics stay put."""
    weak = weak_view(images, generator)
    strong = strong_view(images, generator)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        scores = log_odds(model(torch.cat([weak, strong]).to(device)))
    finally:
        model.train(was_training)
    return scores[:len(images)], scores[len(images):]


def consistency_loss(model, images, generator):
    s_w, s_s = _view_scores(model, images, generator)
    return torch.nn.functional.smooth_l1_loss(s_s, s_w.detach(), beta=1.0)


def view_disagreement(model, images, generator):
    """Mean |s_s - s_w| without a gradient (logging only)."""
    with torch.no_grad():
        s_w, s_s = _view_scores(model, images, generator)
    return float((s_s - s_w).abs().mean())


def tta_parameters(n, height, width, seed, draws=8):
    generator = torch.Generator().manual_seed(seed)
    return [strong_parameters(n, height, width, generator) for _ in range(draws)]


def tta_probabilities(model, chunks, seed, draws=8):
    """Mean softmax over `draws` strong views; the draws depend only on seed and cohort shape."""
    n = sum(len(x) for x in chunks)
    parameters = tta_parameters(n, *chunks[0].shape[-2:], seed, draws)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        total = None
        with torch.no_grad():
            for draw in parameters:
                start, values = 0, []
                for x in chunks:
                    view = apply_strong(x, draw[start:start + len(x)])
                    values.append(model(view.to(device)).softmax(1).cpu())
                    start += len(x)
                values = torch.cat(values)
                total = values if total is None else total + values
    finally:
        model.train(was_training)
    total = total / draws
    if not bool(torch.isfinite(total).all()):
        raise RuntimeError('nonfinite TTA probabilities')
    return total, parameters_sha256(parameters)

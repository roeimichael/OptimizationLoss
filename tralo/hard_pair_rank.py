"""Training-label rank pressure from weak positives and hard negatives.

Unlike a wrong-occupant-only loss, this remains active when the training top-K
quota is pure. Selection is detached at the fixed pre-update model state.
"""

import torch


def _pairs(logits, training_labels, sample_ids, capacity):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            logits.shape[1] != 5 or not torch.is_floating_point(logits) or
            len(logits) < 2 or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite five-class logit matrix required')
    n = len(logits)
    if type(capacity) is not int or not 0 < capacity < n:
        raise ValueError('capacity must be strictly inside the cohort')
    if (len(training_labels) != n or
            any(type(y) is not int or not 0 <= y < 5 for y in training_labels)):
        raise ValueError('training labels must match the cohort')
    if (len(sample_ids) != n or len(set(sample_ids)) != n or
            any(type(value) is not str or not value for value in sample_ids)):
        raise ValueError('unique nonempty training IDs required')
    other = torch.cat((logits[:, :3], logits[:, 4:]), dim=1)
    margins = logits[:, 3] - torch.logsumexp(other, dim=1)
    detached = margins.detach().cpu().tolist()
    positives = [i for i, label in enumerate(training_labels) if label == 3]
    negatives = [i for i, label in enumerate(training_labels) if label != 3]
    size = min(capacity, len(positives), len(negatives))
    weak = sorted(positives, key=lambda i: (detached[i], sample_ids[i]))[:size]
    hard = sorted(negatives, key=lambda i: (-detached[i], sample_ids[i]))[:size]
    details = dict(weak_positive_ids=[sample_ids[i] for i in weak],
                   hard_negative_ids=[sample_ids[i] for i in hard],
                   active_pairs=size * size)
    return margins, other, weak, hard, details


def hard_pair_rank_loss(logits, training_labels, sample_ids, capacity):
    margins, _, weak, hard, details = _pairs(logits, training_labels, sample_ids, capacity)
    if not weak:
        return logits.sum() * 0., details
    differences = margins[hard][None, :] - margins[weak][:, None]
    return torch.nn.functional.softplus(differences).mean(), details


def hard_pair_rank_gradient(logits, training_labels, sample_ids, capacity):
    margins, other, weak, hard, details = _pairs(logits, training_labels,
                                                 sample_ids, capacity)
    gradient = torch.zeros_like(logits)
    if not weak:
        return gradient, details
    derivatives = torch.sigmoid(margins[hard][None, :] - margins[weak][:, None])
    derivatives = derivatives / (len(weak) * len(hard))
    dm = torch.zeros_like(margins)
    dm[weak] = -derivatives.sum(dim=1)
    dm[hard] = derivatives.sum(dim=0)
    gradient[:, 3] = dm
    alternative = -dm[:, None] * other.softmax(dim=1)
    gradient[:, :3] = alternative[:, :3]
    gradient[:, 4:] = alternative[:, 3:]
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('nonfinite hard-pair gradient')
    return gradient, details

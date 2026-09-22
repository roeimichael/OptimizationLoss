"""Optional supervised terms. Labels here must belong to the TRAINING batch."""
import math


def sample_loss(logits, training_labels, caps, kind, margin=1.0):
    """Margin separation or false-positive suppression; neither replaces CE.

    margin: mean relu(m + max_wrong_logit - true_logit).
    false_positive: mean -log(1-p_c) over pairs (i,c) with capped c != y_i.
    The second term may trade recall for precision. It is not a quota guarantee.
    """
    import torch
    from .global_constraint import _validate_caps
    if (logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2 or
            not torch.is_floating_point(logits) or not torch.isfinite(logits).all()):
        raise ValueError('finite nonempty multiclass logits required')
    if (training_labels.shape != (len(logits),) or training_labels.dtype != torch.long
            or training_labels.device != logits.device or (training_labels < 0).any()
            or (training_labels >= logits.shape[1]).any()):
        raise ValueError('training labels must match batch, device and class range')
    _validate_caps(caps, logits.shape[1])
    if type(margin) not in (int,float) or not math.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    if kind == 'none':
        return logits.sum()*0.0
    if kind == 'margin':
        mask = torch.nn.functional.one_hot(training_labels,logits.shape[1]).bool()
        wrong = logits.masked_fill(mask,-torch.inf).max(dim=1).values
        correct = logits.gather(1,training_labels[:,None]).squeeze(1)
        return torch.relu(margin+wrong-correct).mean()
    if kind == 'false_positive':
        total, count = logits.sum()*0.0, 0
        normalizer = logits.logsumexp(dim=1)
        for c,cap in enumerate(caps):
            if cap is None: continue
            selected = training_labels != c
            others = [j for j in range(logits.shape[1]) if j != c]
            # Difference of log-normalizers avoids log(1-softmax) cancellation.
            penalty = normalizer-logits[:,others].logsumexp(dim=1)
            total = total+penalty[selected].sum()
            count += int(selected.sum())
        return total/max(count,1)
    raise ValueError('unknown supervised auxiliary term')

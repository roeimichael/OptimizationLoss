"""Training-label-only ranking of missed positives versus wrong quota occupants."""

import torch


def _selection(logits, training_labels, sample_ids, capacity):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            logits.shape[1] != 5 or not torch.is_floating_point(logits) or
            len(logits) < 2 or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite five-class logit matrix required')
    n=len(logits)
    if type(capacity) is not int or not 0 < capacity < n:
        raise ValueError('capacity must be strictly inside the training cohort')
    if len(training_labels)!=n or any(type(y) is not int or not 0<=y<5
                                     for y in training_labels):
        raise ValueError('training labels must match the five-class cohort')
    if (len(sample_ids)!=n or any(type(value) is not str or not value
                                  for value in sample_ids) or len(set(sample_ids))!=n):
        raise ValueError('unique nonempty training sample IDs required')
    other=torch.cat((logits[:,:3],logits[:,4:]),dim=1)
    margin=logits[:,3]-torch.logsumexp(other,dim=1)
    detached=margin.detach().cpu().tolist()
    selected=sorted(range(n),key=lambda i:(-detached[i],sample_ids[i]))[:capacity]
    included=set(selected)
    missed=[i for i,y in enumerate(training_labels) if y==3 and i not in included]
    wrong=[i for i,y in enumerate(training_labels) if y!=3 and i in included]
    details=dict(selected_ids=[sample_ids[i] for i in selected],
                 missed_true_ids=[sample_ids[i] for i in missed],
                 wrong_occupant_ids=[sample_ids[i] for i in wrong],
                 selected_true_count=sum(training_labels[i]==3 for i in selected),
                 active_pairs=len(missed)*len(wrong))
    return margin,other,missed,wrong,details


def cutoff_rank_loss(logits, training_labels, sample_ids, capacity):
    """Differentiable fixed-selection logistic pair loss for a toy oracle."""
    margin,_,missed,wrong,details=_selection(logits,training_labels,sample_ids,capacity)
    if not missed or not wrong:
        return logits.sum()*0.,details
    differences=margin[wrong][None,:]-margin[missed][:,None]
    return torch.nn.functional.softplus(differences).mean(),details


def cutoff_rank_gradient(logits, training_labels, sample_ids, capacity):
    """Exact dL/dlogits at a fixed top-K selection, for a streamed backward pass."""
    margin,other,missed,wrong,details=_selection(logits,training_labels,sample_ids,capacity)
    gradient=torch.zeros_like(logits)
    if not missed or not wrong:
        return gradient,details
    derivatives=torch.sigmoid(margin[wrong][None,:]-margin[missed][:,None])
    derivatives=derivatives/(len(missed)*len(wrong))
    dm=torch.zeros_like(margin)
    dm[missed]=-derivatives.sum(dim=1)
    dm[wrong]=derivatives.sum(dim=0)
    gradient[:,3]=dm
    alternative=-dm[:,None]*other.softmax(dim=1)
    gradient[:,:3]=alternative[:,:3]
    gradient[:,4:]=alternative[:,3:]
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('nonfinite ranking gradient')
    return gradient,details

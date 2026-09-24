"""One fixed-weight auxiliary step from training rank and/or unlabeled count."""

import torch

from .cutoff_rank import cutoff_rank_gradient, cutoff_rank_loss
from .global_constraint import bounded_count_penalty
from .streamed_constraint import count_logit_gradient


def streamed_rank_count_step(model, train_chunks, training_labels, training_ids,
                             train_capacity, development_chunks, caps, multipliers,
                             rho, optimizer, *, use_rank, use_count):
    """Stream two populations, accumulate analytic logit derivatives, step once.

    Only `training_labels` enter the ranking objective. Development chunks carry
    images without labels. The top-K pair sets are selected at a fixed parameter
    state, and BatchNorm remains in eval mode until after the optimizer step.
    """
    if (not hasattr(train_chunks,'__iter__') or not hasattr(train_chunks,'__len__') or
            len(train_chunks)==0 or not isinstance(development_chunks,(list,tuple)) or
            not development_chunks or
            type(use_rank) is not bool or type(use_count) is not bool or
            not (use_rank or use_count)):
        raise ValueError('nonempty replayable cohorts and an active objective required')
    was_training=model.training
    model.eval()
    try:
        device=next(model.parameters()).device
        with torch.no_grad():
            train_parts=[model(images.to(device)) for images in train_chunks]
            development_parts=[model(images.to(device)) for images in development_chunks]
        train_logits=torch.cat(train_parts)
        development_logits=torch.cat(development_parts)
        if not bool(torch.isfinite(train_logits).all()) or not bool(torch.isfinite(development_logits).all()):
            raise RuntimeError('nonfinite fixed-state logits')
        if use_rank:
            rank_gradient,rank_details=cutoff_rank_gradient(train_logits,training_labels,
                                                              training_ids,train_capacity)
            rank_loss=float(cutoff_rank_loss(train_logits,training_labels,
                                             training_ids,train_capacity)[0])
        else:
            rank_gradient=torch.zeros_like(train_logits)
            rank_details=None
            rank_loss=0.
        probabilities=development_logits.softmax(1)
        if use_count:
            count_gradient=count_logit_gradient(probabilities,caps,multipliers,rho)
            count_loss=float(bounded_count_penalty(development_logits,caps,multipliers,rho))
        else:
            count_gradient=torch.zeros_like(development_logits)
            count_loss=0.
        if not bool(torch.isfinite(rank_gradient).all()) or not bool(torch.isfinite(count_gradient).all()):
            raise RuntimeError('nonfinite auxiliary logit gradient')
        result=dict(applied=False,ranking=rank_details,rank_loss=rank_loss,count_loss=count_loss,
                    rank_logit_gradient_norm=float(rank_gradient.norm()),
                    count_logit_gradient_norm=float(count_gradient.norm()),
                    development_probabilities=probabilities.detach(),
                    soft_counts=probabilities.sum(0).detach())
        if not bool((rank_gradient!=0).any() or (count_gradient!=0).any()):
            return result
        optimizer.zero_grad(set_to_none=True)
        for chunks,expected_parts,gradient in (
                (train_chunks,train_parts,rank_gradient),
                (development_chunks,development_parts,count_gradient)):
            if not bool((gradient!=0).any()):
                continue
            start=0
            for images,expected in zip(chunks,expected_parts):
                logits=model(images.to(device))
                if not torch.allclose(logits.detach(),expected,atol=1e-7,rtol=1e-6):
                    raise RuntimeError('auxiliary replay changed fixed-weight logits')
                end=start+len(images)
                logits.backward(gradient[start:end])
                start=end
            if start!=len(gradient):
                raise RuntimeError('auxiliary cohort length changed')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad):
            raise RuntimeError('invalid auxiliary parameter gradient')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('nonfinite auxiliary parameter')
        result['applied']=True
        return result
    finally:
        model.train(was_training)

"""Powell-Hestenes-Rockafellar inequality AL, with signed normalized counts.

Fixed rho and one inexact (one supervised epoch) primal phase per dual update.
No labels enter these functions. Uncapped classes have no multiplier.
"""
import math
import torch


def residuals(logits, caps):
    if logits.ndim != 2 or logits.shape[1] != len(caps):
        raise ValueError('logit/cap shape mismatch')
    if any(k is not None and (type(k) is not int or k < 0) for k in caps):
        raise ValueError('caps must be nonnegative integers or None')
    counts=logits.softmax(dim=1).sum(dim=0)
    columns=[c for c,k in enumerate(caps) if k is not None]
    if not columns: return counts[:0]
    limits=logits.new_tensor([caps[c] for c in columns])
    return (counts[columns]-limits)/limits.clamp_min(1)


def augmented_penalty(g, multipliers, rho):
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError('ALM rho must be positive and finite')
    if g.shape != multipliers.shape or (multipliers < 0).any():
        raise ValueError('invalid multiplier vector')
    return ((multipliers+rho*g).clamp_min(0).square()-multipliers.square()).sum()/(2*rho)


def update_dual(g, multipliers, rho):
    augmented_penalty(g,multipliers,rho) # validate the same contract
    return (multipliers+rho*g).clamp_min(0).detach()

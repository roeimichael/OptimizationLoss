"""Imbalanced-learning training losses for the TMLR review-response baselines (Track B, B1).

Three drop-in criteria that answer the reviewers' question: does an
imbalanced-learning-aware training objective (+ clipping) close the macro-F1 gap
that the paper attributes to constraint-time training?

  focal          -- Lin et al. (2017), "Focal Loss for Dense Object Detection".
  class_balanced -- Cui et al. (2019), effective-number reweighting.
  logit_adjust   -- Menon et al. (2020), logit-adjusted softmax (loss variant).

Each builder returns a callable criterion(logits, targets) -> scalar, matching
nn.CrossEntropyLoss's signature so it drops straight into the fine-tune loop.
Hyperparameter defaults are the values from each original paper (and from the
advisor's Track-B handoff): focal alpha=0.25/gamma=2.0; CB beta=0.9999;
logit-adjust tau=1.0.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multi-class focal loss: alpha * (1 - p_t)^gamma * CE (Lin et al. 2017)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # softmax prob of the true class
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


class LogitAdjustedLoss(nn.Module):
    """Menon et al. (2020) loss variant: CE(logits + tau * log(prior), targets)."""

    def __init__(self, prior, tau=1.0):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("log_prior", torch.log(prior.clamp(min=1e-12)))

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.tau * self.log_prior, targets)


def _class_counts(y_train, num_classes):
    y = y_train if isinstance(y_train, torch.Tensor) else torch.as_tensor(y_train)
    return torch.bincount(y.to(torch.long).flatten(), minlength=num_classes).float()


def class_balanced_criterion(y_train, num_classes, device, beta=0.9999):
    """Cui et al. (2019): weight class c by (1 - beta) / (1 - beta^{n_c}),
    normalised to mean 1 so the overall loss scale matches plain CE."""
    counts = _class_counts(y_train, num_classes).to(device)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=device), counts.clamp(min=1))
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    log.info("class_balanced weights (beta=%.4f): %s",
             beta, weights.detach().cpu().numpy().round(3))
    return nn.CrossEntropyLoss(weight=weights)


def logit_adjusted_criterion(y_train, num_classes, device, tau=1.0):
    counts = _class_counts(y_train, num_classes)
    prior = (counts / counts.sum()).to(device)
    crit = LogitAdjustedLoss(prior, tau=tau).to(device)
    log.info("logit_adjust prior (tau=%.2f): %s",
             tau, prior.detach().cpu().numpy().round(4))
    return crit


def build_warmup_criterion(warmup_loss, y_train, num_classes, device, hp):
    """Return the imbalanced TRAINING criterion for `warmup_loss`, used by the
    shared warmup phase so the backbone is trained with the imbalanced objective
    from the pretrained init -- NOT merely fine-tuned off a saturated CE model
    (which the smoke test showed to be a no-op: CE-warmup loss is ~0)."""
    if warmup_loss == "focal":
        return FocalLoss(alpha=hp.get("focal_alpha", 0.25),
                         gamma=hp.get("focal_gamma", 2.0)).to(device)
    if warmup_loss == "class_balanced":
        return class_balanced_criterion(y_train, num_classes, device,
                                        beta=hp.get("cb_beta", 0.9999))
    if warmup_loss == "logit_adjust":
        return logit_adjusted_criterion(y_train, num_classes, device,
                                        tau=hp.get("logit_adjust_tau", 1.0))
    raise ValueError(f"Unknown warmup_loss: {warmup_loss!r}")

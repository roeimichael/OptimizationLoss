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


def build_warmup_criterion(warmup_loss, y_train, num_classes, device, hp):
    """Return the imbalanced TRAINING criterion for `warmup_loss`, used by the
    shared warmup phase so the backbone is trained with the imbalanced objective
    from the pretrained init -- NOT merely fine-tuned off a saturated CE model
    (which the smoke test showed to be a no-op: CE-warmup loss is ~0)."""
    if warmup_loss == "focal":
        return FocalLoss(alpha=hp.get("focal_alpha", 0.25),
                         gamma=hp.get("focal_gamma", 2.0)).to(device)
    raise ValueError(f"Unknown warmup_loss: {warmup_loss!r}")

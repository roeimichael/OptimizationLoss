import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: alpha * (1 - p_t)^gamma * CE (Lin et al. 2017)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


def build_warmup_criterion(warmup_loss, y_train, num_classes, device, hp):
    if warmup_loss == "focal":
        return FocalLoss(
            alpha=hp.get("focal_alpha", 0.25), gamma=hp.get("focal_gamma", 2.0)
        ).to(device)
    raise ValueError("Unknown warmup_loss: %r" % warmup_loss)

"""Transductive loss: rational saturation + ALM with KL-divergence regularization.

L_constraint = E/(E+K) + (rho/2)*(E/K)^2 where E = relu(soft_count - limit).
KL term keeps predictions close to warmup model to prevent distribution warping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-8
UNLIMITED = 1e9


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, num_classes=2,
                 use_sum=True, initial_rho=0.5, alpha_kl=0.0):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.alpha_kl = alpha_kl
        self.num_classes = num_classes
        self.use_sum = use_sum
        self.rho = initial_rho  # ALM quadratic penalty
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False

        # Register global constraints as buffer
        if global_constraints is not None:
            self.register_buffer('global_constraints',
                                 torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.register_buffer('global_constraints', torch.tensor([]))

        # Register local constraints as buffers
        self.local_groups = {}
        if local_constraints:
            for group_id, constraints in local_constraints.items():
                name = f'local_{int(group_id)}'
                self.register_buffer(name, torch.tensor(constraints, dtype=torch.float32))
                self.local_groups[group_id] = name

    def forward(self, logits, y_true=None, group_ids=None, warmup_proba=None):
        device = logits.device
        # Softmax for differentiable soft counts (no temperature scaling)
        soft_proba = F.softmax(logits, dim=1)
        hard_preds = logits.argmax(dim=1)

        L_ce = torch.tensor(0.0, device=device)
        if y_true is not None:
            L_ce = F.cross_entropy(logits, y_true)

        L_global = self._global_loss(soft_proba, hard_preds, device, logits)
        L_local = self._local_loss(soft_proba, hard_preds, group_ids, device, logits)

        # KL-divergence regularization against warmup model
        L_kl = self._kl_divergence_loss(logits, warmup_proba, device)

        L_total = (L_ce
                   + self.lambda_global * L_global
                   + self.lambda_local * L_local
                   + self.alpha_kl * L_kl)
        return L_total, L_ce, L_global, L_local, L_kl

    def _global_loss(self, proba, hard_preds, device, logits):
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            return logits.sum() * 0.0

        constraints = self.global_constraints.to(device)
        total_loss = torch.tensor(0.0, device=device)
        num_constrained = 0
        all_satisfied = True

        for c in range(self.num_classes):
            if c >= len(constraints) or constraints[c] >= UNLIMITED:
                continue
            K = constraints[c]
            if K <= 0:
                continue

            soft_count = proba[:, c].sum()
            hard_count = (hard_preds == c).sum().float()

            # Compute excess over limit
            E = F.relu(soft_count - K)

            # Check satisfaction against hard limit
            if hard_count.item() > K.item():
                all_satisfied = False

            E_norm = E / (K + EPSILON)
            loss = E / (E + K + EPSILON) + (self.rho / 2) * (E_norm ** 2)
            total_loss = total_loss + loss
            num_constrained += 1

        self.global_constraints_satisfied = all_satisfied
        if num_constrained == 0:
            return logits.sum() * 0.0

        if self.use_sum:
            return total_loss
        return total_loss / num_constrained

    def _local_loss(self, proba, hard_preds, group_ids, device, logits):
        if not self.local_groups or group_ids is None:
            self.local_constraints_satisfied = True
            return logits.sum() * 0.0

        group_ids = group_ids.to(device)
        total_loss = torch.tensor(0.0, device=device)
        num_constrained_total = 0
        all_satisfied = True

        for gid, buffer_name in self.local_groups.items():
            mask = (group_ids == gid)
            group_size = mask.sum().float()
            if group_size == 0:
                continue
            group_proba = proba[mask]
            group_hard_preds = hard_preds[mask]
            constraints = getattr(self, buffer_name).to(device)

            for c in range(self.num_classes):
                if c >= len(constraints) or constraints[c] >= UNLIMITED:
                    continue
                K = constraints[c]
                if K <= 0:
                    continue

                soft_count = group_proba[:, c].sum()
                hard_count = (group_hard_preds == c).sum().float()

                E = F.relu(soft_count - K)
                if hard_count.item() > K.item():
                    all_satisfied = False

                E_norm = E / (K + EPSILON)
                loss = E / (E + K + EPSILON) + (self.rho / 2) * (E_norm ** 2)

                total_loss = total_loss + loss
                num_constrained_total += 1

        self.local_constraints_satisfied = all_satisfied
        if num_constrained_total == 0:
            return logits.sum() * 0.0

        if self.use_sum:
            return total_loss
        return total_loss / num_constrained_total

    def _kl_divergence_loss(self, logits, warmup_proba, device):
        """KL(current || warmup) to keep predictions close to warmup model."""
        if warmup_proba is None or self.alpha_kl <= 0:
            return torch.tensor(0.0, device=device)

        log_p = F.log_softmax(logits, dim=1)
        p_warmup = warmup_proba.detach().clamp(min=EPSILON)
        p_current = F.softmax(logits, dim=1)
        kl = (p_current * (log_p - torch.log(p_warmup))).sum(dim=1)
        return kl.mean()

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = float(lambda_global)
        if lambda_local is not None:
            self.lambda_local = float(lambda_local)

    def set_alpha_kl(self, alpha_kl):
        self.alpha_kl = float(alpha_kl)

    def update_rho(self, factor=1.5):
        self.rho *= factor

    def get_rho(self):
        return self.rho


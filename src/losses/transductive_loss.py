# Transductive constraint loss: rational saturation + bounded quadratic + KL regularization.
# L_constraint = E/(E+K) + rho * (E/K)^2 / (1 + (E/K)^2), bounded to [0, 1+rho).
# KL term anchors predictions to warmup model to prevent distribution warping.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

EPSILON = 1e-8
UNLIMITED = 1e10


class MulticlassTransductiveLoss(nn.Module):

    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, num_classes=7,
                 use_sum=True, initial_rho=0.5, alpha_kl=0.0):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.alpha_kl = alpha_kl
        self.num_classes = num_classes
        self.use_sum = use_sum
        self.register_buffer('rho', torch.tensor(float(initial_rho)))
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False
        if global_constraints is not None:
            assert len(global_constraints) == num_classes
            self.register_buffer('global_constraints',
                                 torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.register_buffer('global_constraints', torch.tensor([]))
        self.local_groups = {}
        if local_constraints:
            for group_id, constraints in local_constraints.items():
                name = f'local_{int(group_id)}'
                self.register_buffer(name, torch.tensor(constraints, dtype=torch.float32))
                self.local_groups[group_id] = name

    def forward(self, logits, y_true=None, group_ids=None, warmup_proba=None):
        device = logits.device
        soft_proba = F.softmax(logits, dim=1)
        hard_preds = logits.argmax(dim=1)
        L_ce = torch.tensor(0.0, device=device)
        if y_true is not None:
            L_ce = F.cross_entropy(logits, y_true)
        L_global = self._global_loss(soft_proba, hard_preds, device, logits)
        L_local = self._local_loss(soft_proba, hard_preds, group_ids, device, logits)
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
                log.warning("Global constraint class %d has K=%.1f (<=0), skipping", c, K.item())
                continue
            soft_count = proba[:, c].sum()
            hard_count = (hard_preds == c).sum().float()
            E = F.relu(soft_count - K)
            if (hard_count > K).item():
                all_satisfied = False
            E_norm = E / (K + EPSILON)
            sat = E / (E + K + EPSILON)
            quad = (E_norm ** 2) / (1 + E_norm ** 2 + EPSILON)
            loss = sat + self.rho * quad
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
                    log.warning("Local constraint group %s class %d has K=%.1f (<=0), skipping", gid, c, K.item())
                    continue
                soft_count = group_proba[:, c].sum()
                hard_count = (group_hard_preds == c).sum().float()
                E = F.relu(soft_count - K)
                if (hard_count > K).item():
                    all_satisfied = False
                E_norm = E / (K + EPSILON)
                sat = E / (E + K + EPSILON)
                quad = (E_norm ** 2) / (1 + E_norm ** 2 + EPSILON)
                loss = sat + self.rho * quad
                total_loss = total_loss + loss
                num_constrained_total += 1
        self.local_constraints_satisfied = all_satisfied
        if num_constrained_total == 0:
            return logits.sum() * 0.0
        if self.use_sum:
            return total_loss
        return total_loss / num_constrained_total

    def _kl_divergence_loss(self, logits, warmup_proba, device):
        if warmup_proba is None or self.alpha_kl <= 0:
            return torch.tensor(0.0, device=device)
        log_p = F.log_softmax(logits, dim=1)
        p_warmup = warmup_proba.detach().clamp(min=EPSILON)
        p_current = F.softmax(logits, dim=1)
        kl = (p_current * (log_p - torch.log(p_warmup))).sum(dim=1)
        return kl.mean()

    def compute_global_from_counts(self, soft_counts):
        device = soft_counts.device
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            return soft_counts.sum() * 0.0
        constraints = self.global_constraints.to(device)
        total_loss = torch.tensor(0.0, device=device)
        num_constrained = 0
        all_satisfied = True
        for c in range(self.num_classes):
            if c >= len(constraints) or constraints[c] >= UNLIMITED:
                continue
            K = constraints[c]
            if K <= 0:
                log.warning("Global constraint class %d has K=%.1f (<=0), skipping", c, K.item())
                continue
            E = F.relu(soft_counts[c] - K)
            if (soft_counts[c] > K).item():
                all_satisfied = False
            E_norm = E / (K + EPSILON)
            sat = E / (E + K + EPSILON)
            quad = (E_norm ** 2) / (1 + E_norm ** 2 + EPSILON)
            loss = sat + self.rho * quad
            total_loss = total_loss + loss
            num_constrained += 1
        self.global_constraints_satisfied = all_satisfied
        if num_constrained == 0:
            return soft_counts.sum() * 0.0
        return total_loss if self.use_sum else total_loss / num_constrained

    def compute_local_from_counts(self, local_soft_counts):
        if not self.local_groups or not local_soft_counts:
            self.local_constraints_satisfied = True
            for v in local_soft_counts.values():
                return v.sum() * 0.0
            return torch.tensor(0.0)
        device = next(iter(local_soft_counts.values())).device
        total_loss = torch.tensor(0.0, device=device)
        num_constrained = 0
        all_satisfied = True
        for gid, buffer_name in self.local_groups.items():
            if gid not in local_soft_counts:
                continue
            group_soft = local_soft_counts[gid]
            constraints = getattr(self, buffer_name).to(device)
            for c in range(self.num_classes):
                if c >= len(constraints) or constraints[c] >= UNLIMITED:
                    continue
                K = constraints[c]
                if K <= 0:
                    log.warning("Local constraint group %s class %d has K=%.1f (<=0), skipping", gid, c, K.item())
                    continue
                E = F.relu(group_soft[c] - K)
                if (group_soft[c] > K).item():
                    all_satisfied = False
                E_norm = E / (K + EPSILON)
                sat = E / (E + K + EPSILON)
                quad = (E_norm ** 2) / (1 + E_norm ** 2 + EPSILON)
                loss = sat + self.rho * quad
                total_loss = total_loss + loss
                num_constrained += 1
        self.local_constraints_satisfied = all_satisfied
        if num_constrained == 0:
            for v in local_soft_counts.values():
                return v.sum() * 0.0
            return torch.tensor(0.0, device=device)
        return total_loss if self.use_sum else total_loss / num_constrained

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = float(lambda_global)
        if lambda_local is not None:
            self.lambda_local = float(lambda_local)

    def increment_rho(self, step):
        self.rho.add_(step)

    def get_rho(self):
        return self.rho.item()

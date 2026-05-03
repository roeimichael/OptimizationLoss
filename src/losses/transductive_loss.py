# Transductive constraint loss: rational saturation + bounded quadratic + KL regularization.
# L_constraint = E/(E+K) + rho * (E/K)^2 / (1 + (E/K)^2), bounded to [0, 1+rho).
# KL term anchors predictions to warmup model to prevent distribution warping.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import UNLIMITED, EPSILON

log = logging.getLogger(__name__)


class MulticlassTransductiveLoss(nn.Module):

    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, num_classes=7,
                 use_sum=True, initial_rho=0.5, alpha_kl=0.0,
                 per_class_lambda=False):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.alpha_kl = alpha_kl
        self.num_classes = num_classes
        self.use_sum = use_sum
        self.per_class_lambda = per_class_lambda
        self.register_buffer('rho', torch.tensor(float(initial_rho)))
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False
        # Per-class lambda dicts (used when per_class_lambda=True)
        # Keys: class index for global, (group_id, class) for local
        self.lambda_global_per_class = {}
        self.lambda_local_per_key = {}
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

    def set_lambda_per_class(self, class_idx, value, scope='global', group_id=None):
        """Set lambda for a specific class (global) or (group, class) pair (local)."""
        if scope == 'global':
            self.lambda_global_per_class[class_idx] = float(value)
        elif scope == 'local' and group_id is not None:
            self.lambda_local_per_key[(group_id, class_idx)] = float(value)

    def get_lambda_per_class(self, class_idx, scope='global', group_id=None):
        if scope == 'global':
            return self.lambda_global_per_class.get(class_idx, 0.0)
        return self.lambda_local_per_key.get((group_id, class_idx), 0.0)

    def increment_rho(self, step):
        self.rho.add_(step)

    def get_rho(self):
        return self.rho.item()

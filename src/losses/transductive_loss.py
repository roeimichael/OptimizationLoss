"""Transductive loss with global and local constraints."""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
EPSILON = 1e-6
UNLIMITED = 1e9


class MulticlassTransductiveLoss(nn.Module):
    """
    Loss function combining cross-entropy with constraint satisfaction losses.

    Computes:
        L_total = L_ce + lambda_global * L_global + lambda_local * L_local

    Where constraint losses use soft predictions to remain differentiable.
    """

    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, num_classes=NUM_CLASSES):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.num_classes = num_classes
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

    def forward(self, logits, y_true=None, group_ids=None):
        device = logits.device
        proba = F.softmax(logits, dim=1)

        L_ce = torch.tensor(0.0, device=device)
        if y_true is not None:
            L_ce = F.cross_entropy(logits, y_true)

        L_global = self._global_loss(proba, device)
        L_local = self._local_loss(proba, group_ids, device)

        L_total = L_ce + self.lambda_global * L_global + self.lambda_local * L_local
        return L_total, L_ce, L_global, L_local

    def _global_loss(self, proba, device):
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            return torch.tensor(0.0, device=device, requires_grad=True)

        constraints = self.global_constraints.to(device)
        losses = []
        all_satisfied = True

        for c in range(self.num_classes):
            if c >= len(constraints) or constraints[c] > UNLIMITED:
                continue

            K = constraints[c]
            count = proba[:, c].sum()

            if count > K:
                all_satisfied = False
                excess = count - K
                loss = excess / (excess + K + EPSILON)
                losses.append(loss)

        self.global_constraints_satisfied = all_satisfied
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=device, requires_grad=True)

    def _local_loss(self, proba, group_ids, device):
        if not self.local_groups or group_ids is None:
            self.local_constraints_satisfied = True
            return torch.tensor(0.0, device=device, requires_grad=True)

        group_ids = group_ids.to(device)
        total_loss = 0.0
        total_weight = 0.0
        all_satisfied = True

        for gid, buffer_name in self.local_groups.items():
            mask = (group_ids == gid)
            size = mask.sum().float()
            if size == 0:
                continue

            group_proba = proba[mask]
            constraints = getattr(self, buffer_name).to(device)
            group_losses = []

            for c in range(self.num_classes):
                if c >= len(constraints) or constraints[c] > UNLIMITED:
                    continue

                K = constraints[c]
                count = group_proba[:, c].sum()

                if count > K:
                    all_satisfied = False
                    excess = count - K
                    loss = excess / (excess + K + EPSILON)
                    group_losses.append(loss)

            if group_losses:
                total_loss += (sum(group_losses) / len(group_losses)) * size
                total_weight += size

        self.local_constraints_satisfied = all_satisfied
        return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=device, requires_grad=True)

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = float(lambda_global)
        if lambda_local is not None:
            self.lambda_local = float(lambda_local)

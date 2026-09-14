"""Bounded TraLO count penalty. Soft counts provide gradients; hard counts drive the trainer ratchet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.constants import UNLIMITED, EPSILON


class MulticlassTransductiveLoss(nn.Module):
    def __init__(
        self, global_constraints, local_constraints, num_classes, initial_rho=0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("rho", torch.tensor(float(initial_rho)))
        self.lambda_global_per_class = {}
        self.lambda_local_per_key = {}
        if global_constraints is not None:
            assert len(global_constraints) == num_classes
            self.register_buffer(
                "global_constraints",
                torch.tensor(global_constraints, dtype=torch.float32),
            )
        else:
            self.register_buffer("global_constraints", torch.tensor([]))
        self.local_groups = {}
        for group_id, constraints in (local_constraints or {}).items():
            name = "local_%d" % int(group_id)
            self.register_buffer(name, torch.tensor(constraints, dtype=torch.float32))
            self.local_groups[group_id] = name

    def _penalty(self, soft, K):
        E = F.relu(soft - K)
        scale = K if K >= 1 else 1.0
        e = E / (scale + EPSILON)
        return E / (E + scale + EPSILON) + self.rho * e**2 / (1 + e**2 + EPSILON)

    def _sum(self, entries, device):
        total = torch.tensor(0.0, device=device)
        (all_satisfied, n) = (True, 0)
        for soft, K, lam in entries:
            if (soft > K).item():
                all_satisfied = False
            total = total + lam * self._penalty(soft, K)
            n += 1
        return (total, all_satisfied, n)

    def _capped(self, constraints, num_classes):
        return [
            c
            for c in range(num_classes)
            if c < len(constraints) and constraints[c] < UNLIMITED
        ]

    def compute_global_from_counts(self, soft_counts):
        device = soft_counts.device
        if len(self.global_constraints) == 0:
            return soft_counts.sum() * 0.0
        con = self.global_constraints.to(device)
        entries = [
            (soft_counts[c], con[c], self.lambda_global_per_class.get(c, 0.0))
            for c in self._capped(con, self.num_classes)
        ]
        (total, _satisfied, n) = self._sum(entries, device)
        return total if n else soft_counts.sum() * 0.0

    def _zero(self):
        return torch.zeros((), device=self.global_constraints.device)

    def compute_local_from_counts(self, local_soft_counts):
        if not self.local_groups or not local_soft_counts:
            for v in local_soft_counts.values():
                return v.sum() * 0.0
            return self._zero()
        device = next(iter(local_soft_counts.values())).device
        entries = []
        for gid, buffer_name in self.local_groups.items():
            if gid not in local_soft_counts:
                continue
            soft = local_soft_counts[gid]
            con = getattr(self, buffer_name).to(device)
            entries += [
                (soft[c], con[c], self.lambda_local_per_key.get((gid, c), 0.0))
                for c in self._capped(con, self.num_classes)
            ]
        (total, _satisfied, n) = self._sum(entries, device)
        if n:
            return total
        for v in local_soft_counts.values():
            return v.sum() * 0.0
        return self._zero()

    def set_lambda_per_class(self, class_idx, value, scope="global", group_id=None):
        if scope == "global":
            self.lambda_global_per_class[class_idx] = float(value)
        elif scope == "local" and group_id is not None:
            self.lambda_local_per_key[group_id, class_idx] = float(value)
        else:
            raise ValueError(
                "set_lambda_per_class(scope=%r, group_id=%r) would set nothing, leaving lambda at 0.0 and making this arm its own null. scope must be 'global', or 'local' WITH a group_id."
                % (scope, group_id)
            )

    def get_lambda_per_class(self, class_idx, scope="global", group_id=None):
        if scope == "global":
            return self.lambda_global_per_class.get(class_idx, 0.0)
        return self.lambda_local_per_key.get((group_id, class_idx), 0.0)

    def increment_rho(self, step):
        self.rho.add_(step)

    def get_rho(self):
        return self.rho.item()

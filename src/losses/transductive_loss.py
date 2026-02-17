"""
Transductive loss with global and local constraints using Rational Saturation.

Mathematical formulation:
    L_total = L_pred + λ_global * (E_global / (E_global + K_global))
                     + λ_local/M * Σ (E_group^i / (E_group^i + K_group))

Where:
    E_global = max(0, Σŷ - K_global)      # Global violation (soft counts for gradient)
    E_group  = max(0, Σŷ_group - K_group) # Per-group violation (soft counts for gradient)

The saturation term E/(E+K) maps any violation E ∈ [0, ∞) to loss ∈ [0, 1).

Note: Loss uses soft counts (differentiable), but satisfaction check uses hard counts (argmax).

Indexing convention (0-indexed throughout):
    - Labels are 0-1 (binary: 0 = no churn, 1 = churn)
    - Model outputs 2 logits (indices 0-1), where output[i] = probability of class i
    - Constraint arrays use direct indexing: constraints[i] = limit for class i
    - Class 0 is unconstrained, class 1 is the constrained class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 2  # Binary: 0 = no churn, 1 = churn
EPSILON = 1e-8
UNLIMITED = 1e9


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, num_classes=NUM_CLASSES,
                 margin=0.0, use_sum=True, initial_rho=1.0, gumbel_temp=0.1):
        """
        Args:
            global_constraints: Per-class global limits
            local_constraints: Per-group per-class limits
            lambda_global: Weight for global constraint loss
            lambda_local: Weight for local constraint loss
            num_classes: Number of classes
            margin: Safety margin (e.g., 0.1 means aim for 90% of constraint)
            use_sum: If True, sum constraint losses instead of averaging (stronger gradient)
            initial_rho: Initial quadratic penalty for Augmented Lagrangian Method
            gumbel_temp: Temperature for Gumbel-Softmax (lower = more like hard argmax)
        """
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.num_classes = num_classes
        self.margin = margin
        self.use_sum = use_sum
        self.rho = initial_rho  # Augmented Lagrangian quadratic penalty
        self.gumbel_temp = gumbel_temp  # Gumbel-Softmax temperature
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
        # Temperature-scaled softmax for differentiable soft counts
        # Lower temperature = sharper distribution ≈ hard counts but with gradients
        # NOTE: STE was removed because it blocked gradients entirely
        # (forward used hard counts, making E = relu(hard_count - K) have zero gradient)
        soft_proba = F.softmax(logits / self.gumbel_temp, dim=1)
        hard_preds = logits.argmax(dim=1)

        L_ce = torch.tensor(0.0, device=device)
        if y_true is not None:
            L_ce = F.cross_entropy(logits, y_true)

        L_global = self._global_loss(soft_proba, hard_preds, device, logits)
        L_local = self._local_loss(soft_proba, hard_preds, group_ids, device, logits)

        L_total = L_ce + self.lambda_global * L_global + self.lambda_local * L_local
        return L_total, L_ce, L_global, L_local

    def _global_loss(self, proba, hard_preds, device, logits):
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            # Return graph-connected zero (not disconnected leaf tensor)
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

            # Apply margin: target is (1 - margin) * K
            # e.g., with margin=0.1 and K=100, target is 90
            K_target = K * (1.0 - self.margin)

            soft_count = proba[:, c].sum()
            hard_count = (hard_preds == c).sum().float()

            # Compute excess over target (not hard limit)
            E = F.relu(soft_count - K_target)

            # Satisfaction still checked against hard limit
            if hard_count.item() > K.item():
                all_satisfied = False

            # Augmented Lagrangian loss: normalized linear + quadratic penalty
            # L = (E/K) + (rho/2) * (E/K)^2
            # Normalized by K to keep loss bounded and comparable across constraints
            E_norm = E / (K + EPSILON)
            loss = E_norm + (self.rho / 2) * (E_norm ** 2)
            total_loss = total_loss + loss
            num_constrained += 1

        self.global_constraints_satisfied = all_satisfied
        if num_constrained == 0:
            # Return graph-connected zero
            return logits.sum() * 0.0

        # Sum gives stronger gradient signal than averaging
        if self.use_sum:
            return total_loss
        return total_loss / num_constrained

    def _local_loss(self, proba, hard_preds, group_ids, device, logits):
        if not self.local_groups or group_ids is None:
            self.local_constraints_satisfied = True
            # Return graph-connected zero
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

                # Apply margin
                K_target = K * (1.0 - self.margin)

                soft_count = group_proba[:, c].sum()
                hard_count = (group_hard_preds == c).sum().float()

                E = F.relu(soft_count - K_target)
                if hard_count.item() > K.item():
                    all_satisfied = False

                # Augmented Lagrangian loss: normalized linear + quadratic penalty
                E_norm = E / (K + EPSILON)
                loss = E_norm + (self.rho / 2) * (E_norm ** 2)
                total_loss = total_loss + loss
                num_constrained_total += 1

        self.local_constraints_satisfied = all_satisfied
        if num_constrained_total == 0:
            # Return graph-connected zero
            return logits.sum() * 0.0

        # Sum gives stronger gradient signal
        if self.use_sum:
            return total_loss
        return total_loss / num_constrained_total

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = float(lambda_global)
        if lambda_local is not None:
            self.lambda_local = float(lambda_local)

    def update_rho(self, factor=1.5):
        """Increase quadratic penalty when constraints remain violated (Adaptive ALM)."""
        self.rho *= factor

    def get_rho(self):
        """Get current rho value."""
        return self.rho

    def set_gumbel_temp(self, temp):
        """Set Gumbel-Softmax temperature."""
        self.gumbel_temp = temp

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 3
EPSILON = 1e-6
UNLIMITED_THRESHOLD = 1e9


def _compute_single_class_constraint_loss(
        soft_predictions: torch.Tensor,
        hard_predictions: torch.Tensor,
        class_id: int,
        constraint_value: float,
        epsilon: float = EPSILON
) -> tuple[torch.Tensor, bool]:

    if constraint_value > UNLIMITED_THRESHOLD:
        return torch.tensor(0.0, device=soft_predictions.device), True
    hard_count = (hard_predictions == class_id).sum().float()
    is_satisfied = hard_count <= constraint_value
    predicted_count = soft_predictions[:, class_id].sum()
    if predicted_count > constraint_value:
        E = torch.relu(predicted_count - constraint_value)
        loss = E / (E + constraint_value + epsilon)
        return loss, is_satisfied

    return torch.tensor(0.0, device=soft_predictions.device), is_satisfied


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, use_ce=True, num_classes=NUM_CLASSES):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.eps = EPSILON
        self.num_classes = num_classes
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False
        self.ce_loss = nn.CrossEntropyLoss()

        if global_constraints is not None:
            self.register_buffer('global_constraints', torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.register_buffer('global_constraints', torch.tensor([]))

        if local_constraints is not None:
            self.local_constraint_dict = {}
            for group_id, constraints in local_constraints.items():
                buffer_name = f'local_constraint_{group_id}'
                self.register_buffer(buffer_name, torch.tensor(constraints, dtype=torch.float32))
                self.local_constraint_dict[group_id] = buffer_name
        else:
            self.local_constraint_dict = {}

    def forward(self, logits, y_true=None, group_ids=None):
        device = logits.device
        L_pred = torch.tensor(0.0, device=device)
        if y_true is not None:
            L_pred = self.ce_loss(logits, y_true)
        y_proba = F.softmax(logits, dim=1)
        L_target = self._compute_global_constraint_loss(y_proba, device)
        L_feat = self._compute_local_constraint_loss(y_proba, group_ids, device)
        L_total = L_pred + self.lambda_global * L_target + self.lambda_local * L_feat
        return L_total, L_pred, L_target, L_feat

    def _compute_global_constraint_loss(self, y_proba, device):
        L_target = torch.tensor(0.0, device=device, requires_grad=True)
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            return L_target

        g_cons = self.global_constraints.to(device)
        y_hard = torch.argmax(y_proba, dim=1)

        total_violation = 0.0
        num_constrained = 0
        all_satisfied = True

        for class_id in range(self.num_classes):
            if class_id >= len(g_cons):
                continue
            K = g_cons[class_id]

            if K > UNLIMITED_THRESHOLD:
                continue

            class_loss, is_satisfied = _compute_single_class_constraint_loss(
                y_proba, y_hard, class_id, K, self.eps
            )

            if not is_satisfied:
                all_satisfied = False

            if class_loss > 0:
                total_violation += class_loss
                num_constrained += 1

        # Average over constrained classes
        if num_constrained > 0:
            L_target = total_violation / num_constrained

        self.global_constraints_satisfied = all_satisfied
        return L_target

    def _compute_local_constraint_loss(self, y_proba, group_ids, device):
        L_feat = torch.tensor(0.0, device=device, requires_grad=True)
        if not self.local_constraint_dict or group_ids is None:
            self.local_constraints_satisfied = True
            return L_feat

        group_ids_device = group_ids.to(device)
        y_hard = torch.argmax(y_proba, dim=1)

        total_violation = 0.0
        total_weight = 0.0
        all_satisfied = True

        for group_id, buffer_name in self.local_constraint_dict.items():
            in_group = (group_ids_device == group_id)
            group_size = in_group.sum().float()
            if group_size == 0:
                continue

            group_proba = y_proba[in_group]
            group_hard = y_hard[in_group]
            l_cons = getattr(self, buffer_name).to(device)

            group_violation = 0.0
            group_constrained = 0

            for class_id in range(self.num_classes):
                if class_id >= len(l_cons):
                    continue
                K = l_cons[class_id]

                if K > UNLIMITED_THRESHOLD:
                    continue

                class_loss, is_satisfied = _compute_single_class_constraint_loss(
                    group_proba, group_hard, class_id, K, self.eps
                )

                if not is_satisfied:
                    all_satisfied = False

                if class_loss > 0:
                    group_violation += class_loss
                    group_constrained += 1

            # Weight by group size * constrained classes
            if group_constrained > 0:
                avg_group_violation = group_violation / group_constrained
                total_violation += avg_group_violation * group_size
                total_weight += group_size

        # Weighted average across groups
        if total_weight > 0:
            L_feat = total_violation / total_weight

        self.local_constraints_satisfied = all_satisfied
        return L_feat

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = float(lambda_global)  # Ensure scalar
        if lambda_local is not None:
            self.lambda_local = float(lambda_local)

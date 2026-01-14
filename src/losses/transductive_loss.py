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
    """Compute constraint loss for a single class.

    Args:
        soft_predictions: Soft probability predictions for the class [N, num_classes]
        hard_predictions: Hard predictions (argmax) [N]
        class_id: The class ID to compute constraint for
        constraint_value: Maximum allowed count (K)
        epsilon: Numerical stability constant

    Returns:
        tuple: (loss, is_satisfied)
            - loss: Constraint violation loss
            - is_satisfied: Whether hard predictions satisfy the constraint
    """
    # Check if constraint is unlimited
    if constraint_value > UNLIMITED_THRESHOLD:
        return torch.tensor(0.0, device=soft_predictions.device), True

    # Check hard satisfaction
    hard_count = (hard_predictions == class_id).sum().float()
    is_satisfied = hard_count <= constraint_value

    # Compute soft constraint loss
    predicted_count = soft_predictions[:, class_id].sum()
    if predicted_count > constraint_value:
        E = torch.relu(predicted_count - constraint_value)
        loss = E / (E + constraint_value + epsilon)
        return loss, is_satisfied

    return torch.tensor(0.0, device=soft_predictions.device), is_satisfied


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, use_ce=True):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.use_ce = use_ce
        self.eps = EPSILON
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False
        if global_constraints is not None:
            self.register_buffer('global_constraints',
                                 torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.global_constraints = None
        if local_constraints is not None:
            self.local_constraint_dict = {}
            for group_id, constraints in local_constraints.items():
                buffer_name = f'local_constraint_{group_id}'
                self.register_buffer(buffer_name,
                                     torch.tensor(constraints, dtype=torch.float32))
                self.local_constraint_dict[group_id] = buffer_name
        else:
            self.local_constraint_dict = None
        if use_ce:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, y_true=None, group_ids=None):
        device = logits.device
        L_pred = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_pred = self.ce_loss(logits, y_true)
        y_proba = F.softmax(logits, dim=1)
        L_target = self._compute_global_constraint_loss(y_proba, device)
        L_feat = self._compute_local_constraint_loss(y_proba, group_ids, device)
        L_total = L_pred + self.lambda_global * L_target + self.lambda_local * L_feat
        return L_total, L_pred, L_target, L_feat

    def _compute_global_constraint_loss(self, y_proba, device):
        L_target = torch.tensor(0.0, device=device)
        if self.global_constraints is None:
            self.global_constraints_satisfied = True
            return L_target

        g_cons = self.global_constraints.to(device)
        y_hard = torch.argmax(y_proba, dim=1)

        num_constrained = 0
        all_satisfied = True

        for class_id in range(NUM_CLASSES):
            K = g_cons[class_id]

            # Skip unlimited constraints
            if K > UNLIMITED_THRESHOLD:
                continue

            # Compute constraint loss and satisfaction for this class
            class_loss, is_satisfied = _compute_single_class_constraint_loss(
                y_proba, y_hard, class_id, K, self.eps
            )

            if not is_satisfied:
                all_satisfied = False

            if class_loss > 0:
                L_target = L_target + class_loss
                num_constrained += 1

        # Average over constrained classes
        if num_constrained > 0:
            L_target = L_target / num_constrained

        self.global_constraints_satisfied = all_satisfied
        return L_target

    def _compute_local_constraint_loss(self, y_proba, group_ids, device):
        L_feat = torch.tensor(0.0, device=device)
        if self.local_constraint_dict is None or group_ids is None:
            self.local_constraints_satisfied = True
            return L_feat

        group_ids_device = group_ids.to(device)
        y_hard = torch.argmax(y_proba, dim=1)

        num_constrained = 0
        all_satisfied = True

        for group_id, buffer_name in self.local_constraint_dict.items():
            # Get group mask and predictions
            in_group = (group_ids_device == group_id)
            if in_group.sum() == 0:
                continue

            group_proba = y_proba[in_group]
            group_hard = y_hard[in_group]
            l_cons = getattr(self, buffer_name).to(device)

            # Compute constraint for each class in this group
            for class_id in range(NUM_CLASSES):
                K = l_cons[class_id]

                # Skip unlimited constraints
                if K > UNLIMITED_THRESHOLD:
                    continue

                # Compute constraint loss and satisfaction for this class
                class_loss, is_satisfied = _compute_single_class_constraint_loss(
                    group_proba, group_hard, class_id, K, self.eps
                )

                if not is_satisfied:
                    all_satisfied = False

                if class_loss > 0:
                    L_feat = L_feat + class_loss
                    num_constrained += 1

        # Average over constrained classes
        if num_constrained > 0:
            L_feat = L_feat / num_constrained

        self.local_constraints_satisfied = all_satisfied
        return L_feat

    def set_lambda(self, lambda_global=None, lambda_local=None):
        if lambda_global is not None:
            self.lambda_global = lambda_global
        if lambda_local is not None:
            self.lambda_local = lambda_local

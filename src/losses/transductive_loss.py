import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassTransductiveLoss(nn.Module):
    """
    Transductive Loss with Constraint Satisfaction

    Total Loss = L_pred + λ_global * L_target + λ_local * L_feat

    Where:
    - L_pred: Standard cross-entropy loss (classification)
    - L_target: Global constraint penalty (across all students)
    - L_feat: Local constraint penalty (per course)

    Constraint formula: E / (E + K)
    where E = ReLU(predicted_count - K)
    """

    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, use_ce=True):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.use_ce = use_ce
        self.eps = 1e-6

        # Store constraints as PyTorch buffers (pre-cleaned, no sanitization needed)
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
        """
        Compute transductive loss.

        Args:
            logits: Model predictions [batch_size, 3]
            y_true: True labels (optional, for L_pred)
            group_ids: Course IDs (optional, for L_feat)

        Returns:
            L_total: Total loss
            L_pred: Classification loss
            L_target: Global constraint loss
            L_feat: Local constraint loss
        """
        device = logits.device

        # Step 1: Classification loss
        L_pred = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_pred = self.ce_loss(logits, y_true)

        # Step 2: Get soft predictions (probabilities)
        y_proba = F.softmax(logits, dim=1)

        # Step 3: Global constraint loss
        L_target = self._compute_global_constraint_loss(y_proba, device)

        # Step 4: Local constraint loss
        L_feat = self._compute_local_constraint_loss(y_proba, group_ids, device)

        # Step 5: Combine losses
        L_total = L_pred + self.lambda_global * L_target + self.lambda_local * L_feat

        return L_total, L_pred, L_target, L_feat

    def _compute_global_constraint_loss(self, y_proba, device):
        """
        Compute global constraint penalty.

        For each class:
        - Count predicted students (sum of probabilities)
        - If count > limit K, penalize excess
        - Formula: E / (E + K) where E = ReLU(count - K)
        """
        L_target = torch.tensor(0.0, device=device)

        if self.global_constraints is None:
            return L_target

        g_cons = self.global_constraints.to(device)
        num_constrained = 0

        for class_id in range(3):
            K = g_cons[class_id]

            # Skip unconstrained classes (K = 1e10)
            if K > 1e9:
                continue

            # Predicted count = sum of probabilities
            predicted_count = y_proba[:, class_id].sum()

            # Excess over limit
            E = torch.relu(predicted_count - K)

            # Rational saturation formula
            loss = E / (E + K + self.eps)

            L_target = L_target + loss
            num_constrained += 1

        # Average over constrained classes
        if num_constrained > 0:
            L_target = L_target / num_constrained

        return L_target

    def _compute_local_constraint_loss(self, y_proba, group_ids, device):
        """
        Compute local (per-course) constraint penalty.

        For each course and class:
        - Count predicted students in that course
        - If count > limit K, penalize excess
        - Formula: E / (E + K) where E = ReLU(count - K)
        """
        L_feat = torch.tensor(0.0, device=device)

        if self.local_constraint_dict is None or group_ids is None:
            return L_feat

        group_ids_device = group_ids.to(device)
        num_constrained = 0

        for group_id, buffer_name in self.local_constraint_dict.items():
            # Get students in this course
            in_group = (group_ids_device == group_id)

            if in_group.sum() == 0:
                continue

            # Probabilities for this course only
            group_proba = y_proba[in_group]

            # Get constraints for this course
            l_cons = getattr(self, buffer_name).to(device)

            for class_id in range(3):
                K = l_cons[class_id]

                # Skip unconstrained classes
                if K > 1e9:
                    continue

                # Predicted count in this course
                predicted_count = group_proba[:, class_id].sum()

                # Excess over limit
                E = torch.relu(predicted_count - K)

                # Rational saturation formula
                loss = E / (E + K + self.eps)

                L_feat = L_feat + loss
                num_constrained += 1

        # Average over all (course, class) combinations
        if num_constrained > 0:
            L_feat = L_feat / num_constrained

        return L_feat

    def set_lambda(self, lambda_global=None, lambda_local=None):
        """Update constraint weights during training (for adaptive lambdas)."""
        if lambda_global is not None:
            self.lambda_global = lambda_global
        if lambda_local is not None:
            self.lambda_local = lambda_local

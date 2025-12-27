import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        # Store constraints as PyTorch buffers (moved to GPU with model)
        self.global_constraints = self._prepare_global_constraints(global_constraints)
        self.local_constraint_dict = self._prepare_local_constraints(local_constraints)

        if use_ce:
            self.ce_loss = nn.CrossEntropyLoss()

    def _sanitize_value(self, value):
        """
        Convert constraint value to valid number.

        None, NaN, or Inf -> 1e10 (means unconstrained)
        Valid number -> keep as is
        """
        if value is None:
            return 1e10

        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 1e10
            return val
        except (ValueError, TypeError):
            return 1e10

    def _prepare_global_constraints(self, constraints):
        """Convert global constraints list to PyTorch tensor."""
        if constraints is None:
            return None

        # Clean each constraint value
        cleaned = [self._sanitize_value(c) for c in constraints]

        # Register as buffer (moves to GPU with model)
        self.register_buffer('global_constraints',
                           torch.tensor(cleaned, dtype=torch.float32))
        return self.global_constraints

    def _prepare_local_constraints(self, constraints):
        """Convert local constraints dict to PyTorch tensors."""
        if constraints is None:
            return None

        constraint_dict = {}
        for group_id, constraint_list in constraints.items():
            # Clean each constraint value
            cleaned = [self._sanitize_value(c) for c in constraint_list]

            # Register as buffer with unique name
            buffer_name = f'local_constraint_{group_id}'
            self.register_buffer(buffer_name,
                               torch.tensor(cleaned, dtype=torch.float32))
            constraint_dict[group_id] = buffer_name

        return constraint_dict

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

        # Step 1: Classification loss (standard cross-entropy)
        L_pred = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_pred = self.ce_loss(logits, y_true)

        # Step 2: Get soft predictions (probabilities for each class)
        # Shape: [batch_size, 3] where each row sums to 1.0
        y_proba = F.softmax(logits, dim=1)

        # Step 3: Global constraint loss
        L_target = self._compute_global_constraint_loss(y_proba, device)

        # Step 4: Local (per-course) constraint loss
        L_feat = self._compute_local_constraint_loss(y_proba, group_ids, device)

        # Step 5: Combine all losses
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

            # Predicted count = sum of probabilities for this class
            predicted_count = y_proba[:, class_id].sum()

            # Excess = how much we're over the limit
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

        For each course and each class:
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

            # Probabilities for students in this course only
            group_proba = y_proba[in_group]

            # Get constraint limits for this course
            l_cons = getattr(self, buffer_name).to(device)

            for class_id in range(3):
                K = l_cons[class_id]

                # Skip unconstrained classes
                if K > 1e9:
                    continue

                # Predicted count in this course for this class
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

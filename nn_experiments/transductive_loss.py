import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, use_ce=True):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.use_ce = use_ce
        self.eps = 1e-6

        # --- Helper: Sanitize Constraints (Fixes the NaN bug) ---
        def _sanitize(constraints_list):
            cleaned = []
            for c in constraints_list:
                # 1. Check for None
                if c is None:
                    cleaned.append(1e10)
                    continue

                # 2. Check for NaN/Inf (convert to float first to handle numpy types)
                try:
                    val = float(c)
                    if math.isnan(val) or math.isinf(val):
                        cleaned.append(1e10)
                    else:
                        cleaned.append(val)
                except (ValueError, TypeError):
                    # Fallback for unexpected types
                    cleaned.append(1e10)
            return cleaned

        # --------------------------------------------------------

        if global_constraints is not None:
            clean_global = _sanitize(global_constraints)
            self.register_buffer('global_constraints',
                                 torch.tensor(clean_global, dtype=torch.float32))
        else:
            self.global_constraints = None

        if local_constraints is not None:
            self.local_constraint_dict = {}
            for group_id, constraints in local_constraints.items():
                clean_local = _sanitize(constraints)
                self.register_buffer(f'local_constraint_{group_id}',
                                     torch.tensor(clean_local, dtype=torch.float32))
                self.local_constraint_dict[group_id] = f'local_constraint_{group_id}'
        else:
            self.local_constraint_dict = None

        if use_ce:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, y_true=None, group_ids=None):
        """
        Compute total loss according to paper formulation:
        L_total = L_pred + λ_1*L_target + λ_2*L_feat

        Where:
        - L_pred: BCE/Cross-entropy loss on training data
        - L_target: Global constraint loss using rational saturation
        - L_feat: Local/sector constraint loss using rational saturation

        For each constraint: L = E / (E + K)
        where E = ReLU(N_predicted - K)

        CRITICAL: N_predicted uses HARD predictions (argmax), not soft probabilities
        Gradients maintained via straight-through estimator
        """
        device = logits.device

        # ===================================================================
        # 1. Prediction Loss (L_pred) - BCE/CrossEntropy on training data
        # ===================================================================
        L_pred = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_pred = self.ce_loss(logits, y_true)

        # Get predictions: hard for counting, soft for gradients
        y_proba = F.softmax(logits, dim=1)  # Soft probabilities for gradient flow
        y_pred_hard = torch.argmax(logits, dim=1)  # Hard predictions for actual counts

        # ===================================================================
        # 2. Target Constraint Loss (L_target) - Global constraints
        # ===================================================================
        # Formula: L_target = (1/C) * Σ_c [E_c / (E_c + K_c)]
        # where E_c = ReLU(N_predicted_c - K_c)
        # and N_predicted_c is the HARD count of students predicted for class c
        L_target = torch.tensor(0.0, device=device)
        if self.global_constraints is not None:
            g_cons = self.global_constraints.to(device)
            n_constrained = 0

            for class_id in range(3):
                # Skip unconstrained classes (marked with very large K)
                if g_cons[class_id] > 1e9:
                    continue

                # Hard count: actual number of students predicted for this class
                hard_count = (y_pred_hard == class_id).sum().float()

                # Soft count: sum of probabilities (for gradient flow)
                soft_count = y_proba[:, class_id].sum()

                # Straight-through estimator: forward uses hard, backward uses soft
                # Gradients flow through soft_count, not hard_count
                N_predicted = soft_count + (hard_count - soft_count).detach()

                # Rational saturation formula: E / (E + K)
                K = g_cons[class_id]
                E = torch.relu(N_predicted - K)  # Excess over constraint
                constraint_loss = E / (E + K + self.eps)

                L_target = L_target + constraint_loss
                n_constrained += 1

            # Average over constrained classes only
            if n_constrained > 0:
                L_target = L_target / n_constrained

        # ===================================================================
        # 3. Feature/Sector Constraint Loss (L_feat) - Local per-course constraints
        # ===================================================================
        # Formula: L_feat = (1/(M*C)) * Σ_j Σ_c [E_jc / (E_jc + K_jc)]
        # where j indexes courses/sectors, c indexes classes
        L_feat = torch.tensor(0.0, device=device)
        if self.local_constraint_dict is not None and group_ids is not None:
            group_ids_device = group_ids.to(device)
            n_constrained = 0

            for group_id, buffer_name in self.local_constraint_dict.items():
                # Get mask for students in this course/sector
                group_mask = (group_ids_device == group_id)

                if group_mask.sum() == 0:
                    continue

                # Predictions for this group only
                group_preds_hard = y_pred_hard[group_mask]
                group_proba = y_proba[group_mask]
                l_cons = getattr(self, buffer_name).to(device)

                for class_id in range(3):
                    # Skip unconstrained classes
                    if l_cons[class_id] > 1e9:
                        continue

                    # Hard count for this group and class
                    hard_count = (group_preds_hard == class_id).sum().float()

                    # Soft count for gradients
                    soft_count = group_proba[:, class_id].sum()

                    # Straight-through estimator
                    N_predicted = soft_count + (hard_count - soft_count).detach()

                    # Rational saturation formula
                    K = l_cons[class_id]
                    E = torch.relu(N_predicted - K)
                    constraint_loss = E / (E + K + self.eps)

                    L_feat = L_feat + constraint_loss
                    n_constrained += 1

            # Average over all (group, class) combinations
            if n_constrained > 0:
                L_feat = L_feat / n_constrained

        # ===================================================================
        # 4. Total Loss: L_total = L_pred + λ_1*L_target + λ_2*L_feat
        # ===================================================================
        # With λ_1 = λ_2 = 1.0 (equal weights)
        L_total = L_pred + self.lambda_global * L_target + self.lambda_local * L_feat

        return L_total, L_pred, L_target, L_feat
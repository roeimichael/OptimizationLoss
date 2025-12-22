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
        Forward pass with robust Rational Saturation handling.
        """
        y_proba = F.softmax(logits, dim=1)
        device = logits.device

        # 1. CE Loss
        L_ce = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_ce = self.ce_loss(logits, y_true)

        # 2. Global Constraints
        L_global = torch.tensor(0.0, device=device)
        if self.global_constraints is not None:
            class_counts = y_proba.sum(dim=0)
            g_cons = self.global_constraints.to(device)

            # Excess: ReLU(Predicted - Constraint)
            excess = torch.relu(class_counts - g_cons)

            # Rational Saturation: E / (E + K)
            # Safe division: g_cons is now guaranteed to be non-NaN.
            # If g_cons is 1e10 (unconstrained), excess is 0, result is 0.
            constraint_loss = excess / (excess + g_cons + self.eps)
            L_global = constraint_loss.mean()

        # 3. Local Constraints
        L_local = torch.tensor(0.0, device=device)
        if self.local_constraint_dict is not None and group_ids is not None:
            total_loss = 0.0
            num_groups = 0
            group_ids_device = group_ids.to(device)

            for group_id, buffer_name in self.local_constraint_dict.items():
                group_mask = (group_ids_device == group_id)

                # Skip if group not present in this batch/set
                if group_mask.sum() == 0:
                    continue

                group_proba = y_proba[group_mask]
                group_class_counts = group_proba.sum(dim=0)

                l_cons = getattr(self, buffer_name).to(device)

                excess = torch.relu(group_class_counts - l_cons)
                constraint_loss = excess / (excess + l_cons + self.eps)

                total_loss += constraint_loss.mean()
                num_groups += 1

            # Avoid division by zero if no groups were found
            L_local = total_loss / max(num_groups, 1)

        L_total = L_ce + L_global + L_local

        return L_total, L_ce, L_global, L_local
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

        if self.use_ce and y_true is not None:
            L_pred = self.ce_loss(logits, y_true)

        if self.global_constraints is not None:
            g_cons = self.global_constraints.to(device)



        if self.local_constraint_dict is not None and group_ids is not None:
            group_ids_device = group_ids.to(device)

            for group_id, buffer_name in self.local_constraint_dict.items():
                group_mask = (group_ids_device == group_id)

                if group_mask.sum() == 0:
                    continue

                group_proba = y_proba[group_mask]
                l_cons = getattr(self, buffer_name).to(device)





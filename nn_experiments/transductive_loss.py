import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassTransductiveLoss(nn.Module):
    def __init__(self, global_constraints, local_constraints,
                 lambda_global=1.0, lambda_local=1.0, use_ce=True):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.use_ce = use_ce
        self.eps = 1e-6

        if global_constraints is not None:
            global_constraints = [c if c is not None else 1e10 for c in global_constraints]
            self.register_buffer('global_constraints',
                                torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.global_constraints = None

        if local_constraints is not None:
            self.local_constraint_dict = {}
            for group_id, constraints in local_constraints.items():
                constraints = [c if c is not None else 1e10 for c in constraints]
                self.register_buffer(f'local_constraint_{group_id}',
                                   torch.tensor(constraints, dtype=torch.float32))
                self.local_constraint_dict[group_id] = f'local_constraint_{group_id}'
        else:
            self.local_constraint_dict = None

        if use_ce:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, y_true=None, group_ids=None):
        y_proba = F.softmax(logits, dim=1)
        device = logits.device

        L_ce = torch.tensor(0.0, device=device)
        if self.use_ce and y_true is not None:
            L_ce = self.ce_loss(logits, y_true)

        L_global = torch.tensor(0.0, device=device)
        if self.global_constraints is not None:
            class_counts = y_proba.sum(dim=0)
            global_constraints_device = self.global_constraints.to(device)
            excess = torch.relu(class_counts - global_constraints_device)
            constraint_loss = excess / (excess + global_constraints_device + self.eps)
            L_global = constraint_loss.mean()

        L_local = torch.tensor(0.0, device=device)
        if self.local_constraint_dict is not None and group_ids is not None:
            total_loss = 0.0
            num_groups = 0
            group_ids_device = group_ids.to(device) if not group_ids.is_cuda else group_ids

            for group_id, buffer_name in self.local_constraint_dict.items():
                group_mask = (group_ids_device == group_id)
                if group_mask.sum() == 0:
                    continue

                group_proba = y_proba[group_mask]
                group_class_counts = group_proba.sum(dim=0)
                constraints_tensor = getattr(self, buffer_name).to(device)
                excess = torch.relu(group_class_counts - constraints_tensor)
                constraint_loss = excess / (excess + constraints_tensor + self.eps)
                total_loss += constraint_loss.mean()
                num_groups += 1

            L_local = total_loss / max(num_groups, 1)

        L_total = L_ce + self.lambda_global * L_global + self.lambda_local * L_local

        return L_total, L_ce, L_global, L_local

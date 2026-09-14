import csv
import logging
import torch
import torch.nn.functional as F
from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.reordering import capped_scores, reordering_report
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def count_excess(hard_preds, groups_np, constrained_classes, global_con, local_con):
    excess = sum(
        (
            max(0, int((hard_preds == c).sum()) - int(global_con[c]))
            for c in constrained_classes
            if global_con[c] < UNLIMITED
        )
    )
    if local_con:
        for g_id, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                    excess += max(0, gc - int(bounds[c]))
    return excess


def count_fields(constrained_classes):
    return [
        f"{p}_Class{c}"
        for c in sorted(constrained_classes)
        for p in ("Limit", "Hard", "Soft")
    ]


def count_row(hard_preds, total_soft, constrained_classes, global_con):
    row = {}
    soft = total_soft.detach().cpu().numpy()
    for c in sorted(constrained_classes):
        lim = global_con[c]
        row[f"Limit_Class{c}"] = int(lim) if lim < UNLIMITED else UNLIMITED
        row[f"Hard_Class{c}"] = int((hard_preds == c).sum())
        row[f"Soft_Class{c}"] = float(soft[c])
    return row


class TrainingState:
    """Observed satisfaction and applied updates, without model selection."""

    def __init__(self, tag):
        self.tag = tag
        self.satisfaction_epoch = None
        self.steps_applied = 0
        self.steps_attempted = 0

    def record_step(self, applied):
        self.steps_attempted += 1
        self.steps_applied += int(applied)

    def record(self, satisfied, epoch):
        if satisfied and self.satisfaction_epoch is None:
            self.satisfaction_epoch = epoch + 1
            log.info("%s: first satisfaction at epoch %d", self.tag, epoch + 1)


def open_epoch_log(experiment_path, fields):
    path = experiment_path / "training_log.csv"
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    def append(row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

    return append


def read_step_config(hp):
    mode = str(hp.get("constraint_grad_mode", "clip"))
    if mode not in ("clip", "normalize"):
        raise ValueError("constraint_grad_mode must be clip / normalize, got %r" % mode)
    return {
        "clip": _required(hp, "constraint_grad_clip"),
        "mode": mode,
        "fp32": bool(hp.get("constraint_fp32", False)),
    }


def dual_setup(model, inputs, device, lr, batch_size):
    return (
        make_optimizer(model.parameters(), lr, device),
        make_ce_criterion(inputs.config, inputs.y_train, inputs.num_classes, device),
        make_dataloader(inputs.X_train, inputs.y_train, batch_size),
    )


def ce_epoch(
    model, train_loader, optimizer, criterion_ce, device, amp_dtype, use_amp, scaler
):
    model.train()
    ce_losses = []
    correct = seen = 0
    for batch_X, batch_y in train_loader:
        (batch_X, batch_y) = (batch_X.to(device), batch_y.to(device))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits_ce = model(batch_X)
            ce_loss = criterion_ce(logits_ce, batch_y)
        ce_losses.append(ce_loss.item())
        with torch.no_grad():
            correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
            seen += batch_y.size(0)
        if scaler:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            ce_loss.backward()
            optimizer.step()
    return (ce_losses, correct / seen if seen else 0.0)


def transductive_counts(
    model, X_test_dev, groups_np, unique_groups, num_classes, chunk_size, device
):
    model.eval()
    total_soft = torch.zeros(num_classes, device=device)
    group_soft = {
        int(g): torch.zeros(num_classes, device=device) for g in unique_groups
    }
    all_hard = []
    with torch.no_grad():
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i : i + chunk_size])
            chunk_proba = F.softmax(chunk_logits, dim=1)
            total_soft += chunk_proba.sum(dim=0)
            all_hard.append(chunk_logits.argmax(dim=1))
            chunk_groups = groups_np[i : i + chunk_size]
            for g in unique_groups:
                mask = chunk_groups == g
                if mask.any():
                    group_soft[int(g)] += chunk_proba[mask].sum(dim=0)
        hard_preds = torch.cat(all_hard).cpu().numpy()
    return (total_soft, group_soft, hard_preds)


def run_dual_arm(inputs: TrainInputs, train_constraints, tag) -> TrainOutputs:
    hp = inputs.hyperparams
    model = inputs.model
    device = inputs.device
    chunk_size = _required(hp, "constraint_chunk_size", int)
    warmup_scores = capped_scores(
        model, inputs.X_test, inputs.constrained_classes, chunk_size
    )
    ck = train_constraints(model, inputs, device)
    reorder = reordering_report(
        model, inputs.X_test, warmup_scores, inputs.constrained_classes, chunk_size
    )
    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": ck.satisfaction_epoch,
            "constraint_steps_applied": int(ck.steps_applied),
            "constraint_steps_attempted": int(ck.steps_attempted),
            "reordering": reorder,
        },
    )

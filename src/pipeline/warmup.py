"""Shared CE warmup phase + small builder helpers.

Merges the three near-identical warmup loops that lived in
ConstraintTrainer.train_warmup, run_heuristic.train_fixed_warmup, and
run_fioretto._train_warmup. They diverged only in log strings and on
whether to write per-epoch CSV rows.
"""

import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from src.pipeline.setup import setup_runtime
from src.training.logging import log_progress_to_csv
from src.training.metrics import compute_train_accuracy
from src.training.model_cache import load_from_cache, save_to_cache

log = logging.getLogger(__name__)


def make_ce_criterion(config, y_train, num_classes, device):
    """Plain CE, class-weighted CE (hp['class_weighted_ce']), or an imbalanced-
    learning training loss (hp['warmup_loss'] in focal/class_balanced/
    logit_adjust). TMLR Track B / B1 baselines train the backbone WITH the
    imbalanced objective in this shared warmup phase, then LP-clip."""
    hp = config["hyperparams"]
    warmup_loss = hp.get("warmup_loss", "ce")
    if warmup_loss != "ce":
        from src.losses.imbalanced_losses import build_warmup_criterion
        return build_warmup_criterion(warmup_loss, y_train, num_classes, device, hp)
    if not hp.get("class_weighted_ce", False):
        return nn.CrossEntropyLoss()
    counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights = weights / weights.sum() * num_classes
    log.info("Using class-weighted CE: weights=%s", weights.cpu().numpy().round(3))
    return nn.CrossEntropyLoss(weight=weights)


def make_optimizer(params, lr, device):
    use_fused = device.type == "cuda" and hasattr(torch.optim.Adam, "fused")
    try:
        return torch.optim.Adam(params, lr=lr, fused=use_fused)
    except Exception:
        return torch.optim.Adam(params, lr=lr)


def make_dataloader(X, y, batch_size):
    use_workers = os.name != "nt"
    n_workers = 2 if use_workers else 0
    return DataLoader(
        TensorDataset(X, y), batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=use_workers and n_workers > 0,
    )


def run_warmup(config, num_classes, X_train, y_train, device,
               *, input_dim=None, csv_log_path=None):
    """CE-only warmup phase. Loads from cache if available, else trains and saves.

    Returns (model, from_cache). When from_cache=True, no training ran.

    csv_log_path: optional. tralo passes its training_log.csv so the
    warmup phase appears in the log alongside constraint epochs. heuristic +
    fioretto pass None (their warmup is not logged per-epoch to CSV).
    """
    cache_id = config["base_model_id"]
    hp = config["hyperparams"]

    cached = load_from_cache(cache_id, config, input_dim, num_classes, device)
    if cached is not None:
        log.info("Loaded cached warmup model: %s", cache_id)
        return cached, True

    use_amp, amp_dtype, scaler = setup_runtime(device)

    model = get_model(
        config["model_name"], input_dim=input_dim, n_classes=num_classes,
        dropout=hp["dropout"], pretrained=hp.get("pretrained", False),
    ).to(device)

    criterion = make_ce_criterion(config, y_train, num_classes, device)
    optimizer = make_optimizer(model.parameters(), hp["lr"], device)
    loader = make_dataloader(X_train, y_train, hp["batch_size"])

    warmup_epochs = hp["warmup_epochs"]
    log_interval = max(1, warmup_epochs // 5)
    n_batches = len(loader)
    log.info("Warmup: %d epochs, %d batches/epoch (batch_size=%d, samples=%d)",
             warmup_epochs, n_batches, hp["batch_size"], len(X_train))
    log.info("AMP: enabled=%s dtype=%s scaler=%s", use_amp, amp_dtype, scaler is not None)

    epoch_times = []
    for epoch in range(warmup_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(batch_X), batch_y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        should_log = epoch < 3 or (epoch + 1) % log_interval == 0 or epoch == warmup_epochs - 1
        if should_log:
            avg_loss = epoch_loss / n_batches
            train_acc = compute_train_accuracy(model, loader, device)
            if csv_log_path:
                log_progress_to_csv(csv_log_path, epoch, avg_loss, train_acc, num_classes=num_classes)
            log.info("Warmup %d/%d: loss=%.4f acc=%.4f [%.2fs/epoch]",
                     epoch + 1, warmup_epochs, avg_loss, train_acc, epoch_elapsed)

    avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    log.info("Warmup done: avg=%.2fs/epoch total=%.1fs", avg_epoch, sum(epoch_times))
    save_to_cache(model, cache_id, config)
    return model, False

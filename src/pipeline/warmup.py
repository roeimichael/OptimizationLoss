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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.models import get_model
from src.training.rank_loss import budgeted_rank_loss, read_rank_config
from src.pipeline.setup import setup_runtime
from src.training.logging import log_progress_to_csv
from src.training.metrics import compute_train_accuracy
from src.training.model_cache import load_from_cache, save_to_cache

log = logging.getLogger(__name__)


def make_ce_criterion(config, y_train, num_classes, device):
    """Ordinary CE or the declared focal warm-up objective."""
    hp = config["hyperparams"]
    warmup_loss = hp.get("warmup_loss", "ce")
    if warmup_loss != "ce":
        from src.losses.imbalanced_losses import build_warmup_criterion
        return build_warmup_criterion(warmup_loss, y_train, num_classes, device, hp)
    return nn.CrossEntropyLoss()


def make_optimizer(params, lr, device):
    # `fused` is a constructor kwarg, not a class attribute -- the old
    # hasattr(torch.optim.Adam, "fused") probe was False on every GPU, so the
    # fused path never ran. The try/except below is the real capability check.
    use_fused = device.type == "cuda"
    try:
        return torch.optim.Adam(params, lr=lr, fused=use_fused)
    except (RuntimeError, TypeError):
        # Narrowed from a bare `except Exception`. These are the two the
        # capability probe can legitimately raise -- TypeError on a torch too
        # old to know the kwarg, RuntimeError when the build rejects fused on
        # this device. Anything else is a real failure and must propagate
        # rather than be swallowed into a silently slower optimizer.
        return torch.optim.Adam(params, lr=lr)


class AugmentedTensors(Dataset):
    """Random horizontal flip + reflect-padded random crop, per item.

    WHY THIS EXISTS. The trainer had no augmentation of ANY kind --
    `make_dataloader` wrapped a bare `TensorDataset` over the preprocessed
    arrays -- and the model then memorised the train set within 2-5 epochs on
    every backbone/dataset pair measured (`scripts/saturation_gate.py`:
    ViTB16/fmow2 reaches 90.6% train accuracy after ONE constraint epoch). From
    that point cross-entropy is ~0, and because `constraint_step` rescales the
    constraint gradient to a FIXED norm however small the violation is, every
    remaining constraint step is full-size with nothing opposing it. The
    constraint is not reshaping a boundary, it is kicking a frozen one, and no
    arm comparison made in that regime means anything.

    Flip and padded crop ONLY, and deliberately so: both are geometric, so they
    are safe to apply AFTER the ImageNet normalisation the loader has already
    done. A colour transform would not be -- it would have to run before it.
    Both are tensor slices, so the cost is negligible next to the forward pass.
    """

    def __init__(self, X, y, pad=16, groups=None):
        self.X, self.y, self.pad = X, y, int(pad)
        self.groups = groups

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if float(torch.rand(())) < 0.5:
            x = torch.flip(x, dims=(-1,))
        h, w = x.shape[-2], x.shape[-1]
        # reflect padding requires pad < the dimension it pads, so a fixed 16
        # raises on any image under 17px. That is not hypothetical: it made
        # `aug_tralo` UNRUNNABLE and smoke_arms, whose input is 8x8, is the only
        # reason it was caught before the campaign burned three hours on it.
        p = min(self.pad, h - 1, w - 1)
        if p > 0:
            x = F.pad(x.unsqueeze(0), (p, p, p, p), mode="reflect").squeeze(0)
            top = int(torch.randint(0, 2 * p + 1, ()))
            left = int(torch.randint(0, 2 * p + 1, ()))
            x = x[..., top:top + h, left:left + w]
        if self.groups is None:
            return x, self.y[i]
        return x, self.y[i], self.groups[i]


def make_dataloader(X, y, batch_size, augment=False, groups=None):
    use_workers = os.name != "nt"
    n_workers = 2 if use_workers else 0
    # augment=False must stay byte-identical to the unaugmented path: every
    # stored result was produced by it, and a silent change here would make the
    # corpus incomparable rather than merely different.
    #
    # `groups` is opt-in and defaults to None, so every existing arm keeps
    # yielding 2-tuples and its exact byte stream. The sampler draws from a
    # generator that does not see the dataset's contents, so carrying a third
    # tensor cannot perturb the shuffle order either -- which is what makes it
    # safe to add to a corpus whose comparability rests on bit-determinism.
    if groups is None:
        ds = AugmentedTensors(X, y) if augment else TensorDataset(X, y)
    else:
        g = torch.as_tensor(groups, dtype=torch.long)
        ds = (AugmentedTensors(X, y, groups=g) if augment
              else TensorDataset(X, y, g))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=use_workers and n_workers > 0,
    )


def run_warmup(config, num_classes, X_train, y_train, device,
               *, csv_log_path=None, groups_train=None):
    """CE-only warmup phase. Loads from cache if available, else trains and saves.

    Returns (model, from_cache). When from_cache=True, no training ran.

    csv_log_path: the run's training_log.csv, so the warm-up epochs appear in
    the log alongside the constraint epochs. run_experiment passes it for every
    arm.
    """
    cache_id = config["base_model_id"]
    hp = config["hyperparams"]

    cached = load_from_cache(cache_id, config, num_classes, device)
    if cached is not None:
        log.info("Loaded cached warmup model: %s", cache_id)
        return cached, True

    use_amp, amp_dtype, scaler = setup_runtime(device)

    model = get_model(
        config["model_name"], n_classes=num_classes,
        dropout=hp["dropout"], pretrained=hp.get("pretrained", False),
    ).to(device)

    criterion = make_ce_criterion(config, y_train, num_classes, device)
    optimizer = make_optimizer(model.parameters(), hp["lr"], device)
    # The budgeted ranking loss needs train-side groups. It is opt-in, and
    # `rank_weight` is a warm-up IDENTITY key -- without that, a ranking arm
    # would silently load the plain arm's cached warm-up and be byte-identical
    # to it, which is precisely how an inert flag looks healthy in the logs.
    rank_cfg = read_rank_config(hp)
    rank_groups = groups_train if rank_cfg["weight"] > 0 else None
    loader = make_dataloader(X_train, y_train, hp["batch_size"],
                             augment=hp.get("augment", False),
                             groups=rank_groups)
    rank_classes = config.get("dataset_config", {}).get("constrained_class")
    if rank_classes is not None and not isinstance(rank_classes, list):
        rank_classes = [rank_classes]
    _cap = (config.get("constraint") or [1.0, 1.0])[0]
    rank_frac = float(_cap) if not isinstance(_cap, list) else 1.0
    # `rank_cap_fraction` is the value the WARM-UP CACHE was keyed on. If it
    # ever disagreed with the cap actually being trained, a cached warm-up would
    # be reused for a cut it was not trained at -- which is exactly the bug this
    # key was added to close, reappearing through the back door. Refuse rather
    # than train something the digest does not describe.
    if rank_cfg["weight"] > 0 and "rank_cap_fraction" in hp:
        stamped = float(hp["rank_cap_fraction"])
        if abs(stamped - rank_frac) > 1e-12:
            raise ValueError(
                "rank_cap_fraction %r was hashed into base_model_id but the cap "
                "being trained is %r; the warm-up cache would be keyed on a cut "
                "this run does not use" % (stamped, rank_frac))

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
        for batch in loader:
            if len(batch) == 3:
                batch_X, batch_y, batch_g = batch
                batch_g = batch_g.to(device)
            else:
                (batch_X, batch_y), batch_g = batch, None
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                if batch_g is not None and rank_classes:
                    loss = loss + rank_cfg["weight"] * budgeted_rank_loss(
                        logits, batch_y, batch_g, rank_classes, rank_frac,
                        margin=rank_cfg["margin"],
                        min_group=rank_cfg["min_group"])
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

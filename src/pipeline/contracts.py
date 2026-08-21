"""Contracts between the shared pipeline and per-methodology train functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainInputs:
    model: nn.Module
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: np.ndarray
    group_ids: np.ndarray
    global_con: List[float]
    local_con: Dict[int, list]
    constrained_classes: List[int]
    num_classes: int
    config: Dict[str, Any]
    hyperparams: Dict[str, Any]
    device: torch.device
    experiment_path: Path
    csv_log_path: Path


@dataclass
class TrainOutputs:
    model: nn.Module
    summary: Dict[str, Any]
    skip_targeted_correction: bool = False
    precomputed_predictions: Optional[np.ndarray] = None


def _required(hp, key, cast=float):
    """Read a protocol value that must never fall back to an inline default.

    The inline defaults this replaces were the retracted ones -- lr_constraint 1e-5
    against the protocol's 1e-4, constraint_epochs 150 against 29,
    stable_count_threshold 5 against 31 (low enough that the early stop would
    actually fire). A missing key is a generator bug; failing loudly is the
    only safe behaviour.
    """
    if key not in hp:
        raise KeyError(
            "%s is required and has no safe default. configs/protocol.yml is "
            "the source of truth; generate the campaign with "
            "configs.gen_campaign rather than hand-writing a config." % key)
    return cast(hp[key])

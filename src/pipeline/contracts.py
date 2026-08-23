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

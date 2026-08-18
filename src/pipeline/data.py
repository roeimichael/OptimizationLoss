"""Shared data loading: tensors + groups + constraint limits + constrained_classes."""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from src.utils.data_loader import load_experiment_data

log = logging.getLogger(__name__)


@dataclass
class LoadedData:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: np.ndarray
    groups_test: np.ndarray
    global_con: List[float]
    local_con: Dict[int, list]
    constrained_classes: List[int]
    num_classes: int


def _to_numpy(arr):
    return arr.values if hasattr(arr, "values") else arr


def load_data(config) -> LoadedData:
    """Load data, build CPU tensors, derive constrained_classes list."""
    t0 = time.time()
    raw = load_experiment_data(config)
    X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = raw
    log.info("TIMING data_load=%.2fs train=%s test=%s",
             time.time() - t0, X_train.shape, X_test.shape)

    ds = config.get("dataset_config", {})
    # No default. data_loader raises on a missing constrained_class, so
    # defaulting to num_classes-1 here meant the two halves of the same load
    # disagreed -- and a silent cap on whichever class happens to be last is
    # exactly the kind of thing that gets measured for a week before anyone
    # notices.
    if "constrained_class" not in ds:
        raise KeyError(
            "dataset_config.constrained_class is required; there is no sensible "
            "default for which class to cap.")
    constrained_class = ds["constrained_class"]
    if isinstance(constrained_class, (list, tuple)):
        constrained_classes = list(constrained_class)
    else:
        constrained_classes = [constrained_class]

    return LoadedData(
        X_train=torch.FloatTensor(X_train),
        y_train=torch.LongTensor(_to_numpy(y_train)),
        X_test=torch.FloatTensor(X_test),
        y_test=_to_numpy(y_test),
        groups_test=_to_numpy(groups_test),
        global_con=global_con,
        local_con=local_con,
        constrained_classes=constrained_classes,
        num_classes=num_classes,
    )

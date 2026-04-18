# Data loading and constraint computation for imagery experiments.
# Loads npy image arrays, applies ImageNet normalization, computes constraint limits.

import logging
import os

import numpy as np
import pandas as pd

from src.training.constraints import compute_global_constraints, compute_local_constraints

log = logging.getLogger(__name__)

IMAGERY_DATASETS = {'dermmnist', 'tissuemnist', 'cifar100'}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _ensure_3channel(images):
    if images.ndim == 4 and images.shape[1] == 1:
        # Use contiguous copy instead of np.repeat to avoid 3x peak memory.
        # np.broadcast_to is zero-copy but returns read-only view;
        # we need a writable array for in-place normalization.
        return np.ascontiguousarray(np.broadcast_to(images, (images.shape[0], 3, images.shape[2], images.shape[3])))
    return images


def _apply_imagenet_normalization(images):
    images -= IMAGENET_MEAN
    images /= IMAGENET_STD
    return images


def _load_imagery_data(config):
    ds = config['dataset_config']
    data_dir = ds['data_dir']
    num_classes = ds['num_classes']
    constrained_class = ds['constrained_class']
    group_col = ds.get('group_column', 'sex')
    dataset_mode = config.get('dataset_mode', 'unknown')
    X_train = _ensure_3channel(np.load(os.path.join(data_dir, 'train_images.npy')))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    X_test = _ensure_3channel(np.load(os.path.join(data_dir, 'test_images.npy')))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    X_train = _apply_imagenet_normalization(X_train)
    X_test = _apply_imagenet_normalization(X_test)
    test_meta = pd.read_csv(os.path.join(data_dir, 'test_meta.csv'))
    groups_test = test_meta[group_col].values.astype(np.int64)
    local_percent, global_percent = config['constraint']
    test_df = pd.DataFrame({'label': y_test, group_col: groups_test})
    global_con = compute_global_constraints(
        test_df, 'label', global_percent,
        constrained_class=constrained_class, num_classes=num_classes)
    local_con = compute_local_constraints(
        test_df, 'label', local_percent, group_col,
        constrained_class=constrained_class, num_classes=num_classes)
    log.info("mode=%s classes=%d constrained=%s global=%s local_groups=%d test=%d train=%d",
             dataset_mode, num_classes, constrained_class, global_con,
             len(local_con), len(y_test), len(y_train))
    return (X_train, X_test, y_train, y_test,
            groups_test, global_con, local_con, num_classes)


def load_experiment_data(config):
    dataset_mode = config.get('dataset_mode', 'binary')
    if dataset_mode in IMAGERY_DATASETS:
        return _load_imagery_data(config)
    else:
        raise ValueError(f"Unknown dataset_mode='{dataset_mode}'. Supported: {IMAGERY_DATASETS}")

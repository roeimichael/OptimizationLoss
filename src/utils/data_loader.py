"""Data loading and constraint computation for experiments."""

import logging
import os

import numpy as np
import pandas as pd

from src.training.constraints import compute_global_constraints, compute_local_constraints

log = logging.getLogger(__name__)


# ── Tabular (CSV-based) loading — commented out, not used for DermMNIST ──────
#
# from sklearn.preprocessing import LabelEncoder
#
# def load_presplit_data(train_path, test_path):
#     """Load pre-split train and test CSVs."""
#     return pd.read_csv(train_path), pd.read_csv(test_path)
#
#
# def encode_categorical_features(train_df, test_df):
#     """LabelEncode all object columns."""
#     train_enc, test_enc = train_df.copy(), test_df.copy()
#     for col in train_df.select_dtypes(include=['object']).columns:
#         le = LabelEncoder()
#         train_enc[col] = le.fit_transform(train_df[col].astype(str))
#         test_enc[col] = test_df[col].astype(str).map(
#             lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
#         )
#     return train_enc, test_enc
#
#
# def _load_tabular_data(config):
#     """Load tabular CSV data. Returns 8-tuple."""
#     ds = config['dataset_config']
#     train_df, test_df = load_presplit_data(ds['train_path'], ds['test_path'])
#     train_df, test_df = encode_categorical_features(train_df, test_df)
#
#     target_col = ds['target_column']
#     group_col = ds['group_column']
#     num_classes = ds['num_classes']
#     constrained_class = ds['constrained_class']
#
#     local_percent, global_percent = config['constraint']
#
#     global_con = compute_global_constraints(
#         test_df, target_col, global_percent,
#         constrained_class=constrained_class, num_classes=num_classes)
#     local_con = compute_local_constraints(
#         test_df, target_col, local_percent, group_col,
#         constrained_class=constrained_class, num_classes=num_classes)
#
#     log.info("mode=tabular classes=%d constrained=%d global=%s local_groups=%d",
#              num_classes, constrained_class, global_con, len(local_con))
#
#     drop_cols = [target_col, group_col]
#     return (train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols),
#             train_df[target_col], test_df[target_col],
#             test_df[group_col], global_con, local_con, num_classes)


# ── Imagery (npy-based) loading ──────────────────────────────────────────────

IMAGERY_DATASETS = {'dermmnist', 'tissuemnist'}


def _ensure_3channel(images):
    """Convert single-channel grayscale (N, 1, H, W) to 3-channel (N, 3, H, W).

    RGB images (N, 3, H, W) pass through unchanged. This lets pretrained models
    (ResNet18, MobileNetV3) accept grayscale datasets without architecture changes.
    """
    if images.ndim == 4 and images.shape[1] == 1:
        return np.repeat(images, 3, axis=1)
    return images


def _load_imagery_data(config):
    """Load imagery dataset from npy files + CSV metadata. Returns 8-tuple.

    Expected files in data_dir:
        train_images.npy  (N, C, H, W) float32 [0, 1]  — C=1 (grayscale) or C=3 (RGB)
        train_labels.npy  (N,) int
        test_images.npy   (N, C, H, W) float32 [0, 1]
        test_labels.npy   (N,) int
        test_meta.csv     must contain label column + group column

    Grayscale images are automatically expanded to 3-channel for model compatibility.
    """
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

    # Load metadata for test set (contains group column)
    test_meta = pd.read_csv(os.path.join(data_dir, 'test_meta.csv'))
    groups_test = test_meta[group_col].values.astype(np.int64)

    local_percent, global_percent = config['constraint']

    # Constraint computation using test labels + groups
    test_df = pd.DataFrame({'label': y_test, group_col: groups_test})
    global_con = compute_global_constraints(
        test_df, 'label', global_percent,
        constrained_class=constrained_class, num_classes=num_classes)
    local_con = compute_local_constraints(
        test_df, 'label', local_percent, group_col,
        constrained_class=constrained_class, num_classes=num_classes)

    log.info("mode=%s classes=%d constrained=%d global=%s local_groups=%d test=%d train=%d",
             dataset_mode, num_classes, constrained_class, global_con,
             len(local_con), len(y_test), len(y_train))

    return (X_train, X_test, y_train, y_test,
            groups_test, global_con, local_con, num_classes)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def load_experiment_data(config):
    """Load data based on dataset_mode in config. Returns 8-tuple:

    (X_train, X_test, y_train, y_test, groups_test,
     global_constraint, local_constraint, num_classes)
    """
    dataset_mode = config.get('dataset_mode', 'binary')

    if dataset_mode in IMAGERY_DATASETS:
        return _load_imagery_data(config)
    else:
        raise ValueError(f"Unknown dataset_mode='{dataset_mode}'. "
                         f"Supported: {IMAGERY_DATASETS}")
        # return _load_tabular_data(config)

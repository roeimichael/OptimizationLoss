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


# ── DermMNIST (image-based) loading ──────────────────────────────────────────

def _load_dermmnist_data(config):
    """Load DermMNIST npy images + labels. Returns 8-tuple (same interface).

    Images are already (N, 3, 64, 64) float32 [0, 1].
    No group column available — local constraints use global-only mode.
    """
    ds = config['dataset_config']
    data_dir = ds['data_dir']
    num_classes = ds['num_classes']
    constrained_class = ds['constrained_class']

    X_train = np.load(os.path.join(data_dir, 'train_images.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    local_percent, global_percent = config['constraint']

    # Constraint computation using a simple DataFrame
    test_df = pd.DataFrame({'label': y_test})
    global_con = compute_global_constraints(
        test_df, 'label', global_percent,
        constrained_class=constrained_class, num_classes=num_classes)

    # No demographic group column — empty local constraints
    local_con = {}

    # Dummy group column (single group) so the pipeline doesn't break
    groups_test = np.zeros(len(y_test), dtype=np.int64)

    log.info("mode=dermmnist classes=%d constrained=%d global=%s test=%d train=%d",
             num_classes, constrained_class, global_con, len(y_test), len(y_train))

    return (X_train, X_test, y_train, y_test,
            groups_test, global_con, local_con, num_classes)


# ── Dispatcher ───────────────────────────────────────────────────────────────

def load_experiment_data(config):
    """Load data based on dataset_mode in config. Returns 8-tuple:

    (X_train, X_test, y_train, y_test, groups_test,
     global_constraint, local_constraint, num_classes)
    """
    dataset_mode = config.get('dataset_mode', 'binary')

    if dataset_mode == 'dermmnist':
        return _load_dermmnist_data(config)
    else:
        raise ValueError(f"Tabular loading temporarily disabled. Use dataset_mode='dermmnist'.")
        # return _load_tabular_data(config)

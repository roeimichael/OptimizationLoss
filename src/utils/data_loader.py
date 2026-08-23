# Data loading and constraint computation for imagery experiments.
# Loads npy image arrays, applies ImageNet normalization, computes constraint limits.

import logging
import hashlib
import os

import numpy as np
import pandas as pd

from src.training.constraints import (compute_global_constraints,
                                      compute_local_constraints,
                                      normalize_constrained_classes)

log = logging.getLogger(__name__)

# The only dataset in scope (docs/FRAMEWORK.md section 2(n)). dermmnist,
# octmnist and tissuemnist were REMOVED 2026-08-22 after the screen measured
# that none of them can carry a count constraint: octmnist and tissuemnist
# build `synth_group` as `np.arange(len(y)) % 3`, so their groups are i.i.d.
# draws from one distribution and the local scope is empty BY CONSTRUCTION;
# dermmnist clears the screen at +65 items and still nulls, because its test
# groups are the training groups and the model has already learned their
# priors. aider, eurosat, retinamnist, bloodmnist, organamnist and the
# native-resolution variants were dropped earlier. None may come back without
# a `dataset_screen` number recorded in the framework.
IMAGERY_DATASETS = {'iwildcam', 'cct'}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _ensure_3channel(images):
    """Grayscale (N,1,H,W) -> (N,3,H,W). Assumes NCHW, and runs BEFORE the
    NHWC->NCHW coercion, so it has to recognise NHWC grayscale (N,H,W,1) and
    refuse it -- which it does, below, with a message naming the shape. The one
    case it cannot distinguish is H=1, where (N,1,W,1) is ambiguous; that is
    unreachable for the fixed 28x28 MedMNIST sources in scope."""
    if images.ndim == 4 and images.shape[-1] == 1 and images.shape[1] != 1:
        raise ValueError(
            "images look like NHWC grayscale %s. _ensure_3channel expects "
            "NCHW; coerce the layout first, or save the array as (N,1,H,W)."
            % (images.shape,))
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


def _coerce_imagery_layout(images):
    """Convert to float32 NCHW in [0,1]. Accept NHWC uint8 (GTSRB-style) or
    NCHW float32 (MedMNIST/AIDER-style) input."""
    if images.ndim == 4 and images.shape[-1] == 3 and images.shape[1] != 3:
        # NHWC -> NCHW
        images = np.transpose(images, (0, 3, 1, 2))
    if images.dtype == np.uint8:
        images = images.astype(np.float32, copy=False) / 255.0
    elif images.dtype != np.float32:
        images = images.astype(np.float32, copy=False)
    # The /255 was gated on uint8 alone, but the repo has two live storage
    # conventions: dermmnist is written as NCHW float32 ALREADY divided by 255,
    # the medmnist preps write NHWC uint8. A float32 array holding 0..255 -- a
    # re-prep that drops one division -- sailed through and every pixel came out
    # ~255x too large, with no error anywhere downstream.
    hi = float(images.max()) if images.size else 0.0
    if hi > 1.0 + 1e-3:
        raise ValueError(
            "images are float32 but max=%.3f, so they are NOT in [0,1]. Either "
            "they were written already-scaled and divided again, or written as "
            "0..255 float and never divided. Fix the prep script -- do not "
            "normalize this." % hi)
    return images


def data_fingerprint(y_train, y_test, groups_test):
    """Identity of the actual data behind a data_dir.

    Labels and groups only -- they are small, and any re-slice, re-split or
    re-shuffle moves them. Pixels are not hashed: it would cost seconds per run
    to catch a case (same labels, different images) that no prep script here
    can produce.
    """
    h = hashlib.md5()
    for arr in (np.asarray(y_train).ravel(), np.asarray(y_test).ravel(),
                np.asarray(groups_test).ravel()):
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _check_group_leakage(data_dir, group_column, must_be_disjoint):
    """MEASURE whether any TEST group also appears in TRAIN. Never assume it.

    This replaces a dermmnist-specific `lesion_id` check, because the hazard
    changed with the dataset. On a held-out-domain slice the entire premise is
    that the model has never seen the test groups: FRAMEWORK 2(n) selected
    iwildcam precisely because its 7 test cameras are disjoint from the 150
    training cameras, which is what makes the local cap the only source of a
    per-camera prior. One camera appearing on both sides silently restores the
    prior and turns the campaign back into dermmnist, where the same
    measurement nulled -- and nothing downstream would say so.

    `must_be_disjoint` comes from the dataset config, so a slice that is
    SUPPOSED to share groups is not flagged and one that is not supposed to
    raises rather than warns. A warning is the wrong severity for a fault that
    invalidates every number the run produces.
    """
    tr = os.path.join(data_dir, 'train_meta.csv')
    te = os.path.join(data_dir, 'test_meta.csv')
    if not (os.path.exists(tr) and os.path.exists(te)):
        return
    a, b = pd.read_csv(tr), pd.read_csv(te)
    if group_column not in a.columns or group_column not in b.columns:
        log.warning(
            "%s: `%s` is missing from one of the meta files, so train/test "
            "group overlap CANNOT be checked.", data_dir, group_column)
        return
    shared = set(a[group_column]) & set(b[group_column])
    if not shared:
        return
    hit = b[group_column].isin(shared)
    msg = ("%s: %d group(s) appear in BOTH splits, so %.1f%% of the test set "
           "comes from a group the model trained on."
           % (data_dir, len(shared), 100.0 * hit.mean()))
    if must_be_disjoint:
        raise ValueError(
            msg + " This slice declares `disjoint_groups: true`, which is the "
            "property the whole design rests on: with the groups shared the "
            "model already holds their priors and the local cap carries "
            "nothing new. Rebuild the split by GROUP.")
    log.warning(msg)


def _load_imagery_data(config):
    ds = config['dataset_config']
    data_dir = ds['data_dir']
    num_classes = ds['num_classes']
    constrained_class = ds['constrained_class']
    if 'group_column' not in ds:
        raise KeyError("dataset_config.group_column is required (e.g. 'synth_group' for "
                       "TissueMNIST, 'coarse_label' for CIFAR-100). The legacy 'sex' "
                       "default came from the Adult/Churn era and is no longer valid.")
    group_col = ds['group_column']
    dataset_mode = config.get('dataset_mode', 'unknown')
    X_train = _coerce_imagery_layout(_ensure_3channel(
        np.load(os.path.join(data_dir, 'train_images.npy'))))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    X_test = _coerce_imagery_layout(_ensure_3channel(
        np.load(os.path.join(data_dir, 'test_images.npy'))))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        if len(X) != len(y):
            raise ValueError(
                "%s_images.npy has %d rows but %s_labels.npy has %d. They are "
                "paired BY POSITION everywhere downstream. Train would have "
                "died late inside TensorDataset with an unlabelled "
                "AssertionError; test would not have raised at all -- the "
                "chunked loops key off len(X_test) and would simply score "
                "fewer items than the labels describe."
                % (split, len(X), split, len(y)))
    # Does this slice actually contain the dataset the config describes? None
    # of this was checked, so pointing data_dir at the wrong dataset produced a
    # complete, plausible run: the capped class is absent, K rounds to 0, and
    # constraints.py logs a warning rather than raising.
    _classes = sorted(set(np.asarray(y_train).ravel().tolist())
                      | set(np.asarray(y_test).ravel().tolist()))
    if _classes and (_classes[0] < 0 or _classes[-1] >= num_classes):
        raise ValueError(
            "%s: labels span %d..%d but dataset_config.num_classes is %d. This "
            "is a different dataset from the one the config describes, or "
            "num_classes is wrong. Every constraint is indexed by class id."
            % (data_dir, _classes[0], _classes[-1], num_classes))
    for _c in normalize_constrained_classes(ds['constrained_class']):
        if _c not in _classes:
            raise ValueError(
                "%s: constrained class %d does not occur in this slice "
                "(present: %s). The budget would round to K=0 on a class that "
                "is not there, the loss would have nothing to push down, and "
                "the run would complete looking healthy."
                % (data_dir, _c, _classes))
        n_pos = int((np.asarray(y_test).ravel() == _c).sum())
        log.info("capped class %d: %d of %d test items (%.1f%%)",
                 _c, n_pos, len(y_test), 100.0 * n_pos / max(1, len(y_test)))
    X_train = _apply_imagenet_normalization(X_train)
    X_test = _apply_imagenet_normalization(X_test)
    test_meta = pd.read_csv(os.path.join(data_dir, 'test_meta.csv'))
    if len(test_meta) != len(y_test):
        raise ValueError(
            "test_meta.csv has %d rows but test_labels.npy has %d entries; the "
            "group column is joined BY POSITION."
            % (len(test_meta), len(y_test)))
    # Every prep script writes a `label` column beside the group, and one of
    # them even asserts npz-vs-CSV alignment at write time -- but the loader
    # threw it away and trusted row order. A reordered meta file gives every
    # item the wrong group: wrong local budgets, wrong per-group metrics, and
    # Group_ID is written into final_predictions.csv so the scorer inherits it.
    if 'label' in test_meta.columns:
        meta_labels = test_meta['label'].to_numpy()
        if not np.array_equal(meta_labels, np.asarray(y_test).ravel()):
            n_bad = int((meta_labels != np.asarray(y_test).ravel()).sum())
            raise ValueError(
                "test_meta.csv `label` disagrees with test_labels.npy on %d of "
                "%d rows -- the two files are not row-aligned, so the group "
                "column would be assigned to the wrong items."
                % (n_bad, len(meta_labels)))
    else:
        log.warning("test_meta.csv has no `label` column, so the group join "
                    "cannot be verified. It is positional -- if the file was "
                    "ever rewritten, groups may be misaligned.")
    # The same check on the TRAIN split. This was written up as an accepted
    # risk on the reasoning that no train_meta.csv exists, so a
    # same-length-but-shuffled desync between train_images.npy and
    # train_labels.npy had no second source of truth to be caught against.
    # That reasoning was wrong: all three prep scripts DO write a
    # train_meta.csv with a `label` column, and it is on disk in every slice
    # (iwildcam label,class_name,filename,location
    # label,class_name,synth_group). The redundant signal was there all along.
    train_meta_path = os.path.join(data_dir, 'train_meta.csv')
    if os.path.exists(train_meta_path):
        train_meta = pd.read_csv(train_meta_path)
        if len(train_meta) != len(y_train):
            raise ValueError(
                "train_meta.csv has %d rows but train_labels.npy has %d."
                % (len(train_meta), len(y_train)))
        if 'label' in train_meta.columns:
            tm = train_meta['label'].to_numpy()
            if not np.array_equal(tm, np.asarray(y_train).ravel()):
                n_bad = int((tm != np.asarray(y_train).ravel()).sum())
                raise ValueError(
                    "train_meta.csv `label` disagrees with train_labels.npy on "
                    "%d of %d rows -- train_images.npy and train_labels.npy are "
                    "not row-aligned. The length check cannot see a permutation; "
                    "this can." % (n_bad, len(tm)))
        else:
            log.warning("train_meta.csv has no `label` column, so a permutation "
                        "of train_images.npy against train_labels.npy cannot be "
                        "detected.")
    else:
        log.warning("no train_meta.csv in %s, so a permutation of "
                    "train_images.npy against train_labels.npy cannot be "
                    "detected here. Every current prep script writes one.",
                    data_dir)
    if test_meta[group_col].isna().any():
        raise ValueError("group column %r contains nulls; .astype(int64) would "
                         "turn them into a huge negative group id" % group_col)
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
    _check_group_leakage(data_dir, group_col,
                         bool(ds.get('disjoint_groups', False)))
    config["data_fingerprint"] = data_fingerprint(y_train, y_test, groups_test)
    log.info("data fingerprint %s (%s)", config["data_fingerprint"], data_dir)
    return (X_train, X_test, y_train, y_test,
            groups_test, global_con, local_con, num_classes)


def load_experiment_data(config):
    if 'dataset_mode' not in config:
        raise KeyError("config.dataset_mode is required. The legacy 'binary' "
                       "default came from the Adult/Churn era and is no longer valid.")
    dataset_mode = config['dataset_mode']
    if dataset_mode in IMAGERY_DATASETS:
        return _load_imagery_data(config)
    else:
        raise ValueError(f"Unknown dataset_mode='{dataset_mode}'. Supported: {IMAGERY_DATASETS}")

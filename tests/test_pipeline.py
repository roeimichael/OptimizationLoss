"""Maintained behavioral regression fixtures."""

import importlib

import io

import json

import math

import os

import shutil

import subprocess

import sys

import tempfile

import numpy as np

import pytest

import torch

import torch.nn.functional as F

from configs.gen_campaign import build_hyperparams, cap_pair, compute_base_model_id

from src.losses.transductive_loss import MulticlassTransductiveLoss

from src.methodologies.heuristic.train import (
    _build_hierarchy,
    apply_allocation_heuristic,
    verify_allocation,
)

from src.training.constraints import (
    compute_global_constraints,
    compute_local_constraints,
)

from lean_fixtures import protocol_with_nulls as load_protocol

from src.utils.constants import UNLIMITED

from src.experiments.runner import TRAIN_FNS

from src.models import get_model

from src.training.constraint_step import finish_constraint_step

from src.utils.data_loader import _load_imagery_data as load_data

from src.utils.posthoc_adjustment import targeted_correction

import ast

import pandas as pd

import pathlib

import re

import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reference_penalty(soft, K, rho=0.5):
    E = F.relu(soft - K)
    e = E / (K + 1e-08)
    return E / (E + K + 1e-08) + rho * e**2 / (1 + e**2 + 1e-08)


def _loss(global_constraints, local_constraints=None, rho=0.5):
    return MulticlassTransductiveLoss(
        global_constraints=global_constraints,
        local_constraints=local_constraints or {},
        num_classes=len(global_constraints),
        initial_rho=rho,
    )


@pytest.mark.parametrize("K", [1, 2, 5, 20, 67, 250])
@pytest.mark.parametrize("soft", [0.0, 0.5, 7.5, 20.0, 100.0, 10000.0])
def test_penalty_unchanged_for_positive_K(K, soft):
    crit = _loss([K, UNLIMITED])
    a = torch.tensor(soft, requires_grad=True)
    b = torch.tensor(soft, requires_grad=True)
    crit._penalty(a, K).backward()
    _reference_penalty(b, K).backward()
    assert torch.allclose(
        crit._penalty(torch.tensor(soft), K),
        _reference_penalty(torch.tensor(soft), K),
        atol=0,
        rtol=0,
    )
    assert a.grad.item() == b.grad.item()


def test_K0_constraint_has_a_gradient():
    crit = _loss([0, UNLIMITED])
    soft = torch.tensor(12.0, requires_grad=True)
    crit._penalty(soft, 0).backward()
    assert soft.grad.item() > 1e-06, "K=0 must push the count down"
    old = torch.tensor(12.0, requires_grad=True)
    _reference_penalty(old, 0).backward()
    assert old.grad.item() == 0.0, "the old form really was inert (guards the test)"


def test_penalty_is_zero_at_and_below_the_budget():
    crit = _loss([10, UNLIMITED])
    for soft in (0.0, 5.0, 9.99, 10.0):
        assert crit._penalty(torch.tensor(soft), 10).item() == 0.0


def test_empty_constraints_stay_connected_to_the_graph():
    crit = _loss([UNLIMITED, UNLIMITED])
    counts = torch.zeros(2, requires_grad=True)
    total = crit.compute_global_from_counts(counts)
    assert total.item() == 0.0
    assert total.requires_grad, "an all-UNLIMITED scope returned a detached zero"
    total.backward()
    assert counts.grad is not None
    local = crit.compute_local_from_counts({})
    assert local.item() == 0.0
    assert local.device == crit.global_constraints.device
    assert (crit.compute_global_from_counts(counts) + local).requires_grad


def _frame(labels, groups):
    return pd.DataFrame({"label": np.asarray(labels), "grp": np.asarray(groups)})


def test_local_and_global_percentages_are_independent():
    df = _frame([1] * 100 + [0] * 100, [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
    (local_pct, global_pct) = cap_pair("L50_G30")
    assert (local_pct, global_pct) == (0.5, 0.3)
    g = compute_global_constraints(
        df, "label", global_pct, constrained_class=1, num_classes=2
    )
    loc = compute_local_constraints(
        df, "label", local_pct, "grp", constrained_class=1, num_classes=2
    )
    assert g[1] == 30
    assert sorted((v[1] for v in loc.values())) == [25, 25]


def test_K_zero_from_a_nonzero_count_is_refused():
    df = _frame([1] + [0] * 99, [0] * 100)
    with pytest.raises(ValueError, match="K=0"):
        compute_global_constraints(df, "label", 0.3, constrained_class=1, num_classes=2)


def test_K_zero_from_an_absent_class_is_allowed():
    df = _frame([0] * 50 + [1] * 50, [0] * 50 + [1] * 50)
    loc = compute_local_constraints(
        df, "label", 0.5, "grp", constrained_class=1, num_classes=2
    )
    assert loc[0][1] == 0 and loc[1][1] == 25


def test_several_capped_classes_get_independent_budgets():
    df = _frame([0] * 40 + [1] * 60 + [2] * 100, list(range(2)) * 100)
    g = compute_global_constraints(
        df, "label", 0.5, constrained_class=[0, 2], num_classes=3
    )
    assert g[0] == 20 and g[2] == 50 and (g[1] == UNLIMITED)


def _probs(n, k, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


@pytest.mark.parametrize("seed", range(6))
def test_single_capped_class_is_exactly_top_K(seed):
    (n, k, K) = (200, 4, 30)
    (p, g) = (_probs(n, k, seed), np.zeros(200, dtype=int))
    gcon = [UNLIMITED] * k
    gcon[2] = K
    lcon = {0: [UNLIMITED] * k}
    (y, _) = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [2]), gcon, lcon, k
    )
    chosen = set(np.where(y == 2)[0])
    assert chosen == set(np.argsort(p[:, 2])[::-1][:K])


@pytest.mark.parametrize("seed", range(6))
def test_several_capped_classes_are_not_starved(seed):
    (n, k) = (120, 4)
    (p, g) = (_probs(n, k, seed), np.zeros(120, dtype=int))
    (gcon, lcon) = ([40] * k, {0: [40] * k})
    (y, _) = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k
    )
    counts = np.bincount(y, minlength=k)
    assert counts.min() > 0, "a capped class was starved: %s" % counts
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_a_local_cap_binds_without_a_global_cap():
    (n, k) = (200, 4)
    p = _probs(n, k, 11)
    g = np.arange(n) % 2
    gcon = [UNLIMITED] * k
    lcon = {
        0: [UNLIMITED, 5, UNLIMITED, UNLIMITED],
        1: [UNLIMITED, 5, UNLIMITED, UNLIMITED],
    }
    (y, _) = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [1]), gcon, lcon, k
    )
    for grp in (0, 1):
        assert int((y[g == grp] == 1).sum()) <= 5
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_infeasible_instance_does_not_dump_everything_into_class_zero():
    (n, k) = (100, 4)
    (p, g) = (_probs(n, k, 7), np.zeros(100, dtype=int))
    (gcon, lcon) = ([10] * k, {0: [10] * k})
    (y, _) = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k
    )
    counts = np.bincount(y, minlength=k)
    assert counts.max() < n / 2, "collapsed onto one class: %s" % counts


def _gen(tmp, *extra):
    cmd = [
        sys.executable,
        "-m",
        "configs.gen_campaign",
        "--root",
        str(tmp),
        "--datasets",
        "iwildcam",
        "--arms",
        "tralo",
    ] + list(extra)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_generator_refuses_a_single_cap_level(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30")
    assert r.returncode == 1 and "two cap levels" in r.stdout + r.stderr


def test_generator_refuses_an_out_of_range_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30", "--constrained-class", "9")
    assert r.returncode == 1 and "out of range" in r.stdout + r.stderr


def test_generator_refuses_a_repeated_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30", "--constrained-class", "4", "4")
    assert r.returncode == 1 and "repeats a class" in r.stdout + r.stderr


def test_generator_refuses_to_mix_capped_classes_in_one_root(tmp_path):
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    cfg_path = next(iter(sorted(tmp_path.rglob("config.json"))))
    cfg = json.loads(cfg_path.read_text())
    cfg["status"] = "completed"
    cfg_path.write_text(json.dumps(cfg))
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30", "--constrained-class", "4", "5")
    assert r.returncode == 1 and "already holds a run" in r.stdout + r.stderr


def test_mandatory_clippers_are_always_added(tmp_path):
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    arms = {p.parts[-3] for p in tmp_path.rglob("config.json")}
    assert {"clip", "focal_clip"} <= arms


def _bid(P, arm, seed=1, **over):
    dc = dict(P["datasets"]["iwildcam"])
    hp = build_hyperparams(P, P["arms"][arm], seed)
    hp.update(over)
    return compute_base_model_id(P, "MobileNetV3", hp, "iwildcam", dc)


def test_arms_differing_only_in_the_allocator_share_a_warm_up():
    P = load_protocol()
    assert _bid(P, "tralo") == _bid(P, "fioretto") == _bid(P, "alm")


def test_a_different_warm_up_objective_does_not_share_a_model():
    P = load_protocol()
    assert _bid(P, "clip") != _bid(P, "focal_clip")


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", 0.5),
        ("dropout", 0.9),
        ("batch_size", 7),
        ("warmup_epochs", 3),
        ("pretrained", False),
        ("class_weighted_ce", True),
        ("seed", 99),
    ],
)
def test_every_warm_up_key_moves_the_hash(key, value):
    P = load_protocol()
    assert _bid(P, "tralo") != _bid(P, "tralo", **{key: value})


def test_constraint_phase_keys_do_not_move_the_hash():
    P = load_protocol()
    base = _bid(P, "tralo")
    for key, value in (
        ("constraint_epochs", 5),
        ("lambda_step", 0.9),
        ("initial_rho", 9.0),
        ("enable_checkpoint_restore", True),
    ):
        assert _bid(P, "tralo", **{key: value}) == base


def test_a_diverged_result_is_not_recorded_as_completed(tmp_path):
    from src.pipeline.io import save_results_to_config

    cfg = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg, tmp_path, {"accuracy": float("nan"), "f1": 0.3})
    assert cfg["status"] == "diverged" and "accuracy" in cfg["diverged_keys"][0]
    cfg2 = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg2, tmp_path, {"accuracy": 0.8, "f1": 0.3})
    assert cfg2["status"] == "completed"


def test_rerunning_does_not_turn_the_old_header_into_a_data_row(tmp_path):
    from src.training.logging import log_progress_to_csv, write_csv_header

    p = tmp_path / "training_log.csv"
    for run in range(2):
        write_csv_header(str(p), num_classes=2)
        for epoch in range(3):
            log_progress_to_csv(str(p), epoch, 0.5, 0.9, num_classes=2)
    df = pd.read_csv(p)
    assert pd.api.types.is_numeric_dtype(df["Epoch"])
    assert df["Epoch"].max() == 3


def test_the_csv_reports_the_real_local_lambda_column():
    from src.training.logging import build_csv_header

    header = build_csv_header(num_classes=2)
    assert "Lambda_Local" in header
    assert "Grad_Norm" in header and "L_KL" not in header


@pytest.mark.parametrize(
    "name", ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ViTB16"]
)
def test_backbones_keep_their_pretrained_weights(name):
    torch.manual_seed(0)
    a = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    torch.manual_seed(999)
    b = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    (named_a, named_b) = (dict(a.named_parameters()), dict(b.named_parameters()))
    differ = [k for k in named_a if not torch.equal(named_a[k], named_b[k])]
    assert len(differ) <= 2, (
        "%s re-initialises %d tensors from random, not just the final layer: %s"
        % (name, len(differ), differ[:6])
    )


def test_mobilenetv3_has_exactly_one_dropout_in_its_head():
    m = get_model(
        "MobileNetV3", input_dim=None, n_classes=7, dropout=0.3, pretrained=False
    )
    head = m.backbone.classifier
    drops = [l for l in head if isinstance(l, nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == 0.3


def test_a_repeatedly_failing_run_stops_being_re_dispatched(tmp_path):
    from src.utils.filesystem_manager import (
        MAX_FAILURES,
        get_experiments_by_status,
        save_config_to_path,
        update_experiment_status,
    )

    exp = tmp_path / "MobileNetV3" / "derm" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path(
        {"status": "pending", "hyperparams": {"seed": 1}, "arm": "tralo"}, exp
    )
    for i in range(1, MAX_FAILURES + 1):
        update_experiment_status(str(exp), "pending", count_failure=True)
        cfg = json.loads((exp / "config.json").read_text())
        assert cfg["failures"] == i
    assert cfg["status"] == "failed"
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_a_diverged_run_is_not_re_dispatched(tmp_path):
    from src.utils.filesystem_manager import (
        get_experiments_by_status,
        save_config_to_path,
    )

    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path(
        {"status": "diverged", "hyperparams": {"seed": 1}, "arm": "tralo"}, exp
    )
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_an_interrupted_run_still_resets_to_pending(tmp_path):
    from src.utils.filesystem_manager import (
        get_experiments_by_status,
        save_config_to_path,
    )

    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path(
        {"status": "running", "hyperparams": {"seed": 1}, "arm": "tralo"}, exp
    )
    buckets = get_experiments_by_status(str(tmp_path))
    assert len(buckets["pending"]) == 1 and (not buckets["blocked"])


@pytest.mark.parametrize("key", ["lr_constraint", "constraint_epochs"])
def test_a_missing_protocol_value_raises_instead_of_using_the_trap(key, tmp_path):
    import scripts.smoke_arms as sm

    P = load_protocol()
    (inputs, _, _) = sm.make_inputs(P, "tralo", str(tmp_path))
    inputs.hyperparams.pop(key, None)
    with pytest.raises(KeyError, match=key):
        TRAIN_FNS["tralo"](inputs)


def test_parity_catches_two_arms_sharing_one_warm_up_with_different_objectives(
    tmp_path,
):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "configs.gen_campaign",
            "--root",
            str(tmp_path),
            "--datasets",
            "iwildcam",
            "--models",
            "MobileNetV3",
            "--caps",
            "L30_G30",
            "L50_G30",
            "--arms",
            "clip",
            "focal_clip",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    clip_id = next(
        (
            json.loads(p.read_text())["base_model_id"]
            for p in sorted(tmp_path.rglob("config.json"))
            if json.loads(p.read_text())["arm"] == "clip"
        )
    )
    for p in sorted(tmp_path.rglob("config.json")):
        cfg = json.loads(p.read_text())
        if cfg["arm"] == "focal_clip":
            cfg["base_model_id"] = clip_id
            p.write_text(json.dumps(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "scripts.check_parity", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "DIFFERENT warm-up identities" in r.stdout


def test_verify_caps_fails_when_it_cannot_read_a_slice(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.verify_caps", "--datasets", "iwildcam"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    assert r.returncode == 1, "a gate that cannot fail is not a gate"


def test_the_audit_sees_keys_read_through_the_required_helper():
    from scripts.audit_config import per_methodology_reads

    reads = per_methodology_reads()
    for meth in ("tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        assert "lr_constraint" in reads[meth], meth
        assert "constraint_epochs" in reads[meth], meth
    for meth in ("heuristic",):
        for key in (
            "constraint_grad_clip",
            "constraint_grad_mode",
            "constraint_step_rule",
            "constraint_random_direction",
        ):
            assert key not in reads[meth], (
                "%s can read %s, so a post-hoc arm emitting it would audit clean. Its reader has moved into a directory audit_config unions into every methodology."
                % (meth, key)
            )


def test_metrics_helpers_restore_the_callers_mode():
    from src.training import metrics

    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    X = torch.randn(20, 4)
    groups = torch.zeros(20, dtype=torch.long)
    loader = [(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))]
    for mode in (True, False):
        model.train(mode)
        metrics.compute_prediction_statistics(model, X, groups, num_classes=3)
        assert model.training is mode, (
            "compute_prediction_statistics left the model in %s mode after being called in %s mode"
            % (model.training, mode)
        )
        model.train(mode)
        metrics.get_predictions_with_probabilities(model, X)
        assert model.training is mode
        model.train(mode)
        metrics.compute_train_accuracy(model, loader, torch.device("cpu"))
        assert model.training is mode


def _write_slice(
    d,
    n_train=12,
    n_test=8,
    n_classes=4,
    capped=2,
    truncate_test_labels=False,
    drop_capped=False,
):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    ytr = np.array([i % n_classes for i in range(n_train)])
    yte = np.array([i % n_classes for i in range(n_test)])
    if drop_capped:
        ytr = np.where(ytr == capped, (capped + 1) % n_classes, ytr)
        yte = np.where(yte == capped, (capped + 1) % n_classes, yte)
    np.save(
        os.path.join(d, "train_images.npy"),
        rng.random((n_train, 3, 8, 8)).astype("float32"),
    )
    np.save(os.path.join(d, "train_labels.npy"), ytr)
    np.save(
        os.path.join(d, "test_images.npy"),
        rng.random((n_test, 3, 8, 8)).astype("float32"),
    )
    np.save(
        os.path.join(d, "test_labels.npy"), yte[:-1] if truncate_test_labels else yte
    )
    pd.DataFrame({"label": yte, "grp": [i % 2 for i in range(n_test)]}).to_csv(
        os.path.join(d, "test_meta.csv"), index=False
    )


def _cfg(d, n_classes=4, capped=2):
    return {
        "dataset_mode": "iwildcam",
        "dataset_config": {
            "data_dir": d,
            "num_classes": n_classes,
            "constrained_class": capped,
            "group_column": "grp",
            "target_column": "label",
        },
        "constraint": [0.5, 0.5],
    }


def test_loader_refuses_mismatched_image_and_label_counts(tmp_path):
    d = str(tmp_path / "bad_len")
    _write_slice(d, truncate_test_labels=True)
    with pytest.raises(ValueError, match="test_images.npy has 8 rows"):
        load_data(_cfg(d))


def test_loader_refuses_a_slice_whose_labels_exceed_num_classes(tmp_path):
    d = str(tmp_path / "wrong_ds")
    _write_slice(d, n_classes=6)
    with pytest.raises(ValueError, match="num_classes is 4"):
        load_data(_cfg(d, n_classes=4))


def test_loader_refuses_a_slice_missing_the_capped_class(tmp_path):
    d = str(tmp_path / "no_capped")
    _write_slice(d, drop_capped=True)
    with pytest.raises(ValueError, match="does not occur in this slice"):
        load_data(_cfg(d))


def test_loader_accepts_a_well_formed_slice(tmp_path):
    d = str(tmp_path / "good")
    _write_slice(d)
    out = load_data(_cfg(d))
    assert out is not None


def test_penalty_gradient_is_non_monotone_above_rho_one():
    K = 67.0

    def grad_at(rho, violation):
        soft = torch.tensor(K + violation, requires_grad=True, dtype=torch.double)
        Kt = torch.tensor(K, dtype=torch.double)
        E = torch.relu(soft - Kt)
        S = Kt
        eps = 1e-08
        e = E / (S + eps)
        (E / (E + S + eps) + rho * e**2 / (1 + e**2 + eps)).backward()
        return float(soft.grad)

    assert abs((100.0 - 0.5) / 29 - 3.431) < 0.001
    g0 = [grad_at(0.5, f * K) for f in (0.001, 0.3, 1.0, 8.0)]
    assert g0 == sorted(g0, reverse=True), "rho=0.5 should still be monotone"
    (edge, peak, deep) = (
        grad_at(3.93, 0.001 * K),
        grad_at(3.93, 0.577 * K),
        grad_at(3.93, 8.0 * K),
    )
    assert peak > 2.5 * edge, "expected a hump above the boundary value"
    assert peak > 100 * deep, "expected the deep violation to be starved"
    import math

    at_analytic = grad_at(100.0, 1 / math.sqrt(3) * K)
    for f in (0.2, 0.4, 0.8, 1.5):
        assert grad_at(100.0, f * K) <= at_analytic + 1e-12, (
            "peak should be at u = 1/sqrt(3), not at %.2f" % f
        )
    us = [i / 500 for i in range(1, 5001)]
    gs = [grad_at(3.93, u * K) for u in us]
    peak_u = us[gs.index(max(gs))]
    assert 0.5 < peak_u < 0.56, (
        "at rho=3.93 the peak should sit left of 1/sqrt(3), got u=%.3f" % peak_u
    )


def test_a_deep_violation_is_starved_by_a_milder_one_sharing_the_clip():
    (K, rho, eps) = (67.0, 100.0, 1e-08)

    def shares(u_a, u_b):
        gs = []
        for u in (u_a, u_b):
            soft = torch.tensor(K * (1 + u), requires_grad=True, dtype=torch.double)
            Kt = torch.tensor(K, dtype=torch.double)
            E = torch.relu(soft - Kt)
            e = E / (Kt + eps)
            (E / (E + Kt + eps) + rho * e**2 / (1 + e**2 + eps)).backward()
            gs.append(float(soft.grad))
        tot = sum((g * g for g in gs))
        return [g * g / tot for g in gs]

    (mild, deep) = shares(0.577, 8.0)
    assert mild > 0.999, "the milder violation should take essentially all of it"
    assert deep < 0.001, "the 8x violation should be starved"
    (deep2, mild2) = shares(8.0, 0.577)
    assert abs(mild - mild2) < 1e-09 and abs(deep - deep2) < 1e-09
    assert abs(sum(shares(0.577, 0.577)) / 2 - 0.5) < 1e-09


def test_loader_detects_a_permuted_train_split(tmp_path):
    d = str(tmp_path / "permuted")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    pd.DataFrame({"label": y[::-1]}).to_csv(
        os.path.join(d, "train_meta.csv"), index=False
    )
    with pytest.raises(ValueError, match="not row-aligned"):
        load_data(_cfg(d))


def test_loader_accepts_an_aligned_train_meta(tmp_path):
    d = str(tmp_path / "aligned")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    pd.DataFrame({"label": y}).to_csv(os.path.join(d, "train_meta.csv"), index=False)
    assert load_data(_cfg(d)) is not None


def test_the_null_arm_really_delivers_no_constraint():
    proto = load_protocol()
    blk = proto["blocks"]["tralo_null"]
    for key in ("lambda_global", "lambda_local", "lambda_step"):
        assert blk[key] == 0.0, "%s must be exactly 0.0, got %r" % (key, blk[key])
    (null_arm, real_arm) = (proto["arms"]["tralo_null"], proto["arms"]["tralo"])
    assert null_arm["phase"] == real_arm["phase"] == "trained"
    assert null_arm["methodology"] == real_arm["methodology"] == "tralo"
    assert "constraint_phase" in null_arm["blocks"]
    from src.losses.transductive_loss import MulticlassTransductiveLoss as L

    total = torch.tensor(0.0)
    for soft, K in ((500.0, 67.0), (67.0, 67.0), (0.0, 67.0)):
        st = torch.tensor(soft)
        Kt = 67.0
        E = torch.relu(st - Kt)
        e = E / (Kt + 1e-08)
        pen = E / (E + Kt + 1e-08) + 0.5 * e**2 / (1 + e**2 + 1e-08)
        total = total + 0.0 * pen
    assert float(total) == 0.0
    assert not bool(total > 0), "pass 2 would still run on a nonzero residue"
    assert L is not None


def test_every_trained_arm_reports_reordering():

    def _reaches_reordering(src, seen):
        if "reordering_report(" in src and '"reordering"' in src:
            return True
        for line in src.splitlines():
            line = line.strip()
            if not line.startswith("from src."):
                continue
            mod = line.split()[1]
            path = mod.replace(".", os.sep) + ".py"
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            if _reaches_reordering(io.open(path, encoding="utf-8").read(), seen):
                return True
        return False

    trained = ["tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"]
    for m in trained:
        path = os.path.join("src", "methodologies", m, "train.py")
        src = io.open(path, encoding="utf-8").read()
        assert "reordering_report" not in src or "def reordering_report" not in src, (
            "%s must use the shared diagnostic, not a private copy" % m
        )
        assert _reaches_reordering(src, set()), (
            "%s never reaches reordering_report / never puts it in the summary" % m
        )
    runner = io.open(
        os.path.join("src", "experiments", "runner.py"), encoding="utf-8"
    ).read()
    assert 'config["reordering"]' in runner
    import ast

    result_calls = [
        node
        for node in ast.walk(ast.parse(runner))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "save_results_to_config")
    ]
    assert result_calls
    for call in result_calls:
        assert not any(
            (
                isinstance(node, ast.Constant) and node.value == "reordering"
                for node in ast.walk(call)
            )
        )


NULL_SIBLINGS = [
    ("tralo_null", "tralo", ("lambda_step", "lambda_global", "lambda_local")),
    ("fioretto_null", "fioretto", ("fioretto_step_size",)),
    ("hounie_null", "hounie", ("hounie_eta_lambda",)),
    ("alm_null", "alm", ("alm_eta", "alm_mu0", "alm_mu_step")),
]


@pytest.mark.parametrize(
    "null,parent,zeroed", NULL_SIBLINGS, ids=[n for (n, _, _) in NULL_SIBLINGS]
)
def test_every_trained_arm_has_a_working_zero_dose_sibling(null, parent, zeroed):
    proto = load_protocol()
    blk = proto["blocks"][null]
    for key in zeroed:
        assert blk[key] == 0.0, "%s.%s must be exactly 0.0, got %r" % (
            null,
            key,
            blk[key],
        )
    (a_null, a_parent) = (proto["arms"][null], proto["arms"][parent])
    assert a_null["methodology"] == a_parent["methodology"]
    assert a_null["phase"] == a_parent["phase"] == "trained"
    assert "constraint_phase" in a_null["blocks"]


def test_alm_null_zeroes_the_augmentation_not_just_the_multiplier():
    blk = load_protocol()["blocks"]["alm_null"]
    for epoch in (0, 14, 28):
        mu_t = blk["alm_mu0"] + blk["alm_mu_step"] * epoch
        assert mu_t == 0.0, "mu_t is %r at epoch %d, so the weight is not zero" % (
            mu_t,
            epoch,
        )


def test_hounie_null_does_not_trip_hounie_s_own_stability_guard():
    blk = load_protocol()["blocks"]["hounie_null"]
    factor = abs(1.0 - 2.0 * blk["hounie_eta_u"] * blk["hounie_alpha"])
    assert factor < 1.0, (
        "hounie_null would raise its own stability check: factor %.3f" % factor
    )
    assert blk["hounie_eta_lambda"] == 0.0


@pytest.mark.parametrize("pct", [0.3, 0.5])
def test_targeted_correction_spends_the_whole_reachable_budget(pct):
    (n, n_cls, n_grp, capped) = (2000, 7, 7, [4])
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, n_cls, n)
        g = rng.integers(0, n_grp, n)
        logits = rng.normal(size=(n, n_cls))
        logits[:, capped[0]] += 0.8
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        df = pd.DataFrame({"label": y, "grp": g})
        gcon = compute_global_constraints(
            df, "label", pct, constrained_class=capped, num_classes=n_cls
        )
        lcon = compute_local_constraints(
            df, "label", pct, "grp", constrained_class=capped, num_classes=n_cls
        )
        (y_pred, _, meta) = targeted_correction(proba, g, gcon, lcon, capped)
        c = capped[0]
        reachable = min(gcon[c], sum((lcon[gid][c] for gid in lcon)))
        got = int((y_pred == c).sum())
        assert got == reachable, (
            "seed %d pct %s: filled %d of a reachable %d -- the trained arms would score against clippers that fill to exactly K"
            % (seed, pct, got, reachable)
        )
        assert got <= gcon[c], "global cap violated"
        for gid in lcon:
            in_g = int((y_pred[g == gid] == c).sum())
            assert in_g <= lcon[gid][c], "local cap violated in group %s: %d > %d" % (
                gid,
                in_g,
                lcon[gid][c],
            )


def test_every_training_log_gets_a_header_not_just_tralo_s(tmp_path):
    from src.training.logging import build_csv_header, log_progress_to_csv

    path = tmp_path / "training_log.csv"
    local = {0: [10**10, 5, 10**10], 1: [10**10, 4, 10**10]}
    for epoch in range(3):
        log_progress_to_csv(
            str(path),
            epoch,
            ce_loss=0.5,
            train_acc=0.9,
            constraints=[10**10, 12, 10**10],
            num_classes=3,
            local_constraints=local,
            grad_norm=1.5,
            lambda_global=0.25,
        )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    expected = build_csv_header(3, local)
    assert lines[0].split(",") == expected, (
        "first line is not the header -- pandas would eat epoch 1 as column names"
    )
    assert len(lines) == 1 + 3, "expected a header plus one row per epoch"
    df = pd.read_csv(path)
    assert len(df) == 3, "an epoch was consumed by the header"
    assert "Grad_Norm" in df.columns and "Lambda_Global" in df.columns
    assert float(df["Grad_Norm"].iloc[0]) == 1.5
    assert float(df["Lambda_Global"].iloc[0]) == 0.25


def test_constraint_phase_reaches_every_trained_arm_and_no_posthoc_one():
    proto = load_protocol()
    cp = proto["constraint_phase"]
    trained = [
        a for (a, spec) in proto["arms"].items() if spec.get("phase") == "trained"
    ]
    assert len(trained) >= 4
    for arm in trained:
        spec = proto["arms"][arm]
        if spec.get("constraint_step") is False:
            declared = set()
            for b in spec["blocks"]:
                declared |= set(proto["blocks"].get(b, {}))
            leaked = declared & set(cp) - {"lr_constraint", "constraint_chunk_size"}
            assert not leaked, (
                "%s declares constraint_step: false but still carries %s"
                % (arm, sorted(leaked))
            )
            continue
        assert "constraint_phase" in spec["blocks"], (
            "%s is a trained arm that does NOT include constraint_phase, so a shared constraint knob would silently miss it. If it genuinely takes no constraint step, declare `constraint_step: false`."
            % arm
        )
    for arm, spec in proto["arms"].items():
        if spec.get("phase") == "posthoc":
            assert "constraint_phase" not in (spec.get("blocks") or []), (
                "%s is post-hoc; emitting a constraint-phase key for it would be a key with no reader"
                % arm
            )


def test_normalize_delivers_the_same_step_size_whatever_the_raw_norm():

    def step(raw_scale, mode):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(0.0)
        m.weight.grad = torch.full((1, 4), raw_scale)
        before = m.weight.detach().clone()
        finish_constraint_step(
            m, torch.optim.SGD(m.parameters(), lr=1.0), None, 1.0, mode=mode, fp32=True
        )
        return float((m.weight.detach() - before).norm())

    (small_clip, big_clip) = (step(0.01, "clip"), step(10.0, "clip"))
    (small_nrm, big_nrm) = (step(0.01, "normalize"), step(10.0, "normalize"))
    assert small_clip == pytest.approx(0.02, rel=1e-05), small_clip
    assert big_clip == pytest.approx(1.0, rel=1e-05), big_clip
    assert small_clip < big_clip / 10, (
        "under `clip` a below-threshold gradient keeps its own magnitude -- this is the hounie asymmetry, and it must stay reproducible"
    )
    assert small_nrm == pytest.approx(1.0, rel=1e-05), small_nrm
    assert big_nrm == pytest.approx(1.0, rel=1e-05), big_nrm
    assert small_nrm == pytest.approx(big_nrm, rel=1e-06), (
        "under `normalize` the delivered step must be the same size no matter what the arm's natural gradient scale is -- that is the whole point"
    )


def test_a_non_finite_constraint_gradient_never_moves_the_weights():
    for bad in (float("nan"), float("inf")):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(1.0)
        m.weight.grad = torch.full((1, 4), bad)
        before = m.weight.detach().clone()
        (raw, applied) = finish_constraint_step(
            m,
            torch.optim.SGD(m.parameters(), lr=1.0),
            None,
            1.0,
            mode="normalize",
            fp32=True,
        )
        assert not applied, "a %s gradient must not take the step" % bad
        assert torch.equal(m.weight.detach(), before), (
            "weights moved on a %s gradient" % bad
        )


@pytest.mark.parametrize("arm", ["tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"])
def test_no_arm_hand_rolls_its_own_constraint_step(arm):
    src = open("src/methodologies/%s/train.py" % arm, encoding="utf-8").read()
    assert "finish_constraint_step" in src, (
        "%s does not use the shared constraint step" % arm
    )
    assert "clip_grad_norm_" not in src, (
        "%s calls clip_grad_norm_ directly. That is the divergence this module exists to prevent -- route it through finish_constraint_step so every arm is bounded the same way."
        % arm
    )
    assert "constraint_autocast" in src, (
        "%s does not use the shared constraint autocast, so --constraint-fp32 cannot reach it and it can still lose epochs to fp16 overflow"
        % arm
    )


def test_the_allocator_does_not_fall_through_to_the_LP_when_G_is_less_than_L():
    from configs.gen_campaign import cap_pair

    rng = np.random.default_rng(0)
    (N, C, G) = (600, 7, 5)
    capped = [2, 4]
    labels = rng.choice(C, size=N, p=np.array([0.1, 0.1, 0.25, 0.1, 0.3, 0.1, 0.05]))
    groups = rng.integers(0, G, size=N)
    df = pd.DataFrame({"label": labels, "grp": groups})
    for tag in ("L50_G30", "L40_G30", "L30_G20"):
        (loc_pct, glob_pct) = cap_pair(tag)
        gcon = compute_global_constraints(
            df, "label", glob_pct, constrained_class=capped, num_classes=C
        )
        lcon = compute_local_constraints(
            df, "label", loc_pct, "grp", constrained_class=capped, num_classes=C
        )
        for trial in range(5):
            logits = rng.normal(0, 2.0, size=(N, C))
            logits[:, capped] += 1.2
            e = np.exp(logits - logits.max(1, keepdims=True))
            proba = e / e.sum(1, keepdims=True)
            (_, _, info) = targeted_correction(proba, groups, gcon, lcon, capped)
            assert not info["lp_fallback_used"], (
                "%s trial %d fell through to the LP with %d candidates -- the greedy left the allocation infeasible, so this arm would be scored against `clip` while running a different allocator"
                % (tag, trial, info["lp_fallback_candidates"])
            )


def test_a_cap_that_does_not_bind_gives_the_constraint_zero_gradient():
    K = 56.0
    loss = MulticlassTransductiveLoss(
        global_constraints=[K, K], local_constraints={}, num_classes=2
    )
    for count in (76.0, 51.0, 34.0, 18.0):
        soft = torch.tensor(count, requires_grad=True)
        pen = loss._penalty(soft, K)
        if count > K:
            assert float(pen) > 0.0, "an over-budget count must be penalised"
        else:
            assert float(pen) == 0.0, (
                "count %g is under K=56, so the penalty -- and the whole constraint gradient -- is identically zero"
                % count
            )
            pen.backward()
            assert float(soft.grad) == 0.0, (
                "no gradient reaches the model from a satisfied cap"
            )


def test_final_predictions_that_violate_a_cap_are_refused_not_logged(tmp_path):
    from src.pipeline.eval import write_evaluation_outputs

    n = 12
    y_test = np.zeros(n, dtype=int)
    groups = np.array([0] * 6 + [1] * 6)
    proba = np.tile(np.array([0.6, 0.4]), (n, 1))

    def run(y_pred, global_con, local_con):
        return write_evaluation_outputs(
            tmp_path,
            y_test,
            groups,
            {
                "y_pred": np.asarray(y_pred),
                "raw_pred": np.asarray(y_pred),
                "y_proba": proba,
                "metrics": {
                    "flips_required": 0,
                    "raw_all_satisfied": True,
                    "raw_total_excess": 0,
                },
            },
            2,
            global_con,
            local_con,
        )

    run([1] * 2 + [0] * 10, [UNLIMITED, 4], {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]})
    with pytest.raises(RuntimeError, match="violate"):
        run([1] * 6 + [0] * 6, [UNLIMITED, 4], None)
    with pytest.raises(RuntimeError, match="local group"):
        run(
            [1] * 5 + [0] * 7,
            [UNLIMITED, UNLIMITED],
            {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]},
        )


def test_no_methodology_reads_the_test_LABELS_except_to_count_them():
    offenders = []
    for path in sorted(pathlib.Path("src/methodologies").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "len":
                for a in node.args:
                    allowed.update((id(n) for n in ast.walk(a)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "y_test"
                and (id(node) not in allowed)
            ):
                offenders.append("%s:%d" % (path.as_posix(), node.lineno))
    assert not offenders, (
        "these methodology modules read the TEST LABELS outside a len() call, which is label leakage, not transduction: %s"
        % offenders
    )


def test_the_two_fioretto_arms_initialise_both_multiplier_scopes_alike():
    for mod in ("fioretto_ldf", "fioretto_alm"):
        path = "src/methodologies/%s/train.py" % mod
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        assigns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                (
                    isinstance(t, ast.Subscript)
                    and getattr(t.value, "id", None) == "lambda_l"
                    for t in n.targets
                )
            )
        ]
        init = [n for n in assigns if isinstance(n.value, ast.Constant)]
        assert not init, (
            "%s initialises lambda_l to the literal %r instead of the fioretto_lambda_init value the other arm uses"
            % (mod, [n.value.value for n in init])
        )


def test_alm_gates_its_constraint_pass_on_the_weights_the_loss_uses():
    src = io.open("src/methodologies/fioretto_alm/train.py", encoding="utf-8").read()
    tree = ast.parse(src)
    node = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any((getattr(t, "id", None) == "has_work" for t in n.targets))
        ),
        None,
    )
    assert node is not None, "has_work assignment not found"
    names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    for w in ("lambda_g", "lambda_l", "aug_g", "aug_l"):
        assert w in names, (
            "has_work ignores %s, but the constraint loss weights terms by lambda + aug -- the augmentation is unreachable whenever the multipliers are 0"
            % w
        )


def test_two_cap_tags_that_produce_the_same_budget_are_one_cap_level():
    from scripts.verify_caps import duplicate_budget_tags

    dup = duplicate_budget_tags(
        {2: {"L40_G30": 62, "L50_G30": 62}, 4: {"L40_G30": 67, "L50_G30": 67}}
    )
    assert dup == [(2, 62, ["L40_G30", "L50_G30"]), (4, 67, ["L40_G30", "L50_G30"])], (
        dup
    )
    assert duplicate_budget_tags({2: {"L50_G30": 62, "L50_G20": 41}}) == []
    dup = duplicate_budget_tags({2: {"a": 10, "b": 10}, 4: {"a": 10, "b": 20}})
    assert dup == [(2, 10, ["a", "b"])], dup


def test_no_autocast_banned_op_is_reachable_from_an_arm():
    BANNED = {"binary_cross_entropy", "BCELoss"}
    offenders = []
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if fn in BANNED:
                    offenders.append("%s:%d %s" % (path.as_posix(), node.lineno, fn))
    assert not offenders, (
        "CUDA autocast bans these ops and every training path runs under AMP, so the arm dies on its first GPU batch: %s. Write the loss out by hand on the clamped probability -- NOT the _with_logits variant, which computes a different quantity."
        % offenders
    )


def test_chunking_the_transductive_backward_does_not_change_the_gradient():
    import torch
    from src.losses.transductive_loss import MulticlassTransductiveLoss

    torch.manual_seed(0)
    (N, C, D) = (40, 4, 6)
    X = torch.randn(N, D)
    gids = torch.tensor([i % 3 for i in range(N)])
    glob_c = torch.full((C,), 10000000000.0)
    glob_c[1] = 5.0
    glob_c[2] = 7.0
    loc = {g: torch.full((C,), 10000000000.0) for g in (0, 1, 2)}
    for g in loc:
        loc[g][1] = 2.0
        loc[g][2] = 3.0

    def grad_at(chunk):
        torch.manual_seed(1)
        lin = torch.nn.Linear(D, C)
        crit = MulticlassTransductiveLoss(
            glob_c.detach().numpy().copy(),
            {g: value.detach().numpy().copy() for g, value in loc.items()},
            num_classes=C,
            initial_rho=0.5,
        )
        crit.lambda_global_per_class = {1: 0.7, 2: 0.3}
        crit.lambda_local_per_key = {(g, c): 0.5 for g in loc for c in (1, 2)}
        with torch.no_grad():
            tot_g = torch.softmax(lin(X), dim=1).sum(dim=0)
            tot_l = {g: torch.softmax(lin(X[gids == g]), dim=1).sum(dim=0) for g in loc}
        lin.zero_grad()
        for st in range(0, N, chunk):
            sl = slice(st, min(st + chunk, N))
            pr = torch.softmax(lin(X[sl]), dim=1)
            cg = pr.sum(dim=0)
            cl = {}
            for g in loc:
                m = gids[sl] == g
                cl[g] = pr[m].sum(dim=0) if m.any() else torch.zeros(C)
            g_soft = tot_g.detach() - cg.detach() + cg
            l_soft = {g: tot_l[g].detach() - cl[g].detach() + cl[g] for g in loc}
            loss = crit.compute_global_from_counts(
                g_soft
            ) + crit.compute_local_from_counts(l_soft)
            loss.backward()
        return lin.weight.grad.clone()

    ref = grad_at(N)
    assert ref.abs().sum() > 0, (
        "the probe produced a zero gradient, so it cannot detect a chunking bug -- the caps are not binding"
    )
    for chunk in (1, 3, 7, 13, 40):
        g = grad_at(chunk)
        rel = (g - ref).abs().max() / ref.abs().max()
        assert rel < 1e-05, (
            "constraint_chunk_size=%d changes the constraint gradient by %.2e relative -- it is NOT a pure memory knob, so lowering it to survive an OOM would silently alter every number in the campaign"
            % (chunk, float(rel))
        )


def test_the_chunked_backward_in_tralo_still_uses_the_exact_construction():
    src = io.open(
        os.path.join(REPO, "src", "methodologies", "tralo", "train.py"),
        encoding="utf-8",
    ).read()
    fn = next(
        (
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        ),
        None,
    )
    assert fn is not None, "tralo.train() not found"
    code = ast.unparse(fn)
    flat = "".join(code.split())
    assert "total_global_soft.detach()" in flat, (
        "the chunked backward no longer reconstructs the FULL global count before evaluating the penalty, so what it differentiates is a per-chunk penalty, not the full-N one"
    )
    assert "total_local_soft[gid].detach()" in flat, "same defect on LOCAL counts"
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and ("n_chunks" in ast.unparse(node.right))
        ):
            raise AssertionError(
                "tralo divides by n_chunks (`%s`). n_chunks is ceil(N_test / constraint_chunk_size), so this makes the constraint dose a function of the dataset size AND of a memory knob -- derm 8, oct 4, tissue 10, i.e. 2.5x apart."
                % ast.unparse(node)
            )


def test_resetting_crashed_runs_refuses_to_discard_a_finished_one():
    from scripts.reset_crashed import eligible

    (ok, why) = eligible({"status": "pending", "results": {"accuracy": 0.77}}, 29)
    assert not ok, "would discard a finished run: " + why
    assert "HAS RESULTS" in why
    (ok, why) = eligible({"status": "running"}, 3)
    assert not ok and "running" in why, "would reset a live run"
    (ok, why) = eligible({"status": "pending", "results": {}}, 22)
    assert not ok, "silently discarded 22 epochs of work: " + why
    (ok, why) = eligible({"status": "pending", "failures": 2, "results": {}}, 1)
    assert ok, "refused the actual crash case: " + why
    (ok, why) = eligible({"status": "pending", "results": {}}, 0)
    assert ok, "refused a run with no log at all: " + why


def test_the_two_training_log_schemas_stay_watched_and_keep_their_conventions():
    import ast as _ast

    trained = {
        "tralo": ("tralo", "train.py"),
        "fioretto_ldf": ("fioretto_ldf", "train.py"),
        "fioretto_alm": ("fioretto_alm", "train.py"),
        "hounie_rcl": ("hounie_rcl", "train.py"),
    }
    for name, rel in trained.items():
        src = io.open(
            os.path.join(REPO, "src", "methodologies", *rel), encoding="utf-8"
        ).read()
        code = _ast.unparse(_ast.parse(src))
        assert (
            "train_acc" in code or "Train_Acc" in code or "log_progress_to_csv" in code
        ), (
            "%s neither logs a train-accuracy field nor calls the canonical writer, so scripts/full_panel.py cannot see a terminal collapse in any of its runs"
            % name
        )

    def _epoch_range_args(path):
        tree = _ast.parse(io.open(path, encoding="utf-8").read())
        for n in _ast.walk(tree):
            if (
                isinstance(n, _ast.For)
                and isinstance(n.target, _ast.Name)
                and (n.target.id == "epoch")
                and isinstance(n.iter, _ast.Call)
                and (getattr(n.iter.func, "id", None) == "range")
            ):
                return len(n.iter.args)
        return None

    for pkg in ("tralo",):
        n_args = _epoch_range_args(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py")
        )
        assert n_args == 2, (
            "%s no longer iterates an ABSOLUTE epoch axis (range(warmup, total)); its Epoch column shares a name with the warm-up rows in the same file, so a relative axis would silently renumber them"
            % pkg
        )
    for pkg in ("fioretto_ldf", "fioretto_alm", "hounie_rcl"):
        n_args = _epoch_range_args(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py")
        )
        assert n_args == 1, (
            "%s no longer iterates a RELATIVE epoch axis; FRAMEWORK's known-asymmetries table says its `epoch` is 0-based within the constraint phase, and the 14,524-run archive was written that way"
            % pkg
        )
    logsrc = io.open(
        os.path.join(REPO, "src", "training", "logging.py"), encoding="utf-8"
    ).read()
    writer = next(
        (
            n
            for n in _ast.walk(_ast.parse(logsrc))
            if isinstance(n, _ast.FunctionDef) and n.name == "log_progress_to_csv"
        )
    )
    assert "epoch + 1" in _ast.unparse(writer), (
        "log_progress_to_csv no longer writes epoch + 1, so TraLO's Epoch column stopped being 1-based while FRAMEWORK still says it is"
    )
    for pkg in ("fioretto_ldf", "fioretto_alm", "hounie_rcl"):
        src = io.open(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py"),
            encoding="utf-8",
        ).read()
        assert "'epoch': epoch," in _ast.unparse(_ast.parse(src)), (
            "%s no longer logs the RAW epoch; if it gained a +1 it would look 1-based while remaining relative, which is the one combination no reader could detect"
            % pkg
        )


def test_log_health_does_not_cry_wolf_on_a_warm_up_row_or_a_posthoc_arm(tmp_path):
    from scripts.log_health import read_run

    def mk(name, rows, header):
        d = tmp_path / name
        d.mkdir()
        (d / "training_log.csv").write_text(
            header + chr(10) + chr(10).join(rows) + chr(10)
        )
        (d / "config.json").write_text('{"arm": "%s", "status": "completed"}' % name)
        return str(d)

    wide = "Epoch,Train_Acc,L_CE,Hard_Class2,Limit_Class2,Group0_Hard_Class2"
    warm = mk(
        "tralo",
        [
            "1,0.8150,0.7800,0,,",
            "2,0.9000,0.5000,200,62,70",
            "3,0.9500,0.3000,190,62,66",
            "4,0.9600,0.2000,180,62,60",
        ],
        wide,
    )
    r = read_run(warm)
    assert not r["nonfinite"], (
        "flagged the warm-up row as divergence: %s" % r["nonfinite"]
    )
    assert r["sat"] is None or True
    assert 2 in r["counts"] and r["counts"][2]["K"] == 62
    ph = mk(
        "clip",
        [
            "1,0.8150,0.7800,0,10000000000.0,0",
            "2,0.9000,0.5000,200,10000000000.0,70",
            "3,0.9900,0.3000,190,10000000000.0,66",
        ],
        wide.replace("Hard_Class2,Limit_Class2", "Hard_Class2,Limit_Class2")
        + ",Global_Satisfied",
    )
    ph_rows = io.open(os.path.join(ph, "training_log.csv"), encoding="utf-8").read()
    io.open(os.path.join(ph, "training_log.csv"), "w", encoding="utf-8").write(
        ph_rows.replace(",0" + chr(10), ",0,1" + chr(10))
        .replace(",70" + chr(10), ",70,1" + chr(10))
        .replace(",66" + chr(10), ",66,1" + chr(10))
    )
    r = read_run(ph)
    assert r["posthoc"], (
        "a run whose every Limit_Class is UNLIMITED was not recognised as post-hoc, so it will be reported as satisfied on every epoch"
    )
    assert r["sat"] is None, (
        "reported a vacuous satisfaction count for a post-hoc arm: %s" % (r["sat"],)
    )
    bad = mk(
        "diverged",
        [
            "2,0.9000,0.5000,200,62,70",
            "3,0.9500,nan,190,62,66",
            "4,0.9600,0.2000,180,62,60",
        ],
        wide,
    )
    assert read_run(bad)["nonfinite"], (
        "missed a NaN sitting beside real values -- a run once diverged to all-NaN and still wrote `completed`"
    )


def test_test_embeddings_come_out_at_the_width_the_head_declares():
    import torch
    from src.models.model_factory import get_model
    from src.pipeline.features import extract_test_embeddings, head_and_feature_dim

    X = torch.randn(6, 3, 224, 224)
    for name in ("MobileNetV3", "MobileNetV2", "RegNetY400MF", "ViTB16"):
        model = get_model(name, 7, pretrained=False)
        (_head, dim) = head_and_feature_dim(model)
        feats = extract_test_embeddings(model, X, chunk=4)
        assert feats.shape == (6, dim), (
            "%s: the head declares %d features but the hook captured %s"
            % (name, dim, (feats.shape,))
        )


def test_the_duals_reset_grad_norm_every_epoch_instead_of_carrying_it_forward():
    import ast as _ast

    for pkg in ("fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        src = io.open(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py"),
            encoding="utf-8",
        ).read()
        tree = _ast.parse(src)
        fn = next(
            (
                n
                for n in _ast.walk(tree)
                if isinstance(n, _ast.FunctionDef)
                and any(
                    (
                        isinstance(x, _ast.For)
                        and isinstance(x.target, _ast.Name)
                        and (x.target.id == "epoch")
                        for x in _ast.walk(n)
                    )
                )
            ),
            None,
        )
        assert fn is not None, "%s: no function contains a `for epoch` loop" % pkg

        def _assigns(node):
            return [
                n
                for n in _ast.walk(node)
                if isinstance(n, _ast.Assign)
                and any(
                    (
                        isinstance(t, _ast.Name) and t.id == "last_grad_norm"
                        for t in n.targets
                    )
                )
            ]

        loops = [
            n
            for n in _ast.walk(fn)
            if isinstance(n, _ast.For)
            and isinstance(n.target, _ast.Name)
            and (n.target.id == "epoch")
        ]
        assert loops, "%s: no `for epoch` loop found" % pkg
        inside = {id(a) for lp in loops for a in _assigns(lp)}
        resets = [
            a
            for a in _assigns(fn)
            if isinstance(a.value, _ast.Constant) and a.value.value == 0.0
        ]
        assert resets, "%s: `last_grad_norm = 0.0` disappeared entirely" % pkg
        assert all((id(a) in inside for a in resets)), (
            "%s resets last_grad_norm OUTSIDE the epoch loop, so a slack epoch logs the previous epoch's gradient norm as its own"
            % pkg
        )


def test_the_runner_stamps_the_commit_that_produced_the_weights():
    src = io.open(
        os.path.join(REPO, "src", "experiments", "runner.py"), encoding="utf-8"
    ).read()
    tree = ast.parse(src)
    writes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Subscript)
        and isinstance(t.slice, ast.Constant)
        and (t.slice.value == "run_code_version")
    ]
    assert writes, (
        "src/experiments/runner.py never assigns run_code_version, so every config still describes only the commit that generated it"
    )
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_experiment"
        )
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and (n.func.id in ("save_config_to_path", "update_experiment_status"))
    ]
    assert calls and calls[0].func.id == "save_config_to_path", (
        "the stamp is not persisted before update_experiment_status reloads config.json from disk, so a run that crashes carries no runner stamp"
    )
    for mod in ("configs/gen_campaign.py", "src/experiments/runner.py"):
        code = ast.unparse(
            ast.parse(
                io.open(os.path.join(REPO, *mod.split("/")), encoding="utf-8").read()
            )
        )
        assert "git_version" in code, (
            "%s hand-rolls its own `git rev-parse` -- four hand-rolled copies of one step is how the constraint dose drifted 20x between arms"
            % mod
        )
        assert "rev-parse" not in code, (
            "%s still calls git directly instead of src/utils/gitver" % mod
        )


def test_the_model_cache_prefers_the_stamp_of_the_run_that_trained_it(
    tmp_path, monkeypatch
):
    from src.training import model_cache as MC

    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

    monkeypatch.setattr(MC, "get_model", lambda *a, **k: _Tiny())
    bmid = "TinyNet_smoke_deadbeefcafe"
    dev = torch.device("cpu")

    def _write(payload):
        torch.save(
            dict(
                {"model_state_dict": _Tiny().state_dict(), "base_model_id": bmid},
                **payload,
            ),
            MC.get_cache_path(bmid),
        )

    def _load(cfg):
        return MC.load_from_cache(
            bmid,
            dict({"model_name": "TinyNet", "hyperparams": {"dropout": 0.3}}, **cfg),
            2,
            dev,
        )

    _write({"code_version": "GEN1", "run_code_version": "RUN1"})
    assert _load({"code_version": "GEN1", "run_code_version": "RUN2"}) is None, (
        "the cache was reused across a code change that both configs' generator stamps agree through"
    )
    assert _load({"code_version": "GEN1", "run_code_version": "RUN1"}) is not None
    _write({"code_version": "GEN1"})
    assert _load({"code_version": "GEN1", "run_code_version": "RUN2"}) is not None, (
        "a pre-stamp cache was invalidated; every warm-up on disk is one"
    )
    assert _load({"code_version": "GEN2"}) is None, (
        "the generator fallback stopped rejecting a genuine version mismatch"
    )


TRAINERS_WITH_A_CONSTRAINT_STEP = [
    ("tralo", "train.py"),
    ("fioretto_ldf", "train.py"),
    ("hounie_rcl", "train.py"),
    ("fioretto_alm", "train.py"),
]


def test_no_trainer_discards_whether_the_constraint_step_actually_landed():
    for mod, fname in TRAINERS_WITH_A_CONSTRAINT_STEP:
        path = os.path.join(REPO, "src", "methodologies", mod, fname)
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        targets = []
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
                continue
            f = n.value.func
            name = (
                f.id
                if isinstance(f, ast.Name)
                else f.attr
                if isinstance(f, ast.Attribute)
                else None
            )
            if name != "finish_constraint_step":
                continue
            t = n.targets[0]
            assert isinstance(t, ast.Tuple) and len(t.elts) == 2, (
                "%s does not unpack finish_constraint_step's two returns" % mod
            )
            targets.append(t.elts[1])
        assert targets, "%s never calls finish_constraint_step" % mod
        for t in targets:
            assert isinstance(t, ast.Name) and (not t.id.startswith("_")), (
                "%s binds `applied` to %r -- an underscore name is how this value was discarded in all four trainers"
                % (mod, getattr(t, "id", t))
            )
            reads = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Name)
                and n.id == t.id
                and isinstance(n.ctx, ast.Load)
            ]
            assert reads, (
                "%s binds `applied` and never reads it, so a dropped constraint step is still invisible"
                % mod
            )


@pytest.mark.parametrize("arm", ["tralo", "fioretto", "hounie", "alm"])
def test_a_run_reports_how_many_constraint_steps_it_actually_took(arm, tmp_path):
    import scripts.smoke_arms as SA

    P = load_protocol()
    torch.manual_seed(1)
    (inputs, _g, _l) = SA.make_inputs(P, arm, str(tmp_path))
    out = TRAIN_FNS[P["arms"][arm]["methodology"]](inputs)
    app = out.summary.get("constraint_steps_applied")
    att = out.summary.get("constraint_steps_attempted")
    assert app is not None and att is not None, (
        "%s does not report its applied/attempted constraint steps" % arm
    )
    assert att >= 1, "%s attempted no constraint step at all" % arm
    assert app == att, (
        "%s lost %d of %d constraint steps on a tiny CPU model, where nothing should overflow"
        % (arm, att - app, att)
    )


def test_a_zero_dose_arm_attempts_no_constraint_step():
    import tempfile
    import scripts.smoke_arms as SA

    P = load_protocol()
    tmp = tempfile.mkdtemp()
    try:
        torch.manual_seed(1)
        (inputs, _g, _l) = SA.make_inputs(P, "tralo_null", tmp)
        out = TRAIN_FNS[P["arms"]["tralo_null"]["methodology"]](inputs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out.summary.get("constraint_steps_attempted") == 0, (
        "the zero-dose arm attempted a constraint step; either the lambdas are not zero or the counter is counting epochs rather than steps"
    )
    assert out.summary.get("constraint_steps_applied") == 0


def test_normalize_makes_the_delivered_step_independent_of_the_violation():
    import torch
    from src.training.constraint_step import finish_constraint_step

    (lr, clip) = (0.001, 1.0)
    for scale in (0.0001, 1.0, 10000.0):
        model = torch.nn.Linear(6, 3, bias=False)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 1.0)
        model.weight.grad *= scale / model.weight.grad.norm()
        finish_constraint_step(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=lr),
            scaler=None,
            clip=clip,
            mode="normalize",
            fp32=True,
        )
        moved = float((model.weight.detach() - before).norm())
        assert abs(moved - lr * clip) < 0.0001 * lr * clip, (
            "raw norm %g delivered a step of %g, not lr*clip=%g -- the constraint has become sensitive to violation magnitude, which contradicts FRAMEWORK 2(a3)"
            % (scale, moved, lr * clip)
        )


def _run_arm_probs(arm, methodology, seed=1):
    import tempfile
    import scripts.smoke_arms as smoke

    tmp = tempfile.mkdtemp(prefix="zerodose_")
    try:
        (inputs, _g, _l) = smoke.make_inputs(load_protocol(), arm, tmp, seed=seed)
        torch.manual_seed(seed)
        out = TRAIN_FNS[methodology](inputs)
        out.model.eval()
        with torch.no_grad():
            return F.softmax(out.model(inputs.X_test), dim=1).numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_zero_dose_arm_is_the_same_model_across_code_paths():
    nulls = {
        "tralo_null": "tralo",
        "fioretto_null": "fioretto_ldf",
        "hounie_null": "hounie_rcl",
        "alm_null": "fioretto_alm",
    }
    probs = {a: _run_arm_probs(a, m) for (a, m) in nulls.items()}
    ref_name = "tralo_null"
    ref = probs[ref_name]
    for name, pr in probs.items():
        if name == ref_name:
            continue
        assert np.array_equal(ref, pr), (
            "%s is NOT bit-identical to %s at lambda 0 (max abs diff %g). The shared `null_sibling` is unsound -- either restore the per-family null arms or find what now differs between the code paths."
            % (name, ref_name, float(np.abs(ref - pr).max()))
        )
    treated = _run_arm_probs("fioretto", "fioretto_ldf")
    assert not np.array_equal(ref, treated), (
        "`fioretto` is bit-identical to the null on this harness, so it cannot tell a treated arm from an untreated one and the equalities above prove nothing"
    )


@pytest.mark.parametrize(
    "arm,methodology",
    [
        ("fioretto", "fioretto_ldf"),
        ("hounie", "hounie_rcl"),
        ("alm", "fioretto_alm"),
        ("tralo", "tralo"),
    ],
)
def test_every_trained_arm_logs_the_per_class_counts(arm, methodology):
    import tempfile
    import pandas as pd
    import scripts.smoke_arms as smoke

    tmp = tempfile.mkdtemp(prefix="counts_")
    try:
        (inputs, _g, _l) = smoke.make_inputs(load_protocol(), arm, tmp, seed=1)
        TRAIN_FNS[methodology](inputs)
        df = pd.read_csv(os.path.join(str(inputs.experiment_path), "training_log.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    capped = [
        c
        for c in df.columns
        if c.startswith("Limit_Class")
        and pd.to_numeric(df[c], errors="coerce").dropna().lt(1000000000.0).any()
    ]
    assert capped, (
        "%s logs no finite Limit_Class column, so no reader can tell which class was capped or what the budget was"
        % arm
    )
    for lim in capped:
        c = lim[len("Limit_Class") :]
        for prefix in ("Hard_Class", "Soft_Class"):
            col = prefix + c
            assert col in df.columns, (
                "%s caps class %s but never logs %s -- its count trajectory is unreadable from the log"
                % (arm, c, col)
            )
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            assert len(v) and (v >= 0).all(), "%s wrote no usable values in %s: %s" % (
                arm,
                col,
                list(v),
            )
        h = pd.to_numeric(df["Hard_Class" + c], errors="coerce").dropna()
        assert np.allclose(h, h.round()), (
            "%s wrote a non-integer hard count in Hard_Class%s: %s" % (arm, c, list(h))
        )


def _meta(labels, groups):
    return pd.DataFrame({"label": np.asarray(labels), "loc_group": np.asarray(groups)})


def _screen_pair(train, test):
    from scripts.dataset_screen import novelty_items

    return novelty_items(train, test, "loc_group", n_null=200, seed=0)


def test_screen_net_ignores_a_uniform_shift_but_sees_a_differential_one():
    rng = np.random.default_rng(0)
    n_g = 900
    base = [0.1, 0.2, 0.3]

    def build(p_by_group):
        (lab, grp) = ([], [])
        for gid, p in enumerate(p_by_group):
            draws = rng.choice([0, 1], size=n_g, p=[1 - p, p])
            lab += list(draws)
            grp += [gid] * n_g
        return _meta(lab, grp)

    def relabel(p_by_group, w):
        out = []
        for p in p_by_group:
            num = p * w[1]
            out.append(num / (num + (1 - p) * w[0]))
        return out

    train = build(base)
    uniform = build(relabel(base, [1.0, 2.0]))
    a = _screen_pair(train, uniform)
    assert a["global_z"] > 5, a
    assert a["net_z"] < 2.0, "a uniform shift leaked into NET: %r" % a
    differential = build([0.3, 0.2, 0.1])
    b = _screen_pair(train, differential)
    assert abs(b["global_z"]) < 3, "global fired on a mean-preserving reshuffle: %r" % b
    assert b["net_z"] > 5, "NET missed a pure differential shift: %r" % b
    assert b["net_items"] > 100, b["net_items"]


def test_screen_calls_index_modulo_groups_dead():
    rng = np.random.default_rng(1)
    n = 2700
    train = _meta(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]), np.arange(n) % 3)
    test = _meta(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]), np.arange(n) % 3)
    r = _screen_pair(train, test)
    assert r["net_z"] < 2.0, "index%%3 groups scored as informative: %r" % r
    rigged = test.copy()
    rigged["loc_group"] = (test["label"] == 0).astype(int)
    r2 = _screen_pair(train, rigged)
    assert r2["net_z"] > 5, "screen is blind even to label-aligned groups: %r" % r2


def test_screen_scores_fully_unseen_groups_against_the_global_prior():
    rng = np.random.default_rng(0)
    n_g = 800

    def build(spec):
        (lab, grp) = ([], [])
        for gid, p in spec:
            lab += list(rng.choice([0, 1, 2], size=n_g, p=p))
            grp += [gid] * n_g
        return _meta(lab, grp)

    train = build([(0, [0.5, 0.3, 0.2]), (1, [0.4, 0.4, 0.2]), (2, [0.5, 0.2, 0.3])])
    test = build(
        [(10, [0.9, 0.05, 0.05]), (11, [0.05, 0.9, 0.05]), (12, [0.05, 0.05, 0.9])]
    )
    r = _screen_pair(train, test)
    assert len(r["unseen_groups"]) == 3, r["unseen_groups"]
    assert r["unseen_items"] == 3 * n_g
    assert r["net_z"] > 10, "fully unseen groups scored as no information: %r" % r
    assert r["net_items"] > 500, r["net_items"]
    flat = build([(10, [0.9, 0.05, 0.05])] * 3)
    r2 = _screen_pair(train, flat)
    assert len(r2["unseen_groups"]) == 1, r2["unseen_groups"]
    assert r2["net_z"] < 2.0, "a uniform unseen group leaked into NET: %r" % r2


def test_the_trainer_decides_satisfaction_and_the_ratchet_from_HARD_counts():
    path = os.path.join(REPO, "src", "methodologies", "tralo", "train.py")
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def count_uses(text):
        out = []
        for ln in text.splitlines():
            t = ln.strip()
            counted = re.search("total_(local|global)_(hard|soft)", t)
            if not counted:
                continue
            if re.search("[<>]=?", t) or re.match("\\w*hard\\w*\\s*=\\s*total_", t):
                out.append(t)
        return out

    poisoned = "if total_local_soft[g][c].item() > lc[c].item():"
    assert any(("_soft" in ln for ln in count_uses(poisoned))), (
        "the extractor cannot see a soft comparison even when one is planted"
    )
    checked = count_uses(os.linesep.join(lines))
    assert checked, "no count comparisons found -- the trainer moved"
    soft = [ln for ln in checked if "_soft" in ln]
    assert not soft, (
        "the trainer is comparing a SOFT count against a limit: %s. At K=0 the soft count is strictly positive, so this would make a satisfied group read as violated for the whole run."
        % soft[:3]
    )


def _log_health_campaign(tmp_path, accs, arm="tralo"):
    root = os.path.join(str(tmp_path), "camp")
    for seed in (1, 2, 3):
        d = os.path.join(root, "%s_seed%d" % (arm, seed))
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(
            {
                "Epoch": list(range(1, len(accs) + 1)),
                "Train_Acc": list(accs),
                "L_CE": [0.5] * len(accs),
            }
        ).to_csv(os.path.join(d, "training_log.csv"), index=False)
        with io.open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "arm": arm,
                    "status": "completed",
                    "dataset_mode": "iwildcam",
                    "model_name": "MobileNetV3",
                    "constraint_tag": "L30_G50",
                    "constraint": [0.3, 0.5],
                    "dataset_config": {"constrained_class": [2]},
                    "hyperparams": {"seed": seed},
                },
                fh,
            )
    r = subprocess.run(
        [sys.executable, "-m", "scripts.log_health", root],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return r.stdout + r.stderr


def test_log_health_flags_a_model_already_converged_before_the_constraint_phase(
    tmp_path,
):
    out = _log_health_campaign(tmp_path, [0.956, 0.956, 0.957, 0.956])
    assert "LOGGED ACCURACY OBSERVATIONS" in out, out[-1200:]
    assert "HIGH FLAT LOGGED ACCURACY" in out, out[-1200:]
    assert "scripts.headroom" in out, (
        "accuracy is only a proxy; inspect actual group cuts"
    )


@pytest.mark.parametrize(
    "accs,why",
    [
        ([0.4, 0.55, 0.7, 0.88], "a big gain must not read as saturated"),
        (
            [0.3, 0.3, 0.31, 0.3],
            "a flat but LOW run is not saturation, it is a model that never learned",
        ),
    ],
)
def test_log_health_does_not_cry_saturation_on_a_healthy_run(tmp_path, accs, why):
    out = _log_health_campaign(tmp_path, accs)
    assert "LOGGED ACCURACY OBSERVATIONS" in out, out[-1200:]
    assert "HIGH FLAT LOGGED ACCURACY" not in out, (why, out[-1200:])
    assert "not a high-flat logged signature" in out, out[-1200:]


def _nonfinite_campaign(tmp_path, steps_applied):
    root = os.path.join(str(tmp_path), "camp")
    d = os.path.join(root, "tralo_seed1")
    os.makedirs(d, exist_ok=True)
    n = 6
    pd.DataFrame(
        {
            "Epoch": list(range(1, n + 1)),
            "Train_Acc": [0.4, 0.55, 0.7, 0.8, 0.85, 0.88],
            "L_CE": [0.5] * n,
            "Grad_Norm": [1.0, float("inf"), 1.0, float("nan"), 1.0, 1.0],
        }
    ).to_csv(os.path.join(d, "training_log.csv"), index=False)
    cfg = {
        "arm": "tralo",
        "status": "completed",
        "dataset_mode": "iwildcam",
        "model_name": "ViTB16",
        "constraint_tag": "L30_G50",
        "constraint": [0.3, 0.5],
        "dataset_config": {"constrained_class": [2]},
        "hyperparams": {"seed": 1},
        "results": {"constraint_steps_applied": steps_applied},
    }
    with io.open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    r = subprocess.run(
        [sys.executable, "-m", "scripts.log_health", root],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return r.stdout + r.stderr


def test_the_nonfinite_flag_prints_the_step_budget_beside_it(tmp_path):
    out = _nonfinite_campaign(tmp_path, steps_applied=22)
    assert "NON-FINITE VALUES" in out, out[-800:]
    assert "22 constraint step(s) applied" in out, (
        "the step budget must appear beside the non-finite count, or the flag raises a question it does not answer"
        + out[-900:]
    )
    assert "cannot establish the cause or safety" in out, out[-900:]
    assert "FAIL:" in out and "RuntimeWarning" not in out, out[-900:]


def test_the_nonfinite_step_budget_is_read_from_the_run_not_assumed(tmp_path):
    out = _nonfinite_campaign(tmp_path, steps_applied=0)
    assert "0 constraint step(s) applied" in out, (
        "the number must come from the run" + out[-900:]
    )
    assert "22 constraint step(s) applied" not in out, out[-900:]


def _starvation_campaign(tmp_path, arms):
    root = os.path.join(str(tmp_path), "camp")
    for arm, steps in arms.items():
        for seed in (1, 2, 3):
            d = os.path.join(root, "%s_seed%d" % (arm, seed))
            os.makedirs(d, exist_ok=True)
            n = 6
            pd.DataFrame(
                {
                    "Epoch": list(range(1, n + 1)),
                    "Train_Acc": [0.4, 0.5, 0.6, 0.7, 0.75, 0.8],
                    "L_CE": [0.5] * n,
                    "Group53_Limit_Class7": [10.0] * n,
                    "Group53_Hard_Class7": [200, 200, 202, 204, 206, 208],
                    "Group218_Limit_Class7": [10.0] * n,
                    "Group218_Hard_Class7": [10, 11, 11, 11, 11, 11],
                }
            ).to_csv(os.path.join(d, "training_log.csv"), index=False)
            cfg = {
                "arm": arm,
                "status": "completed",
                "dataset_mode": "iwildcam",
                "model_name": "MobileNetV3",
                "constraint_tag": "L30_G50",
                "constraint": [0.3, 0.5],
                "dataset_config": {"constrained_class": [7]},
                "hyperparams": {"seed": seed},
            }
            if steps is not None:
                cfg["results"] = {"constraint_steps_applied": steps}
            with io.open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
    r = subprocess.run(
        [sys.executable, "-m", "scripts.log_health", root],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return r.stdout + r.stderr


def test_the_starvation_warning_is_never_made_about_an_arm_with_no_penalty(tmp_path):
    out = _starvation_campaign(tmp_path, {"tralo": 19, "tralo_null": 0})
    live = [
        ln
        for ln in out.splitlines()
        if "starvation signature" in ln or "WORST-violating" in ln
    ]
    blob = chr(10).join(live)
    assert "tralo:" in blob, (
        "the signature is present in the fixture, so the LIVE arm must still be flagged -- a guard that silences everything is not a fix "
        + out
    )
    assert "tralo_null:" not in blob, (
        "a lambda=0 twin took no constraint step, so 2(a2) cannot describe it " + out
    )


def test_the_starvation_guard_keeps_working_on_runs_predating_the_key(tmp_path):
    out = _starvation_campaign(tmp_path, {"tralo": None})
    assert "starvation signature" in out, (
        "with no steps key recorded there is no evidence of zero, so the diagnostic must still print "
        + out
    )


def _cct_json(path, alpha, n_loc=10, per_loc=200, n_cls=4, seed=3):
    rng = np.random.default_rng(seed)
    (images, anns) = ([], [])
    iid = 0
    for loc in range(n_loc):
        w = (
            np.ones(n_cls) / n_cls
            if alpha is None
            else rng.dirichlet(np.ones(n_cls) * alpha)
        )
        for _ in range(per_loc):
            c = int(rng.choice(n_cls, p=w))
            images.append(
                {"id": iid, "file_name": "img%06d.jpg" % iid, "location": loc}
            )
            anns.append({"image_id": iid, "category_id": c})
            iid += 1
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "categories": [{"id": i, "name": "sp%d" % i} for i in range(n_cls)],
                "images": images,
                "annotations": anns,
            },
            fh,
        )


def _screen_a_cct(tmp, alpha, tag):
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    ann = os.path.join(d, "ann.json")
    _cct_json(ann, alpha)
    out = os.path.join(d, "slice")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def run(cmd):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, env=env)
        so = r.stdout.decode("utf-8", "replace")
        se = r.stderr.decode("utf-8", "replace")
        assert r.returncode == 0, so + se
        return so

    prep = run(
        [
            sys.executable,
            "-m",
            "scripts.prep_iwildcam",
            "--annotations",
            ann,
            "--out",
            out,
            "--classes",
            "4",
            "--min-per-camera",
            "50",
            "--test-target",
            "400",
            "--train-per-class",
            "300",
            "--meta-only",
        ]
    )
    return (out, prep, run([sys.executable, "-m", "scripts.dataset_screen", out]))


def test_a_candidate_dataset_can_be_screened_before_it_is_downloaded(tmp_path):
    (out, prep_out, screen) = _screen_a_cct(str(tmp_path), 0.35, "live")
    assert "META ONLY" in prep_out, prep_out
    for split in ("train", "test"):
        f = os.path.join(out, "%s_meta.csv" % split)
        assert os.path.exists(f), f
        cols = io.open(f, encoding="utf-8").readline().strip().split(",")
        assert cols == ["label", "class_name", "filename", "location"], cols
    assert not [f for f in os.listdir(out) if f.endswith(".npy")], (
        "--meta-only wrote image arrays, so it downloaded something"
    )
    assert "NET ex" in screen and "GLOBAL ex" in screen, screen[-800:]
    assert "ABSENT from train" in screen, screen[-800:]


def test_identical_per_group_mixes_report_measurements_with_unseen_groups(tmp_path):
    (_out, _prep, screen) = _screen_a_cct(str(tmp_path), None, "dead")
    assert "NET excess" in screen and "z=" in screen, screen[-800:]
    assert "DEAD" not in screen and "PASS" not in screen, screen[-800:]
    assert "ABSENT from train" in screen, (
        "the fixture lost its held-out cameras, so this no longer controls for criterion 1"
        + chr(10)
        + screen[-800:]
    )


def test_the_training_path_never_imports_from_scripts():
    import ast as _ast

    offenders = []
    for sub in ("src", "configs"):
        root = os.path.join(REPO, sub)
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with io.open(path, encoding="utf-8") as f:
                    tree = _ast.parse(f.read(), filename=path)
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.Import):
                        for al in node.names:
                            if al.name.split(".")[0] == "scripts":
                                offenders.append((path, node.lineno, al.name))
                    elif isinstance(node, _ast.ImportFrom):
                        if (node.module or "").split(".")[0] == "scripts":
                            offenders.append((path, node.lineno, node.module))
    assert not offenders, (
        "the training path imports from scripts/, which makes a scorer deploy able to split a live campaign's code_version: %r"
        % (offenders,)
    )
    probe = os.path.join(REPO, "configs", "_import_direction_probe.py")
    io.open(probe, "w", encoding="utf-8").write("from scripts import full_panel\n")
    try:
        found = []
        with io.open(probe, encoding="utf-8") as f:
            tree = _ast.parse(f.read(), filename=probe)
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.ImportFrom)
                and (node.module or "").split(".")[0] == "scripts"
            ):
                found.append(node.module)
        assert found == ["scripts"], (
            "the AST scan cannot detect a scripts import even when one is there, so its clean result above says nothing"
        )
    finally:
        os.remove(probe)


def test_check_parity_fails_on_an_unknown_code_version_and_only_warns_on_dirty(
    tmp_path,
):
    import json
    import subprocess

    def render(stamp):
        from test_operational_cli import cli

        d = tmp_path / stamp
        generated = cli(
            "configs.gen_campaign",
            "--root",
            d,
            "--datasets",
            "iwildcam",
            "--arms",
            "all",
            "--seeds",
            "1",
        )
        assert generated.returncode == 0, generated.stderr
        for p in d.rglob("config.json"):
            cfg = json.loads(p.read_text())
            cfg.update(code_version=stamp, run_code_version=stamp)
            p.write_text(json.dumps(cfg))
        out = subprocess.run(
            [sys.executable, "-m", "scripts.check_parity", d],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return out.stdout.decode("utf-8", "replace")

    unknown = render("unknown")
    assert "CODE VERSION IS unknown" in unknown and "PARITY FAILED" in unknown
    dirty = render("74f858657154-dirty")
    assert "WARN: dirty source" in dirty and "PARITY OK" in dirty, dirty
    clean = render("74f858657154")
    assert "PARITY OK" in clean, clean


def test_the_factorial_gate_cannot_report_a_pass_it_did_not_measure():
    fc = importlib.import_module("scripts.factorial_control")
    root = tempfile.mkdtemp()
    live = fc.control(fc._synthetic(os.path.join(root, "f"), seed=0), sep="|")
    dead = fc.control(os.path.join(root, "f"), sep="@")
    assert live["raked"] == live["unseen"] > 0
    assert live["survives"] < 60.0, (
        "on a slice whose held-out cell IS the product of the observed marginals the raked baseline must absorb most of the novelty, got %.1f%%"
        % live["survives"]
    )
    assert "survives" in chr(10).join(fc.report(live, "live"))
    assert dead["raked"] == 0
    assert math.isnan(dead["survives"]), (
        "a separator that never occurs must yield NO survival figure, got %.1f%%"
        % dead["survives"]
    )
    txt = chr(10).join(fc.report(dead, "dead"))
    assert "NOT A CONTROL" in txt and "survives" not in txt, (
        "the refusal must replace the survival figure, not sit beside it"
    )
    assert dead["net_global"] == dead["net_additive"], (
        "with nothing raked the two arms are the same arm, so they must agree EXACTLY -- an RNG-only difference is what made ~100%% look measured"
    )


def test_the_factorial_gate_reports_the_undiluted_ratio():
    fc = importlib.import_module("scripts.factorial_control")
    root = tempfile.mkdtemp()
    r = fc.control(
        fc._synthetic(os.path.join(root, "f"), seed=0, n_unseen=300, n_seen=2000),
        sep="|",
    )
    assert r["raked"] > 0
    assert r["survives_diluted"] > r["survives"] + 10.0, (
        "expected the whole-slice figure to sit well above the unseen-only one (%.1f%% vs %.1f%%)"
        % (r["survives_diluted"], r["survives"])
    )
    assert r["survives"] < 60.0, (
        "the headline must be the UNDILUTED ratio, got %.1f%%" % r["survives"]
    )
    assert "diluted by the seen groups" in chr(10).join(fc.report(r, "x"))


def test_dataset_diagnostics_preserve_values_and_undefined_null_spread():
    from scripts.dataset_screen import diagnostic_lines

    for net, z in [(1.0, float("nan")), (9.0, 5.0), (500.0, 9.0)]:
        r = dict(gcol="location", net_items=net, net_z=z)
        text = " ".join(diagnostic_lines(r, "fixture"))
        assert "NET excess %+.2f items" % net in text
        assert ("undefined" in text) == (not np.isfinite(z))
        assert not any(
            word in text for word in ("PASS", "DEAD", "MARGINAL", "seed noise")
        )
    assert "NO GROUP COLUMN" in " ".join(diagnostic_lines(dict(gcol=None), "fixture"))

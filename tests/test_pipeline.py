"""Regression tests for the invariants this project has broken before.

Every test here corresponds to a defect that actually shipped. The point is not
coverage -- it is that the specific ways this pipeline has produced wrong
numbers are now mechanically checked.

    python -m pytest tests -q

Runs in a few seconds on CPU and needs no dataset.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.gen_campaign import (build_hyperparams, cap_pair,      # noqa: E402
                                  compute_base_model_id, load_protocol)
from src.losses.transductive_loss import MulticlassTransductiveLoss  # noqa: E402
from src.methodologies.heuristic.train import (                      # noqa: E402
    _build_hierarchy, apply_allocation_heuristic, verify_allocation)
from src.training.constraints import (compute_global_constraints,    # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED                            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- the loss --

def _reference_penalty(soft, K, rho=0.5):
    """The penalty as it stood before the K=0 fix. For K >= 1 the current code
    must match this exactly -- the fix must not move any existing result."""
    E = F.relu(soft - K)
    e = E / (K + 1e-8)
    return E / (E + K + 1e-8) + rho * (e ** 2) / (1 + e ** 2 + 1e-8)


def _loss(global_constraints, local_constraints=None, rho=0.5):
    return MulticlassTransductiveLoss(
        global_constraints=global_constraints,
        local_constraints=local_constraints or {},
        num_classes=len(global_constraints), initial_rho=rho)


@pytest.mark.parametrize("K", [1, 2, 5, 20, 67, 250])
@pytest.mark.parametrize("soft", [0.0, 0.5, 7.5, 20.0, 100.0, 1e4])
def test_penalty_unchanged_for_positive_K(K, soft):
    """max(K,1) is the identity for K >= 1: value AND gradient."""
    crit = _loss([K, UNLIMITED])
    a = torch.tensor(soft, requires_grad=True)
    b = torch.tensor(soft, requires_grad=True)
    crit._penalty(a, K).backward()
    _reference_penalty(b, K).backward()
    assert torch.allclose(crit._penalty(torch.tensor(soft), K),
                          _reference_penalty(torch.tensor(soft), K), atol=0, rtol=0)
    assert a.grad.item() == b.grad.item()


def test_K0_constraint_has_a_gradient():
    """A group with no true instance of the capped class gets K=0 legitimately.
    It used to sit pinned at the penalty's own bound: a nonzero CONSTANT with
    exactly zero gradient, permanently unsatisfiable, holding the ratchet gate
    open for every other constraint."""
    crit = _loss([0, UNLIMITED])
    soft = torch.tensor(12.0, requires_grad=True)
    crit._penalty(soft, 0).backward()
    assert soft.grad.item() > 1e-6, "K=0 must push the count down"
    old = torch.tensor(12.0, requires_grad=True)
    _reference_penalty(old, 0).backward()
    assert old.grad.item() == 0.0, "the old form really was inert (guards the test)"


def test_penalty_is_zero_at_and_below_the_budget():
    crit = _loss([10, UNLIMITED])
    for soft in (0.0, 5.0, 9.99, 10.0):
        assert crit._penalty(torch.tensor(soft), 10).item() == 0.0


def test_empty_constraints_stay_connected_to_the_graph():
    """An all-UNLIMITED scope must still return an autograd-connected zero on
    the right device, or .backward() raises on a run with no capped class."""
    crit = _loss([UNLIMITED, UNLIMITED])
    counts = torch.zeros(2, requires_grad=True)
    total = crit.compute_global_from_counts(counts)
    assert total.item() == 0.0
    assert total.requires_grad, "an all-UNLIMITED scope returned a detached zero"
    total.backward()                       # must not raise
    assert counts.grad is not None

    # An empty local scope has no count tensor to hang the graph on, so its
    # zero is a constant -- but it must be on the MODULE's device, or adding it
    # to a CUDA global term raises. It used to be a bare CPU tensor(0.0).
    local = crit.compute_local_from_counts({})
    assert local.item() == 0.0
    assert local.device == crit.global_constraints.device
    # and the sum must stay differentiable as long as either scope is live
    assert (crit.compute_global_from_counts(counts) + local).requires_grad


# ------------------------------------------------------------- the budgets --

def _frame(labels, groups):
    import pandas as pd
    return pd.DataFrame({"label": np.asarray(labels), "grp": np.asarray(groups)})


def test_local_and_global_percentages_are_independent():
    df = _frame([1] * 100 + [0] * 100, [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
    local_pct, global_pct = cap_pair("L50_G30")
    assert (local_pct, global_pct) == (0.5, 0.3)
    g = compute_global_constraints(df, "label", global_pct,
                                   constrained_class=1, num_classes=2)
    loc = compute_local_constraints(df, "label", local_pct, "grp",
                                    constrained_class=1, num_classes=2)
    assert g[1] == 30                                  # 30% of 100
    assert sorted(v[1] for v in loc.values()) == [25, 25]   # 50% of 50, twice


def test_K_zero_from_a_nonzero_count_is_refused():
    """A percentage that rounds a real class to zero would vanish silently."""
    df = _frame([1] + [0] * 99, [0] * 100)
    with pytest.raises(ValueError, match="K=0"):
        compute_global_constraints(df, "label", 0.3,
                                   constrained_class=1, num_classes=2)


def test_K_zero_from_an_absent_class_is_allowed():
    """count == 0 is a real, tightest-possible budget, not a config error."""
    df = _frame([0] * 50 + [1] * 50, [0] * 50 + [1] * 50)
    loc = compute_local_constraints(df, "label", 0.5, "grp",
                                    constrained_class=1, num_classes=2)
    assert loc[0][1] == 0 and loc[1][1] == 25


def test_several_capped_classes_get_independent_budgets():
    df = _frame([0] * 40 + [1] * 60 + [2] * 100, list(range(2)) * 100)
    g = compute_global_constraints(df, "label", 0.5,
                                   constrained_class=[0, 2], num_classes=3)
    assert g[0] == 20 and g[2] == 50 and g[1] == UNLIMITED


# ----------------------------------------------------------- the allocator --

def _probs(n, k, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


@pytest.mark.parametrize("seed", range(6))
def test_single_capped_class_is_exactly_top_K(seed):
    n, k, K = 200, 4, 30
    p, g = _probs(n, k, seed), np.zeros(200, dtype=int)
    gcon = [UNLIMITED] * k
    gcon[2] = K
    lcon = {0: [UNLIMITED] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [2]), gcon, lcon, k)
    chosen = set(np.where(y == 2)[0])
    assert chosen == set(np.argsort(p[:, 2])[::-1][:K])


@pytest.mark.parametrize("seed", range(6))
def test_several_capped_classes_are_not_starved(seed):
    """Filling classes one at a time let each take its full budget from the
    whole pool, so the last capped class got nothing: [40, 40, 40, 0]."""
    n, k = 120, 4
    p, g = _probs(n, k, seed), np.zeros(120, dtype=int)
    gcon, lcon = [40] * k, {0: [40] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k)
    counts = np.bincount(y, minlength=k)
    assert counts.min() > 0, "a capped class was starved: %s" % counts
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_a_local_cap_binds_without_a_global_cap():
    """The leftover pass gated the local check on the GLOBAL cap, so a class
    capped only locally was never blocked locally."""
    n, k = 200, 4
    p = _probs(n, k, 11)
    g = np.arange(n) % 2
    gcon = [UNLIMITED] * k
    lcon = {0: [UNLIMITED, 5, UNLIMITED, UNLIMITED],
            1: [UNLIMITED, 5, UNLIMITED, UNLIMITED]}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [1]), gcon, lcon, k)
    for grp in (0, 1):
        assert int((y[g == grp] == 1).sum()) <= 5
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_infeasible_instance_does_not_dump_everything_into_class_zero():
    """argmax of an all -1 vector returns 0, so every impossible item was
    assigned class 0 far past its cap, silently."""
    n, k = 100, 4
    p, g = _probs(n, k, 7), np.zeros(100, dtype=int)
    gcon, lcon = [10] * k, {0: [10] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k)
    counts = np.bincount(y, minlength=k)
    assert counts.max() < n / 2, "collapsed onto one class: %s" % counts


# ------------------------------------------------------- the generator gate --

def _gen(tmp, *extra):
    cmd = [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp),
           "--datasets", "dermmnist", "--arms", "tralo"] + list(extra)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_generator_refuses_a_single_cap_level(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30")
    assert r.returncode == 1 and "two cap levels" in r.stdout + r.stderr


def test_generator_refuses_an_out_of_range_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30", "--constrained-class", "9")
    assert r.returncode == 1 and "out of range" in r.stdout + r.stderr


def test_generator_refuses_a_repeated_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30",
             "--constrained-class", "4", "4")
    assert r.returncode == 1 and "repeats a class" in r.stdout + r.stderr


def test_generator_refuses_to_mix_capped_classes_in_one_root(tmp_path):
    """A completed run is never reset, so a second pass with a different capped
    class would leave both in one cell and the scorer would pool them."""
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    cfg_path = next(iter(sorted(tmp_path.rglob("config.json"))))
    cfg = json.loads(cfg_path.read_text())
    cfg["status"] = "completed"
    cfg_path.write_text(json.dumps(cfg))
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30",
             "--constrained-class", "4", "5")
    assert r.returncode == 1 and "already holds a run" in r.stdout + r.stderr


def test_mandatory_clippers_are_always_added(tmp_path):
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    arms = {p.parts[-3] for p in tmp_path.rglob("config.json")}
    assert {"clip", "focal_clip"} <= arms


# -------------------------------------------------- the warm-up cache identity

def _bid(P, arm, seed=1, **over):
    dc = dict(P["datasets"]["dermmnist"])
    hp = build_hyperparams(P, P["arms"][arm], seed)
    hp.update(over)
    return compute_base_model_id(P, "MobileNetV3", hp, "dermmnist", dc)


def test_arms_differing_only_in_the_allocator_share_a_warm_up():
    P = load_protocol()
    assert _bid(P, "clip") == _bid(P, "lp")
    assert _bid(P, "focal_clip") == _bid(P, "focal_lp")
    assert _bid(P, "tralo") == _bid(P, "fioretto") == _bid(P, "alm")


def test_a_different_warm_up_objective_does_not_share_a_model():
    """clip and focal_clip once hashed identically, so focal_clip silently
    loaded clip's model and became a second clip."""
    P = load_protocol()
    assert _bid(P, "clip") != _bid(P, "focal_clip")
    assert _bid(P, "clip") != _bid(P, "cb_lp") != _bid(P, "la_lp")


@pytest.mark.parametrize("key,value", [
    ("lr", 0.5), ("dropout", 0.9), ("batch_size", 7), ("warmup_epochs", 3),
    ("pretrained", False), ("class_weighted_ce", True), ("seed", 99),
])
def test_every_warm_up_key_moves_the_hash(key, value):
    P = load_protocol()
    assert _bid(P, "tralo") != _bid(P, "tralo", **{key: value})


def test_constraint_phase_keys_do_not_move_the_hash():
    """They do not change what the warm-up optimizes, so the cache must be
    shared -- that sharing is the whole point."""
    P = load_protocol()
    base = _bid(P, "tralo")
    for key, value in (("constraint_epochs", 5), ("lambda_step", 0.9),
                       ("initial_rho", 9.0), ("enable_checkpoint_restore", True)):
        assert _bid(P, "tralo", **{key: value}) == base


# ------------------------------------------------------------- the lifecycle --

def test_a_diverged_result_is_not_recorded_as_completed(tmp_path):
    """An all-NaN model still scores like a degenerate but healthy classifier;
    `completed` means the dispatcher never revisits it."""
    from src.pipeline.io import save_results_to_config
    cfg = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg, tmp_path, {"accuracy": float("nan"), "f1": 0.3})
    assert cfg["status"] == "diverged" and "accuracy" in cfg["diverged_keys"][0]
    cfg2 = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg2, tmp_path, {"accuracy": 0.8, "f1": 0.3})
    assert cfg2["status"] == "completed"


def test_rerunning_does_not_turn_the_old_header_into_a_data_row(tmp_path):
    """A crashed run is reset to pending and re-dispatched into the same dir;
    rows are appended, so df["Epoch"].max() returned the STRING 'Epoch'."""
    import pandas as pd
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
    """Lambda_Local was passed as a literal 0.0 and read 0.0000 forever."""
    from src.training.logging import build_csv_header
    header = build_csv_header(num_classes=2)
    assert "Lambda_Local" in header
    assert "Grad_Norm" in header and "L_KL" not in header


# --------------------------------------------------------- the backbone heads

@pytest.mark.parametrize("name", ["MobileNetV3", "MobileNetV2",
                                  "RegNetY400MF", "ViTB16"])
def test_backbones_keep_their_pretrained_weights(name):
    """Every backbone must replace ONLY its final layer. MobileNetV3 -- the
    headline backbone -- used to rebuild its whole classifier, discarding the
    pretrained 960->1280 projection. That projection is trained during warm-up
    only, and trained arms get ONE warm-up epoch against the post-hoc arms'
    thirty, so it biased the headline comparison on the headline backbone."""
    from src.models import get_model
    torch.manual_seed(0)
    a = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    torch.manual_seed(999)
    b = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    named_a, named_b = dict(a.named_parameters()), dict(b.named_parameters())
    differ = [k for k in named_a
              if not torch.equal(named_a[k], named_b[k])]
    # only the randomly-initialised final layer may differ between two seeds
    assert len(differ) <= 2, (
        "%s re-initialises %d tensors from random, not just the final layer: %s"
        % (name, len(differ), differ[:6]))


def test_mobilenetv3_has_exactly_one_dropout_in_its_head():
    """The whole reason the classifier was rebuilt was to avoid a double
    dropout. Reusing the pretrained head must not reintroduce one."""
    import torch.nn as nn
    from src.models import get_model
    m = get_model("MobileNetV3", input_dim=None, n_classes=7,
                  dropout=0.3, pretrained=False)
    head = m.backbone.classifier
    drops = [l for l in head if isinstance(l, nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == 0.3


# ------------------------------------------------------ the failure lifecycle

def test_a_repeatedly_failing_run_stops_being_re_dispatched(tmp_path):
    """A config that fails deterministically used to reset to `pending` and be
    picked up again by every subsequent dispatch, forever, with nothing on disk
    saying why."""
    from src.utils.filesystem_manager import (MAX_FAILURES,
                                              get_experiments_by_status,
                                              save_config_to_path,
                                              update_experiment_status)
    exp = tmp_path / "MobileNetV3" / "derm" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "pending", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    for i in range(1, MAX_FAILURES + 1):
        update_experiment_status(str(exp), "pending", count_failure=True)
        cfg = json.loads((exp / "config.json").read_text())
        assert cfg["failures"] == i
    assert cfg["status"] == "failed"
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_a_diverged_run_is_not_re_dispatched(tmp_path):
    from src.utils.filesystem_manager import (get_experiments_by_status,
                                              save_config_to_path)
    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "diverged", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_an_interrupted_run_still_resets_to_pending(tmp_path):
    """Load-bearing: this is what makes overnight re-dispatch idempotent.
    `running` must NOT be terminal."""
    from src.utils.filesystem_manager import (get_experiments_by_status,
                                              save_config_to_path)
    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "running", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    buckets = get_experiments_by_status(str(tmp_path))
    assert len(buckets["pending"]) == 1 and not buckets["blocked"]

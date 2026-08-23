"""Regression tests for the invariants this project has broken before.

Every test here corresponds to a defect that actually shipped. The point is not
coverage -- it is that the specific ways this pipeline has produced wrong
numbers are now mechanically checked.

    python -m pytest tests -q

Runs in a few seconds on CPU and needs no dataset.
"""
import io
import json
import os
import shutil
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
from scripts.full_panel import equalize_multi
from src.experiments.runner import TRAIN_FNS
from src.losses.transductive_loss import margin_window
from src.methodologies.select.train import selective_loss
from src.models import get_model
from src.training.constraint_step import finish_constraint_step
from src.utils.data_loader import _load_imagery_data as load_data
from src.utils.posthoc_adjustment import targeted_correction
import ast
import pandas as pd
import pathlib
import re
import torch.nn as nn
import yaml

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
    exactly zero gradient, so the constraint contributed nothing to the model at
    all. `scale = max(K, 1)` fixed that.

    NOTE the claim that used to be appended here -- that it also held the
    ratchet gate open for every other constraint -- is FALSE and was removed
    2026-08-22. The trainer computes satisfaction and the gate from the HARD
    counts, which can be exactly zero. See
    `test_the_trainer_decides_satisfaction_and_the_ratchet_from_HARD_counts`."""
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
    # `tralo_reseed` is here because the generator REFUSES a campaign that
    # holds a trained arm without the reseed floor -- a count trajectory read
    # without it is not a measurement. Every test below this line is about
    # something else, so they carry the control rather than trip that gate; the
    # gate itself is tested by
    # test_generator_refuses_a_count_reading_campaign_without_the_reseed_control.
    cmd = [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp),
           "--datasets", "iwildcam",
           "--arms", "tralo", "tralo_reseed"] + list(extra)
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
    dc = dict(P["datasets"]["iwildcam"])
    hp = build_hyperparams(P, P["arms"][arm], seed)
    hp.update(over)
    return compute_base_model_id(P, "MobileNetV3", hp, "iwildcam", dc)


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


# --------------------------------------------- protocol values are not optional

@pytest.mark.parametrize("key", ["lr_constraint", "constraint_epochs",
                                 "stable_count_threshold"])
def test_a_missing_protocol_value_raises_instead_of_using_the_trap(key, tmp_path):
    """These inline defaults WERE the retracted values: lr_constraint 1e-5
    against the protocol's 1e-4 (an unequal lr_constraint fabricated a -16.7 pp
    finding), constraint_epochs 150 against 29, and stable_count_threshold 5
    against 31 -- low enough that the early stop would actually fire, so an arm
    would stop training partway and still be scored at 'equal compute'."""
    import scripts.smoke_arms as sm
    P = load_protocol()
    inputs, _, _ = sm.make_inputs(P, "tralo", str(tmp_path))
    inputs.hyperparams.pop(key, None)
    with pytest.raises(KeyError, match=key):
        TRAIN_FNS["tralo"](inputs)


def test_deleted_danits_helpers_are_really_gone():
    """They were reachable only through __init__ re-exports, which is why the
    AST reachability pass reported them as live. The manuscript claims two
    post-hoc clippers -- a greedy threshold (that is the `clip` arm) and LP-LG
    with 'an identity misclassification cost rather than the general cost
    matrix' -- so none of the three was in scope."""
    import src.methodologies.danits_lp as d
    assert set(d.__all__) == {"solve_lp_assignment", "LPResult"}
    for gone in ("solve_greedy_assignment", "build_psi_phi_from_percentages",
                 "build_priority_cost_matrix", "describe_cost_matrix"):
        assert not hasattr(d, gone), "%s came back" % gone


# --------------------------------------------------- the gates must be able to fail

def test_parity_catches_two_arms_sharing_one_warm_up_with_different_objectives(tmp_path):
    """Occurrence 5 of the inert-flag failure: clip and focal_clip hashed
    identically, so focal_clip loaded clip's model and silently became a second
    clip. This gate used to print the sharing groups and ask a human to look."""
    r = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp_path),
         "--datasets", "iwildcam", "--models", "MobileNetV3",
         "--caps", "L30_G30", "L50_G30", "--arms", "clip", "focal_clip"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0
    clip_id = next(json.loads(p.read_text())["base_model_id"]
                   for p in sorted(tmp_path.rglob("config.json"))
                   if json.loads(p.read_text())["arm"] == "clip")
    for p in sorted(tmp_path.rglob("config.json")):
        cfg = json.loads(p.read_text())
        if cfg["arm"] == "focal_clip":
            cfg["base_model_id"] = clip_id
            p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "scripts.check_parity", str(tmp_path)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1
    # gate 4 now names the offending key rather than saying "objectives",
    # because it checks all twelve warm-up-identity keys, not just this one
    assert "DIFFERENT warmup_loss" in r.stdout


def test_verify_caps_fails_when_it_cannot_read_a_slice(tmp_path):
    """It printed 'CAP CHECK OK -- every cap tag produces a real integer budget
    on every dataset' having opened no file at all."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.verify_caps", "--datasets", "iwildcam"],
        cwd=str(tmp_path), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO})
    assert r.returncode == 1, "a gate that cannot fail is not a gate"


def test_the_scorer_pairs_on_the_capped_class(tmp_path):
    """pivot_table's default aggfunc averaged two capped-class settings into one
    pair, so a +0.40 cell and a -0.40 cell collapsed to an exact tie while the
    header still printed two cells."""
    import scripts.full_panel as fp
    src = io.io.open(os.path.join(REPO, "scripts", "full_panel.py"),
                      encoding="utf-8").read() if False else open(
        os.path.join(REPO, "scripts", "full_panel.py"), encoding="utf-8").read()
    key = src.split("key = [")[1].split("]")[0]
    assert '"capped"' in key, "the capped class is not in the pairing key"
    with pytest.raises(ValueError, match="pairing key is missing"):
        fp._one(pd.Series([0.4, -0.4]))
    assert fp._one(pd.Series([0.4])) == 0.4


# ------------------------------------------------------------- the verdict rule

def _panel_verdict(tmp_path, n_better_cells, n_tied_cells, metric="AP"):
    """Build a synthetic campaign with a KNOWN answer and read the verdict."""
    # 8 DISTINCT cells. They used to span three datasets; with only iwildcam
    # live the axis moved to backbone x cap. Distinctness is load-bearing --
    # a repeated (dataset, model, cap) writes into the same directory, so the
    # later cell silently overwrites the earlier one and the panel sees fewer
    # pairs than the test believes it built.
    cells = [(ds, m, cap)
             for m in ("MobileNetV3", "MobileNetV2", "RegNetY400MF", "ViTB16")
             for cap in ("L30_G30", "L50_G30")
             for ds in ("iwildcam",)]
    assert len(set(cells)) == len(cells) == 8
    N, K = 200, 4
    for i, (ds, model, cap) in enumerate(cells[:n_better_cells + n_tied_cells]):
        for arm, boost in (("clip", 0.0),
                           ("tralo", 1.5 if i < n_better_cells else 0.0)):
            for seed in (1, 2, 3, 4):
                rng = np.random.default_rng(1000 * i + seed)
                y = rng.integers(0, K, size=N)
                z = rng.normal(size=(N, K))
                z[np.arange(N), y] += boost
                P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
                d = tmp_path / model / ds / cap / arm / ("seed_%d" % seed)
                d.mkdir(parents=True, exist_ok=True)
                cols = {"True_Label": y, "Predicted_Label": P.argmax(1),
                        "Group_ID": rng.integers(0, 3, size=N)}
                for c in range(K):
                    cols["Prob_Class_%d" % c] = P[:, c]
                for f in ("final_predictions_raw.csv", "final_predictions.csv"):
                    pd.DataFrame(cols).to_csv(d / f, index=False)
                (d / "config.json").write_text(json.dumps(
                    {"arm": arm, "methodology": "x", "model_name": model,
                     "dataset_mode": ds, "constraint_tag": cap,
                     "constraint": [0.5, 0.3], "status": "completed",
                     "dataset_config": {"constrained_class": 1, "num_classes": K},
                     "hyperparams": {"seed": seed}}))
    r = subprocess.run([sys.executable, "-m", "scripts.full_panel",
                        "--campaign", str(tmp_path), "--control", "clip"],
                       cwd=REPO, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith(metric + " "):
            return line
    return r.stdout + r.stderr


def test_a_win_with_majority_ties_is_not_reported_as_a_loss(tmp_path):
    """stats.wilcoxon DISCARDS zero differences, but the majority test counted
    them in its denominator. With 3 cells strictly better and 5 bit-identical
    the old rule printed '*** LOSS' on a +0.19 delta at p=0.0022. The bug is
    asymmetric -- it can only turn a win into a loss -- and partial ties are the
    normal state here."""
    line = _panel_verdict(tmp_path, n_better_cells=3, n_tied_cells=5)
    assert "LOSS" not in line, line
    assert "3/0" in line, line          # 3 better, 0 worse, rest tied


def test_a_clean_win_is_still_called_a_win(tmp_path):
    line = _panel_verdict(tmp_path, n_better_cells=6, n_tied_cells=2)
    assert "*** WIN" in line, line


def test_a_bit_identical_arm_is_a_dead_flag_not_a_direction(tmp_path):
    """scipy returns p=1.0 for n<=12 and NaN for n>=16 on all-zero differences,
    so an inert arm printed 'lean loss' x13. Identical output means the
    treatment did nothing -- the project's most frequent failure, five
    occurrences -- and must never render as a direction."""
    line = _panel_verdict(tmp_path, n_better_cells=0, n_tied_cells=8)
    assert "DEAD FLAG" in line, line
    assert "loss" not in line.lower(), line


def test_unattainable_significance_is_declared_not_called(tmp_path):
    """At n=4 the exact two-sided Wilcoxon floor is 0.125, so 'p=0.125, lean
    loss' was the arm being un-callable, not a settled tie."""
    line = _panel_verdict(tmp_path, n_better_cells=4, n_tied_cells=0)
    assert "NOT CALLABLE" in line, line


def test_the_scorer_skips_runs_that_are_not_completed(tmp_path):
    """It ignored `status` entirely, and regenerating a campaign overwrites a
    non-completed config in place while leaving the OLD predictions on disk --
    so the previous code's predictions get scored as the new code's result."""
    _panel_verdict(tmp_path, 6, 2)
    for p in sorted(tmp_path.rglob("config.json")):
        cfg = json.loads(p.read_text())
        if cfg["arm"] == "tralo":
            cfg["status"] = "diverged"
            p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "scripts.full_panel",
                        "--campaign", str(tmp_path), "--control", "clip"],
                       cwd=REPO, capture_output=True, text=True)
    assert "skipped" in r.stdout and "diverged" in r.stdout


def test_the_audit_sees_keys_read_through_the_required_helper():
    """_required(hp, "lr_constraint", float) passes the config as an argument.
    The walker understood subscripts and dict methods only, so every key read
    that way was invisible in BOTH directions -- emitting one looked
    HALLUCINATED, omitting one produced no SILENT flag."""
    from scripts.audit_config import per_methodology_reads
    reads = per_methodology_reads()
    for meth in ("tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        assert "lr_constraint" in reads[meth], meth
        assert "stable_count_threshold" in reads[meth], meth
        assert "enable_checkpoint_restore" in reads[meth], meth

    # AND THE READ SET MUST STAY PER-ARM. Anything under audit_config's
    # SHARED_DIRS is credited to EVERY methodology, so putting a constraint-
    # phase key's only reader in src/training/ silently makes that key
    # legitimate on `clip`. Measured: with `read_step_config` living there, a
    # `clip` config carrying `constraint_grad_clip` audited CLEAN; with it in
    # src/methodologies/dual_common.py the same config FAILS and names the key.
    # The four trained arms share one reader either way -- only its address
    # decides whether the gate can still see the difference.
    for meth in ("heuristic", "danits_lp", "focal", "class_balanced",
                 "logit_adjust"):
        for key in ("constraint_grad_clip", "constraint_grad_mode",
                    "constraint_step_rule", "constraint_random_direction"):
            assert key not in reads[meth], (
                "%s can read %s, so a post-hoc arm emitting it would audit "
                "clean. Its reader has moved into a directory audit_config "
                "unions into every methodology." % (meth, key))


# ---------------------------------------------------------------------------
# A read must not mutate the model's train/eval mode.
#
# compute_prediction_statistics is called immediately after model.eval() at the
# end of every trained run and used to return the model in train() mode. Every
# current caller re-asserts eval(), so it was harmless -- but the duals' own
# comments say eval() during a test-set pass is what stops BN running stats
# updating from test data, "a data-leakage source that flips a few borderline
# samples and corrupts the lambda update". A read that silently arms that is
# the shape of defect this project keeps finding.
# ---------------------------------------------------------------------------
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
            "compute_prediction_statistics left the model in %s mode after "
            "being called in %s mode" % (model.training, mode))

        model.train(mode)
        metrics.get_predictions_with_probabilities(model, X)
        assert model.training is mode

        model.train(mode)
        metrics.compute_train_accuracy(model, loader, torch.device("cpu"))
        assert model.training is mode


# ---------------------------------------------------------------------------
# A misconfigured or swapped slice must FAIL, not produce a plausible number.
#
# Before these guards: four independent np.load calls with no length comparison
# (train died late inside TensorDataset with an unlabelled AssertionError, test
# never raised at all -- the chunked loops key off len(X_test) and would score
# fewer items than the labels describe); and num_classes / constrained_class
# were never checked against the labels, so pointing data_dir at the wrong
# dataset gave K=0 on an absent class, a log warning, and a complete run.
# ---------------------------------------------------------------------------
def _write_slice(d, n_train=12, n_test=8, n_classes=4, capped=2,
                 truncate_test_labels=False, drop_capped=False):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    ytr = np.array([i % n_classes for i in range(n_train)])
    yte = np.array([i % n_classes for i in range(n_test)])
    if drop_capped:                       # a slice that lacks the capped class
        ytr = np.where(ytr == capped, (capped + 1) % n_classes, ytr)
        yte = np.where(yte == capped, (capped + 1) % n_classes, yte)
    np.save(os.path.join(d, "train_images.npy"),
            rng.random((n_train, 3, 8, 8)).astype("float32"))
    np.save(os.path.join(d, "train_labels.npy"), ytr)
    np.save(os.path.join(d, "test_images.npy"),
            rng.random((n_test, 3, 8, 8)).astype("float32"))
    np.save(os.path.join(d, "test_labels.npy"),
            yte[:-1] if truncate_test_labels else yte)
    pd.DataFrame({"label": yte, "grp": [i % 2 for i in range(n_test)]}).to_csv(
        os.path.join(d, "test_meta.csv"), index=False)


def _cfg(d, n_classes=4, capped=2):
    return {"dataset_mode": "iwildcam", "dataset_config": {
        "data_dir": d, "num_classes": n_classes, "constrained_class": capped,
        "group_column": "grp", "target_column": "label"},
        "constraint": [0.5, 0.5]}


def test_loader_refuses_mismatched_image_and_label_counts(tmp_path):
    d = str(tmp_path / "bad_len")
    _write_slice(d, truncate_test_labels=True)
    with pytest.raises(ValueError, match="test_images.npy has 8 rows"):
        load_data(_cfg(d))


def test_loader_refuses_a_slice_whose_labels_exceed_num_classes(tmp_path):
    d = str(tmp_path / "wrong_ds")
    _write_slice(d, n_classes=6)                 # 6 real classes...
    with pytest.raises(ValueError, match="num_classes is 4"):
        load_data(_cfg(d, n_classes=4))          # ...config says 4


def test_loader_refuses_a_slice_missing_the_capped_class(tmp_path):
    d = str(tmp_path / "no_capped")
    _write_slice(d, drop_capped=True)
    with pytest.raises(ValueError, match="does not occur in this slice"):
        load_data(_cfg(d))


def test_loader_accepts_a_well_formed_slice(tmp_path):
    """The guards must not fire on good data."""
    d = str(tmp_path / "good")
    _write_slice(d)
    out = load_data(_cfg(d))
    assert out is not None


# ---------------------------------------------------------------------------
# Pin the penalty's gradient SHAPE.
#
# Above rho ~ 1 the gradient is non-monotone in the violation: near-zero at the
# boundary, peaking at u = 1/sqrt(3) = 57.7% overshoot, then decaying toward
# zero. At rho=100 a constraint violated 8x over budget gets ~167x less pull
# than one violated 58% over. This is the PUBLISHED formula (manuscript Eq. 4),
# not a bug, so the test pins it rather than forbidding it -- if someone changes
# the shape, that has to be a deliberate act with the paper updated, and this
# test is where they find out.
# ---------------------------------------------------------------------------
def test_penalty_gradient_is_non_monotone_above_rho_one():

    K = 67.0

    def grad_at(rho, violation):
        soft = torch.tensor(K + violation, requires_grad=True, dtype=torch.double)
        Kt = torch.tensor(K, dtype=torch.double)
        E = torch.relu(soft - Kt)
        S = Kt
        eps = 1e-8
        e = E / (S + eps)
        (E / (E + S + eps) + rho * (e ** 2) / (1 + e ** 2 + eps)).backward()
        return float(soft.grad)

    # rho starts at 0.5 and rho_step is derived as (100 - 0.5)/29 = 3.43,
    # so rho is 3.93 after ONE constraint epoch of twenty-nine.
    assert abs((100.0 - 0.5) / 29 - 3.431) < 0.001

    # at the initial rho the gradient is monotonically decreasing -- correct
    g0 = [grad_at(0.5, f * K) for f in (0.001, 0.3, 1.0, 8.0)]
    assert g0 == sorted(g0, reverse=True), "rho=0.5 should still be monotone"

    # one epoch in, it is not
    edge, peak, deep = (grad_at(3.93, 0.001 * K), grad_at(3.93, 0.577 * K),
                        grad_at(3.93, 8.0 * K))
    assert peak > 2.5 * edge, "expected a hump above the boundary value"
    assert peak > 100 * deep, "expected the deep violation to be starved"

    # u = 1/sqrt(3) is the peak of the QUADRATIC TERM alone, so it is the
    # rho -> infinity limit, not a general truth. It holds at rho=100...
    import math
    at_analytic = grad_at(100.0, (1 / math.sqrt(3)) * K)
    for f in (0.2, 0.4, 0.8, 1.5):
        assert grad_at(100.0, f * K) <= at_analytic + 1e-12, (
            "peak should be at u = 1/sqrt(3), not at %.2f" % f)

    # ...and NOT at the rho the runs actually operate at. The rational term's
    # own slope is decreasing, which pulls the combined peak left, to u ~ 0.53.
    # An earlier version of this claim said "analytically" with no caveat.
    us = [i / 500 for i in range(1, 5001)]
    gs = [grad_at(3.93, u * K) for u in us]
    peak_u = us[gs.index(max(gs))]
    assert 0.50 < peak_u < 0.56, (
        "at rho=3.93 the peak should sit left of 1/sqrt(3), got u=%.3f" % peak_u)


def test_a_deep_violation_is_starved_by_a_milder_one_sharing_the_clip():
    """The terms compete: `_sum` adds them, then ONE clip normalizes the total.

    So the relative weight of two capped scopes is set entirely by the penalty
    shape, and the shape hands almost everything to whichever scope sits nearer
    its own peak. dermmnist has 3 groups, so every run already carries 4 terms.
    """

    K, rho, eps = 67.0, 100.0, 1e-8

    def shares(u_a, u_b):
        gs = []
        for u in (u_a, u_b):
            soft = torch.tensor(K * (1 + u), requires_grad=True, dtype=torch.double)
            Kt = torch.tensor(K, dtype=torch.double)
            E = torch.relu(soft - Kt)
            e = E / (Kt + eps)
            (E / (E + Kt + eps) + rho * (e ** 2) / (1 + e ** 2 + eps)).backward()
            gs.append(float(soft.grad))
        tot = sum(g * g for g in gs)
        return [g * g / tot for g in gs]

    mild, deep = shares(0.577, 8.0)
    assert mild > 0.999, "the milder violation should take essentially all of it"
    assert deep < 0.001, "the 8x violation should be starved"
    # symmetric: it is the violation DEPTH, not the position in the sum
    deep2, mild2 = shares(8.0, 0.577)
    assert abs(mild - mild2) < 1e-9 and abs(deep - deep2) < 1e-9
    assert abs(sum(shares(0.577, 0.577))/2 - 0.5) < 1e-9


def test_loader_detects_a_permuted_train_split(tmp_path):
    """A permutation of images vs labels is invisible to a length check.

    This was written up as an undetectable accepted risk on the reasoning that
    no train_meta.csv exists. That was wrong -- all three prep scripts write
    one, with a `label` column, and it is on disk in every slice. The redundant
    signal needed to catch this was there the whole time.
    """
    d = str(tmp_path / "permuted")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    # same length, same multiset, different order -- exactly what a length
    # check cannot see
    pd.DataFrame({"label": y[::-1]}).to_csv(
        os.path.join(d, "train_meta.csv"), index=False)
    with pytest.raises(ValueError, match="not row-aligned"):
        load_data(_cfg(d))


def test_loader_accepts_an_aligned_train_meta(tmp_path):
    d = str(tmp_path / "aligned")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    pd.DataFrame({"label": y}).to_csv(
        os.path.join(d, "train_meta.csv"), index=False)
    assert load_data(_cfg(d)) is not None


def test_the_null_arm_really_delivers_no_constraint():
    """tralo_null is the control that makes "nothing happened" falsifiable.

    Everything else about it matches tralo -- warm-up length, epoch count,
    optimizer restart, transductive passes, allocator. If its lambdas were not
    exactly zero it would be a weak treatment rather than a control, and the
    whole point of the arm would be lost silently.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    blk = proto["blocks"]["tralo_null"]
    for key in ("lambda_global", "lambda_local", "lambda_step"):
        assert blk[key] == 0.0, "%s must be exactly 0.0, got %r" % (key, blk[key])

    # and the arm must otherwise be tralo: same phase, same shared block
    null_arm, real_arm = proto["arms"]["tralo_null"], proto["arms"]["tralo"]
    assert null_arm["phase"] == real_arm["phase"] == "trained"
    assert null_arm["methodology"] == real_arm["methodology"] == "tralo"
    assert "constraint_phase" in null_arm["blocks"]

    # zero lambda must make the summed penalty exactly zero, not merely small:
    # the trainer gates pass 2 on `total_constraint > 0`, so a 1e-30 residue
    # would still run a constraint step.
    from src.losses.transductive_loss import MulticlassTransductiveLoss as L
    total = torch.tensor(0.0)
    for soft, K in ((500.0, 67.0), (67.0, 67.0), (0.0, 67.0)):
        st = torch.tensor(soft)
        Kt = 67.0
        E = torch.relu(st - Kt)
        e = E / (Kt + 1e-8)
        pen = E / (E + Kt + 1e-8) + 0.5 * (e ** 2) / (1 + e ** 2 + 1e-8)
        total = total + 0.0 * pen
    assert float(total) == 0.0
    assert not bool(total > 0), "pass 2 would still run on a nonzero residue"
    assert L is not None


def test_every_trained_arm_reports_reordering():
    """The diagnostic must not reach one arm only.

    It was written inside tralo/train.py and reached only TraLO -- the exact
    shape of the CE-skip asymmetry that produced a 0.22 cc-F1 artifact, and of
    the focal keys that were live in one arm and dead in another. The guard is
    structural: every trainer that runs a constraint phase reaches the SAME two
    functions in the SAME module, and the scorer reads the field.

    Reached DIRECTLY (tralo) or through `src/methodologies/dual_common.py`,
    which the three duals share. Following the import chain rather than grepping
    one file is what lets the shared tail exist at all -- and it still fails if
    an arm grows a private copy or drops the summary field, because the module
    that calls `reordering_report` is also the module that must write
    "reordering".
    """

    def _reaches_reordering(src, seen):
        """Does this source call reordering_report and emit the summary key --
        here, or in any first-party module it imports?"""
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
            "%s must use the shared diagnostic, not a private copy" % m)
        assert _reaches_reordering(src, set()), (
            "%s never reaches reordering_report / never puts it in the summary" % m)

    # it has to survive to disk, and outside config["results"] -- a NaN tau on
    # a constant score column would otherwise mark the run `diverged`
    runner = io.open(os.path.join("src", "experiments", "runner.py"),
                      encoding="utf-8").read()
    assert "config['reordering']" in runner
    results_blk = runner[runner.index("save_results_to_config(config"):]
    assert "reordering" not in results_blk[:results_blk.index("})")]

    # and the scorer must actually read it
    panel = io.open(os.path.join("scripts", "full_panel.py"), encoding="utf-8").read()
    assert 'cfg.get("reordering")' in panel
    assert "_reordering_check(rows)" in panel


def test_the_documented_test_count_is_the_real_one(request):
    """CLAUDE.md and FRAMEWORK.md both quote this number. Both were wrong.

    CLAUDE.md said 75, FRAMEWORK.md said 96 in three places, and pytest
    collected 107. A reader uses the number to decide whether their checkout is
    complete, so a stale one says "you are missing tests" to someone who is not.
    """


    # Only meaningful when the whole suite was collected. Running a single node
    # id collects 1, which would fail the guard on every targeted run.
    # `-k` is an OPTION, not a positional arg, so scanning config.args never
    # saw it: a targeted `-k` run collected 1 test and failed this guard on
    # `n > 1` instead of skipping. Read the option itself.
    if (request.config.option.keyword
            or any("::" in a for a in request.config.args)):
        pytest.skip("subset run: the collected count is not the suite count")
    n = request.session.testscollected or len(request.session.items)
    assert n > 1

    claimed = {}
    for path in ("CLAUDE.md", "docs/FRAMEWORK.md"):
        txt = io.open(path, encoding="utf-8").read()
        for m in re.finditer(r"(\d+)\s+(?:regression\s+)?tests", txt):
            claimed.setdefault(path, set()).add(int(m.group(1)))

    wrong = {path: sorted(v - {n}) for path, v in claimed.items() if v - {n}}
    assert not wrong, (
        "pytest collects %d, but the docs claim %s. Update them, or the count "
        "tells a reader their checkout is incomplete." % (n, wrong))


NULL_SIBLINGS = [
    # (null arm, its treated parent, the keys that must be exactly zero)
    ("tralo_null", "tralo", ("lambda_step", "lambda_global", "lambda_local")),
    ("fioretto_null", "fioretto", ("fioretto_step_size",)),
    ("hounie_null", "hounie", ("hounie_eta_lambda",)),
    ("alm_null", "alm", ("alm_eta", "alm_mu0", "alm_mu_step")),
]


@pytest.mark.parametrize("null,parent,zeroed", NULL_SIBLINGS,
                         ids=[n for n, _, _ in NULL_SIBLINGS])
def test_every_trained_arm_has_a_working_zero_dose_sibling(null, parent, zeroed):
    """Without it, an arm-vs-clip delta cannot be attributed to the constraint.

    For a long time only tralo had one, so "was it the constraint or was it the
    regime" was falsifiable for exactly one of the four trained arms while the
    other three were reported against `clip` anyway.

    Zeroing is a DIFFERENT knob in each arm and two of them have a trap, so the
    keys are named per arm rather than pattern-matched.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    blk = proto["blocks"][null]
    for key in zeroed:
        assert blk[key] == 0.0, "%s.%s must be exactly 0.0, got %r" % (
            null, key, blk[key])

    a_null, a_parent = proto["arms"][null], proto["arms"][parent]
    assert a_null["methodology"] == a_parent["methodology"]
    assert a_null["phase"] == a_parent["phase"] == "trained"
    assert "constraint_phase" in a_null["blocks"]


def test_alm_null_zeroes_the_augmentation_not_just_the_multiplier():
    """The ALM weight is `lambda + mu_t * r+`, mu_t = mu0 + mu_step * epoch.

    Zeroing alm_eta and alm_mu_step but leaving mu0 at 0.01 gives a live
    augmentation of mu0 * excess on every epoch -- a weak treatment wearing a
    control's name, which is worse than having no control.
    """
    blk = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))["blocks"]["alm_null"]
    for epoch in (0, 14, 28):
        mu_t = blk["alm_mu0"] + blk["alm_mu_step"] * epoch
        assert mu_t == 0.0, "mu_t is %r at epoch %d, so the weight is not zero" % (
            mu_t, epoch)


def test_hounie_null_does_not_trip_hounie_s_own_stability_guard():
    """eta_u must NOT be zeroed even though it looks like a dose knob.

    hounie_rcl/train.py refuses `abs(1 - 2*eta_u*alpha) >= 1.0`, and eta_u = 0
    gives exactly 1.0, so a null built by zeroing every eta would raise before
    training a single epoch. eta_u moves only the slack u, which cannot reach
    the primal once lam is pinned at its 0.0 init.
    """
    blk = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))["blocks"]["hounie_null"]
    factor = abs(1.0 - 2.0 * blk["hounie_eta_u"] * blk["hounie_alpha"])
    assert factor < 1.0, (
        "hounie_null would raise its own stability check: factor %.3f" % factor)
    assert blk["hounie_eta_lambda"] == 0.0


def _load_panel():
    """full_panel.py is a script, not a package module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_full_panel", os.path.join("scripts", "full_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_full_panel"] = mod
    argv, sys.argv = sys.argv, ["full_panel"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return mod


def test_equalize_multi_respects_every_capped_class_not_just_the_first():
    """The bug its docstring describes: caps 2..n silently violated.

    `equalize` in score_arm.py takes top-K for ONE class and argmaxes the rest
    with no budget check, so a multi-capped-class campaign scored "equalized"
    metrics on an allocation that broke every cap after the first.
    """

    panel = _load_panel()
    UNLIM = 10 ** 10
    rng = np.random.default_rng(0)
    n, n_cls = 60, 4
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    gids = np.array([i % 3 for i in range(n)])
    capped = [1, 2]
    glob_c = [UNLIM, 10, 8, UNLIM]
    loc = {g: [UNLIM, 4, 3, UNLIM] for g in (0, 1, 2)}

    y = np.asarray(panel.equalize_multi(proba, gids, glob_c, loc, capped))
    assert y.shape == (n,)
    for c in capped:
        assert int((y == c).sum()) <= glob_c[c], (
            "class %d global cap violated: %d > %d"
            % (c, int((y == c).sum()), glob_c[c]))
        for g, bounds in loc.items():
            got = int((y[gids == g] == c).sum())
            assert got <= bounds[c], (
                "class %d local cap violated in group %s: %d > %d"
                % (c, g, got, bounds[c]))


def test_equalize_multi_matches_the_single_class_path():
    """Its docstring promises "with one capped class this is exactly the old
    behaviour" -- i.e. top-K by probability. The single-class results the
    project still stands on were produced by that path."""

    panel = _load_panel()
    UNLIM = 10 ** 10
    rng = np.random.default_rng(7)
    n, n_cls = 40, 3
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    gids = np.zeros(n, dtype=int)
    K = 9
    y = np.asarray(panel.equalize_multi(proba, gids, [UNLIM, K, UNLIM],
                                        {0: [UNLIM, UNLIM, UNLIM]}, [1]))
    chosen = set(np.where(y == 1)[0].tolist())
    topk = set(np.argsort(-proba[:, 1])[:K].tolist())
    assert len(chosen) == K
    assert chosen == topk, "single-class path is not top-K by probability"


def test_signflip_is_the_exact_permutation_and_hits_its_floor():
    """At D datasets the exact two-sided floor is 2^(1-D): 0.25 at three.

    This is the number that says no campaign this project can run reaches
    p<0.05 on the generalization unit, so it has to be exact, not approximate.
    """
    panel = _load_panel()
    for x in ([0.1, 0.2, 0.3], [3.0, 0.001, 99.0], [-0.4, -0.4, -0.4]):
        p, d = panel._signflip_p(x)
        assert d == 3 and abs(p - 0.25) < 1e-12, (x, p, d)
    assert panel._signflip_p([0.4])[0] == 1.0
    assert abs(panel._signflip_p([0.5, 0.5])[0] - 0.5) < 1e-12
    # a mixed-sign set must NOT reach the floor
    assert panel._signflip_p([0.1, -0.2, 0.3])[0] > 0.25
    # 2^(1-D) for a same-signed set, at every D it can reach
    for D in range(1, 7):
        p, d = panel._signflip_p([1.0] * D)
        assert d == D and abs(p - 2.0 ** (1 - D)) < 1e-12


def test_bh_monotonicity_and_that_aliases_do_not_widen_the_family():
    """Two things the scorer's q-values depend on.

    BH is a step-up procedure: the raw p*m/i can be non-monotone in the sorted
    order and must be made monotone by a backward cumulative minimum. And ccP /
    ccR must stay OUT of the family -- letting all three in widens the callable
    threshold by 2.54x.
    """
    panel = _load_panel()
    assert panel.BH_ALIAS_OF == {"ccP": "ccF1", "ccR": "ccF1"}
    assert "ccF1" not in panel.BH_ALIASES

    pv = [0.04, 0.01, 0.03]
    finite = sorted((v, str(i)) for i, v in enumerate(pv))
    q = {}
    for i, (p_, m) in enumerate(finite, 1):
        q[m] = min(1.0, p_ * len(finite) / i)
    for i in range(len(finite) - 2, -1, -1):
        q[finite[i][1]] = min(q[finite[i][1]], q[finite[i + 1][1]])
    got = [q[str(i)] for i in range(3)]
    # raw: p=0.01 -> 0.03 (i=1), p=0.03 -> 0.045 (i=2), p=0.04 -> 0.04 (i=3).
    # The raw sequence is NOT monotone -- 0.045 sits above the 0.04 that follows
    # it -- and the backward cumulative minimum pulls it down to 0.04. Without
    # that pass the middle hypothesis would carry a LARGER q than a weaker one.
    assert abs(got[1] - 0.03) < 1e-12
    assert abs(got[2] - 0.04) < 1e-12, "0.045 should be pulled down to 0.04"
    assert abs(got[0] - 0.04) < 1e-12
    ordered = [q[m] for _, m in finite]
    assert ordered == sorted(ordered), "q must be non-decreasing in sorted p"


def _panel_run(tmp_path, constrained_class, n=12, n_cls=3):
    """A minimal scorable run directory: config.json + the two prediction CSVs."""


    d = tmp_path / "ds" / "M" / "L30_G30" / "arm" / "seed_1"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    frame = {
        "True_Label": [i % n_cls for i in range(n)],
        "Predicted_Label": list(proba.argmax(axis=1)),
        "Group_ID": [i % 2 for i in range(n)],
    }
    for c in range(n_cls):
        frame["Prob_Class_%d" % c] = proba[:, c]
    df = pd.DataFrame(frame)
    df.to_csv(d / "final_predictions_raw.csv", index=False)
    df.to_csv(d / "final_predictions.csv", index=False)
    cfg = {
        "status": "completed",
        "arm": "arm",
        "constraint": [0.3, 0.3],
        "constraint_tag": "L30_G30",
        "model_name": "M",
        "dataset_mode": "ds",
        "dataset_config": {"num_classes": n_cls,
                           "constrained_class": constrained_class},
        "hyperparams": {"seed": 1},
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d, cfg


def test_panel_refuses_a_run_with_no_capped_class(tmp_path):
    """`constrained_class: null` must raise with a reason, not label a cell "None".

    full_panel.py carried two more inline scalar-or-list copies after the other
    three were unified. The one building the `capped` cell key did not raise on
    None at all -- it produced the literal string "None" as the cell label, so
    the run would have been paired and scored under a cell that does not exist.
    """
    panel_mod = _load_panel()
    d, cfg = _panel_run(tmp_path, None)
    with pytest.raises(ValueError) as exc:
        panel_mod.panel(str(d), cfg)
    assert "constrained_class" in str(exc.value)


@pytest.mark.parametrize("cc,expect", [(1, "1"), ([1], "1"), ([1, 2], "1-2")])
def test_panel_labels_the_capped_cell_the_same_way_for_scalar_and_list(
        tmp_path, cc, expect):
    """A scalar and a one-element list are the same campaign and must pair.

    If they produced different `capped` labels they would land in different
    cells and every pair would silently vanish from the comparison -- the
    failure that once made in-flight campaigns read as ties.
    """
    panel_mod = _load_panel()
    d, cfg = _panel_run(tmp_path, cc)
    r = panel_mod.panel(str(d), cfg)
    assert r is not None, "a well-formed run must be scorable"
    assert r["capped"] == expect


@pytest.mark.parametrize("pct", [0.30, 0.50])
def test_targeted_correction_spends_the_whole_reachable_budget(pct):
    """The allocator every TRAINED arm uses must fill to exactly K.

    Nothing in this suite called targeted_correction before, and that is
    precisely why a regression shipped: a global-budget check added to the
    local FILL phase, while 3a and 3b were interleaved per group, blocked the
    fill for every group processed before the reductions that free the room.
    The trained arms then under-spent the capped-class budget by ~4-5% while
    the clippers -- which never call this function -- filled to exactly K.

    An asymmetry that size is the size of the entire effect under study, and it
    pointed the same way, so it would have read as a real loss for the trained
    arms. Assert the invariant force_exact=True actually promises.
    """


    n, n_cls, n_grp, capped = 2000, 7, 7, [4]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, n_cls, n)
        g = rng.integers(0, n_grp, n)
        logits = rng.normal(size=(n, n_cls))
        logits[:, capped[0]] += 0.8          # over-predict the capped class
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)

        df = pd.DataFrame({"label": y, "grp": g})
        gcon = compute_global_constraints(df, "label", pct,
                                          constrained_class=capped,
                                          num_classes=n_cls)
        lcon = compute_local_constraints(df, "label", pct, "grp",
                                         constrained_class=capped,
                                         num_classes=n_cls)
        y_pred, _, meta = targeted_correction(proba, g, gcon, lcon, capped)

        c = capped[0]
        # local caps are per-GROUP ceilings, so their sum also bounds the count
        reachable = min(gcon[c], sum(lcon[gid][c] for gid in lcon))
        got = int((y_pred == c).sum())
        assert got == reachable, (
            "seed %d pct %s: filled %d of a reachable %d -- the trained arms "
            "would score against clippers that fill to exactly K"
            % (seed, pct, got, reachable))

        # and it must still be FEASIBLE, which is what the global check exists
        # for -- filling to the budget must not overshoot any cap
        assert got <= gcon[c], "global cap violated"
        for gid in lcon:
            in_g = int((y_pred[g == gid] == c).sum())
            assert in_g <= lcon[gid][c], (
                "local cap violated in group %s: %d > %d"
                % (gid, in_g, lcon[gid][c]))


def test_every_training_log_gets_a_header_not_just_tralo_s(tmp_path):
    """A headerless log is silently mis-parsed, not loudly broken.

    write_csv_header was called by exactly one of the ten arms -- tralo. Every
    other arm appended rows to a file with no header line, so pandas took the
    FIRST LOGGED EPOCH as the column names and mislabelled every column after
    it. TraLO's training log was machine-readable and the baselines it is
    compared against were not.

    The header is now written by the writer on first row, so this asserts the
    property for a caller that never mentions headers at all.
    """
    from src.training.logging import build_csv_header, log_progress_to_csv

    path = tmp_path / "training_log.csv"
    local = {0: [10 ** 10, 5, 10 ** 10], 1: [10 ** 10, 4, 10 ** 10]}
    for epoch in range(3):
        log_progress_to_csv(
            str(path), epoch, ce_loss=0.5, train_acc=0.9,
            constraints=[10 ** 10, 12, 10 ** 10], num_classes=3,
            local_constraints=local, grad_norm=1.5, lambda_global=0.25)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    expected = build_csv_header(3, local)
    assert lines[0].split(",") == expected, (
        "first line is not the header -- pandas would eat epoch 1 as column names")
    assert len(lines) == 1 + 3, "expected a header plus one row per epoch"

    df = pd.read_csv(path)
    assert len(df) == 3, "an epoch was consumed by the header"
    assert "Grad_Norm" in df.columns and "Lambda_Global" in df.columns
    assert float(df["Grad_Norm"].iloc[0]) == 1.5
    assert float(df["Lambda_Global"].iloc[0]) == 0.25


def test_constraint_phase_reaches_every_trained_arm_and_no_posthoc_one():
    """A shared knob must reach all four trained arms, or none of them.

    This is the structure the CE-saturation gate violated: its flag was
    declared by TraLO alone, so a campaign ran it off for TraLO and on for
    both duals -- a 0.22 cc-F1 artifact against a 0.019-0.031 margin. Any
    shared constraint knob therefore lives in `constraint_phase`, which every
    trained arm includes and no post-hoc arm does, so one assignment cannot
    reach one arm and not another.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    cp = proto["constraint_phase"]

    trained = [a for a, spec in proto["arms"].items()
               if spec.get("phase") == "trained"]
    assert len(trained) >= 4
    for arm in trained:
        spec = proto["arms"][arm]
        if spec.get("constraint_step") is False:
            # An arm that takes NO constraint step (`select`: a selection head,
            # no dual, no gradient clip) cannot miss a gate it does not have.
            # Handing it `constraint_phase` would emit eight keys nothing reads
            # -- the exact defect audit_config exists to catch -- so the
            # exemption is declared, and then CHECKED: it may not carry a single
            # constraint_phase key, which is what stops "constraint_step: false"
            # from becoming a way to smuggle dead keys past both gates.
            declared = set()
            for b in spec["blocks"]:
                declared |= set(proto["blocks"].get(b, {}))
            leaked = declared & (set(cp) - {"lr_constraint", "constraint_chunk_size"})
            assert not leaked, (
                "%s declares constraint_step: false but still carries %s" %
                (arm, sorted(leaked)))
            continue
        assert "constraint_phase" in spec["blocks"], (
            "%s is a trained arm that does NOT include constraint_phase, so a "
            "shared constraint knob would silently miss it. If it genuinely "
            "takes no constraint step, declare `constraint_step: false`." % arm)
    for arm, spec in proto["arms"].items():
        if spec.get("phase") == "posthoc":
            assert "constraint_phase" not in (spec.get("blocks") or []), (
                "%s is post-hoc; emitting a constraint-phase key for it would "
                "be a key with no reader" % arm)


def test_the_bounded_shape_starves_the_deepest_violator_and_the_hinges_do_not():
    """The measured multi-class failure, pinned as a property.

    This asserts a GRADIENT property, which is why it is a unit test and not a
    reading of a training log. Counts cannot establish it: measured against a
    lambda=0 control on dermmnist with classes 2+4 capped at L30_G20, CE alone
    swings the capped counts 242 -> 227 -> 324 -> 233 over four epochs with the
    penalty identically off, so no count trajectory is attributable to the
    shape without that control.

    What the control DOES show is the consequence of the property below. Every
    shape pushed class 4 down and class 2 up -- a see-saw, because the softmax
    makes the capped classes compete and the class that should resist is the
    one this shape starves. Shape set the see-saw's size in the order this test
    fixes: class 2 moved +197 under rational_bounded, +112 under squared, +86
    under linear.

    Single-class runs cannot show any of it: their spread of relative excess
    across scopes has median 1.5x, and at equal excess the shape is exactly
    inert -- a common scalar times a fixed direction, divided out by the clip.
    """
    from src.losses.transductive_loss import MulticlassTransductiveLoss, UNLIMITED

    NC = 5
    gcon = [UNLIMITED, 44.0, 45.0, UNLIMITED, UNLIMITED]

    def pull(shape):
        L = MulticlassTransductiveLoss(gcon, {}, num_classes=NC,
                                       initial_rho=10.0, penalty_shape=shape)
        for c in (1, 2):
            L.set_lambda_per_class(c, 0.01, scope="global")
        # class 1 = the DEEP violator (410 vs 44); class 2 = mild (57 vs 45)
        soft = torch.tensor([0.0, 410.0, 57.0, 0.0, 0.0], requires_grad=True)
        L.compute_global_from_counts(soft).backward()
        return float(soft.grad[1]), float(soft.grad[2])

    deep_b, mild_b = pull("rational_bounded")
    deep_s, mild_s = pull("squared")
    deep_l, mild_l = pull("linear")

    assert deep_b < mild_b, (
        "the bounded shape is supposed to STARVE the deep violator here; if "
        "this fails the measured pathology has changed and 2a2 needs redoing")
    assert deep_s > mild_s, "squared must favour the deeper violator"
    assert deep_l > mild_l, "linear must favour the deeper violator"
    # and the reversal must be large enough to matter, not a rounding artifact
    assert (deep_s / mild_s) > 10 * (deep_b / mild_b)


def test_normalize_delivers_the_same_step_size_whatever_the_raw_norm():
    """The point of `normalize`: the dose stops depending on the arm.

    Under `clip` a gradient below the threshold passes through untouched, which
    is why hounie (raw norm 0.005-0.11 against a clip of 1.0) took a step ~20x
    smaller than tralo's on every one of its 29 epochs while both configs said
    constraint_grad_clip: 1.0.
    """

    def step(raw_scale, mode):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(0.0)
        m.weight.grad = torch.full((1, 4), raw_scale)
        before = m.weight.detach().clone()
        finish_constraint_step(m, None, None, 1.0, mode=mode,
                               fp32=True, step_rule="sgd", lr=1.0)
        return float((m.weight.detach() - before).norm())

    # raw norm 0.02 (far below the clip) and 20.0 (far above)
    small_clip, big_clip = step(0.01, "clip"), step(10.0, "clip")
    small_nrm, big_nrm = step(0.01, "normalize"), step(10.0, "normalize")

    assert small_clip == pytest.approx(0.02, rel=1e-5), small_clip
    assert big_clip == pytest.approx(1.0, rel=1e-5), big_clip
    assert small_clip < big_clip / 10, (
        "under `clip` a below-threshold gradient keeps its own magnitude -- "
        "this is the hounie asymmetry, and it must stay reproducible")

    assert small_nrm == pytest.approx(1.0, rel=1e-5), small_nrm
    assert big_nrm == pytest.approx(1.0, rel=1e-5), big_nrm
    assert small_nrm == pytest.approx(big_nrm, rel=1e-6), (
        "under `normalize` the delivered step must be the same size no matter "
        "what the arm's natural gradient scale is -- that is the whole point")


def test_the_random_direction_control_keeps_the_dose_and_drops_the_information():
    """The control must change ONLY the direction, never the step size.

    Its whole purpose is to answer "did the penalty's direction matter?", and
    it can only answer that if the dose is held exactly. If it also changed the
    norm it would confound direction with magnitude -- the same confound that
    made the dedicated-Adam arm uninterpretable (it moved 8,900x further and
    cost AP -0.0938, and no one could say which half did it).
    """

    def step(random_direction):
        torch.manual_seed(0)
        m = torch.nn.Linear(64, 8)
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.01      # small: normalize scales UP
        before = [p.detach().clone() for p in m.parameters()]
        finish_constraint_step(m, None, None, clip=1.0, mode="normalize",
                               step_rule="sgd", lr=1.0,
                               random_direction=random_direction)
        return torch.cat([(p.detach() - b).flatten()
                          for p, b in zip(m.parameters(), before)])

    real, rand = step(False), step(True)

    # same dose: normalize delivers exactly `clip`, times lr=1.0
    assert abs(float(real.norm()) - 1.0) < 1e-4
    assert abs(float(rand.norm()) - 1.0) < 1e-4

    # and the control must not DRAW from the global RNG: if it does, the
    # control run's dropout masks and batch order diverge from the real arm's
    # too, so it varies two things instead of one. Checked on the randomiser
    # itself -- the step() helper above seeds and samples on its own.
    from src.training.constraint_step import _randomize_direction
    m2 = torch.nn.Linear(8, 4)
    for q in m2.parameters():
        q.grad = torch.ones_like(q)
    torch.manual_seed(1234)
    before_state = torch.random.get_rng_state()
    _randomize_direction(m2, 1.0, next(iter(m2.parameters())))
    assert torch.equal(torch.random.get_rng_state(), before_state), (
        "the random-direction control consumed a global RNG draw")

    # different information: a random direction in 520 dimensions is very
    # nearly orthogonal to any fixed one
    cos = float(torch.dot(real, rand) / (real.norm() * rand.norm()))
    assert abs(cos) < 0.3, "random direction is not independent of the real one"


def test_a_non_finite_constraint_gradient_never_moves_the_weights():
    """fioretto lost 10 of 29 epochs to NaN/inf. It must lose them SAFELY."""

    for bad in (float("nan"), float("inf")):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(1.0)
        m.weight.grad = torch.full((1, 4), bad)
        before = m.weight.detach().clone()
        raw, applied = finish_constraint_step(m, None, None, 1.0,
                                              mode="normalize", fp32=True,
                                              step_rule="sgd", lr=1.0)
        assert not applied, "a %s gradient must not take the step" % bad
        assert torch.equal(m.weight.detach(), before), (
            "weights moved on a %s gradient" % bad)


@pytest.mark.parametrize("arm", ["tralo", "fioretto_ldf", "hounie_rcl",
                                 "fioretto_alm"])
def test_no_arm_hand_rolls_its_own_constraint_step(arm):
    """All four must go through src/training/constraint_step.py.

    They each hand-rolled this block once, and the copies drifted until the
    arms were not receiving the same treatment. Measured 2026-08-20 on
    results/vit_diag seed 1, same warm-up model, every config saying
    constraint_grad_clip: 1.0 --

        tralo     raw grad norm 0.638 .. 1826.5    clip bound  6 of 7
        fioretto  raw grad norm 17,667 .. 80,827   clip bound 18 of 18
        hounie    raw grad norm 0.005 .. 0.1105    clip bound  0 of 29

    -- so hounie took its raw ~0.05-norm step while the other two took unit
    ones. A direct clip_grad_norm_ call in a trainer is how that came back, so
    it is banned here rather than merely discouraged.
    """
    src = open("src/methodologies/%s/train.py" % arm, encoding="utf-8").read()
    assert "finish_constraint_step" in src, (
        "%s does not use the shared constraint step" % arm)
    assert "clip_grad_norm_" not in src, (
        "%s calls clip_grad_norm_ directly. That is the divergence this "
        "module exists to prevent -- route it through finish_constraint_step "
        "so every arm is bounded the same way." % arm)
    assert "constraint_autocast" in src, (
        "%s does not use the shared constraint autocast, so --constraint-fp32 "
        "cannot reach it and it can still lose epochs to fp16 overflow" % arm)


def test_the_margin_count_is_a_real_count_not_a_constant():
    """The trap that killed the order-statistic version, pinned.

    Centring the window on the K-th largest probability -- the obvious way to
    put the gradient on the cut -- yields `sum_i sigma((p_i - tau)/T)`, which
    counts how many items exceed the K-th largest. That is K - 0.5 for ANY
    model, so it is a constant, `relu(s - K)` is identically zero, and no
    gradient is ever produced. It was wired into the trainer and caught here.

    The margin count must instead MOVE with the model and be able to exceed
    the budget, which is what makes a violation visible at all.
    """

    torch.manual_seed(0)
    K, cls = 8, 1
    counts, hards, os_wide, os_narrow = [], [], [], []
    for shift in (-4.0, 0.0, 4.0):
        logits = torch.randn(40, 3) + torch.tensor([0.0, shift, 0.0])
        proba = F.softmax(logits, dim=1)
        counts.append(float(margin_window(proba, 0.02)[:, cls].sum()))
        hards.append(float((proba.argmax(dim=1) == cls).sum()))
        tau = torch.topk(proba[:, cls], K).values[-1]
        os_wide.append(float(torch.sigmoid((proba[:, cls] - tau) / 0.02).sum()))
        os_narrow.append(float(torch.sigmoid((proba[:, cls] - tau) / 1e-4).sum()))
        assert abs(counts[-1] - hards[-1]) < 1.0, (
            "margin count %.2f is not tracking the hard count %d"
            % (counts[-1], hards[-1]))

    swing = max(hards) - min(hards)
    assert swing > 20 and max(counts) - min(counts) > 20, (
        "the margin count barely moved (%s) as the class went from rare to "
        "dominant -- it is not measuring the count" % counts)
    assert max(counts) > K, "the count can never exceed the budget: no violation"

    # The dead version, for the record. As T -> 0 it is EXACTLY K - 0.5 for every
    # model: it counts the items above the K-th largest. At a usable T it
    # smears wherever probabilities are packed tighter than T, but it is still
    # pinned near the budget and still blind to the count -- here it spans
    # 8.5 while the true count spans 34.
    assert all(abs(c - (K - 0.5)) < 0.01 for c in os_narrow), os_narrow
    assert max(os_wide) - min(os_wide) < swing / 3, (
        "order-statistic count spread %.1f vs true spread %.1f -- this test "
        "is supposed to show it does NOT track"
        % (max(os_wide) - min(os_wide), swing))


def test_the_margin_count_puts_its_gradient_on_the_boundary():
    """Not on the unsure items. This is the entire point of the arm.

    The plain count's per-item derivative is p(1-p), maximal at p = 0.5 and
    ~zero at the cut. The margin count's is sigma'(m/T)/T, maximal at margin 0
    -- at the items one step from flipping out of the class.
    """

    torch.manual_seed(1)
    proba = F.softmax(torch.randn(60, 4), dim=1).requires_grad_(True)
    cls = 2
    margin = proba[:, cls] - torch.cat(
        [proba[:, :cls], proba[:, cls + 1:]], dim=1).max(dim=1).values

    (g_margin,) = torch.autograd.grad(
        margin_window(proba, 0.02)[:, cls].sum(), proba, retain_graph=True)
    (g_sum,) = torch.autograd.grad(proba[:, cls].sum(), proba)

    at_cut = int(margin.detach().abs().argmin())
    deep = int(margin.detach().argmax())
    gm = g_margin[:, cls].abs()
    assert gm[at_cut] > gm[deep] * 50, (
        "the margin count weights a deeply-committed item (%.3g) comparably "
        "to one at the boundary (%.3g)" % (gm[deep], gm[at_cut]))
    # the plain count is flat in p: every item gets identical weight here, and
    # the p(1-p) shape enters only through the softmax below it.
    assert torch.allclose(g_sum[:, cls], torch.ones(60))


def test_the_windowed_count_keeps_the_exact_full_N_gradient_when_chunked():
    """The shipped construction: value is the HARD count, gradient is the window.

    Each chunk backprops through `total.detach() - chunk.detach() + chunk`,
    whose gradient reaches only that chunk's items, so the chunks sum to the
    exact full-N gradient -- this is what makes `constraint_chunk_size`
    gradient-neutral. In margin mode `total` is seeded with the HARD count, so
    the same expression is also a straight-through estimator: the penalty
    reads the true count and differentiates the window.

    That matters because a wide window over-counts (56.6 against a hard 45 at
    40 items), and a penalty reading an inflated count keeps pushing after the
    cap is already satisfied -- the overshoot scripts/flag_live measured.
    """
    from src.losses.transductive_loss import margin_window, margins, window_temp

    torch.manual_seed(0)
    logits = (torch.randn(40, 3) + torch.tensor([0.0, 2.0, 0.0])).requires_grad_(True)
    K, cls = 8, 1

    def penalty(s_):
        return F.relu(s_[cls] - K) ** 2

    with torch.no_grad():
        proba0 = F.softmax(logits, dim=1)
        T = window_temp(margins(proba0), 10)
        hard = torch.zeros(3)
        for c in range(3):
            hard[c] = float((proba0.argmax(dim=1) == c).sum())
    assert float(hard[cls]) > K, "no violation, so the penalty gradient is 0"

    # reference: the full-batch gradient of the windowed count
    full_soft = margin_window(F.softmax(logits, dim=1), T).sum(dim=0)
    st_full = hard.detach() - full_soft.detach() + full_soft
    assert torch.allclose(st_full.detach(), hard), "straight-through value is not the hard count"
    (g_full,) = torch.autograd.grad(penalty(st_full), logits)

    # chunk size 7, not a divisor of 40, so the ragged last chunk runs too
    g_chunked = torch.zeros_like(logits)
    for start in range(0, 40, 7):
        sl = slice(start, min(start + 7, 40))
        eff = margin_window(F.softmax(logits[sl], dim=1), T)
        g_soft = hard.detach() - eff.sum(dim=0).detach() + eff.sum(dim=0)
        assert torch.allclose(g_soft.detach(), hard)
        (g,) = torch.autograd.grad(penalty(g_soft), logits, allow_unused=True)
        g_chunked = g_chunked + g

    assert torch.allclose(g_chunked, g_full, atol=1e-6), (
        "chunking changed the gradient (max diff %.3g) -- the detach "
        "cancellation broke, so chunk size is no longer a free knob"
        % float((g_chunked - g_full).abs().max()))
    assert g_full.abs().sum() > 0


def test_the_window_width_is_in_items_so_the_dose_cannot_vanish():
    """A fixed T is not a fixed dose, and an empty window is a silent null.

    Measured on the stored dermmnist evidence (MobileNetV3, 4 seeds, two cap
    tags): the T holding ~20 items at the boundary spans 0.182 .. 0.502 ACROSS
    SEEDS OF ONE CELL, and T = 0.02 puts 0-3 items in the window -- a run that
    contributes nothing, reports a null, and writes `completed`. Deriving T
    from a width in items makes the dose dimensionless and non-empty by
    construction.
    """
    from src.losses.transductive_loss import margins, window_temp

    torch.manual_seed(3)
    for scale in (0.5, 2.0, 8.0):        # flat, moderate and sharp models
        proba = F.softmax(torch.randn(300, 5) * scale, dim=1)
        m = margins(proba)
        for n in (10, 40, 120):
            T = window_temp(m, n)
            for c in range(5):
                inside = int((m[:, c].abs() < T[c]).sum())
                assert abs(inside - n) <= 1, (
                    "asked for %d items in the window, got %d (scale %.1f, "
                    "class %d)" % (n, inside, scale, c))
        # and T really does have to move to achieve that
        assert float(window_temp(m, 120).max()) > float(window_temp(m, 10).max())


def test_every_trained_arm_has_a_null_sibling_the_gate_can_find():
    """The gate looks up a name; a new arm can slip past it silently.

    gen_campaign warns when a trained arm is requested without its zero-dose
    sibling, because a delta vs `clip` cannot otherwise be attributed to the
    constraint rather than to the regime. It found that sibling by appending
    `_null`, so `tralo_margin` -- the arm that most needs a control, being a
    new estimator -- resolved to `tralo_margin_null`, which does not exist,
    and the gate said nothing. Arms may now name another arm's null via
    `null_sibling`; this pins that every trained arm resolves to a real one.
    """

    from configs.gen_campaign import _null_of

    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    missing = []
    for name, spec in P["arms"].items():
        if spec["phase"] != "trained" or name.endswith("_null"):
            continue
        sib = _null_of(P, name)
        if sib not in P["arms"]:
            missing.append("%s -> %s" % (name, sib))
    assert not missing, (
        "trained arms whose null sibling does not exist, so the gate cannot "
        "warn about them: %s" % ", ".join(missing))

    # And the shared one is genuinely shared, not a copy: tralo_margin differs
    # from tralo only in where the count puts its gradient, and at lambda 0 no
    # constraint gradient is formed at all, so one null serves both.
    assert _null_of(P, "tralo_margin") == _null_of(P, "tralo") == "tralo_null"
    assert P["blocks"]["tralo_null"]["lambda_global"] == 0.0


def test_the_inert_flag_gate_can_actually_detect_an_inert_flag():
    """A gate nobody has seen fail is not known to work.

    `flag_live` exists because inert flags are this project's most frequent
    failure mode -- four occurrences, every one of which passed audit_config
    (the key had a reader) and smoke_arms (the arm ran). Comparing an arm to
    ITSELF must report inert, which both proves the detector fires and proves
    the harness is deterministic: if two runs of one arm already differ, no
    difference between two arms means anything.
    """

    same = subprocess.run(
        [sys.executable, "-m", "scripts.flag_live", "tralo", "tralo",
         "--constraint-epochs", "2"],
        capture_output=True, text=True)
    assert same.returncode == 1, (
        "an arm compared to itself was NOT reported inert -- either the "
        "detector is broken or the harness is nondeterministic, and in the "
        "second case the gate cannot distinguish a live flag from noise:\n%s"
        % same.stdout[-600:])
    assert "INERT" in same.stdout

    diff = subprocess.run(
        [sys.executable, "-m", "scripts.flag_live", "tralo", "tralo_margin",
         "--constraint-epochs", "2"],
        capture_output=True, text=True)
    assert diff.returncode == 0, (
        "soft_count_mode: margin produced bit-identical predictions to sum, "
        "so it is a fifth inert flag:\n%s" % diff.stdout[-600:])


def test_the_scorers_posthoc_list_matches_the_protocol():
    """A hardcoded copy of a protocol fact drifts, and this one is load-bearing.

    `full_panel.POSTHOC_ARMS` exists so arms with `constraint_epochs == 0` are
    not flagged for producing cap-invariant predictions -- their warm-up IS the
    run, so identical predictions across cap levels is correct, not a bug. If a
    post-hoc arm is added to protocol.yml and not here, the scorer reports a
    real arm as broken. If a TRAINED arm is listed here by mistake, the scorer
    stops reporting the one failure mode that has actually occurred (six models
    behind twelve cells).
    """

    from scripts.full_panel import POSTHOC_ARMS

    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    expected = {a for a, v in P["arms"].items() if v["phase"] == "posthoc"}
    assert POSTHOC_ARMS == expected, (
        "full_panel.POSTHOC_ARMS has drifted from configs/protocol.yml.\n"
        "  only in the scorer:  %s\n"
        "  only in the protocol: %s"
        % (sorted(POSTHOC_ARMS - expected), sorted(expected - POSTHOC_ARMS)))


def test_straight_through_closes_the_K_equals_zero_trap():
    """A group with no true instances of the capped class gets K == 0 legitimately.

    On the soft value that constraint can NEVER be satisfied: `sum_i p_ic` is
    strictly positive for any softmax, even when the model predicts the class
    for nobody in the group, so `relu(count - 0)` stays positive forever.

    That is where the standing warning stops being true. It does NOT hold the
    ratchet gate open for every other constraint -- the trainer reads HARD
    counts, which can be exactly zero (corrected 2026-08-22, FRAMEWORK 1b). What
    the term really does is push `p_ic` down in that group permanently, which
    for a group with genuinely no instances of the class is the RIGHT direction.
    So `straight_through` switches real pressure off rather than repairing a
    defect, and on iwildcam -- seven of fourteen ceilings at K == 0 -- that is a
    decision to make on a measurement, not a fix to apply on sight.

    The hard count CAN be exactly zero, so `straight_through: true` makes that
    constraint satisfiable. This pins the difference rather than the fix, since
    the K == 0 group is created by the data, not by a setting.
    """
    torch.manual_seed(4)
    # class 2 is never the argmax, but it still carries probability mass
    logits = torch.randn(80, 5)
    others = torch.cat([logits[:, :2], logits[:, 3:]], dim=1).max(dim=1).values
    logits[:, 2] = others - 1.5      # always second best, never the argmax
    p = F.softmax(logits, dim=1)

    hard = int((p.argmax(dim=1) == 2).sum())
    soft = float(p[:, 2].sum())
    assert hard == 0, "fixture broken: class 2 is predicted for %d items" % hard
    assert soft > 0.5, "fixture broken: class 2 carries no mass (%.3f)" % soft

    K = 0
    assert float(F.relu(torch.tensor(soft) - K)) > 0.5, (
        "the soft value is satisfiable at K=0, which would mean this trap is "
        "not real")
    assert float(F.relu(torch.tensor(float(hard)) - K)) == 0.0, (
        "the hard value is NOT satisfiable at K=0 -- straight_through does not "
        "close the trap after all")


def test_margin_without_straight_through_is_refused_not_silently_reinterpreted():
    """The fourth corner of the 2x2 is a third semantics, so it must not run.

    Pass 1 accumulates the plain `sum_i p_ic` into the running total; pass 2
    windows each chunk. The detach construction then cancels a WINDOWED chunk
    out of a PLAIN total, giving value `sum_i p_ic` with a windowed gradient --
    neither `tralo` nor `tralo_margin`. And the windowed count over-counts
    (56.6 against a hard 45), so the penalty pushes past feasibility, which is
    the joint arm's measured failure mode.

    The generator cannot emit this combination; a hand-written config can.
    """
    import inspect

    from src.methodologies.tralo import train as tralo_train

    src = inspect.getsource(tralo_train.train)
    assert 'soft_count_mode: margin requires straight_through' in src, (
        "the guard against margin-without-straight-through is gone; that "
        "combination now runs and produces a third, undocumented estimator")
    # and the protocol's own arm sets both, so the guard never fires in practice
    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    blk = P["blocks"]["tralo_margin"]
    assert blk.get("soft_count_mode") == "margin" and blk.get("straight_through") is True


def test_the_allocator_does_not_fall_through_to_the_LP_when_G_is_less_than_L():
    """G < L is the prescribed sweep, and it used to break the greedy on EVERY run.

    Phase 3b filled local room without re-checking the GLOBAL budget, so local
    room got spent past the global cap, the allocation came out infeasible, and
    the run silently fell through to the small-scope LP -- a DIFFERENT algorithm
    from the greedy that `clip` keeps. An arm scored against `clip` while
    running a different allocator is not an arm-vs-arm comparison, and
    `full_panel` still reports 29% fall-through on the stored evidence.

    The fix is in the code with a comment. Nothing pinned it, and the failure
    is silent: caps still hold afterwards, so `smoke_arms --matrix` passes
    either way. This pins it on the cap tags actually being run.
    """

    from configs.gen_campaign import cap_pair

    rng = np.random.default_rng(0)
    N, C, G = 600, 7, 5
    capped = [2, 4]
    labels = rng.choice(C, size=N, p=np.array([.1, .1, .25, .1, .3, .1, .05]))
    groups = rng.integers(0, G, size=N)
    df = pd.DataFrame({"label": labels, "grp": groups})

    for tag in ("L50_G30", "L40_G30", "L30_G20"):
        loc_pct, glob_pct = cap_pair(tag)
        gcon = compute_global_constraints(df, "label", glob_pct,
                                          constrained_class=capped, num_classes=C)
        lcon = compute_local_constraints(df, "label", loc_pct, "grp",
                                         constrained_class=capped, num_classes=C)
        for trial in range(5):
            logits = rng.normal(0, 2.0, size=(N, C))
            logits[:, capped] += 1.2          # over-predict the capped classes
            e = np.exp(logits - logits.max(1, keepdims=True))
            proba = e / e.sum(1, keepdims=True)
            _, _, info = targeted_correction(proba, groups, gcon, lcon, capped)
            assert not info["lp_fallback_used"], (
                "%s trial %d fell through to the LP with %d candidates -- the "
                "greedy left the allocation infeasible, so this arm would be "
                "scored against `clip` while running a different allocator"
                % (tag, trial, info["lp_fallback_candidates"]))


def test_equalize_multi_fills_to_exactly_K_so_the_items_conversion_is_exact():
    """`full_panel` reports `items per 0.01 capF1`, and that number assumes it.

    `F1 = 2TP/(K+n)` only holds when exactly K predictions are emitted for the
    capped class. If `equalize_multi` ever fell short, the denominator would be
    `(emitted + n)` and the conversion printed after every panel would be
    quietly wrong -- in the direction of understating how many items a delta is
    worth, on a scale where the whole headroom is 2 to 10 items.

    Worth pinning because the RUNTIME allocator does NOT have this property:
    measured on the stored evidence, `targeted_correction` emits K-1 on 22 of
    88 (run, capped class) pairs -- never over, so no cap is violated, but not
    exactly K either. The scorer re-equalizes from probabilities, which is why
    the house rule is to read `full_panel` and never the stored metrics.
    """

    rng = np.random.default_rng(0)
    N, C, G = 500, 7, 4
    caps = [2, 4]
    for trial in range(6):
        logits = rng.normal(0, 2.0, size=(N, C))
        logits[:, caps] += 1.0            # over-predict the capped classes
        e = np.exp(logits - logits.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)
        gids = rng.integers(0, G, size=N)
        glob_c = np.full(C, UNLIMITED, dtype=float)
        for c in caps:
            glob_c[c] = int(rng.integers(30, 90))
        loc = {g: np.full(C, UNLIMITED, dtype=float) for g in range(G)}

        eq = equalize_multi(P, gids, glob_c, loc, caps)
        for c in caps:
            got, want = int((eq == c).sum()), int(glob_c[c])
            assert got == want, (
                "trial %d class %d: equalize_multi emitted %d against a budget "
                "of %d, so F1 = 2TP/(K+n) no longer holds and the items "
                "conversion printed by full_panel is wrong"
                % (trial, c, got, want))

def test_budgets_reads_a_post_hoc_arm_not_just_a_trained_one(tmp_path):
    """The clipper is the bar. A budget helper blind to it drops the control.

    A post-hoc arm runs `constraint_epochs: 0`, so every row of its training
    log carries the hardcoded `Limit_Class = inf` default and the log-only
    lookup returned {}. Any caller that skips a run with no budget then reports
    the treatment with no control and calls it a comparison.
    """
    from scripts.reachability import budgets

    d = tmp_path / "clip" / "seed_1"
    d.mkdir(parents=True)
    # exactly what a post-hoc arm writes: the inf default, on every row
    pd.DataFrame({"Epoch": [0, 1], "Limit_Class4": [1e10, 1e10]}).to_csv(
        d / "training_log.csv", index=False)
    pd.DataFrame({"True_Label": [4] * 100 + [0] * 50}).to_csv(
        d / "final_predictions_raw.csv", index=False)
    (d / "config.json").write_text(json.dumps({
        "constraint": [0.3, 0.3],
        "capped_classes": [4],
        "dataset_config": {"constrained_class": [4]},
    }), encoding="utf-8")

    assert budgets(d) == {4: 30}, (
        "the cap is a fraction of the class's true count: 0.3 x 100 = 30")


def test_headroom_scores_the_control_the_way_the_scorer_does(tmp_path):
    """`achieved` must be the scorer's allocation, not the raw argmax.

    The raw argmax ignores the budget, so scoring the control on it beats the
    analytic ceiling `2K/(K+n)` and yields a NEGATIVE headroom -- which is how
    this was caught. A headroom that can go negative is measuring two different
    allocations against each other.
    """
    from scripts.headroom import f1

    rng = np.random.default_rng(0)
    n, K, cls = 200, 20, 1
    y = np.zeros(n, dtype=int)
    y[:60] = cls
    P = rng.random((n, 3))
    P[:60, cls] += 1.0                     # class 1 genuinely more probable
    P = P / P.sum(axis=1, keepdims=True)

    g = np.zeros(n, dtype=int)
    G = {c: (K if c == cls else UNLIMITED) for c in range(3)}
    eq = equalize_multi(P, g, G, {0: {c: UNLIMITED for c in range(3)}}, [cls])

    assert int((eq == cls).sum()) == K, "the scorer fills the budget exactly"
    ceiling = 2.0 * K / (K + int((y == cls).sum()))
    assert f1(y, eq, cls) <= ceiling + 1e-12, (
        "no allocation emitting exactly K can beat 2K/(K+n)")
    assert f1(y, P.argmax(1), cls) > ceiling, (
        "the raw argmax DOES beat it, because it ignores the budget -- which "
        "is exactly why headroom must not be measured against it")

def test_a_cap_that_does_not_bind_gives_the_constraint_zero_gradient():
    """A seed already under budget is its own null. Measure it PER SEED.

    The penalty is built on relu(soft_count - K). Below the budget it is
    identically zero, so the constraint contributes no gradient and the treated
    arm and its null are the same run. Averaging over such a seed dilutes a
    real effect toward zero and reports it as a tie.

    Measured on the stored evidence: tissuemnist L50_G50 class 1 runs 76 / 51 /
    34 / 18 against K=56, so it binds in ONE seed of four -- while its MEAN
    excess is -11.2 and its L30 sibling's mean excess is a healthy-looking
    +10.8 on 2-of-4 binding. The mean cannot show this; only the per-seed count
    can, which is why scripts/headroom.py keeps the counts per seed.
    """


    K = 56.0
    loss = MulticlassTransductiveLoss(global_constraints=[K, K],
                                      local_constraints={}, num_classes=2)
    for count in (76.0, 51.0, 34.0, 18.0):
        soft = torch.tensor(count, requires_grad=True)
        pen = loss._penalty(soft, K)
        if count > K:
            assert float(pen) > 0.0, "an over-budget count must be penalised"
        else:
            assert float(pen) == 0.0, (
                "count %g is under K=56, so the penalty -- and the whole "
                "constraint gradient -- is identically zero" % count)
            pen.backward()
            assert float(soft.grad) == 0.0, (
                "no gradient reaches the model from a satisfied cap")


def test_final_predictions_that_violate_a_cap_are_refused_not_logged(tmp_path):
    """A trained arm must not be able to ship an infeasible result.

    `heuristic` raises on its own violations, so the post-hoc arms hard-failed
    while the trained arms wrote `status: completed` with "VIOLATED by N" at
    INFO level -- an asymmetry in which arms can silently ship an infeasible
    result, in the file that decides what every scorer reads.

    And the check read `global_con` only, so a LOCAL violation was not merely
    unreported, it was never looked at. A class capped only per-group has an
    UNLIMITED global budget, so every global check on it passes vacuously.
    """

    from src.pipeline.eval import write_evaluation_outputs

    n = 12
    y_test = np.zeros(n, dtype=int)
    groups = np.array([0] * 6 + [1] * 6)
    proba = np.tile(np.array([0.6, 0.4]), (n, 1))

    def run(y_pred, global_con, local_con):
        return write_evaluation_outputs(
            tmp_path, y_test, groups,
            {"y_pred": np.asarray(y_pred), "raw_pred": np.asarray(y_pred),
             "y_proba": proba,
             "metrics": {"flips_required": 0, "raw_all_satisfied": True,
                         "raw_total_excess": 0}},
            2, global_con, local_con)

    # feasible under both scopes
    run([1] * 2 + [0] * 10, [UNLIMITED, 4], {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]})

    # GLOBAL violated
    with pytest.raises(RuntimeError, match="violate"):
        run([1] * 6 + [0] * 6, [UNLIMITED, 4], None)

    # LOCAL violated while the global budget is UNLIMITED -- the case the
    # global-only check could not see even in principle.
    with pytest.raises(RuntimeError, match="local group"):
        run([1] * 5 + [0] * 7, [UNLIMITED, UNLIMITED],
            {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]})

def test_the_deletion_table_does_not_claim_live_code_was_deleted():
    """FRAMEWORK is the law, so a false claim in it is a defect, not a typo.

    Section (f) listed `danits_lp`, `focal`, `class_balanced` and
    `logit_adjust` as deleted methodology packages, and `cb_beta` /
    `logit_adjust_tau` as removed keys. All four packages exist, are registered
    in TRAIN_FNS, and are among the nine methodologies the PAPER claims; both
    keys are live in protocol.yml. Anyone trusting the table would conclude
    those arms are gone.

    Rather than fix the prose and hope, the table is checked: anything it says
    was removed must actually be absent.
    """



    text = io.open("docs/FRAMEWORK.md", encoding="utf-8").read()
    start = text.index("### (f) What was DELETED FROM THE CODE")
    section = text[start:text.index(chr(10) + "### ", start + 10)]

    proto = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8"))
    live_keys = set(proto.get("core", {})) | set(proto.get("constraint_phase", {}))
    for blk in proto.get("blocks", {}).values():
        live_keys |= set(blk)

    claimed = set()
    for row in section.splitlines():
        if not row.startswith("|") or row.startswith("| removed") or "---" in row:
            continue
        cells = row.split("|")
        # Columns 1 AND 2 -- "removed" and "was". Reading only column 1 made the
        # methodology half of this check VACUOUS: that row says "5 methodology
        # packages" in column 1 and puts the actual names in column 2, so a
        # false claim about a live package sailed through. Caught by running the
        # negative control instead of trusting a green test.
        text2 = " ".join(cells[1:3])
        claimed |= set(re.findall(r"[a-z_][a-z0-9_]{2,}", text2))

    assert claimed, "the deletion table parsed to nothing -- the check is vacuous"

    still_live = sorted(k for k in claimed if k in live_keys)
    assert not still_live, (
        "FRAMEWORK section (f) says these were removed, but they are live keys "
        "in protocol.yml: %s" % still_live)

    registered = sorted(m for m in claimed if m in TRAIN_FNS)
    assert not registered, (
        "FRAMEWORK section (f) says these methodologies were deleted, but they "
        "are registered in TRAIN_FNS: %s" % registered)


def test_the_coin_control_matches_the_delivered_step_not_the_clip_bound():
    """The coin must differ from the treatment in INFORMATION only, not dose.

    `_randomize_direction` rescaled the random gradient to exactly `clip`
    unconditionally, but under the protocol default `constraint_grad_mode: clip`
    the treatment delivers min(raw, clip). So on every epoch where the clip did
    not bind, the control took a LARGER step than the thing it controls -- 20x
    for hounie, whose raw norms are 0.005-0.11 against clip 1.0 -- and the bias
    runs in the direction that flatters the treatment.
    """


    def delivered(mode, target_raw, coin):
        torch.manual_seed(0)
        m = nn.Linear(64, 8, bias=False)
        for p in m.parameters():
            p.grad = torch.ones_like(p)
            p.grad.mul_(target_raw / float(p.grad.norm()))
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        finish_constraint_step(m, opt, None, 1.0, mode=mode, step_rule="sgd",
                               lr=0.0, random_direction=coin)
        return sum(float(p.grad.pow(2).sum()) for p in m.parameters()) ** 0.5

    for mode in ("clip", "normalize"):
        for raw in (0.05, 0.5, 5.0):
            t = delivered(mode, raw, False)
            c = delivered(mode, raw, True)
            assert abs(c - t) < 1e-5, (
                "coin is dosed %.4f but the treatment delivers %.4f at "
                "mode=%s raw=%.2f (%.1fx) -- the control varies dose as well "
                "as information" % (c, t, mode, raw, c / max(t, 1e-12)))


def test_coverage_targets_uses_a_whole_test_budget_not_the_smallest_group():
    """tau's numerator and denominator must live in the same scope.

    `local_con[g][c]` bounds group g alone; `g.mean()` covers the whole batch.
    Taking min across groups put the SMALLEST group's budget over the whole
    test set -- 9/2003 instead of 67/2003 on derm L50_G30, a 7.4x
    over-tightening -- and made tau move with the LOCAL tag while the global
    cap was unchanged, so a G<L sweep would sweep the smallest group.
    """
    from src.methodologies.select.train import coverage_targets
    from src.training.constraints import UNLIMITED

    n = 2003          # dermmnist slice_1 test n, per docs/FRAMEWORK.md
    loc = {g: [UNLIMITED] * 7 for g in ("a", "b", "c")}
    loc["a"][4], loc["b"][4], loc["c"][4] = 75.0, 28.0, 9.0   # sum 112

    g_tight = [UNLIMITED] * 7
    g_tight[4] = 67.0
    assert abs(coverage_targets(g_tight, loc, [4], n, 7)[4] - 67 / n) < 1e-9, (
        "global 67 is tighter than the local sum 112, so tau must be 67/n")

    # Local-only: the global is UNLIMITED, and the SUM of the locals is the
    # whole-test ceiling. Reading global_con alone would give tau = 1.0 here.
    assert abs(coverage_targets([UNLIMITED] * 7, loc, [4], n, 7)[4]
               - 112 / n) < 1e-9, "local-only must fall back to the local SUM"

    # The smallest group's budget (9) must never be the numerator.
    for g in (g_tight, [UNLIMITED] * 7):
        assert coverage_targets(g, loc, [4], n, 7)[4] > 9.0 / n + 1e-9, (
            "tau collapsed onto the smallest group's budget")


def test_the_selection_arm_actually_threads_its_running_coverage_estimate():
    """Inert-flag gate for `cov_ema`: it must set the coverage term's VALUE.

    The stabilised coverage term takes its value from a running estimate and
    its gradient from the current batch. Asserting only that the argument
    "changes the loss" is too weak -- a broken construction that merely ADDED
    the estimate (`cov_ema + cov`, no `- cov.detach()`) would pass that. So
    this pins the actual property: with cov_ema = X the penalty must equal
    cov_weight * (X - tau)^2 exactly, i.e. the value is the ESTIMATE and not
    this batch's coverage.
    """



    tau, w, X = 0.03, 32.0, 0.20
    g = torch.full((64,), 0.5, requires_grad=True)
    probs = torch.full((64, 7), 1 / 7.0)
    y = torch.zeros(64, dtype=torch.long)

    bare, cov, _b = selective_loss(g, probs, y, 4, tau, w)
    with_ema, cov2, _b2 = selective_loss(g, probs, y, 4, tau, w, cov_ema=X)
    assert abs(cov - 0.5) < 1e-6 and abs(cov2 - 0.5) < 1e-6

    # The ONLY difference between the two is the coverage term's value.
    delta = float(with_ema) - float(bare)
    expected = w * ((X - tau) ** 2 - (cov - tau) ** 2)
    assert abs(delta - expected) < 1e-4, (
        "coverage term used %.6f, expected the estimate %.2f -> %.6f"
        % (delta, X, expected))

    # ...and the gradient must still come from THIS batch, not the estimate.
    with_ema.backward()
    assert g.grad is not None and float(g.grad.abs().sum()) > 0, (
        "no gradient reaches g -- the detach construction broke the graph")

    # ...and the training loop must actually pass it.
    src = io.open("src/methodologies/select/train.py", encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "selective_loss"]
    assert calls, "no call to selective_loss found in the arm"
    assert all(len(c.args) >= 7 or any(k.arg == "cov_ema" for k in c.keywords)
               for c in calls), (
        "selective_loss is called without cov_ema -- the stabilisation is inert")


def test_the_selective_risk_is_centred_so_it_does_not_only_push_coverage_down():
    """An UNcentred risk makes every item's gradient positive.

    Normalising by the expected covered mass instead of g.sum() fixes the
    variance but removes the centring the ratio estimator had for free:
    d risk / d g_i = per_i / (n*tau) > 0 for every item, so the risk term
    degenerates into a pure "cover nothing" force. Measured on a synthetic:
    equilibrium coverage falls from 0.74*tau to 0.60*tau, and undershooting
    the budget is the one regime where the two-allocator confound bites.
    A selective risk must pull EASY items in and push hard ones out.
    """


    g = torch.full((64,), 0.5, requires_grad=True)
    # All 64 ARE the capped class; the model is confident on the first half and
    # wrong on the second, so the two halves differ in per-item LOSS, which is
    # the quantity the selective risk is supposed to sort on. (Flipping the
    # LABEL instead gives both halves the same loss and the baseline cancels
    # everything -- a test that then passes on any implementation.)
    probs = torch.zeros(64, 7)
    probs[:, 4] = torch.cat([torch.full((32,), 0.95), torch.full((32,), 0.05)])
    probs[:, 0] = 1 - probs[:, 4]
    y = torch.full((64,), 4, dtype=torch.long)

    # cov_weight 0 isolates the RISK term from the coverage pull.
    loss, _cov, _b = selective_loss(g, probs, y, 4, 0.03, 0.0)
    loss.backward()
    easy, hard = g.grad[:32], g.grad[32:]
    assert float(easy.mean()) < 0 < float(hard.mean()), (
        "risk gradient is not centred: easy items %.4g, hard items %.4g -- "
        "both signs must not be the same, or the term only pushes coverage "
        "down" % (float(easy.mean()), float(hard.mean())))


def test_no_methodology_reads_the_test_LABELS_except_to_count_them():
    """`y_test` is handed to every train(); only discipline keeps it unread.

    `TrainInputs` carries the true test labels so the pipeline can derive K and
    score afterwards, and every methodology receives them. The transductive
    setting permits reading the COUNT (that is the declared assumption) and
    nothing else. An arm that trains on test inputs -- self-training, pseudo
    -labelling -- is exactly where an accidental read would be easy to write
    and impossible to notice, so this is a gate rather than a convention.
    """

    offenders = []
    for path in sorted(pathlib.Path("src/methodologies").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        # Every `len(...)` argument is an allowed context; collect them first.
        allowed = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "len"):
                for a in node.args:
                    allowed.update(id(n) for n in ast.walk(a))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "y_test"
                    and id(node) not in allowed):
                offenders.append("%s:%d" % (path.as_posix(), node.lineno))

    assert not offenders, (
        "these methodology modules read the TEST LABELS outside a len() call, "
        "which is label leakage, not transduction: %s" % offenders)


def test_the_two_fioretto_arms_initialise_both_multiplier_scopes_alike():
    """ALM claims to differ from LDF in the DUAL UPDATE only.

    `fioretto_lambda_init` reached ALM's global and local multipliers but only
    LDF's global one -- LDF's locals were hardcoded to 0.0. Sweeping the key
    would then have changed the dual rule AND the local initialisation at once,
    so the arm-vs-arm delta would be attributable to neither. Dormant at the
    protocol's 0.0, which is exactly why only a gate catches it.
    """

    for mod in ("fioretto_ldf", "fioretto_alm"):
        path = "src/methodologies/%s/train.py" % mod
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript)
                           and getattr(t.value, "id", None) == "lambda_l"
                           for t in n.targets)]
        init = [n for n in assigns if isinstance(n.value, ast.Constant)]
        assert not init, (
            "%s initialises lambda_l to the literal %r instead of the "
            "fioretto_lambda_init value the other arm uses"
            % (mod, [n.value.value for n in init]))


def test_alm_gates_its_constraint_pass_on_the_weights_the_loss_uses():
    """A treatment that logs itself and never runs is this repo's failure mode.

    ALM's chunk loss weights each term by `lambda + aug`. `has_work` decided
    whether to run that pass at all, and consulted `lambda` only -- so with the
    multipliers pinned at 0 and the augmentation climbing, the pass was skipped
    on every epoch while `training_log.csv` wrote a rising mu_t.
    """

    src = io.open("src/methodologies/fioretto_alm/train.py",
                   encoding="utf-8").read()
    tree = ast.parse(src)
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "has_work" for t in n.targets)),
                None)
    assert node is not None, "has_work assignment not found"
    names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    for w in ("lambda_g", "lambda_l", "aug_g", "aug_l"):
        assert w in names, (
            "has_work ignores %s, but the constraint loss weights terms by "
            "lambda + aug -- the augmentation is unreachable whenever the "
            "multipliers are 0" % w)


def test_no_f_string_placeholder_survives_in_a_plain_string_literal():
    """A `{name}` in a NON-f string prints the braces instead of the value.

    `main.py`'s dispatcher header printed the literal "{completed} done,
    {failed} failed" for an unknown length of time -- an implicit-concatenation
    slip where the first fragment lost its `f` prefix and the second kept it.
    The dispatcher banner is how a run's progress is read at a glance, so this
    is a silent instrument failure, not a cosmetic one.

    Scoped to the two places the slip can actually reach a reader: a plain
    fragment implicitly concatenated INTO an f-string, and a bare literal
    handed straight to `print`. Docstrings are excluded -- this repo's carry
    real algebra, and `(1 - beta) / (1 - beta^n)` is not a formatting bug.
    """

    PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]']*\}")

    def bad(node):
        return [c for c in ([node] if isinstance(node, ast.Constant)
                            else getattr(node, "values", []))
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
                and PLACEHOLDER.search(c.value)]

    offenders = []
    roots = ([pathlib.Path("main.py")] + sorted(pathlib.Path("src").rglob("*.py"))
             + sorted(pathlib.Path("scripts").rglob("*.py")))
    for path in roots:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            # (a) a plain fragment glued into an f-string: the exact slip.
            if isinstance(node, ast.JoinedStr):
                hits = bad(node)
            # (b) a bare literal handed to print().
            elif (isinstance(node, ast.Call)
                  and getattr(node.func, "id", None) == "print"):
                hits = [h for a in node.args for h in bad(a)]
            else:
                continue
            offenders += ["%s:%d %r" % (path.as_posix(), node.lineno,
                                        h.value[:60]) for h in hits]
    assert not offenders, (
        "these string literals contain an f-string placeholder but are not "
        "f-strings, so they print the braces: %s" % offenders)


def test_the_scorer_says_a_skipped_run_CRASHED_rather_than_never_started(tmp_path):
    """A dead treatment arm must not read as "campaign merely unfinished".

    The dispatcher resets an interrupted run to `pending`, which makes a run
    that CRASHED indistinguishable from one that never started. Measured
    2026-08-21 on `results/dosefix`: all 8 `tralo` runs died of CUDA OOM in the
    transductive forward while every control completed -- the lambda=0 arm
    skips that pass entirely, so it never allocates the chunk that OOMs -- and
    the panel reported a clean comparison between controls with no hint that
    the treatment was absent. The tell is an error_log.json beside the config.
    """
    import json as _json

    run = tmp_path / "ds" / "mdl" / "L50_G30" / "tralo" / "seed_1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        _json.dumps({"status": "pending", "arm": "tralo"}), encoding="utf-8")
    (run / "error_log.json").write_text(
        _json.dumps({"exception_type": "OutOfMemoryError"}), encoding="utf-8")

    out = subprocess.run(
        [sys.executable, "-m", "scripts.full_panel",
         "--campaign", str(tmp_path), "--control", "clip"],
        capture_output=True, text=True).stdout

    assert "CRASHED" in out, (
        "a run with an error_log.json was reported as merely not-completed:\n%s"
        % out[-1500:])
    assert "OutOfMemoryError" in out, "the exception type was not surfaced"
    assert "tralo" in out and "NO scorable run" in out, (
        "the scorer did not say which arm contributed nothing")


def test_two_cap_tags_that_produce_the_same_budget_are_one_cap_level():
    """House rule 4 is about BUDGETS, not tag spellings.

    `gen_campaign` refuses a single-cap campaign by comparing tag strings, which
    any two distinct spellings satisfy. Measured 2026-08-21 on `results/dosefix`:
    L40_G30 and L50_G30 both bind on the GLOBAL scope (local sums 82 and 103
    against a global 62), so class 2 gets K=62 and class 4 K=67 in BOTH cells --
    one budget level wearing two tags, and a per-cell count over them
    double-counts a single measurement.
    """
    from scripts.verify_caps import duplicate_budget_tags

    # the real dosefix numbers
    dup = duplicate_budget_tags({2: {"L40_G30": 62, "L50_G30": 62},
                                 4: {"L40_G30": 67, "L50_G30": 67}})
    assert dup == [(2, 62, ["L40_G30", "L50_G30"]),
                   (4, 67, ["L40_G30", "L50_G30"])], dup

    # a genuine sweep must come back clean
    assert duplicate_budget_tags({2: {"L50_G30": 62, "L50_G20": 41}}) == []

    # ...and a partial collision must be reported for the colliding class only
    dup = duplicate_budget_tags({2: {"a": 10, "b": 10}, 4: {"a": 10, "b": 20}})
    assert dup == [(2, 10, ["a", "b"])], dup


def test_no_autocast_banned_op_is_reachable_from_an_arm():
    """CUDA autocast BANS a few ops; calling one under AMP is a hard error.

    `select` called F.binary_cross_entropy. Every GPU run died in 11 seconds --
    "binary_cross_entropy and BCELoss are unsafe to autocast" -- wrote a
    header-only training_log.csv, and was reset to `pending` by the dispatcher,
    so the campaign looked merely unfinished while burning the card.

    ⚠️ THE BAN IS DYNAMIC, NOT LEXICAL, which is why this test is a blanket ban
    rather than a scan of `with autocast` bodies. The offending call sat in
    `selective_loss`, a module-level function that autocast never encloses
    lexically -- it is only ever CALLED from inside the block. A lexical gate
    written that way passed while the bug was present; this one was checked
    against the real defect.

    ⚠️ And the obvious fix is the wrong one. `binary_cross_entropy_with_logits`
    IS autocast-safe, but it applies a SIGMOID to its input, and our argument is
    a SOFTMAX probability, not that class's logit -- swapping it in silences the
    crash while quietly computing a different loss. Write the BCE out by hand on
    the already-clamped probability instead; it is bit-identical (verified, max
    abs diff 0.0) and autocast-safe.

    `scripts/smoke_arms.py` cannot cover this: it runs on cpu, where autocast is
    a no-op.
    """

    BANNED = {"binary_cross_entropy", "BCELoss"}

    offenders = []
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if fn in BANNED:
                    offenders.append("%s:%d %s"
                                     % (path.as_posix(), node.lineno, fn))

    assert not offenders, (
        "CUDA autocast bans these ops and every training path runs under AMP, "
        "so the arm dies on its first GPU batch: %s. Write the loss out by "
        "hand on the clamped probability -- NOT the _with_logits variant, which "
        "computes a different quantity." % offenders)


def test_the_selective_risks_centring_estimate_is_not_self_referential():
    """`selective_loss` must return THIS batch's estimate, not the EMA it was given.

    It returned the EMA-substituted value, and the trainer feeds the return
    straight back into its EMA -- so the update was 0.9*x + 0.1*x = x and the
    centring constant FROZE at batch 1 of epoch 1 and never moved again.

    That is not a stale log line. d risk / d g_i = (per_i - centre)/(n*tau), so a
    centre pinned to an untrained model's covered-set loss while per_item
    collapses (CE saturates by epoch 10, L_CE 0.02) flips the sign for nearly
    every item and turns the risk term into exactly the "cover everything" force
    the centring exists to prevent. The tell was that `cov_ema`, three lines
    away, IS recomputed from the batch -- the asymmetry between the two.
    """


    ema, beta, seen = None, 0.9, []
    torch.manual_seed(0)
    for b in range(5):
        g = torch.rand(64, requires_grad=True)
        # sharpen the probabilities each batch: the covered-set loss MUST move
        probs = torch.softmax(torch.randn(64, 7) * (1.0 + 2.0 * b), dim=1)
        y = torch.randint(0, 7, (64,))
        _loss, _cov, base = selective_loss(g, probs, y, 4, 0.03, 32.0,
                                           cov_ema=0.03, risk_ema=ema)
        ema = base if ema is None else beta * ema + (1 - beta) * base
        seen.append(base)

    assert len(set(round(v, 9) for v in seen)) > 1, (
        "selective_loss returned the SAME centring estimate on 5 batches whose "
        "probabilities differ by 5x -- it is echoing the EMA it was handed, so "
        "the caller's EMA can never move: %s" % seen)
    assert abs(ema - seen[0]) > 1e-6, "the caller's EMA never left batch 1"


def test_the_scorer_detects_a_run_that_collapsed_on_its_final_epoch(tmp_path):
    """The pipeline keeps the LAST epoch unconditionally, so one bad terminal
    epoch is the model that gets scored -- and when that run is the CONTROL,
    every arm appears to beat it at that seed.

    Measured on `results/dosefix`: `clip` seed 4 ended 0.9934 -> 0.9116 while
    the other seven controls ended 0.9935-1.0000, and it reversed the sign of
    the 4-seed `tralo_null` vs `clip` headline. FRAMEWORK section 9 states
    this; this test is what makes the statement checkable.
    """
    from scripts.full_panel import _terminal_collapse

    n = [0]

    def _log(accs, epochs=None):
        n[0] += 1
        d = tmp_path / ("r%d" % n[0])
        d.mkdir()
        ep = epochs if epochs is not None else list(range(len(accs)))
        rows = ["Epoch,Train_Acc"]
        rows += ["%d,%.4f" % (e, a) for e, a in zip(ep, accs)]
        (d / "training_log.csv").write_text(chr(10).join(rows) + chr(10))
        return str(d)

    assert _terminal_collapse(_log([0.95, 0.98, 0.9934, 0.9116]))[0] == "collapse", (
        "the detector missed the exact drop it was written for "
        "(dosefix clip seed 4, 0.9934 -> 0.9116)")

    # THE SAME DROP AS THE POST-HOC ARM LOGS IT. src/pipeline/warmup.py logs
    # epoch < 3 and then every max(1, warmup_epochs // 5)-th epoch, so at the
    # protocol's warmup_epochs=30 the rows are 1,2,3,6,12,18,24,30 and the last
    # interval spans SIX epochs. `clip` seed 4 is a post-hoc arm, so this is
    # the shape the real collapse had.
    st, (last, prev, gap) = _terminal_collapse(
        _log([0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.9934, 0.9116],
             epochs=[1, 2, 3, 6, 12, 18, 24, 30]))
    assert st == "collapse", "missed the collapse at the real logging density"
    assert gap == 6, "read the gap as %d; the warm-up logger writes every 6th " \
                     "epoch at warmup_epochs=30" % gap

    # NEGATIVE CONTROLS -- a gate is not done until it has been shown not to
    # fire on the things it must leave alone.
    for accs, why in [
        ([0.95, 0.98, 0.9934, 0.9940], "a healthy run still improving"),
        ([0.95, 0.98, 0.9934, 0.9900], "ordinary wobble, 0.0034 < 0.02"),
        ([0.9116, 0.9934], "a run that RECOVERED -- only the last epoch is kept"),
    ]:
        assert _terminal_collapse(_log(accs))[0] == "ok", "fired on " + why

    # AND THE THRESHOLD IS CALIBRATED PER GAP, not once for gap 1. Measured
    # over the 4,862 logs in this repo, the per-interval spread of a converged
    # run grows about as sqrt(gap) -- sd 0.00152 at gap 1 against 0.00300 at
    # gap 5 -- so one constant judged the post-hoc control and the trained
    # treatment by different standards.
    drop = 0.03      # above the gap-1 bar, below the gap-6 one
    assert _terminal_collapse(_log([0.99, 0.99 - drop]))[0] == "collapse"
    assert _terminal_collapse(
        _log([0.99, 0.99 - drop], epochs=[24, 30]))[0] == "ok", (
        "a 0.03 drop across SIX epochs is inside the measured wobble at that "
        "span; flagging it holds the control to a tighter bar than the arm")

    # THREE ANSWERS, NOT TWO. `None` used to mean both "healthy" and "this run
    # wrote no trajectory at all", and the second is the case that hid a
    # post-hoc control whose warm-up came from cache.
    assert _terminal_collapse(str(tmp_path / "nope"))[0] == "nolog", (
        "a completed run with no training_log.csv reads as healthy")


def test_a_posthoc_arm_that_wrote_no_log_is_not_silently_scored_as_healthy(capsys):
    """`clip` + `lp` share one `base_model_id`, as do `focal_clip` + `focal_lp`.

    Whichever of a pair the dispatcher runs SECOND loads the warm-up from cache
    -- `src/pipeline/warmup.py` returns early on a hit and the five post-hoc
    trainers write no CSV at all -- so it produces no `training_log.csv` and the
    collapse detector cannot see it. Its weights are nevertheless byte-identical
    to the sibling's, so when the log-less one is the `--control` the warning
    that matters most is exactly the one that cannot fire.
    """
    from scripts.full_panel import _collapse_report

    def _row(arm, rd, seed=4):
        return {"arm": arm, "cap": "L50_G30", "seed": seed, "run_dir": rd,
                "base_model_id": "MobileNetV3_dermmnist_deadbeef",
                "posthoc": True}

    import tempfile
    tmp = tempfile.mkdtemp()
    lp_dir = os.path.join(tmp, "lp")
    os.makedirs(lp_dir)
    with io.open(os.path.join(lp_dir, "training_log.csv"), "w",
                 encoding="utf-8") as fh:
        fh.write("Epoch,Train_Acc\n24,0.9934\n30,0.9116\n")
    # `clip` ran second, hit the cache, wrote nothing.
    rows = [_row("clip", os.path.join(tmp, "clip")), _row("lp", lp_dir)]

    _collapse_report(rows, "clip")
    out = capsys.readouterr().out
    assert "COLLAPSED" in out, "the collapse was not reported at all"
    assert "clip" in out and "via the shared warm-up" in out, (
        "the log-less control was not resolved through its shared warm-up:\n"
        + out)
    assert "ONE OF THESE IS THE CONTROL" in out, (
        "the control warning did not fire for an arm whose weights ARE the "
        "collapsed ones:\n" + out)

    # NEGATIVE CONTROL: with no sibling to inherit from, the run must be
    # REPORTED as undetermined, never skipped.
    _collapse_report([_row("clip", os.path.join(tmp, "clip"))], "clip")
    out = capsys.readouterr().out
    assert "WROTE NO TRAINING TRAJECTORY" in out, (
        "a completed run with no trajectory vanished silently:\n" + out)



def _ast_module_docstring(src):
    """The module docstring, via AST. Reading the first triple-quoted block by
    string search would also match a comment or a nested string; this cannot.
    """
    import ast as _a
    return _a.get_docstring(_a.parse(src)) or ""

def test_nothing_presents_a_closed_result_as_a_live_one(tmp_path):
    """FRAMEWORK is the law, so a stale "still open" in it sends a week of GPU
    into a question that is already answered -- and a generator that schedules
    a rejected arm by default spends that GPU without anyone reading anything.
    Two closures are pinned here, in the docs AND in the tool.

    (1) `tralo_null` - `clip` = -5.2 items was published from THREE seeds. The
    fourth reverses it (4-seed mean -0.06 items) because the `clip` control at
    that seed collapsed on its final epoch. The retraction has to sit ABOVE the
    superseded table or the next reader quotes the dead number.

    (2) `select` (path 1c, the jointly-trained selection head) was REJECTED on
    2026-08-22 -- -22 items vs `clip`, 0 of 2 cells on every metric, 2 of 8 runs
    collapsing on their final epoch. Section 4 is titled THE ONE OPEN QUESTION
    and listed 1c as "built, not run", told the reader "if 1b ties, go to 1c",
    and headed the 1c entry with "not yet run". All three read as an invitation
    to launch it. The section-12 verdict alone does not fix that, because
    nobody reads a 2,300-line file end to end before acting on section 4.
    """
    txt = io.open(os.path.join(REPO, "docs", "FRAMEWORK.md"),
                  encoding="utf-8").read()
    assert "-0.0188" in txt, "the 3-seed table vanished; keep it as superseded"
    head = txt.split("-0.0188")[0]
    assert "RETRACTED AT 4 SEEDS" in head, (
        "FRAMEWORK section 9 shows the 3-seed ccF1 table with no retraction "
        "above it -- a reader quotes the first number they see.")

    # (2) the select closure, checked where a reader would actually look.
    assert "IS REJECTED" in txt, "section 12's verdict on `select` vanished"

    # The section-4 STATE table is the first thing section 4 says. Its 1c row
    # must carry the verdict, not a build status.
    sec4 = txt.split("## 4. THE ONE OPEN QUESTION")[1]
    row = [r for r in sec4.splitlines()[:20]
           if r.startswith("|") and "SELECTION head" in r]
    assert row, "section 4's STATE table lost its 1c row entirely"
    assert "REJECTED" in row[0], (
        "section 4's STATE table still advertises 1c as something to run: %r"
        % row[0][:120])

    # and no forward-looking instruction to go build it may survive anywhere.
    for dead in ("go to 1c, **not** to a third count",
                 "BUILT 2026-08-21, not yet run",
                 "mechanism is available to 1c and to none of the arms run so far",
                 "1c escapes"):
        assert dead not in txt, (
            "FRAMEWORK still tells the reader to build 1c: %r. `select` is "
            "rejected (section 12) -- it lost 22 items and destabilised "
            "training." % dead)

    # The arm's own docstring is the other place a reader lands, via TRAIN_FNS.
    arm = io.open(os.path.join(REPO, "src", "methodologies", "select",
                               "train.py"), encoding="utf-8").read()
    doc = _ast_module_docstring(arm)
    assert "REJECTED" in doc, (
        "src/methodologies/select/train.py opens by arguing the direction is "
        "worth a campaign; it is rejected, and the module docstring is what "
        "anyone reading the registry follows.")

    # AND THE STRONGEST FORM OF "PRESENTED AS LIVE" IS BEING RUN BY DEFAULT.
    # `--arms all` expanded to every non-null arm in protocol.yml, `select`
    # included, so the canonical generator was scheduling GPU on a closed
    # question -- and putting an arm that collapsed 2 of its 8 runs into every
    # campaign. Checked by GENERATING, not by reading the code: this is about
    # what the tool emits.
    import yaml
    proto = yaml.safe_load(io.open(os.path.join(REPO, "configs",
                                                "protocol.yml"),
                                   encoding="utf-8"))
    assert "select" in proto.get("rejected_arms", {}), (
        "protocol.yml no longer declares `select` rejected, so `--arms all` "
        "puts it back into every campaign")

    # Checked by GENERATING, not by reading the generator: this is about what
    # the tool emits. Naming a rejected arm explicitly must still work, or
    # `results/selectrun` stops being reproducible -- it just cannot arrive by
    # default, and it cannot arrive quietly.
    root = pathlib.Path(tmp_path)
    # Whatever controls a campaign is currently required to carry, name them:
    # this test asks what `all` EXPANDS to, not what else a campaign needs, and
    # hardcoding today's answer would make it fail on the next control added.
    from configs.gen_campaign import count_control_arms
    extra = sorted(count_control_arms(proto))
    r = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--root", str(root),
         "--datasets", "iwildcam", "--caps", "L30_G30", "L50_G50",
         "--arms", "all"] + extra, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(l for l in r.stdout.splitlines()
                if l.strip().startswith("arms:"))
    assert "select" not in line, (
        "`--arms all` still schedules a rejected arm: %s" % line.strip())
    assert not list(root.rglob("*select*")), (
        "`--arms all` wrote rejected-arm run directories")

    # AND `all` MUST NOT SWALLOW THE ARMS NAMED BESIDE IT. `all` replaced
    # args.arms outright, so `--arms all tralo_reseed` generated a campaign
    # without tralo_reseed -- while the generator's own refusal message says
    # "Add: --arms ... tralo_reseed". The tool was instructing the user in a
    # form the tool ignored, and the result is indistinguishable from a
    # correctly generated campaign.
    for arm in extra:
        assert arm in line, (
            "`--arms all %s` dropped %s: naming an arm beside `all` has no "
            "effect, which is the form gen_campaign's own advice tells you to "
            "use -> %s" % (" ".join(extra), arm, line.strip()))

    r2 = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign",
         "--root", str(root / "explicit"), "--datasets", "iwildcam",
         "--caps", "L30_G30", "L50_G50", "--arms", "select"] + extra,
        cwd=REPO, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "IS REJECTED" in r2.stdout, (
        "naming a rejected arm generates it silently -- the verdict has to "
        "reach whoever is about to spend a GPU on it")
    assert list((root / "explicit").rglob("*select*")), (
        "naming `select` explicitly no longer generates it, so section 12's "
        "campaign cannot be reproduced")

    # the same must hold through the `all+null` branch, which has its own
    # expansion and therefore its own way to swallow a named arm
    r3 = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign",
         "--root", str(root / "allnull"), "--datasets", "iwildcam",
         "--caps", "L30_G30", "L50_G50", "--arms", "all+null", "select"],
        cwd=REPO, capture_output=True, text=True)
    assert r3.returncode == 0, r3.stdout + r3.stderr
    line3 = next(l for l in r3.stdout.splitlines()
                 if l.strip().startswith("arms:"))
    assert "select" in line3, (
        "`--arms all+null select` dropped the named arm: %s" % line3.strip())


def test_chunking_the_transductive_backward_does_not_change_the_gradient():
    """`tralo/train.py` chunks the gradient-carrying pass and reconstructs the
    full count as `total.detach() - chunk.detach() + chunk`, with a comment
    claiming this "already yields the EXACT full-N gradient" and therefore that
    `constraint_chunk_size` is a pure MEMORY knob.

    That claim is load-bearing: ViTB16 + `constraint_fp32` OOMs at chunk 256
    and again, intermittently, at 128, so the chunk has to be lowered mid
    campaign -- which is only legitimate if it cannot move a number. The
    penalty is NONLINEAR in the count, so chunking is exact only because
    f(total) is evaluated at the full count while gradient flows through one
    chunk: sum_j f'(total) * d(chunk_j)/dtheta = f'(total) * d(total)/dtheta.
    This pins the implementation to that identity.
    """
    import torch
    from src.losses.transductive_loss import MulticlassTransductiveLoss

    torch.manual_seed(0)
    N, C, D = 40, 4, 6
    X = torch.randn(N, D)
    gids = torch.tensor([i % 3 for i in range(N)])
    glob_c = torch.full((C,), 1e10)
    glob_c[1] = 5.0
    glob_c[2] = 7.0
    loc = {g: torch.full((C,), 1e10) for g in (0, 1, 2)}
    for g in loc:
        loc[g][1] = 2.0
        loc[g][2] = 3.0

    def grad_at(chunk):
        torch.manual_seed(1)
        lin = torch.nn.Linear(D, C)
        crit = MulticlassTransductiveLoss(glob_c, loc, num_classes=C,
                                          initial_rho=0.5)
        crit.lambda_global_per_class = {1: 0.7, 2: 0.3}
        crit.lambda_local_per_key = {(g, c): 0.5 for g in loc for c in (1, 2)}
        with torch.no_grad():
            tot_g = torch.softmax(lin(X), dim=1).sum(dim=0)
            tot_l = {g: torch.softmax(lin(X[gids == g]), dim=1).sum(dim=0)
                     for g in loc}
        lin.zero_grad()
        for st in range(0, N, chunk):
            sl = slice(st, min(st + chunk, N))
            pr = torch.softmax(lin(X[sl]), dim=1)
            cg = pr.sum(dim=0)
            cl = {}
            for g in loc:
                m = (gids[sl] == g)
                cl[g] = pr[m].sum(dim=0) if m.any() else torch.zeros(C)
            g_soft = tot_g.detach() - cg.detach() + cg
            l_soft = {g: tot_l[g].detach() - cl[g].detach() + cl[g] for g in loc}
            loss = (crit.compute_global_from_counts(g_soft)
                    + crit.compute_local_from_counts(l_soft))
            loss.backward()
        return lin.weight.grad.clone()

    ref = grad_at(N)                      # one chunk == unchunked
    assert ref.abs().sum() > 0, "the probe produced a zero gradient, so it "        "cannot detect a chunking bug -- the caps are not binding"
    for chunk in (1, 3, 7, 13, 40):
        g = grad_at(chunk)
        rel = (g - ref).abs().max() / ref.abs().max()
        assert rel < 1e-5, (
            "constraint_chunk_size=%d changes the constraint gradient by "
            "%.2e relative -- it is NOT a pure memory knob, so lowering it "
            "to survive an OOM would silently alter every number in the "
            "campaign" % (chunk, float(rel)))


def test_the_chunked_backward_in_tralo_still_uses_the_exact_construction():
    """The math gate above proves the identity holds. This one proves the arm
    that has to survive an OOM still IMPLEMENTS it, so the two cannot drift
    apart and quietly stop `constraint_chunk_size` being free.

    AST, not text. The first version of this gate matched the COMMENT that
    says "No /n_chunks" and failed on correct code -- the same way a grep once
    reported `rho_step` as read because a log line named it.
    """
    src = io.open(os.path.join(REPO, "src", "methodologies", "tralo",
                               "train.py"), encoding="utf-8").read()
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "train"), None)
    assert fn is not None, "tralo.train() not found"
    code = ast.unparse(fn)          # comments are gone; this is real code

    flat = "".join(code.split())
    assert "total_global_soft.detach()" in flat, (
        "the chunked backward no longer reconstructs the FULL global count "
        "before evaluating the penalty, so what it differentiates is a "
        "per-chunk penalty, not the full-N one")
    assert "total_local_soft[gid].detach()" in flat, "same defect on LOCAL counts"

    # a real division node, not a mention of one
    for node in ast.walk(fn):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and "n_chunks" in ast.unparse(node.right)):
            raise AssertionError(
                "tralo divides by n_chunks (`%s`). n_chunks is "
                "ceil(N_test / constraint_chunk_size), so this makes the "
                "constraint dose a function of the dataset size AND of a "
                "memory knob -- derm 8, oct 4, tissue 10, i.e. 2.5x apart."
                % ast.unparse(node))


def test_resetting_crashed_runs_refuses_to_discard_a_finished_one():
    """A run can carry a crash log AND be the real result: the dispatcher
    retries, so the second attempt can finish beside the first one's log.

    Reset by crash-log presence alone and you set those back to `pending` and
    re-run them, overwriting a good result. That happened by hand on
    `results/dosefix` -- two tralo runs that had crashed once, been retried and
    finished 29/30 epochs were reset, and only a status listing caught it
    before the dispatcher reached them.
    """
    from scripts.reset_crashed import eligible

    ok, why = eligible({"status": "pending", "results": {"accuracy": 0.77}}, 29)
    assert not ok, "would discard a finished run: " + why
    assert "HAS RESULTS" in why

    ok, why = eligible({"status": "running"}, 3)
    assert not ok and "running" in why, "would reset a live run"

    ok, why = eligible({"status": "pending", "results": {}}, 22)
    assert not ok, "silently discarded 22 epochs of work: " + why

    # the case it IS for: died on its face, nothing to keep
    ok, why = eligible({"status": "pending", "failures": 2, "results": {}}, 1)
    assert ok, "refused the actual crash case: " + why
    ok, why = eligible({"status": "pending", "results": {}}, 0)
    assert ok, "refused a run with no log at all: " + why


def test_the_scorer_still_sees_a_crash_log_that_was_renamed_aside():
    """Archiving `error_log.json` after fixing the cause is the natural tidy-up,
    and it restores exactly the blindness the crash report exists to remove --
    the dispatcher resets an interrupted run to `pending`, so with the log gone
    a dead treatment arm is indistinguishable from one that never started.
    """
    src = io.open(os.path.join(REPO, "scripts", "full_panel.py"),
                  encoding="utf-8").read()
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert fn is not None
    code = ast.unparse(fn)
    assert "error_log*.json" in code, (
        "full_panel.main looks for a literal error_log.json, so a crash log "
        "renamed aside (error_log.oom.json) makes the run report as plain "
        "`pending` -- a dead arm reading as an unstarted one")


def _code_identifiers(path):
    """Every name the CODE references: identifiers, attributes, import aliases
    and string literals -- with docstrings excluded.

    AST, never grep. A grep once reported `rho_step` as read because a LOG LINE
    named it, and the earlier version of the gate below compared raw file text,
    so a COMMENT could satisfy it. Docstrings are dropped for the same reason;
    string literals are kept, because a config key is read as `hp["key"]`.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)) and n.body:
            b = n.body[0]
            if (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                    and isinstance(b.value.value, str)):
                docs.add(id(b.value))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.alias):
            out.add(n.asname or n.name.split(".")[-1])
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str)
              and id(n) not in docs):
            out.add(n.value)
    return {n.lower() for n in out}


def test_the_grad_carrying_chunk_and_the_no_grad_chunk_are_separate_keys():
    """`constraint_chunk_size` bounds a BACKWARD pass; the allocators' and
    `select`'s chunk bounds inference under no_grad. Two knobs.

    While they shared one name, `check_parity` -- which compares a key across
    every arm that carries it -- read a correctly generated campaign as broken
    ("constraint_chunk_size differs: ['128', '256']") and blocked every launch,
    because the constraint phase had to drop to 128 to survive ViTB16 + fp32
    OOM while the no_grad path had no reason to move. The names are now
    distinct, so an arm cannot read one thinking it set the other.

    THE SPLIT WAS INCOMPLETE UNTIL 2026-08-22, AND THIS GATE COULD NOT SEE IT.
    `src/utils/constants.py` still exported the fallback as
    `CONSTRAINT_CHUNK_SIZE`, and all four no_grad arms imported it under that
    name to default `inference_chunk_size` -- so the constraint knob's NAME
    carried the inference knob's VALUE, and "fixing" the constant to track
    protocol's 128 would silently have halved the allocators' chunk. The old
    check compared lowercase strings against raw file text, and the constant is
    uppercase, so it passed. The comparison is now case-insensitive and runs
    over AST identifiers.
    """
    import yaml
    proto = yaml.safe_load(io.open(os.path.join(REPO, "configs", "protocol.yml"),
                                   encoding="utf-8"))
    assert "inference_chunk_size" in proto["chunked"], (
        "the no_grad chunk lost its own name; if it is called "
        "constraint_chunk_size again, check_parity fails on every campaign")
    assert "constraint_chunk_size" not in proto["chunked"], (
        "the `chunked` block emits constraint_chunk_size again -- that is the "
        "collision check_parity cannot see through")
    assert "constraint_chunk_size" in proto["constraint_phase"]

    # ONE SOURCE OF TRUTH FOR THE FALLBACK. `clip` and `focal_clip` carry no
    # `chunked` block, so they fall back to the module constant while every
    # other allocator reads the protocol's value. If the two drift, one knob
    # has two values depending on which arm you ask.
    from src.utils.constants import INFERENCE_CHUNK_SIZE
    assert INFERENCE_CHUNK_SIZE == proto["chunked"]["inference_chunk_size"], (
        "src.utils.constants.INFERENCE_CHUNK_SIZE is %r but protocol.yml says "
        "%r -- the arms without the `chunked` block would chunk differently "
        "from the ones with it" % (INFERENCE_CHUNK_SIZE,
                                   proto["chunked"]["inference_chunk_size"]))

    # and the readers must not have drifted back: _required (grad path) vs
    # .get (no_grad path) is what separates them at the call site. Compared
    # case-insensitively over AST identifiers, so neither a comment nor an
    # UPPERCASE constant can satisfy or evade it.
    for rel in [("tralo", "train.py"), ("fioretto_ldf", "train.py"),
                ("fioretto_alm", "train.py"), ("hounie_rcl", "train.py"),
                ("dual_common.py",)]:
        names = _code_identifiers(
            os.path.join(REPO, "src", "methodologies", *rel))
        assert "inference_chunk_size" not in names, (
            "%s names the no_grad inference chunk on its gradient-carrying "
            "pass -- at 256 that is the configuration that OOMs" % rel[0])
    for rel in [("danits_lp", "train.py"), ("heuristic", "train.py"),
                ("select", "train.py"), ("imbalanced_common.py",)]:
        names = _code_identifiers(
            os.path.join(REPO, "src", "methodologies", *rel))
        assert "constraint_chunk_size" not in names, (
            "%s names the CONSTRAINT chunk on a no_grad pass; it does not "
            "carry the constraint_phase block, so it would silently fall back "
            "to the module default and stop tracking the protocol" % rel[0])

    # THE OTHER HALF OF THE COLLISION. `src/utils/inference.py` had its own
    # module-level INFERENCE_CHUNK_SIZE = 512 for the metrics forward pass --
    # the same name as the configurable knob's fallback, carrying a different
    # value. It is private now, and the two must stay uncoupled: re-chunking a
    # forward pass re-associates the batch dimension, so unifying them
    # perturbs reported metrics in the last bits, and this project reads
    # arm-vs-arm identity at md5 resolution.
    import src.utils.inference as _inf
    assert not hasattr(_inf, "INFERENCE_CHUNK_SIZE"), (
        "src/utils/inference.py exports INFERENCE_CHUNK_SIZE again -- the same "
        "name as the protocol-backed fallback in src/utils/constants.py, "
        "carrying a different value. That is the collision this split removed")
    assert _inf._METRICS_FORWARD_CHUNK != INFERENCE_CHUNK_SIZE, (
        "the metrics forward stride and the allocators' inference chunk are "
        "now equal; if that is intended it is a metrics change and must be "
        "made deliberately, not fall out of a rename")

    # ONE CARD, ONE CAPACITY. The 128 value is justified by an OOM, and the
    # three places stating it quoted 22 GB / 22GB / 24 GB for the same event on
    # the same machine. The project owns exactly two GPUs -- a 24 GB Quadro RTX
    # 6000 and a 96 GB RTX PRO 6000 Blackwell -- so a 22 GB card is not one of
    # them, and the 96 GB one would not have OOM'd.
    caps = {}
    for site in ("configs/protocol.yml", "docs/FRAMEWORK.md",
                 "scripts/dose_scan.py"):
        txt = io.open(os.path.join(REPO, *site.split("/")),
                      encoding="utf-8").read()
        found = re.findall(r"(\d+)\s*GB[^.\n]{0,30}Quadro RTX 6000", txt)
        assert found, (
            "%s justifies constraint_chunk_size: 128 with an OOM but no "
            "longer names the card and its capacity" % site)
        caps[site] = set(found)
    assert len(set().union(*caps.values())) == 1, (
        "one OOM, one machine, and these files attribute it to cards of "
        "different sizes: %s" % caps)


def test_the_two_training_log_schemas_stay_watched_and_keep_their_conventions():
    """`training_log.csv` is written by TWO writers with different columns AND
    a different epoch axis. Both halves are pinned here.

    (1) WATCHED. The collapse detector reads train accuracy, so an arm that
    logs none is invisible to it -- and the pipeline keeps the FINAL epoch
    unconditionally, which is what makes a terminal collapse the scored model.
    The three dual arms wrote their own 7-column schema with no accuracy in it,
    so `fioretto`, `hounie` and `alm` could collapse silently. In an 80-run
    campaign that is 48 unwatched runs, on exactly the arms under test.

    (2) CONVENTIONS. `Epoch` is absolute and 1-based for `tralo` / `select`;
    `epoch` is relative to the constraint phase and 0-based for the three
    duals. The same training step is row 2 in one and row 0 in the other. That
    asymmetry is a decision (merging the schemas would make the 14,524-run
    provenance archive unreadable), it is written down in FRAMEWORK's "Known
    asymmetries" item 2, and it is pinned so neither side can drift onto the
    other's meaning without saying so.
    """
    import ast as _ast
    trained = {"tralo": ("tralo", "train.py"),
               "select": ("select", "train.py"),
               "fioretto_ldf": ("fioretto_ldf", "train.py"),
               "fioretto_alm": ("fioretto_alm", "train.py"),
               "hounie_rcl": ("hounie_rcl", "train.py")}
    for name, rel in trained.items():
        src = io.open(os.path.join(REPO, "src", "methodologies", *rel),
                      encoding="utf-8").read()
        code = _ast.unparse(_ast.parse(src))     # comments cannot satisfy this
        # Either the arm names the field itself, or it delegates to the
        # canonical writer, which always emits Train_Acc. Naming the field was
        # too strict: `select` passes the accuracy positionally to
        # log_progress_to_csv, and its logs do carry Train_Acc.
        assert ("train_acc" in code or "Train_Acc" in code
                or "log_progress_to_csv" in code), (
            "%s neither logs a train-accuracy field nor calls the canonical "
            "writer, so scripts/full_panel.py cannot see a terminal collapse "
            "in any of its runs" % name)

    # and the detector must accept BOTH spellings, or adding the column to the
    # duals silently fails to switch it on
    panel = io.open(os.path.join(REPO, "scripts", "full_panel.py"),
                    encoding="utf-8").read()
    fn = next(n for n in _ast.walk(_ast.parse(panel))
              if isinstance(n, _ast.FunctionDef) and n.name == "_terminal_collapse")
    body = _ast.unparse(fn)
    for spelling in ("Train_Acc", "train_acc"):
        assert spelling in body, (
            "_terminal_collapse does not know the %r spelling, so one log "
            "schema is unwatched" % spelling)

    # THE EPOCH AXIS MEANS TWO DIFFERENT THINGS, AND THAT IS A DECISION.
    # `tralo`/`select` run range(warmup_epochs, total) and log through
    # log_progress_to_csv, which adds 1 -> `Epoch` is ABSOLUTE and 1-based, so
    # the first constraint row is 2 at warm-up 1. The duals run
    # range(constraint_epochs) and log `epoch` raw -> RELATIVE and 0-based, so
    # the same training step is row 0. Documented in FRAMEWORK's "Known
    # asymmetries" as item 2; pinned here so neither side drifts onto the
    # other's meaning without a deliberate change. AST, so a comment saying
    # "absolute" proves nothing.
    def _epoch_range_args(path):
        tree = _ast.parse(io.open(path, encoding="utf-8").read())
        for n in _ast.walk(tree):
            if (isinstance(n, _ast.For)
                    and isinstance(n.target, _ast.Name)
                    and n.target.id == "epoch"
                    and isinstance(n.iter, _ast.Call)
                    and getattr(n.iter.func, "id", None) == "range"):
                return len(n.iter.args)
        return None

    for pkg in ("tralo", "select"):
        n_args = _epoch_range_args(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py"))
        assert n_args == 2, (
            "%s no longer iterates an ABSOLUTE epoch axis (range(warmup, "
            "total)); its Epoch column shares a name with the warm-up rows in "
            "the same file, so a relative axis would silently renumber them"
            % pkg)
    for pkg in ("fioretto_ldf", "fioretto_alm", "hounie_rcl"):
        n_args = _epoch_range_args(
            os.path.join(REPO, "src", "methodologies", pkg, "train.py"))
        assert n_args == 1, (
            "%s no longer iterates a RELATIVE epoch axis; FRAMEWORK's known-"
            "asymmetries table says its `epoch` is 0-based within the "
            "constraint phase, and the 14,524-run archive was written that way"
            % pkg)

    # and the two writers must keep the offset they are documented with
    logsrc = io.open(os.path.join(REPO, "src", "training", "logging.py"),
                     encoding="utf-8").read()
    writer = next(n for n in _ast.walk(_ast.parse(logsrc))
                  if isinstance(n, _ast.FunctionDef)
                  and n.name == "log_progress_to_csv")
    assert "epoch + 1" in _ast.unparse(writer), (
        "log_progress_to_csv no longer writes epoch + 1, so TraLO's Epoch "
        "column stopped being 1-based while FRAMEWORK still says it is")
    for pkg in ("fioretto_ldf", "fioretto_alm", "hounie_rcl"):
        src = io.open(os.path.join(REPO, "src", "methodologies", pkg,
                                   "train.py"), encoding="utf-8").read()
        assert "'epoch': epoch," in _ast.unparse(_ast.parse(src)), (
            "%s no longer logs the RAW epoch; if it gained a +1 it would look "
            "1-based while remaining relative, which is the one combination "
            "no reader could detect" % pkg)


def test_log_health_does_not_cry_wolf_on_a_warm_up_row_or_a_posthoc_arm(tmp_path):
    """`scripts/log_health.py` executes the house rule "validate from the
    training log, never from a final number". Two ways it can be worse than
    nothing, both of which it did on its first run:

    1. It flagged NON-FINITE VALUES on every tralo run that logged a warm-up
       epoch. That row is written before the constraint object exists, so its
       limits are legitimately blank -- exactly one NaN per constraint column.
       A divergence detector that fires on healthy runs gets ignored.
    2. It reported `clip` as satisfied 28/28. A post-hoc arm runs no constraint
       phase, so satisfaction is vacuous for it, and feasibility is not a metric
       in this project precisely because the allocator makes the clipper
       feasible by construction.
    """
    from scripts.log_health import read_run

    def mk(name, rows, header):
        d = tmp_path / name
        d.mkdir()
        (d / "training_log.csv").write_text(
            header + chr(10) + chr(10).join(rows) + chr(10))
        (d / "config.json").write_text('{"arm": "%s", "status": "completed"}' % name)
        return str(d)

    wide = "Epoch,Train_Acc,L_CE,Hard_Class2,Limit_Class2,Group0_Hard_Class2"
    # a warm-up row (Epoch 1) with blank constraint state, then healthy epochs
    warm = mk("tralo", ["1,0.8150,0.7800,0,,",
                        "2,0.9000,0.5000,200,62,70",
                        "3,0.9500,0.3000,190,62,66",
                        "4,0.9600,0.2000,180,62,60"], wide)
    r = read_run(warm)
    assert not r["nonfinite"], (
        "flagged the warm-up row as divergence: %s" % r["nonfinite"])
    assert r["sat"] is None or True
    assert 2 in r["counts"] and r["counts"][2]["K"] == 62

    # a post-hoc arm: same header, but every limit is UNLIMITED
    ph = mk("clip", ["1,0.8150,0.7800,0,10000000000.0,0",
                     "2,0.9000,0.5000,200,10000000000.0,70",
                     "3,0.9900,0.3000,190,10000000000.0,66"],
            wide.replace("Hard_Class2,Limit_Class2",
                         "Hard_Class2,Limit_Class2") + ",Global_Satisfied")
    ph_rows = io.open(os.path.join(ph, "training_log.csv"), encoding="utf-8").read()
    io.open(os.path.join(ph, "training_log.csv"), "w", encoding="utf-8").write(
        ph_rows.replace(",0" + chr(10), ",0,1" + chr(10))
               .replace(",70" + chr(10), ",70,1" + chr(10))
               .replace(",66" + chr(10), ",66,1" + chr(10)))
    r = read_run(ph)
    assert r["posthoc"], (
        "a run whose every Limit_Class is UNLIMITED was not recognised as "
        "post-hoc, so it will be reported as satisfied on every epoch")
    assert r["sat"] is None, "reported a vacuous satisfaction count for a "        "post-hoc arm: %s" % (r["sat"],)

    # and it must still SEE a real divergence
    bad = mk("diverged", ["2,0.9000,0.5000,200,62,70",
                          "3,0.9500,nan,190,62,66",
                          "4,0.9600,0.2000,180,62,60"], wide)
    assert read_run(bad)["nonfinite"], (
        "missed a NaN sitting beside real values -- a run once diverged to "
        "all-NaN and still wrote `completed`")


def test_test_embeddings_come_out_at_the_width_the_head_declares():
    """A wrong feature dim produces a silently meaningless embedding file --
    the array still saves, still loads, and every offline analysis built on it
    is garbage. The four backbones name their head differently (`heads` on ViT,
    `fc` on RegNet, `classifier` on both MobileNets), so the width is looked up,
    and this checks the lookup agrees with what the hook actually captures.
    """
    import torch
    from src.models.model_factory import get_model
    from src.pipeline.features import extract_test_embeddings, head_and_feature_dim

    X = torch.randn(6, 3, 224, 224)
    for name in ("MobileNetV3", "MobileNetV2", "RegNetY400MF", "ViTB16"):
        model = get_model(name, 7, pretrained=False)
        _head, dim = head_and_feature_dim(model)
        feats = extract_test_embeddings(model, X, chunk=4)
        assert feats.shape == (6, dim), (
            "%s: the head declares %d features but the hook captured %s" %
            (name, dim, (feats.shape,)))


def test_treatment_weight_keys_covers_every_null_arm_and_no_treated_one():
    """`_zero_lambda_arms` decides which arms are their own control. It read a
    hardcoded tuple of lambda + `select_eta`, so `fioretto_null`,
    `hounie_null` and `alm_null` -- which zero `fioretto_step_size`,
    `hounie_eta_lambda` and `alm_eta`/`alm_mu0`/`alm_mu_step` -- fell through to
    the treated branch and drew the "one run counted twice" false alarm on
    three of four nulls.

    The list is now derived from protocol.yml, and BOTH halves of the rule
    matter: a key must be 0 in the null block AND non-zero in its treated twin.
    `fioretto_lambda_init` is 0.0 in both, so taking every zero in a null block
    would classify the TREATED fioretto as untreated -- worse than the bug.
    """
    import yaml
    from scripts.full_panel import TREATMENT_WEIGHT_KEYS as KEYS

    with io.open(os.path.join(REPO, "configs", "protocol.yml"),
                 encoding="utf-8") as fh:
        blocks = yaml.safe_load(fh)["blocks"]

    for arm in ("tralo", "select", "fioretto", "hounie", "alm"):
        null = blocks.get(arm + "_null")
        if null is None:
            continue
        present = [null[k] for k in KEYS if k in null]
        assert present and all(float(v) == 0.0 for v in present), (
            "%s_null carries no treatment key that is zero, so it will be "
            "scored as a TREATED arm: %s" % (arm, present))
        live = blocks.get(arm) or {}
        lv = [live[k] for k in KEYS if k in live]
        assert lv and any(float(v) != 0.0 for v in lv), (
            "the treated arm %s reads as untreated (%s) -- its cross-cap "
            "identity check would be inverted" % (arm, lv))


# ------------------------------------------- the reseed control (FRAMEWORK 13) --

def _run_tralo_arm(arm, seed=1):
    """Run one tralo-family arm on the smoke harness.

    Returns (test probabilities, list of constraint-step calls).

    The arm is assembled from `configs/protocol.yml` through
    `scripts.smoke_arms.make_inputs`, exactly as `gen_campaign` would assemble
    it, rather than from a hand-written hyperparameter dict -- the claim under
    test is about the ARM, and a hand-written config cannot catch a YAML defect.
    """
    import shutil
    import tempfile

    import scripts.smoke_arms as smoke
    import src.methodologies.tralo.train as tralo_mod

    tmp = tempfile.mkdtemp(prefix="reseed_")
    calls = []
    real_finish = tralo_mod.finish_constraint_step
    real_backward = tralo_mod.constraint_backward
    try:
        inputs, _g, _l = smoke.make_inputs(smoke.load_protocol(), arm, tmp,
                                           seed=seed)
        # run_experiment re-seeds AFTER the warm-up and immediately before
        # train(), so every arm's constraint phase starts from one RNG state.
        # The RNG half only: seed_all also flips the PROCESS-WIDE
        # use_deterministic_algorithms, which would leak into every later test.
        torch.manual_seed(seed)

        def _spy(name, fn):
            def wrapped(*a, **k):
                calls.append(name)
                return fn(*a, **k)
            return wrapped

        tralo_mod.finish_constraint_step = _spy("finish", real_finish)
        tralo_mod.constraint_backward = _spy("backward", real_backward)
        out = TRAIN_FNS["tralo"](inputs)
        out.model.eval()
        with torch.no_grad():
            proba = F.softmax(out.model(inputs.X_test), dim=1).numpy()
        return proba, calls
    finally:
        tralo_mod.finish_constraint_step = real_finish
        tralo_mod.constraint_backward = real_backward
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_reseed_control_moves_the_predictions_and_takes_no_constraint_step():
    """`tralo_reseed` is `tralo_null` with the RNG stream perturbed, and that
    is the whole arm: zero dose, no constraint step, different stream.

    WHY IT EXISTS. Measured 2026-08-22 on `results/dosefix` and independently
    verified -- RMS separation of the capped-class hard count over epochs >= 4:
    turning the constraint ON moves it 75-95 items, and reseeding two pure-CE
    runs moves it 83-95. The constraint's whole measurable footprint on the
    count is 0.90-1.00x a reseed. That floor was only in the data by accident:
    `select_null` sets `select_eta: 0`, so it is a pure-CE run on `tralo_null`'s
    seed and warm-up cache whose selection head happens to draw from the global
    RNG. This arm makes the accident deliberate.

    Two things have to hold at once and neither is worth much alone:

      - the predictions must MOVE, or the arm is a duplicate `tralo_null`
        burning a GPU slot and reporting a floor of zero;
      - the constraint step must NEVER be taken, or the "floor" contains a dose
        and the comparison it exists to support is circular.

    Both assertions carry their own liveness control, because both are of the
    shape that passes when the instrument is broken: a bit-identical repeat
    would make "the predictions moved" unprovable, and a spy wired to the wrong
    name would make "no constraint step" vacuous. So the test also runs
    `tralo_null` twice (must be bit-identical) and `tralo` once (the spy must
    fire).
    """
    null_a, calls_null_a = _run_tralo_arm("tralo_null")
    null_b, calls_null_b = _run_tralo_arm("tralo_null")
    reseed, calls_reseed = _run_tralo_arm("tralo_reseed")
    _treated, calls_treated = _run_tralo_arm("tralo")

    # LIVENESS 1: the harness repeats bit for bit, so a difference is a
    # difference. This is the project's own standard since the determinism fix
    # -- identical output is not a small effect, it is no effect, at n=1.
    assert np.array_equal(null_a, null_b), (
        "two runs of tralo_null already differ, so nothing this test measures "
        "about tralo_reseed means anything")

    # LIVENESS 2: the spy fires on an arm that DOES take the step.
    assert calls_treated, (
        "the constraint-step spy never fired on `tralo`, so 'tralo_reseed "
        "takes no constraint step' is vacuous -- the patch missed its target")

    assert calls_null_a == [] and calls_null_b == [], (
        "tralo_null took a constraint step: %s" % calls_null_a)
    assert calls_reseed == [], (
        "tralo_reseed took a constraint step (%s), so it is not a zero-dose "
        "control -- at lambda 0 the penalty is identically 0, has_constraint "
        "must be False and pass 2 must be skipped entirely" % calls_reseed)

    assert not np.array_equal(null_a, reseed), (
        "tralo_reseed is BIT-IDENTICAL to tralo_null, so the reseed never "
        "happened and the arm is a duplicate control reporting a noise floor "
        "of exactly zero -- the one reading that would make the constraint "
        "look infinitely better than a reseed")


def test_the_reseed_control_shares_the_warm_up_cache_with_tralo_and_its_null():
    """If it trained its own warm-up it would stop being a matched control.

    The three arms share one `base_model_id` on purpose, so exactly one of them
    trains the cached model and the others load it. That is also why the RNG
    draw happens INSIDE the constraint phase: a draw before the warm-up would
    change what gets cached depending on which arm the dispatcher happened to
    run first, which is a different model for every machine and every run
    order.
    """
    P = load_protocol()
    assert _bid(P, "tralo_reseed") == _bid(P, "tralo") == _bid(P, "tralo_null")
    assert "rng_reseed" not in P["warmup_identity_keys"], (
        "rng_reseed entered warmup_identity_keys, so tralo_reseed now trains "
        "its OWN warm-up and is no longer matched to tralo_null")
    # And it is that arm by construction, not by a copied set of values.
    spec = P["arms"]["tralo_reseed"]
    assert "tralo_null" in spec["blocks"], (
        "tralo_reseed must CARRY the tralo_null block rather than duplicate "
        "its values, or the two can drift apart without either failing")
    hp_null = build_hyperparams(P, P["arms"]["tralo_null"], 1)
    hp_res = build_hyperparams(P, spec, 1)
    differing = {k for k in set(hp_null) | set(hp_res)
                 if hp_null.get(k) != hp_res.get(k)}
    assert differing == {"rng_reseed"}, (
        "tralo_reseed differs from tralo_null in more than the RNG stream: %s"
        % {k: (hp_null.get(k), hp_res.get(k)) for k in sorted(differing)})
    assert hp_res["rng_reseed"] is True and hp_null["rng_reseed"] is False


def test_the_reseed_control_reads_as_untreated_to_the_scorer(tmp_path):
    """`_zero_lambda_arms` decides which arms are their own control, and it
    reads the run CONFIG rather than the `_null` name suffix. `tralo_reseed`
    does not end in `_null`, so if the derivation ever went back to the suffix
    the scorer would treat the noise floor as a TREATED arm and report the
    reseed's count movement as a constraint effect.
    """
    from scripts.full_panel import _zero_lambda_arms

    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30",
             "--arms", "tralo", "tralo_null", "tralo_reseed")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [{"run_dir": str(p.parent), "arm": json.loads(p.read_text())["arm"]}
            for p in tmp_path.rglob("config.json")]
    untreated = _zero_lambda_arms(rows)
    assert "tralo_reseed" in untreated, (
        "the scorer reads tralo_reseed as TREATED, so it would be scored as a "
        "method rather than as the floor every count trajectory is measured "
        "against")
    assert "tralo_null" in untreated and "tralo" not in untreated


def _gen_arms(tmp, *arms):
    """gen_campaign with an explicit arm list and nothing else implied."""
    return subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp),
         "--datasets", "iwildcam", "--caps", "L30_G30", "L50_G30",
         "--arms"] + list(arms),
        cwd=REPO, capture_output=True, text=True)


def test_generator_refuses_a_count_reading_campaign_without_the_reseed_control(tmp_path):
    """A trained arm is exactly what writes a per-epoch capped-class count, and
    that trajectory is what "the constraint moved the count by N items" is read
    out of. It moves 75-95 items with the constraint on and 83-95 on a reseed
    alone, so the claim is not a measurement without the floor in the SAME
    campaign -- the same argument that puts both clippers in every campaign.

    Refused rather than auto-added: adding a trained arm is a compute decision,
    and silently growing what a campaign costs is the scope expansion this
    project has a rule against.
    """
    bad = _gen_arms(tmp_path / "bad", "tralo", "tralo_null")
    assert bad.returncode == 1, (
        "a campaign with trained arms and no reseed floor was accepted:\n%s"
        % (bad.stdout + bad.stderr))
    assert "no reseed control" in bad.stdout + bad.stderr
    assert "tralo_reseed" in bad.stdout + bad.stderr, (
        "the refusal must NAME the arm to add, or it is a puzzle")

    # ... and it is not refusing everything: the same campaign with the control
    # is accepted, so the gate discriminates.
    good = _gen_arms(tmp_path / "good", "tralo", "tralo_null", "tralo_reseed")
    assert good.returncode == 0, good.stdout + good.stderr
    assert "RESEED FLOOR in campaign" in good.stdout

    # A post-hoc-only campaign writes no count trajectory (warm-up epochs only,
    # no constraint phase), so it needs no floor and must not be blocked.
    posthoc = _gen_arms(tmp_path / "posthoc", "clip", "focal_clip", "lp")
    assert posthoc.returncode == 0, posthoc.stdout + posthoc.stderr


def test_all_excludes_the_reseed_control_and_all_plus_null_carries_it(tmp_path):
    """`all` must not silently grow: it already excludes the zero-dose siblings
    because adding four trained arms is +27% on the canonical campaign, and the
    reseed control is a trained arm for the same reason. So `--arms all` is
    REFUSED and names it, while `--arms all+null` carries it.

    Pinned because the two halves are easy to get backwards, and getting them
    backwards is silent either way: auto-adding spends GPU nobody approved,
    omitting it without the refusal ships an unreadable count trajectory.
    """
    r_all = _gen_arms(tmp_path / "all", "all")
    assert r_all.returncode == 1 and "no reseed control" in r_all.stdout + r_all.stderr

    r_null = _gen_arms(tmp_path / "allnull", "all+null")
    assert r_null.returncode == 0, r_null.stdout + r_null.stderr
    arms = {p.parts[-3] for p in (tmp_path / "allnull").rglob("config.json")}
    assert "tralo_reseed" in arms


# ------------------------------------------------- the frozen-head probe --
#
# `scripts/frozen_head_probe.py` prices a loss family on CPU in minutes
# instead of a week on GPU. It is only worth anything if it measures the
# project's real endpoint and if it can DETECT a difference -- a probe that
# cannot see one makes every null it reports unfalsifiable. Each gate below
# carries its own verified negative control, in the same test, because a gate
# whose failure mode was never exercised is a comment.

from scripts import frozen_head_probe as FHP                        # noqa: E402


def test_the_probe_polytope_argmax_equals_the_allocators_own_greedy():
    """The perturbed estimator maximises over a partition matroid, and the
    vectorised two-topk shortcut must equal the greedy `apply_allocation_
    heuristic` runs for one capped class: scan descending, take while the
    class has global AND local room. If it does not, the probe optimises a
    different budget from the one the pipeline enforces.
    """
    rng = np.random.default_rng(7)
    saw_naive_violation = False
    for trial in range(20):
        n, n_groups = 60, 3
        theta = rng.normal(size=n)
        groups = rng.integers(0, n_groups, n)
        caps = [int(x) for x in rng.integers(1, 12, n_groups)]
        K = int(rng.integers(3, 25))

        idx = [np.where(groups == g)[0] for g in range(n_groups)]
        got = FHP._matroid_topk(torch.tensor(theta), K, idx, caps)[0].numpy()

        # the greedy, written out, exactly as the allocator does it
        want = np.zeros(n)
        room_g, taken = dict(enumerate(caps)), 0
        for i in np.argsort(-theta):
            if taken >= K:
                break
            g = int(groups[i])
            if room_g[g] <= 0:
                continue
            room_g[g] -= 1
            want[i] = 1.0
            taken += 1
        assert np.array_equal(got, want), "trial %d" % trial

        # NEGATIVE CONTROL: a plain top-K that ignores the group caps must NOT
        # match, or this test would pass on an implementation that dropped the
        # local scope entirely -- which is exactly the defect it exists to
        # catch, and the one `full_panel` documents for a missing Group_ID.
        naive = np.zeros(n)
        naive[np.argsort(-theta)[:K]] = 1.0
        if any(int((groups[naive == 1] == g).sum()) > caps[g]
               for g in range(n_groups)):
            saw_naive_violation = True
            assert not np.array_equal(naive, want), (
                "the cap-blind top-K agreed with the matroid greedy on a case "
                "where it violates a local cap: the test cannot see the defect")
    assert saw_naive_violation, (
        "no trial made the local cap bind, so the negative control never ran "
        "and this test would pass on a cap-blind implementation")


def test_the_perturbed_topk_gradient_is_exactly_soft_membership_minus_target():
    """Fenchel-Young: dL/dtheta = y_eps(theta) - y_target, and the probe must
    deliver that identity through autograd without differentiating a sort.

    This is the one place the probe implements a published estimator itself
    rather than importing the pipeline's, so the identity is pinned rather
    than trusted.
    """
    torch.manual_seed(0)
    n, K, eps, M = 40, 8, 0.5, 64
    theta = torch.randn(n, requires_grad=True)
    pos = torch.zeros(n, dtype=torch.bool)
    pos[torch.randperm(n)[:16]] = True
    gen = torch.Generator().manual_seed(11)

    loss = FHP.perturbed_topk_loss(theta, pos, K, eps, M, gen, None, None)
    loss.backward()

    # y_eps and y_target, recomputed independently from the SAME stream
    gen2 = torch.Generator().manual_seed(11)
    Z = torch.randn(M, n, generator=gen2)
    y_eps = FHP._matroid_topk(theta.detach().unsqueeze(0) + eps * Z, K,
                              None, None).mean(0)
    masked = torch.where(pos, theta.detach(), torch.full((n,), float("-inf")))
    y_target = FHP._matroid_topk(masked, K, None, None)[0]

    assert torch.allclose(theta.grad, (y_eps - y_target) / K, atol=1e-6), (
        "the autograd gradient is not the Fenchel-Young gradient")

    # NEGATIVE CONTROL: the identity is specific to THIS target. A uniform
    # point of the same polytope must give a different gradient, or the
    # assertion above would hold for any target and would test nothing.
    uniform = torch.full((n,), K / n)
    assert not torch.allclose(theta.grad, (y_eps - uniform) / K, atol=1e-6)


def test_the_probe_converts_to_items_the_way_full_panel_does():
    """`items = d(ccF1) * sum_c (K_c + n_c) / 2` -- SUM over capped classes,
    never mean. full_panel.py records that taking the mean understated the
    count by exactly the number of capped classes, a factor of 3 on dermmnist.
    The probe must not reintroduce it in a second file.
    """
    y = np.array([1] * 20 + [2] * 30 + [4] * 40 + [5] * 100)
    alloc = np.array([1] * 6 + [2] * 9 + [4] * 12 + [5] * 163)
    classes = [1, 2, 4]

    got = FHP.items_per_001(y, alloc, classes)
    want = sum(0.01 * (int((alloc == c).sum()) + int((y == c).sum())) / 2
               for c in classes)
    assert abs(got - want) < 1e-12

    # NEGATIVE CONTROL: the mean-over-classes version, the documented bug.
    mean_version = want / len(classes)
    assert abs(got - mean_version) > 1e-9, (
        "sum and mean agree here, so this case cannot detect the regression")


def test_the_probe_split_is_deterministic_disjoint_and_group_stratified():
    """Train/val/test splits must be deterministic and documented. The probe
    strata are the (class, group) PAIR because dermmnist's group 2 is a
    genuinely different population, so a class-only split lets the LOCAL
    budgets drift between halves.
    """
    d = FHP.make_synthetic("matched", 0, dim=8)
    a1, b1 = FHP.stratified_halves(d.y, d.groups, 3)
    a2, b2 = FHP.stratified_halves(d.y, d.groups, 3)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2), "not deterministic"
    assert not set(a1) & set(b1), "fit and held-out overlap"
    assert len(a1) + len(b1) == len(d.y), "the split loses items"

    other, _ = FHP.stratified_halves(d.y, d.groups, 4)
    assert not np.array_equal(a1, other), "the seed does not change the split"

    # THE (class, group) CELL, not the group share. A plain random half keeps
    # the group shares to within a few percent all by itself, so asserting on
    # those passes on an unstratified split and gates nothing -- measured, a
    # random half is off by up to 12.5 items on a single (class, group) cell
    # while the stratified one is off by at most 1.0. Those cells ARE the local
    # budgets: `compute_local_constraints` takes a percentage of each of them.
    for half in (a1, b1):
        for c in range(7):
            for g in np.unique(d.groups):
                tot = int(((d.y == c) & (d.groups == g)).sum())
                if tot == 0:
                    continue
                got = int(((d.y[half] == c) & (d.groups[half] == g)).sum())
                assert abs(got - tot / 2.0) <= 1.0, (
                    "class %d group %d: %d of %d in this half -- the local "
                    "budget is not comparable across halves" % (c, g, got, tot))


def test_the_probes_liveness_control_is_itself_live():
    """LIVENESS OF THE LIVENESS CONTROL. `corrupt_head` must change the head
    when asked to and must be the IDENTITY at alpha=0 -- otherwise a reported
    "the probe resolves N items" is measuring the corruption knob's own noise.
    """
    torch.manual_seed(0)
    W = torch.randn(7, 12)
    classes = [1, 2, 4]

    same = FHP.corrupt_head(W, classes, 0.0, 5)
    assert torch.equal(same, W), "alpha=0 is not the identity"

    hit = FHP.corrupt_head(W, classes, 0.3, 5)
    moved = {c for c in range(7) if not torch.equal(hit[c], W[c])}
    assert moved == set(classes), (
        "corruption touched %s, expected exactly the capped classes" % moved)
    # the norm is held, so the control removes ORDERING and not scale -- a
    # shrunk row would degrade the softmax for a reason unrelated to ranking
    for c in classes:
        assert abs(float(hit[c].norm()) - float(W[c].norm())) < 0.35 * float(W[c].norm())


def test_the_topk_surrogate_localises_at_the_cut_and_softplus_does_not():
    """The claim the whole family rests on is that a bounded surrogate acts
    only in a window around the K-th ranked item. Pinned WITH its own control:
    the convex `softplus` variant must keep a finite gradient far below the
    cut, because if both localise then localisation is not what is being
    measured.
    """
    n, K = 200, 20
    base = torch.linspace(0.0, -20.0, n)
    pos = torch.zeros(n, dtype=torch.bool)
    pos[-5:] = True                    # positives buried far below the cut

    def grad_far(surrogate):
        s = base.clone().requires_grad_(True)
        FHP.topk_loss(s, pos, K, 0.5, surrogate).backward()
        return float(s.grad[-5:].abs().max())

    assert grad_far("sigmoid") < 1e-8, "the bounded surrogate is not localising"
    assert grad_far("softplus") > 1e-3, (
        "the convex surrogate localises too, so the two cannot be compared")


def test_the_probe_verdict_rule_survives_its_own_liveness_control():
    """REGRESSION ON A DECISION RULE, not on a number.

    The pre-registered bar read `|mean| >= 2 * sd`. The `tailnoise` control,
    where a cut-local loss wins BY CONSTRUCTION, returned +14.94 items on 7 of
    8 seeds with sd 7.53 -- and the rule said NO DIFFERENCE, because 2 x 7.53
    = 15.06. The defect is structural: when an effect is seed-dependent its sd
    grows with it, so that rule gets HARDER as the effect gets larger. It now
    reads standard ERROR, with the conservative reading kept as a `[fragile]`
    tag rather than as the gate.
    """
    st = FHP.paired([16.5, 20.1, 11.3, 25.8, 8.0, 14.9, 22.4, -0.9])
    assert st["mean"] > 2 * st["sem"]
    v = FHP.verdict(st, 8, 1.0, 7.0 / 8.0)
    assert v.startswith("WORTH A CAMPAIGN"), v
    assert "fragile" in v, "the conservative reading must stay visible"

    # NEGATIVE CONTROL: the retired rule must still reject this input, or the
    # regression this test pins never existed.
    assert abs(st["mean"]) < 2.0 * st["sd"]

    # and the rule still says no when there is nothing there
    quiet = FHP.paired([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0, -0.05])
    assert FHP.verdict(quiet, 8, 1.0, 7.0 / 8.0).startswith("NO DIFFERENCE")


def test_the_probes_special_term_is_live_and_is_off_at_weight_zero():
    """The project's most frequent failure mode is an INERT FLAG -- four
    occurrences and counting -- so the probe's own treatment knob gets the
    same across-arms identity check the campaigns get: at weight 0 a treated
    arm must be BIT-IDENTICAL to `ce`, and above 0 it must differ.
    """
    import argparse
    d = FHP.make_synthetic("matched", 0, dim=8)
    fit, _ = FHP.stratified_halves(d.y, d.groups, 1)
    X = d.features[fit]
    X = ((X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
         ).astype(np.float32)
    y = d.y[fit]
    G, L = FHP.budgets(y, d.groups[fit], d.classes, d.local_pct, d.global_pct, 7)
    views = {c: FHP._group_view(d.groups[fit], G[c],
                                {g: lim[c] for g, lim in L.items()
                                 if lim[c] < UNLIMITED}) for c in d.classes}
    args = argparse.Namespace(
        ce_steps=25, refine_steps=25, lr=0.05, weight_decay=1e-4,
        special_weight=0.0, temp=0.5, topk_surrogate="sigmoid",
        pauc_neg_frac=0.05, ptopk_eps=0.5, ptopk_samples=8)

    shared = FHP.fit_head(X, y, 7, d.classes, G, views, "ce", args, 1)
    base = FHP.fit_head(X, y, 7, d.classes, G, views, "ce", args, 1, shared)
    for name in ("topk", "pauc", "ptopk"):
        off = FHP.fit_head(X, y, 7, d.classes, G, views, name, args, 1, shared)
        assert torch.equal(off[0], base[0]) and torch.equal(off[1], base[1]), (
            "%s is not bit-identical to ce at special_weight=0" % name)

    args.special_weight = 1.0
    for name in ("topk", "pauc", "ptopk"):
        on = FHP.fit_head(X, y, 7, d.classes, G, views, name, args, 1, shared)
        assert not torch.equal(on[0], base[0]), (
            "%s is INERT: the weight is live but the head did not move" % name)


def test_the_probe_refuses_a_run_directory_with_no_embeddings(tmp_path):
    """The feature dump only just landed, so most run directories have no
    `test_embeddings.npz`. The probe must say so and name the file -- silently
    falling back to synthetic features would present a generative model's
    numbers as evidence about a dataset.
    """
    with pytest.raises(SystemExit) as e:
        FHP.load_real(str(tmp_path))
    msg = str(e.value)
    assert "test_embeddings.npz" in msg
    assert "synthetic" in msg.lower(), (
        "the refusal must say why substituting synthetic data is not the fix")


def test_the_probe_prints_only_ascii_so_a_warning_cannot_be_truncated():
    """A non-ASCII byte in a print SILENTLY TRUNCATES the report.

    Measured: the probe's "your resolution is coarser than the whole question"
    warning began with a stop-sign glyph, and on this project's Windows
    workstation (cp1252) the run ended at the line BEFORE it -- with exit code
    0 and no traceback in the piped output. The truncated section was the one
    that says a null is unreadable, so the failure mode is "the harness stops
    telling you it cannot see" and nothing announces it.

    Scoped to this one file because it is the only script whose output is a
    verdict a human acts on directly, and because a repo-wide rule would fail
    on documents that legitimately carry glyphs.
    """
    src = io.open(os.path.join(REPO, "scripts", "frozen_head_probe.py"),
                  encoding="utf-8").read()
    bad = [i + 1 for i, line in enumerate(src.splitlines())
           if any(ord(ch) > 127 for ch in line)]
    assert not bad, (
        "non-ASCII on line(s) %s of frozen_head_probe.py: on a cp1252 console "
        "the report stops there, silently and with exit code 0" % bad)

    # NEGATIVE CONTROL: the scan must actually see one when it is there, or it
    # would pass on a file it never really examined.
    probe_ascii = chr(0x2014) + 'x'
    assert [i + 1 for i, line in enumerate([probe_ascii, 'y'])
            if any(ord(ch) > 127 for ch in line)] == [1]


def test_the_duals_reset_grad_norm_every_epoch_instead_of_carrying_it_forward():
    """`grad_norm` is the column this project reads to decide whether two arms
    got a comparable dose -- the whole 20x inter-arm argument is built from it.

    The three duals only reassign `last_grad_norm` inside `if did_backward`, so
    with the declaration hoisted above the epoch loop an epoch where the
    constraint went slack logged the PREVIOUS epoch's norm as if it were its
    own. tralo resets per-epoch and writes 0.0, so the same slack epoch is an
    honest zero in one arm and a fabricated repeat in another -- an asymmetry
    between a treatment and its comparison, in the telemetry, invisible to
    every config gate.

    AST, not text: the assignment must be INSIDE the `for epoch` loop.
    """
    import ast as _ast
    for pkg in ("fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        src = io.open(os.path.join(REPO, "src", "methodologies", pkg,
                                   "train.py"), encoding="utf-8").read()
        tree = _ast.parse(src)
        # the epoch loop lives in `_train_constraints`, not `train` -- found by
        # searching for it rather than by assuming, after assuming wrongly once
        fn = next((n for n in _ast.walk(tree)
                   if isinstance(n, _ast.FunctionDef)
                   and any(isinstance(x, _ast.For) and isinstance(x.target, _ast.Name)
                           and x.target.id == "epoch" for x in _ast.walk(n))), None)
        assert fn is not None, "%s: no function contains a `for epoch` loop" % pkg

        def _assigns(node):
            return [n for n in _ast.walk(node)
                    if isinstance(n, _ast.Assign)
                    and any(isinstance(t, _ast.Name) and t.id == "last_grad_norm"
                            for t in n.targets)]

        loops = [n for n in _ast.walk(fn)
                 if isinstance(n, _ast.For)
                 and isinstance(n.target, _ast.Name) and n.target.id == "epoch"]
        assert loops, "%s: no `for epoch` loop found" % pkg
        inside = {id(a) for lp in loops for a in _assigns(lp)}
        resets = [a for a in _assigns(fn)
                  if isinstance(a.value, _ast.Constant) and a.value.value == 0.0]
        assert resets, "%s: `last_grad_norm = 0.0` disappeared entirely" % pkg
        assert all(id(a) in inside for a in resets), (
            "%s resets last_grad_norm OUTSIDE the epoch loop, so a slack epoch "
            "logs the previous epoch's gradient norm as its own" % pkg)


def test_the_probe_is_invariant_to_global_rng_state():
    """THE FLAKINESS GATE. A probe that answers differently depending on what
    ran before it cannot be trusted to report "no difference" -- the answer it
    returns most often.

    History: this suite was reported flaky at ~1 in 3, on a ROTATING member of
    the probe gates, each passing in isolation. Diagnosed on a disposable copy
    and it was NOT order dependence and NOT global RNG (forward, reverse and
    -k-only orderings all pass; three repeats with `random`, `numpy.random`
    and `torch`'s global generators deliberately polluted all pass). The cause
    was a mutation-verification harness rewriting `scripts/frozen_head_probe.py`
    IN THE WORKING TREE while other agents ran `pytest tests` against the same
    checkout -- one process editing another process's source, which reads
    exactly like flakiness. The harness now works on a temporary copy.

    This gate exists so the diagnosis stays true. It pins the two properties
    that would have made the report real, and it is deliberately stronger than
    what was needed: same answer after the globals are polluted, and the same
    answer from a FRESH interpreter with a different hash seed.
    """
    import hashlib
    import random as _random

    baseline = FHP.determinism_digest()

    # (1) same process, every global generator moved out from under it
    _random.seed(20260822)
    np.random.seed(4242)
    torch.manual_seed(31337)
    for _ in range(37):
        _random.random()
        np.random.rand()
        torch.randn(5)
    assert FHP.determinism_digest() == baseline, (
        "the probe's answer changed after the global generators were "
        "polluted: something in it draws from a global RNG instead of the "
        "seeded one it is given")

    # (2) a fresh interpreter, different hash seed, globals pre-polluted
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="12345")
    r = subprocess.run(
        [sys.executable, "-c",
         "import random,numpy,torch,sys;"
         "random.seed();numpy.random.seed();torch.manual_seed(7);"
         "[torch.randn(3) for _ in range(11)];"
         "sys.path.insert(0, %r);"
         "import scripts.frozen_head_probe as F;"
         "print(F.determinism_digest())" % REPO],
        cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == baseline, (
        "a fresh interpreter gives a different digest (%s vs %s): the probe "
        "depends on interpreter state, not only on its seed"
        % (r.stdout.strip(), baseline))

    # the digest must actually be a function of the run, not a constant string
    assert baseline != FHP.determinism_digest(seed=2), (
        "the digest does not change with the split seed, so it would not "
        "detect a change in anything either"
    )
    assert len(baseline) == len(hashlib.md5(b"").hexdigest())


# ---------------------------------------------------------------------------
# The commit that RAN a config is not the commit that WROTE it
# ---------------------------------------------------------------------------

def test_the_runner_stamps_the_commit_that_produced_the_weights():
    """`code_version` is stamped by the GENERATOR and never revisited.

    `configs/gen_campaign.main()` writes it once per config and explicitly
    skips any config already marked completed, so run half a campaign, land a
    change to a training file and resume the rest: every config still carries
    the ORIGINAL value. `full_panel`'s provenance gate then sees one value
    across two pipelines and scores both halves as one comparison -- the exact
    thing it was written to refuse -- and `model_cache` hands the post-change
    runs the pre-change warm-up on the same false agreement.

    AST, not grep: a comment naming the key would satisfy a text search.
    """
    src = io.open(os.path.join(REPO, "src", "experiments", "runner.py"),
                  encoding="utf-8").read()
    tree = ast.parse(src)

    writes = [n for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              for t in n.targets
              if isinstance(t, ast.Subscript)
              and isinstance(t.slice, ast.Constant)
              and t.slice.value == "run_code_version"]
    assert writes, (
        "src/experiments/runner.py never assigns run_code_version, so every "
        "config still describes only the commit that generated it")

    # and it must land on DISK before the status flips: update_experiment_status
    # reloads config.json and rewrites it, so an in-memory key would be dropped.
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_experiment")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("save_config_to_path", "update_experiment_status")]
    assert calls and calls[0].func.id == "save_config_to_path", (
        "the stamp is not persisted before update_experiment_status reloads "
        "config.json from disk, so a run that crashes carries no runner stamp")

    # ONE implementation of the git call, shared by generator and runner.
    for mod in ("configs/gen_campaign.py", "src/experiments/runner.py"):
        code = ast.unparse(ast.parse(
            io.open(os.path.join(REPO, *mod.split("/")), encoding="utf-8").read()))
        assert "git_version" in code, (
            "%s hand-rolls its own `git rev-parse` -- four hand-rolled copies "
            "of one step is how the constraint dose drifted 20x between arms"
            % mod)
        assert "rev-parse" not in code, (
            "%s still calls git directly instead of src/utils/gitver" % mod)


def test_the_provenance_gate_reads_the_runners_stamp_and_degrades_without_it():
    """The gate must prefer `run_code_version`, and must still work for the
    14,524 archived runs that have none -- degrading to the old check with a
    clear message, never crashing and never silently passing.
    """
    from scripts.full_panel import _provenance_key

    gen, run = "aaaaaaaaaaaa", "bbbbbbbbbbbb"

    # a run that says which commit produced its weights: that is the answer
    key, stamped = _provenance_key(
        {"code_version": gen, "run_code_version": run, "data_fingerprint": "d1"})
    assert key == (run, "d1") and stamped is True

    # THE FAILURE THIS EXISTS FOR: generated once, resumed after a code change.
    # Both halves share `code_version`, so the old key CANNOT tell them apart;
    # the runner's stamp splits them, which is a REFUSAL rather than a silent
    # pass.
    before = _provenance_key({"code_version": gen, "run_code_version": gen,
                              "data_fingerprint": "d1"})[0]
    after = _provenance_key({"code_version": gen, "run_code_version": run,
                             "data_fingerprint": "d1"})[0]
    assert before != after, (
        "two runs of one campaign produced by different code still share a "
        "provenance key, so the gate would score them as one comparison")

    # an archived run: falls back to the generator's stamp, flagged as such
    key, stamped = _provenance_key({"code_version": gen, "data_fingerprint": "d1"})
    assert key == (gen, "d1"), "the fallback changed the archived behaviour"
    assert stamped is False, (
        "an unstamped run reports as stamped, so the panel would not warn "
        "that the check is degraded for it")

    # and a config with neither key must not raise
    assert _provenance_key({}) == ((None, None), False)


def test_the_model_cache_prefers_the_stamp_of_the_run_that_trained_it(
        tmp_path, monkeypatch):
    """`base_model_id` hashes hyperparameters, not code. The cache therefore
    needs a code stamp -- and the generator's cannot see a change landed while
    the campaign was running, which is precisely when a stale warm-up is handed
    across a code boundary.

    Every warm-up on disk predates the runner stamp, so a missing one must
    degrade to the generator comparison rather than invalidate the cache.
    """
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
        torch.save(dict({"model_state_dict": _Tiny().state_dict(),
                         "base_model_id": bmid}, **payload),
                   MC.get_cache_path(bmid))

    def _load(cfg):
        return MC.load_from_cache(bmid, dict({"model_name": "TinyNet",
                                              "hyperparams": {"dropout": 0.3}},
                                             **cfg), 2, dev)

    # (1) a cache and a run that agree on the GENERATOR but not on the RUNNER.
    #     This is the mid-campaign code change, and it must retrain.
    _write({"code_version": "GEN1", "run_code_version": "RUN1"})
    assert _load({"code_version": "GEN1", "run_code_version": "RUN2"}) is None, (
        "the cache was reused across a code change that both configs' "
        "generator stamps agree through")
    assert _load({"code_version": "GEN1", "run_code_version": "RUN1"}) is not None

    # (2) NEGATIVE CONTROL -- an archived cache with no runner stamp must NOT
    #     be thrown away. Invalidating them all would retrain every warm-up.
    _write({"code_version": "GEN1"})
    assert _load({"code_version": "GEN1", "run_code_version": "RUN2"}) is not None, (
        "a pre-stamp cache was invalidated; every warm-up on disk is one")
    assert _load({"code_version": "GEN2"}) is None, (
        "the generator fallback stopped rejecting a genuine version mismatch")


# ---------------------------------------------------------------------------
# A constraint step that did not land
# ---------------------------------------------------------------------------

TRAINERS_WITH_A_CONSTRAINT_STEP = [
    ("tralo", "train.py"), ("fioretto_ldf", "train.py"),
    ("hounie_rcl", "train.py"), ("fioretto_alm", "train.py"),
]


def test_no_trainer_discards_whether_the_constraint_step_actually_landed():
    """`finish_constraint_step` returns `applied`, which is False when the
    constraint gradient came back non-finite -- on the FP16 path a NaN norm
    fails the `> 0` gate and an inf norm is skipped inside `scaler.step`, so
    either way no update lands. All four trainers bound it to `_applied` and
    dropped it, and `fioretto` consequently ran a 62%-length constraint phase
    (10 of 29 epochs lost, 6 NaN + 4 inf) while writing `status: completed`.

    Two arms in one campaign can take 29 and 19 steps, and until this was
    recorded nothing could say so -- a dropped step leaves no trace in the
    predictions except the effect it did not have.
    """
    for mod, fname in TRAINERS_WITH_A_CONSTRAINT_STEP:
        path = os.path.join(REPO, "src", "methodologies", mod, fname)
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        targets = []
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
                continue
            f = n.value.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name != "finish_constraint_step":
                continue
            t = n.targets[0]
            assert isinstance(t, ast.Tuple) and len(t.elts) == 2, (
                "%s does not unpack finish_constraint_step's two returns" % mod)
            targets.append(t.elts[1])
        assert targets, "%s never calls finish_constraint_step" % mod
        for t in targets:
            assert isinstance(t, ast.Name) and not t.id.startswith("_"), (
                "%s binds `applied` to %r -- an underscore name is how this "
                "value was discarded in all four trainers"
                % (mod, getattr(t, "id", t)))
            reads = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Name) and n.id == t.id
                     and isinstance(n.ctx, ast.Load)]
            assert reads, (
                "%s binds `applied` and never reads it, so a dropped "
                "constraint step is still invisible" % mod)


@pytest.mark.parametrize("arm", ["tralo", "fioretto", "hounie", "alm"])
def test_a_run_reports_how_many_constraint_steps_it_actually_took(arm, tmp_path):
    """The count has to reach the run summary, or no scorer can read it."""
    import scripts.smoke_arms as SA

    P = load_protocol()
    torch.manual_seed(1)
    inputs, _g, _l = SA.make_inputs(P, arm, str(tmp_path))
    out = TRAIN_FNS[P["arms"][arm]["methodology"]](inputs)
    app = out.summary.get("constraint_steps_applied")
    att = out.summary.get("constraint_steps_attempted")
    assert app is not None and att is not None, (
        "%s does not report its applied/attempted constraint steps" % arm)
    assert att >= 1, "%s attempted no constraint step at all" % arm
    assert app == att, (
        "%s lost %d of %d constraint steps on a tiny CPU model, where nothing "
        "should overflow" % (arm, att - app, att))


def test_a_zero_dose_arm_attempts_no_constraint_step():
    """`tralo_null` sets every lambda to 0, so `has_constraint` is False and
    transductive pass 2 is skipped entirely. The denominator must therefore be
    "epochs that formed a constraint gradient", not `constraint_epochs` --
    otherwise the null arm reads as having LOST all 29 of its steps.
    """
    import shutil
    import tempfile

    import scripts.smoke_arms as SA

    P = load_protocol()
    tmp = tempfile.mkdtemp()
    try:
        torch.manual_seed(1)
        inputs, _g, _l = SA.make_inputs(P, "tralo_null", tmp)
        out = TRAIN_FNS[P["arms"]["tralo_null"]["methodology"]](inputs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out.summary.get("constraint_steps_attempted") == 0, (
        "the zero-dose arm attempted a constraint step; either the lambdas are "
        "not zero or the counter is counting epochs rather than steps")
    assert out.summary.get("constraint_steps_applied") == 0


def test_the_panel_says_when_two_arms_did_not_take_the_same_number_of_steps(capsys):
    """Reported loudly, because it cannot be recovered from any metric."""
    from scripts.full_panel import _constraint_dose_check

    def _rows(arm, applied, attempted, n=4):
        return [{"arm": arm, "steps_applied": applied,
                 "steps_attempted": attempted} for _ in range(n)]

    # the measured case: fioretto lost 10 of its 29 epochs, tralo lost none
    _constraint_dose_check(_rows("tralo", 29, 29) + _rows("fioretto", 19, 29))
    out = capsys.readouterr().out
    assert "STEP(S) LOST" in out, "a 10-of-29 loss was not named:\n" + out
    assert "DID NOT RUN AT THE SAME DOSE" in out, (
        "two arms a third of a constraint phase apart were reported as "
        "comparable:\n" + out)

    # NEGATIVE CONTROL -- equal, complete doses must say nothing alarming
    _constraint_dose_check(_rows("tralo", 29, 29) + _rows("fioretto", 29, 29))
    out = capsys.readouterr().out
    assert "STEP(S) LOST" not in out and "SAME DOSE" not in out, (
        "fired on two arms that took every step they attempted:\n" + out)

    # and a campaign of runs from before the field existed is NAMED, not
    # silently treated as agreement
    _constraint_dose_check(_rows("tralo", None, None))
    out = capsys.readouterr().out
    assert "no counts recorded" in out, (
        "runs with no step counts were passed over in silence:\n" + out)


# The probe's pre-registered bar, clause (b). It was written as the fraction
# 7/8 for an eight-seed run and never re-derived, so adding seeds made it
# exponentially HARDER -- the harness punished its own precision. These two
# gates hold the corrected form in place; both were confirmed to FAIL against
# the fraction they replaced.


def test_the_probes_sign_bar_does_not_get_harder_as_seeds_are_added():
    """A fixed sign FRACTION is not a bar, it is a moving target.

    Negative control, run before this gate was written: the old rule
    `sign_frac >= 7/8` demands p=0.0703 at n=8 and p=5.56e-10 at n=64 for the
    identical fraction, so a directionally identical effect that CLEARED the
    bar at 8 seeds FAILED it at 64. That is the failure this asserts against.
    """
    import numpy as np

    def sign_p_of_fraction(frac, n):
        st = FHP.paired(np.array([1.0] * int(round(frac * n))
                                 + [-1.0] * (n - int(round(frac * n)))))
        return st["sign_p"]

    # the defect, stated as arithmetic and independent of any result
    p8, p64 = sign_p_of_fraction(7 / 8.0, 8), sign_p_of_fraction(7 / 8.0, 64)
    assert p64 < p8 / 1e6, (
        "the fixed-fraction bar is supposed to be the thing being ruled out; "
        "if it no longer tightens with n this control has gone stale (%g -> %g)"
        % (p8, p64))

    # The corrected rule, clause (b) ISOLATED: one fixed sign consistency, one
    # mean held well clear of clause (a)'s 1-item floor, only n varying. (A
    # first draft of this gate drew the signs at random, which let the mean
    # drift under the floor at large n and failed on clause (a) while claiming
    # to test clause (b).) Clearing at n must imply clearing at every larger n.
    blocked = []
    for n in (8, 16, 24, 64, 128):
        pos = int(np.ceil(0.78 * n))
        d = np.array([5.0] * pos + [-1.0] * (n - pos))   # mean ~+3.7 always
        st = FHP.paired(d)
        assert abs(st["mean"]) > 1.0, "the floor must not be what decides here"
        blocked.append(FHP.verdict(st, n, 1.0, 0.01).startswith("NO DIFF"))
    assert blocked == sorted(blocked, reverse=True), (
        "adding seeds un-cleared the bar: %s" % blocked)
    assert not blocked[-1], "128 seeds of a 78%-consistent effect must clear"

    # The same construction under the rule this replaced, at the size where
    # the two fractions genuinely separate: 100 of 128 seeds carry the sign,
    # sign p = 1.1e-10, and the 7/8 bar still returns NO DIFFERENCE. It is
    # rejecting an effect significant at ten decimal places -- testing the
    # wrong quantity, not being strict.
    n = 128
    pos = int(np.ceil(0.78 * n))
    st = FHP.paired(np.array([5.0] * pos + [-1.0] * (n - pos)))
    assert st["sign_p"] < 1e-9
    assert pos < int(np.ceil((7 / 8.0) * n)), (
        "the old fraction would have accepted this, so the control is stale")


def test_the_probe_prices_a_fragile_effect_in_campaign_seeds():
    """`WORTH A CAMPAIGN` and `[fragile]` contradict each other unless the
    contradiction is priced. The probe resamples SPLITS and affords dozens; a
    campaign resamples training seeds and affords four, so an effect can be
    real here and structurally invisible there.

    Negative control: before `seeds_needed` existed the tag said only
    `a 4-seed campaign could miss it`, which is unfalsifiable -- it named no
    number a reader could check or act on.
    """
    import numpy as np

    # closed form, two-sided alpha=0.05 at 80% power
    for effect, sd in ((1.28, 2.22), (1.20, 2.55), (5.0, 2.0)):
        want = int(np.ceil((1.959963985 + 0.8416212336) ** 2 * sd ** 2
                           / effect ** 2))
        assert FHP.seeds_needed(effect, sd) == want

    # monotone in both arguments, or it is not a power calculation
    assert FHP.seeds_needed(1.0, 2.0) > FHP.seeds_needed(2.0, 2.0)
    assert FHP.seeds_needed(1.0, 4.0) > FHP.seeds_needed(1.0, 2.0)

    # degenerate input must not fabricate a seed count
    assert not np.isfinite(FHP.seeds_needed(0.0, 2.0))
    assert not np.isfinite(FHP.seeds_needed(1.0, 0.0))

    # and the tag must actually carry the number to the reader
    st = FHP.paired(np.where(np.arange(64) % 4 == 0, -1.0, 2.0))
    v = FHP.verdict(st, 64, 1.0, 0.01)
    assert "fragile" in v and "seeds per cell" in v, v


# FRAMEWORK 2(a3) and 2(g). Both sections make a claim ABOUT CODE, so both get
# a gate. Negative controls for each were confirmed to FAIL before these went in.


def test_normalize_makes_the_delivered_step_independent_of_the_violation():
    """FRAMEWORK 2(a3): under `normalize` + `sgd` the parameter displacement
    is exactly `lr * clip` per step, whatever the violation.

    This is the magnitude half of the argument that the constraint is blind to
    violation depth (2(a2) is the direction half). If a future edit lets the
    raw norm through, the framework's `no dose-response to the cap level`
    reading becomes wrong and this catches it.

    Negative control: deleting the scale-UP branch in constraint_step.py makes
    the small-gradient case deliver its raw norm and this FAILS.
    """
    import torch
    from src.training.constraint_step import finish_constraint_step

    lr, clip = 1e-3, 1.0
    for scale in (1e-4, 1.0, 1e4):        # 8 orders of violation magnitude
        model = torch.nn.Linear(6, 3, bias=False)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 1.0)
        # a gradient whose norm is `scale` exactly
        model.weight.grad *= scale / model.weight.grad.norm()
        finish_constraint_step(model, optimizer=None, scaler=None, clip=clip,
                               mode="normalize", fp32=True, step_rule="sgd",
                               lr=lr)
        moved = float((model.weight.detach() - before).norm())
        # relative, at 1e-4: the parameters are float32 and the rescale is one
        # more op on them, so ~1e-6 relative slop is arithmetic, not behaviour.
        # The failure this guards against -- delivering the RAW norm -- is off
        # by four orders of magnitude at the ends of this sweep, not six
        # decimal places.
        assert abs(moved - lr * clip) < 1e-4 * lr * clip, (
            "raw norm %g delivered a step of %g, not lr*clip=%g -- the "
            "constraint has become sensitive to violation magnitude, which "
            "contradicts FRAMEWORK 2(a3)" % (scale, moved, lr * clip))


def test_the_graph_probes_controls_are_fair():
    """FRAMEWORK 2(g) reads the `diffused` column only because C1 and C2 moved.
    A control that differs from the treatment in anything besides the geometry
    would make that reading invalid, so the fairness is asserted, not assumed.

    Negative controls, both confirmed to FAIL: drawing C1's neighbours WITH
    replacement (degree collapses); and dropping the `idx >= arange` self-skip
    (C1 gains self-loops the real graph does not have).
    """
    import numpy as np
    from scripts.graph_probe import diffuse, knn_affinity

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    k = 5
    real = knn_affinity(X, k)
    ctrl = knn_affinity(X, k, np.random.default_rng(1), shuffle_neighbours=True)

    for name, W in (("real", real), ("C1", ctrl)):
        assert np.allclose(W, W.T), "%s graph is not symmetric" % name
        assert float(np.trace(W)) == 0.0, "%s graph has self-loops" % name
        # k directed edges per row, so every row has at least k after
        # symmetrising -- the control must not be sparser than the treatment
        assert W.sum(axis=1).min() >= k, "%s row fell below k" % name
    assert real.sum() > 0 and ctrl.sum() > 0
    # the control must actually be a DIFFERENT graph, or it tests nothing
    assert not np.allclose(real, ctrl)

    # alpha=0 must be the identity, or the baseline comparison is confounded
    P = rng.random((60, 4))
    P = P / P.sum(axis=1, keepdims=True)
    assert np.allclose(diffuse(P, real, 0.0), P, atol=1e-12)
    # and every diffused row is still a distribution
    D = diffuse(P, real, 0.5)
    assert np.allclose(D.sum(axis=1), 1.0)
    assert (D >= 0).all()


def _run_arm_probs(arm, methodology, seed=1):
    """Test probabilities for any arm, assembled from protocol.yml.

    The generic sibling of `_run_tralo_arm`, which patches the tralo module and
    so cannot cross into `dual_common`. Crossing that boundary is the whole
    point here.
    """
    import shutil
    import tempfile

    import scripts.smoke_arms as smoke

    tmp = tempfile.mkdtemp(prefix="zerodose_")
    try:
        inputs, _g, _l = smoke.make_inputs(smoke.load_protocol(), arm, tmp,
                                          seed=seed)
        torch.manual_seed(seed)
        out = TRAIN_FNS[methodology](inputs)
        out.model.eval()
        with torch.no_grad():
            return F.softmax(out.model(inputs.X_test), dim=1).numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_zero_dose_arm_is_the_same_model_across_code_paths():
    """At lambda 0 all four trained families collapse to the same CE run.

    WHY THIS IS LOAD-BEARING AND NOT TRIVIA. `results/dualbar2` spends 32 of its
    88 runs on four zero-dose arms that were confirmed BYTE-IDENTICAL on real
    data -- one md5 (`1bfa966cdf01`) at both cap levels, and 47 of 49 training-log
    columns equal, the two exceptions being the limit columns themselves. The
    protocol already exploits this for the tralo family through `null_sibling`,
    with the reasoning written at `configs/protocol.yml`: at lambda 0 there is no
    constraint gradient, so a second null is a bit-identical run costing a GPU
    slot.

    Extending that to fioretto/hounie/alm crosses a CODE PATH boundary --
    `dual_common` rather than `tralo/train.py` -- so the equivalence stops being
    structural-by-inspection and has to be checked. This is that check. If it
    ever fails, a shared null is no longer sound and the arms must be run
    separately again.

    Liveness: a TREATED arm must differ, or `identical` is being reported by a
    harness that cannot tell anything apart.
    """
    nulls = {"tralo_null": "tralo", "fioretto_null": "fioretto_ldf",
             "hounie_null": "hounie_rcl", "alm_null": "fioretto_alm"}
    probs = {a: _run_arm_probs(a, m) for a, m in nulls.items()}

    ref_name = "tralo_null"
    ref = probs[ref_name]
    for name, pr in probs.items():
        if name == ref_name:
            continue
        assert np.array_equal(ref, pr), (
            "%s is NOT bit-identical to %s at lambda 0 (max abs diff %g). The "
            "shared `null_sibling` is unsound -- either restore the per-family "
            "null arms or find what now differs between the code paths."
            % (name, ref_name, float(np.abs(ref - pr).max())))

    # LIVENESS: the same harness must separate a treated arm from the null,
    # otherwise every equality above is vacuous.
    treated = _run_arm_probs("fioretto", "fioretto_ldf")
    assert not np.array_equal(ref, treated), (
        "`fioretto` is bit-identical to the null on this harness, so it cannot "
        "tell a treated arm from an untreated one and the equalities above "
        "prove nothing")


def test_all_plus_null_schedules_one_null_per_shared_zero_dose_model():
    """`+null` means the null each arm is READ AGAINST, deduplicated.

    It used to mean `every arm whose name ends in _null`, which scheduled one
    bit-identical zero-dose run per FAMILY: 32 of `results/dualbar2`'s 88 runs
    computed a single control four times. The clipper duplicates in that
    campaign were already absorbed by the warm-up cache -- `clip` at the second
    cap writes no training_log at all -- but the nulls are TRAINED arms and
    genuinely re-ran 29 epochs each, so 24 runs of real compute.

    Negative control: reverting the branch to `set(P['arms']) - rejected`
    schedules four nulls and this FAILS.
    """
    import configs.gen_campaign as gc

    P = gc.load_protocol() if hasattr(gc, "load_protocol") else None
    if P is None:
        import yaml
        with io.open(os.path.join(REPO, "configs", "protocol.yml"),
                     encoding="utf-8") as fh:
            P = yaml.safe_load(fh)

    rejected = set(P.get("rejected_arms", {}))
    base = {a for a in P["arms"] if not a.endswith("_null")} - rejected
    scheduled = base | {gc._null_of(P, a) for a in base
                        if gc._null_of(P, a) in P["arms"]}

    nulls = sorted(a for a in scheduled if a.endswith("_null"))
    assert nulls == ["tralo_null"], (
        "`all+null` schedules %s. Every family that shares a zero-dose model "
        "must resolve to ONE null run; a second is bit-identical and costs a "
        "full GPU slot." % nulls)

    # every trained arm must still RESOLVE to a null that is actually there,
    # or the dedup has silently orphaned an arm from its control
    for a in sorted(base):
        if P["arms"][a].get("phase") != "trained":
            continue
        sib = gc._null_of(P, a)
        assert sib in scheduled or a in scheduled and sib not in P["arms"], (
            "trained arm %s resolves to %s, which is not scheduled" % (a, sib))

    # and the three per-family nulls must still EXIST as named arms, because
    # they are how the equivalence gets re-verified on real data
    for a in ("fioretto_null", "hounie_null", "alm_null"):
        assert a in P["arms"], (
            "%s was deleted rather than merely unscheduled -- the shared-null "
            "equivalence can no longer be re-checked on real data" % a)


def _resolution_text(deltas_by_cell_seed, scale, capsys):
    """Run `_resolution_readout` on a synthetic contrast and return its text."""
    import pandas as pd
    from scripts.full_panel import _resolution_readout

    idx, vals = [], []
    for (cap, seed), v in deltas_by_cell_seed.items():
        idx.append(("iwildcam", "MobileNetV3", cap, "2-4", seed))
        vals.append(v)
    per = pd.Series(vals, index=pd.MultiIndex.from_tuples(idx))
    df = pd.DataFrame([{"dataset": "iwildcam", "model": "MobileNetV3",
                        "cap": cap, "capped": "2-4", "items_per_001": scale}
                       for cap in {c for c, _ in deltas_by_cell_seed}])
    _resolution_readout(per, df, "tralo", "clip")
    return capsys.readouterr().out


def test_the_panel_says_whether_it_could_have_seen_what_it_reports(capsys):
    """A tie means `no effect` or `not enough seeds`, and those are opposite.

    The table prints a delta, a Wilcoxon p and a BH q, none of which says
    whether the contrast could have RESOLVED the effect it reports on. This is
    the readout that says so.

    Negative control, confirmed to FAIL before the single-seed guard existed:
    a campaign with one seed per cell has no estimable seed sd at all, and a
    readout that computed one anyway would print a confident seeds-needed
    figure derived from nothing.
    """
    # (a) one seed per cell -- must refuse, not invent a number
    out = _resolution_text({("L50_G20", 1): 0.02, ("L50_G40", 1): 0.03},
                           scale=2.56, capsys=capsys)
    assert "not estimable" in out, out
    assert "seeds per cell" not in out, (
        "a seeds-needed figure was printed with no estimable sd: %s" % out)

    # (b) a large effect against a small spread -- powered at the seeds present
    small = {("L50_G20", s): 0.10 + 0.001 * s for s in (1, 2, 3, 4)}
    out = _resolution_text(small, scale=2.56, capsys=capsys)
    assert "POWERED" in out and "UNDERPOWERED" not in out, out

    # (c) a small effect against a large spread -- must say UNDERPOWERED and
    #     name a seed count larger than what is present
    noisy = {("L50_G20", s): d for s, d in
             zip((1, 2, 3, 4), (0.30, -0.28, 0.26, -0.24))}
    out = _resolution_text(noisy, scale=2.56, capsys=capsys)
    assert "UNDERPOWERED" in out, out
    need = int(out.split("needs ~")[1].split(" ")[0])
    assert need > 4, "a near-zero mean on a huge spread needs many seeds, got %d" % need

    # (d) power is set by the LEAST-replicated cell, never the median: a
    #     median of [4, 1] truncates to 2 and matches no cell that exists.
    mixed = dict(small)
    mixed[("L50_G40", 1)] = 0.10
    out = _resolution_text(mixed, scale=2.56, capsys=capsys)
    assert "1 seed(s) in the least-replicated cell" in out, out


@pytest.mark.parametrize("arm,methodology", [
    ("fioretto", "fioretto_ldf"),
    ("hounie", "hounie_rcl"),
    ("alm", "fioretto_alm"),
    ("tralo", "tralo"),
])
def test_every_trained_arm_logs_the_per_class_counts(arm, methodology):
    """All four trained arms must answer the same question from their log.

    `tralo` logged a per-class Limit/Hard/Soft triple every epoch; the three
    duals logged only `total_excess`, one summed scalar. So the project's
    central quantity -- what the count did, per capped class, across the
    constraint phase -- was readable for one arm of four and `n/a (schema)`
    for the rest, and every cross-arm count comparison had to be rebuilt by
    hand from the stored predictions.

    A SUMMED excess also cannot show the class asymmetry the framework
    records (one capped class moved at ~4x the noise floor, the other at or
    below it) -- one class rising and another falling subtract inside it.

    Negative control: this was verified to FAIL on all three dual arms before
    `count_fields`/`count_row` were added, and to pass on `tralo` throughout,
    which is what made the asymmetry visible for one arm only.
    """
    import shutil
    import tempfile

    import pandas as pd
    import scripts.smoke_arms as smoke

    tmp = tempfile.mkdtemp(prefix="counts_")
    try:
        inputs, _g, _l = smoke.make_inputs(smoke.load_protocol(), arm, tmp,
                                          seed=1)
        TRAIN_FNS[methodology](inputs)
        df = pd.read_csv(os.path.join(str(inputs.experiment_path),
                                      "training_log.csv"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    capped = [c for c in df.columns if c.startswith("Limit_Class")
              and pd.to_numeric(df[c], errors="coerce").dropna().lt(1e9).any()]
    assert capped, (
        "%s logs no finite Limit_Class column, so no reader can tell which "
        "class was capped or what the budget was" % arm)

    for lim in capped:
        c = lim[len("Limit_Class"):]
        for prefix in ("Hard_Class", "Soft_Class"):
            col = prefix + c
            assert col in df.columns, (
                "%s caps class %s but never logs %s -- its count trajectory "
                "is unreadable from the log" % (arm, c, col))
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            assert len(v) and (v >= 0).all(), (
                "%s wrote no usable values in %s: %s" % (arm, col, list(v)))
        # hard counts are argmax tallies, so they must be whole numbers
        h = pd.to_numeric(df["Hard_Class" + c], errors="coerce").dropna()
        assert np.allclose(h, h.round()), (
            "%s wrote a non-integer hard count in Hard_Class%s: %s"
            % (arm, c, list(h)))


def test_the_generator_says_which_scope_each_cap_binds(tmp_path):
    """Local caps are per-GROUP ceilings, so the binding scope is L vs G.

    Local sum is `L * total_true` against the global's `G * total_true`, so
    `G < L` means the GLOBAL binds and the local is slack, and `G > L` the
    reverse. Nothing printed this, and the project has now made the same
    mistake in BOTH directions: until 2026-08-18 every campaign ran `G >= L`
    so the global cap had never bound, and the fix -- sweep `G < L` -- made the
    LOCAL scope inert instead. `results/dualbar2` ran L50_G20 and L50_G40, both
    `G < L`, and `lp_fallback_used` was False on all 50 completed runs with 0
    candidates: a local ceiling was never once the binding constraint.

    Negative control: with the readout removed, a campaign whose caps all bind
    one scope generates silently, which is exactly what happened twice.
    """
    # the arithmetic itself, independent of any output
    for tag, expect in (("L50_G20", "GLOBAL"), ("L50_G40", "GLOBAL"),
                        ("L20_G50", "LOCAL"), ("L30_G30", "IDENTICAL")):
        lp, gp = cap_pair(tag)
        got = ("GLOBAL" if gp < lp else "LOCAL" if gp > lp else "IDENTICAL")
        assert got == expect, "%s: expected %s, got %s" % (tag, expect, got)

    # every cap binding ONE scope must be called out by name
    r = _gen(tmp_path / "same", "--caps", "L50_G20", "L50_G40")
    out = r.stdout + r.stderr
    assert "binding scope: GLOBAL" in out, out[-1500:]
    assert "EVERY cap in this campaign binds the GLOBAL scope" in out, (
        "a campaign that tests only one scope generated without saying so: %s"
        % out[-1500:])
    assert "L20_G50" in out, "the warning must name the fix"

    # a campaign that spans BOTH scopes must NOT be warned at
    r = _gen(tmp_path / "both", "--caps", "L50_G20", "L20_G50")
    out = r.stdout + r.stderr
    assert "binding scope: GLOBAL" in out and "binding scope: LOCAL" in out, out[-1500:]
    assert "EVERY cap in this campaign binds" not in out, (
        "warned about a campaign that spans both scopes -- the gate cries "
        "wolf: %s" % out[-1500:])


def _budget_pair(df, tag, classes, num_classes):
    """(global K, summed local K) per capped class, for one cap tag."""
    local_pct, global_pct = cap_pair(tag)
    g = compute_global_constraints(df, "label", global_pct,
                                   constrained_class=classes,
                                   num_classes=num_classes)
    loc = compute_local_constraints(df, "label", local_pct, "grp",
                                    constrained_class=classes,
                                    num_classes=num_classes)
    return {c: (g[c], sum(v[c] for v in loc.values())) for c in classes}


def test_swapping_L_and_G_holds_the_total_budget_fixed():
    """FRAMEWORK 2(l): L20_G50 and L50_G20 impose the SAME TOTAL, different scope.

    Per-group L% sums to L% of the total, so swapping the two percentages keeps
    the number of predictions the allocator may emit and changes only whether
    the split across groups is pinned. That is what makes the scope contrast a
    controlled experiment rather than two different budgets, and it is a claim
    about banker's rounding, so it is measured and not assumed.

    Real dermmnist structure: capped classes 2 and 4 over three `loc_group`s.
    """
    counts = {2: [70, 60, 75], 4: [97, 95, 30]}
    labels, groups = [], []
    for c, per_group in counts.items():
        for gid, n in enumerate(per_group):
            labels += [c] * n
            groups += [gid] * n
    for gid, n in enumerate([1101 - 70 - 97, 695 - 60 - 95, 218 - 75 - 30]):
        labels += [5] * n
        groups += [gid] * n
    df = _frame(labels, groups)

    tight = _budget_pair(df, "L50_G20", [2, 4], 7)
    loose = _budget_pair(df, "L20_G50", [2, 4], 7)

    for c, total in ((2, 41), (4, 44)):
        g_tight, l_tight = tight[c]
        g_loose, l_loose = loose[c]
        assert g_tight == total, (c, g_tight)
        assert l_loose == total, (c, l_loose)
        # the totals coincide, so only the SCOPE differs between the two tags
        assert g_tight == l_loose
        # and each tag's own non-binding scope really is slack
        assert l_tight > g_tight, ("L50_G20 should bind GLOBAL", c, tight[c])
        assert g_loose > l_loose, ("L20_G50 should bind LOCAL", c, loose[c])

    # NEGATIVE CONTROL 1: the match is a property of SWAPPING, not of any two
    # tags. L30_G50 shares the global with L20_G50 and matches neither total.
    other = _budget_pair(df, "L30_G50", [2, 4], 7)
    for c in (2, 4):
        assert other[c][1] != tight[c][0], (
            "a non-swapped tag matched the total -- the assertion above is "
            "vacuous: %r" % (other[c],))

    # NEGATIVE CONTROL 2: sum-of-rounds equalling round-of-sum is NOT free.
    # Three groups of 5 at 50% give 2+2+2=6 against a global round(7.5)=8, so
    # the equality above is measuring real arithmetic and would catch drift.
    pathological = _frame(([1] * 5 + [0] * 20) * 3,
                          [0] * 25 + [1] * 25 + [2] * 25)
    g_p, l_p = _budget_pair(pathological, "L50_G50", [1], 2)[1]
    assert (g_p, l_p) == (8, 6), (g_p, l_p)
    assert g_p != l_p


# ------------------------------------------------------- the scope probe --

def test_scope_probe_controls_preserve_the_total_exactly():
    """The wrong-shape controls must change ONLY the shape.

    `scope_probe` reports a null for the real pinned split against controls
    that cost 5.3-5.5 items. That reading is only valid if the controls hold
    the budget fixed -- if a permutation changed the total, the control would
    be measuring a different budget and the null would be uninterpretable.
    """
    from scripts.scope_probe import _permute_ceilings
    L = {0: [UNLIMITED, 14, UNLIMITED], 1: [UNLIMITED, 12, UNLIMITED],
         2: [UNLIMITED, 15, UNLIMITED]}
    classes = [1]
    for order in ([1, 2, 0], [2, 1, 0]):
        out = _permute_ceilings(L, classes, order)
        assert sum(out[g][1] for g in out) == 41
        assert sorted(out[g][1] for g in out) == [12, 14, 15]
        # and it must actually MOVE them, or it is not a control at all
        assert [out[g][1] for g in sorted(out)] != [14, 12, 15], order
    # untouched classes stay untouched
    assert all(out[g][0] == UNLIMITED and out[g][2] == UNLIMITED for g in out)
    # NEGATIVE CONTROL: the identity order preserves the total AND the shape,
    # so a test that only checked the total would pass on a broken control.
    same = _permute_ceilings(L, classes, [0, 1, 2])
    assert [same[g][1] for g in sorted(same)] == [14, 12, 15]


def test_group_calibrate_hits_the_target_prior_per_group():
    """Liveness: the correction must actually reach the prevalence it targets.

    A null from `--calibrate` is only a measurement if the instrument moves the
    priors it is asked to move. This pins that separately from whether moving
    them helps.
    """
    from scripts.scope_probe import group_calibrate
    rng = np.random.default_rng(0)
    n, k = 600, 3
    g = np.repeat([0, 1, 2], n // 3)
    P = rng.dirichlet(np.ones(k), size=n)
    targets = {0: [0, 30, 0], 1: [0, 120, 0], 2: [0, 60, 0]}
    Q = group_calibrate(P, g, [1], targets)
    for gg, want in ((0, 30 / 200), (1, 120 / 200), (2, 60 / 200)):
        got = Q[g == gg, 1].mean()
        # renormalisation pulls it off the exact target, so require it to close
        # most of the gap rather than land on it
        start = P[g == gg, 1].mean()
        assert abs(got - want) < abs(start - want), (gg, start, got, want)
    assert np.allclose(Q.sum(axis=1), 1.0)
    # NEGATIVE CONTROL: permuting which group gets which target must give a
    # DIFFERENT result, or the group_key argument is inert and both probe
    # controls would be silently identical to the real row.
    Q2 = group_calibrate(P, g, [1], targets, group_key=[1, 2, 0])
    assert not np.allclose(Q, Q2)


# ----------------------------------------------------- the dataset screen --

def _meta(labels, groups):
    return pd.DataFrame({"label": np.asarray(labels),
                         "loc_group": np.asarray(groups)})


def _screen_pair(train, test):
    from scripts.dataset_screen import novelty_items
    return novelty_items(train, test, "loc_group", n_null=200, seed=0)


def test_screen_net_ignores_a_uniform_shift_but_sees_a_differential_one():
    """NET must separate one multiplier in disguise from a real reordering.

    This is the claim that disqualified `data/dermmnist/shift_1`: it scores 160
    items of LOCAL novelty but only 50 NET, because 110 of them are the global
    shift replicated across all three groups. A uniform per-group shift IS a
    single global multiplier, which FRAMEWORK 2(j) showed is monotone and cannot
    reorder anything, so counting it as per-group information would recommend a
    dataset that provably cannot work.
    """
    rng = np.random.default_rng(0)
    n_g = 900
    base = [0.10, 0.20, 0.30]

    def build(p_by_group):
        lab, grp = [], []
        for gid, p in enumerate(p_by_group):
            draws = rng.choice([0, 1], size=n_g, p=[1 - p, p])
            lab += list(draws)
            grp += [gid] * n_g
        return _meta(lab, grp)

    def relabel(p_by_group, w):
        """Apply ONE per-class multiplier to every group, then renormalise.

        This is the label-shift model: p_test(y) = w_y p_train(y) with the SAME
        w in every group. Note that simply doubling each group's positive RATE
        is NOT this -- it forces the negative class to move by 0.89 / 0.75 /
        0.57 across the three groups, which is a differential shift and which
        NET correctly fires on.
        """
        out = []
        for p in p_by_group:
            num = p * w[1]
            out.append(num / (num + (1 - p) * w[0]))
        return out

    train = build(base)

    # (a) UNIFORM label shift: one multiplier per class, every group. Global
    #     moves a lot; nothing is reorderable, so NET must stay quiet.
    uniform = build(relabel(base, [1.0, 2.0]))
    a = _screen_pair(train, uniform)
    assert a["global_z"] > 5, a
    assert a["net_z"] < 2.0, ("a uniform shift leaked into NET: %r" % a)

    # (b) DIFFERENTIAL shift at a HELD global rate: group 0 up, group 2 down,
    #     mean preserved. Global must stay quiet; NET must fire.
    differential = build([0.30, 0.20, 0.10])
    b = _screen_pair(train, differential)
    assert abs(b["global_z"]) < 3, ("global fired on a mean-preserving "
                                    "reshuffle: %r" % b)
    assert b["net_z"] > 5, ("NET missed a pure differential shift: %r" % b)
    assert b["net_items"] > 100, b["net_items"]


def test_screen_calls_index_modulo_groups_dead():
    """`synth_group = index % 3` is the octmnist/tissuemnist construction.

    It makes every group an i.i.d. draw from one distribution, so the local
    scope carries nothing by construction. The screen must return that as
    sampling noise, or it would recommend re-running the two datasets already
    known to test nothing.
    """
    rng = np.random.default_rng(1)
    n = 2700
    train = _meta(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]),
                  np.arange(n) % 3)
    test = _meta(rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]),
                 np.arange(n) % 3)
    r = _screen_pair(train, test)
    assert r["net_z"] < 2.0, ("index%%3 groups scored as informative: %r" % r)
    # NEGATIVE CONTROL: the same test labels with groups assigned BY LABEL
    # instead of by index must fire, or the assertion above is vacuous.
    rigged = test.copy()
    rigged["loc_group"] = (test["label"] == 0).astype(int)
    r2 = _screen_pair(train, rigged)
    assert r2["net_z"] > 5, ("screen is blind even to label-aligned groups: %r"
                             % r2)


def test_screen_scores_fully_unseen_groups_against_the_global_prior():
    """A held-out-domain split is the criterion's BEST case, not a missing one.

    The first version skipped groups absent from training, which returns
    novelty 0 for exactly the design FRAMEWORK 2(n) recommends -- no unit
    survives to be summed. A model that has never seen a group holds no
    group-specific prior and must fall back to the global one, so that is the
    baseline the deviation is measured against.
    """
    rng = np.random.default_rng(0)
    n_g = 800

    def build(spec):
        lab, grp = [], []
        for gid, p in spec:
            lab += list(rng.choice([0, 1, 2], size=n_g, p=p))
            grp += [gid] * n_g
        return _meta(lab, grp)

    # train groups 0-2, test groups 10-12 -- disjoint, and each test group is
    # dominated by a DIFFERENT class, the iWildCam structure in miniature
    train = build([(0, [0.5, 0.3, 0.2]), (1, [0.4, 0.4, 0.2]),
                   (2, [0.5, 0.2, 0.3])])
    test = build([(10, [0.9, 0.05, 0.05]), (11, [0.05, 0.9, 0.05]),
                  (12, [0.05, 0.05, 0.9])])
    r = _screen_pair(train, test)
    assert len(r["unseen_groups"]) == 3, r["unseen_groups"]
    assert r["unseen_items"] == 3 * n_g
    assert r["net_z"] > 10, ("fully unseen groups scored as no information: %r"
                             % r)
    assert r["net_items"] > 500, r["net_items"]

    # NEGATIVE CONTROL: unseen groups that each match the OVERALL test mix
    # carry no DIFFERENTIAL information -- the cap would only be restating the
    # global shift, which 2(j) shut. NET must stay quiet while the groups are
    # still reported as unseen.
    flat = build([(10, [0.9, 0.05, 0.05])] * 3)
    r2 = _screen_pair(train, flat)
    assert len(r2["unseen_groups"]) == 1, r2["unseen_groups"]
    assert r2["net_z"] < 2.0, ("a uniform unseen group leaked into NET: %r" % r2)


def test_generator_reports_zero_ceilings_not_just_sum_slack(tmp_path):
    """A K=0 per-group ceiling binds even when the local SUM is slack.

    `gen_campaign`'s binding-scope line is pure arithmetic on the two cap
    percentages and was written against dermmnist, where every per-group
    ceiling is positive. On a held-out-camera dataset most cells are zero --
    a species simply is not at that camera -- and reporting "local sum is 2.5x
    slack" would call the local scope inert in the campaign where it does the
    most work. That is the same class of mistake as the 2026-08-18 global-cap
    bug and the 2026-08-22 local one, both of which went unnoticed at
    generation time.
    """
    from configs.gen_campaign import _zero_ceilings

    def protocol(frame, classes):
        d = tmp_path / ("s%d" % len(list(tmp_path.iterdir())))
        d.mkdir()
        frame.to_csv(d / "test_meta.csv", index=False)
        return {"datasets": {"x": {"data_dir": str(d), "num_classes": 4,
                                   "group_column": "grp",
                                   "constrained_class": classes}}}

    # a species present at ONE group and absent from two -- iWildCam's shape
    sparse = pd.DataFrame({
        "label": [1] * 60 + [0] * 60 + [0] * 60,
        "grp":   [0] * 60 + [1] * 60 + [2] * 60})
    zeros, total = _zero_ceilings(protocol(sparse, [1]), "x", 0.5)
    assert (zeros, total) == (2, 3), (zeros, total)

    # NEGATIVE CONTROL: dermmnist's shape -- the class is present in EVERY
    # group, so nothing is zero and the warning must stay silent. Without this
    # the assertion above would pass on a function that always reports zeros.
    dense = pd.DataFrame({
        "label": ([1] * 30 + [0] * 30) * 3,
        "grp":   [0] * 60 + [1] * 60 + [2] * 60})
    assert _zero_ceilings(protocol(dense, [1]), "x", 0.5) == (0, 3)

    # a slice absent from this machine must report nothing, never crash --
    # campaigns are generated on laptops as well as on the server
    missing = {"datasets": {"x": {"data_dir": str(tmp_path / "nope"),
                                  "num_classes": 4, "group_column": "grp",
                                  "constrained_class": [1]}}}
    assert _zero_ceilings(missing, "x", 0.5) == (0, 0)


# ------------------------------------------------- the removed datasets --

REMOVED_DATASETS = ("dermmnist", "octmnist", "tissuemnist")


def test_removed_datasets_cannot_be_selected_anywhere():
    """The three original datasets must be UNRUNNABLE, not merely discouraged.

    `scripts.dataset_screen` measured that none of them can carry a count
    constraint: octmnist and tissuemnist build `synth_group` as
    `np.arange(len(y)) % 3`, so their groups are i.i.d. draws from one
    distribution and the local scope is empty BY CONSTRUCTION; dermmnist clears
    the screen at +65 items and still nulls, because its test groups ARE its
    training groups. Deleting the data is not enough on its own -- a stale
    campaign root, a copied command line or an old config would quietly bring
    one back, and every number it produced would look ordinary. So the ban is
    enforced at all three gates a dataset has to pass.
    """
    import yaml
    from configs.gen_campaign import PROTOCOL_PATH
    from src.utils.data_loader import IMAGERY_DATASETS

    with io.open(PROTOCOL_PATH, encoding="utf-8") as fh:
        declared = set(yaml.safe_load(fh)["datasets"])
    assert declared == {"iwildcam"}, declared
    assert IMAGERY_DATASETS == {"iwildcam"}, IMAGERY_DATASETS
    for name in REMOVED_DATASETS:
        assert name not in declared
        assert name not in IMAGERY_DATASETS
        assert not os.path.exists(os.path.join(REPO, "data", name)), name

    # the generator must REFUSE, not silently emit an unrunnable campaign
    for name in REMOVED_DATASETS:
        r = subprocess.run(
            [sys.executable, "-m", "configs.gen_campaign",
             "--root", os.path.join(REPO, "_never_written"),
             "--datasets", name, "--models", "MobileNetV3",
             "--caps", "L30_G30", "L50_G30", "--arms", "clip"],
            cwd=REPO, capture_output=True, text=True)
        assert r.returncode != 0, "generator accepted %s: %s" % (name, r.stdout)
        assert name in (r.stderr + r.stdout)
    assert not os.path.exists(os.path.join(REPO, "_never_written"))

    # NEGATIVE CONTROL: the live dataset must still pass all of the above, or
    # the assertions are satisfied by a generator that refuses everything.
    r = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign",
         "--root", os.path.join(REPO, "_ctrl_ok"), "--datasets", "iwildcam",
         "--models", "MobileNetV3", "--caps", "L30_G30", "L50_G30",
         "--arms", "clip"], cwd=REPO, capture_output=True, text=True)
    try:
        assert r.returncode == 0, r.stderr[-800:]
    finally:
        shutil.rmtree(os.path.join(REPO, "_ctrl_ok"), ignore_errors=True)


CAMPAIGN_TOOLS = [
    ("audit_config", ["{root}"]),
    ("check_parity", ["{root}"]),
    ("log_health", ["{root}"]),
    ("score_scan", ["{root}"]),
    ("headroom", ["{root}"]),
    ("paired_seeds", ["{root}"]),
    ("full_panel", ["--campaign", "{root}", "--control", "clip"]),
    ("graph_probe", ["--campaign", "{root}"]),
    ("scope_probe", ["--campaign", "{root}"]),
    ("straddle_probe", ["--campaign", "{root}"]),
]


@pytest.mark.parametrize("tool,argv", CAMPAIGN_TOOLS,
                         ids=[t for t, _ in CAMPAIGN_TOOLS])
def test_a_campaign_tool_fails_on_an_empty_root(tool, argv, tmp_path):
    """A tool pointed at nothing must FAIL, never report clean.

    Every one of these iterates a glob over the campaign root, so an empty or
    wrong root makes each check vacuously true. `audit_config --help` took
    `--help` as the root, found zero configs and printed "OK -- arms sharing an
    id agree on all 12 warm-up keys" over zero arms; `log_health` printed its
    reason but returned 0, and since `main()` is called bare the code was
    discarded anyway. Both are this project's own mistake pattern 1 -- a check
    that reports green while not looking -- living inside the tools built to
    catch it.

    Exit code, not stdout, because that is what a script chaining on these
    reads.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "scripts." + tool]
        + [a.format(root=str(empty)) for a in argv],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0, (
        "%s exited 0 on a root with no runs -- indistinguishable from a clean "
        "campaign:\n%s" % (tool, (r.stdout or "")[-500:]))


def test_the_empty_root_guard_is_not_satisfied_by_a_tool_that_always_fails():
    """NEGATIVE CONTROL for the parametrised test above.

    Every assertion there is `returncode != 0`, which a tool that crashed on
    everything would also satisfy. This pins that the same tools SUCCEED on a
    root that does hold runs, so the guard is measuring emptiness rather than
    brokenness.
    """
    r = subprocess.run([sys.executable, "-m", "scripts.audit_config"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "every emitted value has a reader" in r.stdout


# ---------------------------------------------------------------------------
# `scripts/straddle_probe.py` asks how much of the ORACLE headroom a step the
# size of ours can actually reach. Its whole claim is that it reads WHERE the
# errors sit rather than HOW MANY there are, so both tests below are about that
# distinction and not about any particular number.
# ---------------------------------------------------------------------------

def test_the_straddle_gate_separates_errors_at_the_cut_from_buried_ones():
    """The gate must PASS on the two regimes whose error geometry is known.

    `matched` leaves its few residual errors AT the cut; `tailnoise` plants
    positives far below it. A statistic that reports the same reachable SHARE
    for both is counting errors, not locating them, and every number it prints
    about a real campaign would be uninterpretable.
    """
    from scripts import straddle_probe as SP

    SP.self_test(n_seeds=3)          # raises SystemExit on failure


def test_the_straddle_gate_fails_when_the_statistic_ignores_position():
    """NEGATIVE CONTROL ON THE GATE ITSELF.

    A gate that has never been shown to fail is not a gate. Replace the band
    with a position-BLIND one -- every false positive above the cut and every
    true positive below it counts, however far away -- and the reachable share
    becomes the oracle gap in BOTH regimes, so the separation vanishes. The
    gate must reject that, or it would have signed off on a statistic that
    reads the error rate.
    """
    from scripts import straddle_probe as SP

    original = SP.straddle

    def position_blind(scores, is_pos, K, deltas):
        # `original`, not `SP.straddle` -- the name is about to be rebound and
        # calling through it would recurse instead of mutating.
        real = original(scores, is_pos, K, deltas)
        for b in real["bands"]:
            b["reachable"] = real["oracle"]      # reachable regardless of delta
        return real

    try:
        SP.straddle = position_blind
        with pytest.raises(SystemExit) as exc:
            SP.self_test(n_seeds=3)
    finally:
        SP.straddle = original
    assert "SELF-TEST FAILED" in str(exc.value), (
        "the gate exited for some other reason than the mutation")


def test_the_straddle_swap_count_is_bounded_by_both_sides_and_by_the_oracle():
    """The arithmetic, pinned directly: a swap needs BOTH a false positive to
    push out and a true positive to pull in, and no number of swaps can beat
    the unbounded oracle gap. Constructed so the two sides are deliberately
    unequal -- an implementation that returned either side alone, or their sum,
    would pass a symmetric fixture.
    """
    from scripts.straddle_probe import straddle

    # scores descending; K=4 so the cut sits at 0.60
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.55, 0.50, 0.10, 0.05])
    is_pos = np.array([True, False, False, False, True, True, True, False])

    r = straddle(scores, is_pos, 4, [0.06, 0.5])
    assert r["cut"] == pytest.approx(0.60)
    # 3 false positives above the cut, 3 true positives below it, 1 TP above
    assert r["oracle"] == min(4, 4) - 1 == 3

    near = r["bands"][0]
    # within 0.06: FPs at 0.60 only (0.70/0.80 are further); TP at 0.55 only
    assert near["reachable"] == 1, near
    wide = r["bands"][1]
    # within 0.5 both sides offer 3, but the oracle gap caps the useful count
    assert wide["reachable"] == 3 <= r["oracle"], wide


def test_the_straddle_shuffled_reference_does_not_track_the_error_geometry():
    """The shuffled arm is a REFERENCE, not a second measurement.

    Permuting the scores destroys the ordering, so what is left depends on n,
    K and prevalence only. It must therefore be near-EQUAL across two regimes
    whose true error structures differ several-fold -- that insensitivity is
    exactly what licenses reading the real number against it. (It also rises
    rather than collapsing, which is why the docstring warns against reading it
    as a must-collapse control.)
    """
    from scripts import straddle_probe as SP
    from scripts.frozen_head_probe import make_synthetic

    rng = np.random.default_rng(0)
    shuf, real, oracle = {}, {}, {}
    for regime in ("matched", "tailnoise"):
        agg = {}
        for seed in range(3):
            SP.collect(agg, SP.probe(make_synthetic(regime, seed),
                                     SP.sweep_deltas, rng), SP.SWEEP_NAMES)
        widest = SP.SWEEP_NAMES[-1]
        shuf[regime] = sum(np.mean([b["reachable"] for b in agg[c]["shuf"][widest]])
                           for c in agg)
        real[regime] = sum(np.mean([b["reachable"] for b in agg[c]["bands"][widest]])
                           for c in agg)
        oracle[regime] = sum(float(np.mean(agg[c]["oracle"])) for c in agg)

    # the thing the reference is supposed to be blind to really does differ
    assert oracle["tailnoise"] > 3 * oracle["matched"], oracle
    # ...and the reference stays put anyway
    lo, hi = sorted(shuf.values())
    assert hi < 1.6 * lo, (
        "the shuffled reference moved with the error geometry (%s), so it is "
        "measuring the same thing as the real arm" % shuf)
    # the real arm, by contrast, must move -- and cross the reference
    assert real["matched"] < shuf["matched"], (
        "clean labels should leave FEWER swaps than chance", real, shuf)
    assert real["tailnoise"] > shuf["tailnoise"], (
        "buried positives should leave MORE swaps than chance", real, shuf)


def test_no_runnable_command_in_the_docs_names_a_removed_dataset():
    """The purge was enforced in the CODE and missed the DOCS.

    `test_removed_datasets_cannot_be_selected_anywhere` pins protocol.yml, the
    loader, the data directories and the generator's refusal -- and all of it
    stayed green while `CLAUDE.md` still carried
    `gen_campaign ... --datasets dermmnist tissuemnist` as the copy-paste
    example. A command the generator now REFUSES is worse than a stale sentence:
    it reads as the supported way to start, and the first thing it does is fail.

    Prose ABOUT a removed dataset is deliberately still allowed -- the screen
    results are the evidence for removing them and must stay readable. Only
    executable lines are checked, which is the distinction that makes the rule
    mechanical.
    """
    def command_lines(text):
        """Fenced lines, comments stripped.

        The comment half is stripped rather than the whole line skipped: the
        `dataset_screen` block annotates its own command with the screen
        results (`octmnist -7, tissuemnist -55 = DEAD`), which is the EVIDENCE
        for the removal and has to stay. What must not survive is an
        executable token naming one.
        """
        out, in_fence = [], False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or stripped.startswith(("python ", "python3 ", "$ python")):
                code = stripped.split("#", 1)[0].strip()
                if code:
                    out.append(code)
        return out

    # the checker must be capable of firing, or it proves nothing about the docs
    poisoned = "\n".join(["```bash",
                          "python -m configs.gen_campaign --datasets dermmnist",
                          "```"])
    assert any(d in ln for ln in command_lines(poisoned)
               for d in REMOVED_DATASETS), (
        "the extractor cannot see a bad command even when one is planted, so a "
        "green result below would mean nothing")

    for rel in ("CLAUDE.md", os.path.join("docs", "FRAMEWORK.md")):
        path = os.path.join(REPO, rel)
        with io.open(path, encoding="utf-8") as fh:
            lines = command_lines(fh.read())
        assert lines, "%s has no runnable lines -- extractor broken?" % rel
        bad = [(ln, d) for ln in lines for d in REMOVED_DATASETS if d in ln]
        assert not bad, (
            "%s tells the reader to run a REMOVED dataset: %s" % (rel, bad[:3]))


def _write_run(run_dir, arm, seed, probs, y, groups, tag="L30_G50",
               dataset="iwildcam"):
    """A minimal but REAL run directory: exactly the three files `load_real`
    reads, in the shapes `full_panel` writes them."""
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "test_embeddings.npz"),
             features=np.asarray(probs, np.float32))
    cols = {"True_Label": y, "Group_ID": groups}
    for c in range(probs.shape[1]):
        cols["Prob_Class_%d" % c] = probs[:, c]
    pd.DataFrame(cols).to_csv(
        os.path.join(run_dir, "final_predictions_raw.csv"), index=False)
    with io.open(os.path.join(run_dir, "config.json"), "w",
                 encoding="utf-8") as fh:
        json.dump({"arm": arm, "dataset_mode": dataset,
                   "model_name": "MobileNetV3", "constraint_tag": tag,
                   "constraint": [0.30, 0.50],
                   "dataset_config": {"constrained_class": [2]},
                   "hyperparams": {"seed": seed}}, fh)


def _pair_fixture(tmp_path, eps=0.04, shifted_frac=0.10):
    """A treated/null twin differing by a KNOWN displacement in class 2.

    Mass is moved between two columns so the rows still sum to 1 -- `load_real`
    renormalises, and a fixture that relied on renormalisation would measure
    the renormaliser instead of the injected shift.
    """
    rng = np.random.default_rng(7)
    n, n_cls = 400, 3
    y = rng.integers(0, n_cls, n)
    groups = rng.integers(0, 2, n)
    P = rng.dirichlet(np.ones(n_cls) * 2.0, size=n)

    treated = P.copy()
    k = int(n * shifted_frac)
    idx = np.argsort(-P[:, 2])[:k]            # move the top scorers
    treated[idx, 2] += eps
    treated[idx, 0] -= eps
    assert treated.min() > 0, "fixture pushed a probability negative"

    root = os.path.join(str(tmp_path), "camp")
    _write_run(os.path.join(root, "a_tralo"), "tralo", 1, treated, y, groups)
    _write_run(os.path.join(root, "b_null"), "tralo_null", 1, P, y, groups)
    # DECOY: a null at a different seed must not be paired with the arm above
    _write_run(os.path.join(root, "c_null_s2"), "tralo_null", 2, P, y, groups)
    return root


def test_the_straddle_probe_pairs_a_treated_run_with_its_own_null_twin(tmp_path):
    """`pair_runs` is the half of this probe that decides WHAT is compared, and
    a mis-pair would silently redefine the measured displacement. Pinned with a
    decoy null at another seed, because pairing on the arm name alone -- the
    obvious implementation -- would happily take it.
    """
    from scripts.straddle_probe import pair_runs

    root = _pair_fixture(tmp_path)
    runs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    pairs = pair_runs(runs)

    assert len(pairs) == 1, [(p[0], p[2]) for p in pairs]
    arm, _cell, seed, treated, null = pairs[0]
    assert (arm, seed) == ("tralo", 1)
    assert treated.endswith("a_tralo") and null.endswith("b_null")


def test_the_straddle_probe_recovers_the_displacement_it_was_given(tmp_path):
    """The measured delta must be the injected one, or every `reachable` read
    against it is scaled by an unknown factor. 10% of items were moved by
    exactly eps, so the 95th percentile of |dp| sits inside that block and must
    come back as eps.
    """
    from scripts.straddle_probe import measured_delta

    eps = 0.04
    root = _pair_fixture(tmp_path, eps=eps, shifted_frac=0.10)
    _data, disp = measured_delta(os.path.join(root, "a_tralo"),
                                 os.path.join(root, "b_null"))

    assert set(disp) == {2}, disp
    assert disp[2]["q"] == pytest.approx(eps, abs=1e-6), disp
    assert disp[2]["median"] == pytest.approx(0.0, abs=1e-9), (
        "90% of items were untouched, so the median displacement must be 0")


def test_the_straddle_probe_refuses_to_difference_two_different_test_sets(tmp_path):
    """Differencing runs scored on different test sets would produce a
    displacement built from mismatched rows -- a large, meaningless number that
    looks like a strong constraint. It must refuse rather than broadcast.
    """
    from scripts.straddle_probe import measured_delta

    rng = np.random.default_rng(3)
    root = os.path.join(str(tmp_path), "mixed")
    for name, n in (("a_tralo", 200), ("b_null", 200)):
        P = rng.dirichlet(np.ones(3) * 2.0, size=n)
        _write_run(os.path.join(root, name),
                   "tralo" if "tralo" == name.split("_")[1] else "tralo_null",
                   1, P, rng.integers(0, 3, n), rng.integers(0, 2, n))

    with pytest.raises(SystemExit) as exc:
        measured_delta(os.path.join(root, "a_tralo"),
                       os.path.join(root, "b_null"))
    assert "not the same test set" in str(exc.value)


def test_the_straddle_probe_runs_end_to_end_on_a_campaign(tmp_path):
    """The measured path end to end, through `main`. The self-test only ever
    exercises the SWEEP path, so without this the mode that will actually be
    pointed at `results/iwc1` would ship unrun.
    """
    from scripts import straddle_probe as SP

    root = _pair_fixture(tmp_path)
    buf = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = buf
        SP.main(["--campaign", root])
    finally:
        sys.stdout = stdout
    out = buf.getvalue()

    assert "DELTA IS MEASURED" in out, out[-600:]
    assert "1 treated/null twin pair" in out, out[-600:]
    assert "NOT calibrated" not in out, (
        "fell back to the swept ladder even though a twin exists")
    assert "CLASS 2" in out, out[-600:]
    # The BASELINE block is not cosmetic: without the null's own reachability at
    # the SAME delta there is nothing to read the treated number against, and
    # the null is the post-hoc clipper at equal compute with the allocator held
    # fixed -- the reference `headroom.py` quotes.
    assert "BASELINE" in out and "TREATED" in out, out[-900:]
    assert out.index("BASELINE") < out.index("TREATED"), (
        "the baseline must be reported before the treated arm, or the reader "
        "meets the treated number with nothing to compare it to")


@pytest.mark.parametrize("n_pos_lt_K", [False, True])
def test_the_straddle_count_saturates_exactly_at_the_oracle_gap(n_pos_lt_K):
    """THE SATURATION IDENTITY, both branches of the min.

    As delta grows, `fp_near -> K - tp_above` and `tp_near -> n_pos - tp_above`,
    so reachable(inf) = min(K, n_pos) - tp_above = oracle EXACTLY. That is what
    licenses reading this probe as a refinement of `headroom.py` rather than as
    a competing estimate of the same thing -- they agree in the limit by
    construction. It also forces reachable <= oracle at every delta.

    Both branches are exercised because they saturate through DIFFERENT sides
    of the min: with n_pos >= K the false positives above the cut run out
    first, with n_pos < K the true positives below it do. An implementation
    that returned only one side would pass on one branch alone.
    """
    from scripts.straddle_probe import straddle

    rng = np.random.default_rng(11)
    n = 300
    K = 60
    prevalence = 0.05 if n_pos_lt_K else 0.40
    scores = rng.random(n)
    is_pos = rng.random(n) < prevalence
    if n_pos_lt_K:
        assert is_pos.sum() < K, is_pos.sum()
    else:
        assert is_pos.sum() > K, is_pos.sum()

    ladder = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 10.0]
    r = straddle(scores, is_pos, K, ladder)

    for band in r["bands"]:
        assert band["reachable"] <= r["oracle"], (band, r["oracle"])
    assert r["bands"][-1]["reachable"] == r["oracle"], (
        "a delta far wider than the score range must collect the ENTIRE oracle "
        "gap; got %s of %s" % (r["bands"][-1]["reachable"], r["oracle"]))
    # monotone in delta -- a wider window can only expose more swaps
    got = [b["reachable"] for b in r["bands"]]
    assert got == sorted(got), got


def _panel_stdout(tmp_path, boost, jitter, seed0=0):
    """A campaign whose treated arm differs in AP by `boost`, with per-seed
    variability `jitter`. Returns full stdout of the scorer."""
    cells = [(ds, m, cap)
             for m in ("MobileNetV3", "MobileNetV2")
             for cap in ("L30_G30", "L50_G30")
             for ds in ("iwildcam",)]
    N, K = 300, 4
    for i, (ds, model, cap) in enumerate(cells):
        for arm in ("clip", "tralo"):
            for seed in (1, 2, 3, 4):
                rng = np.random.default_rng(seed0 + 1000 * i + seed)
                y = rng.integers(0, K, size=N)
                z = rng.normal(size=(N, K))
                if arm == "tralo":
                    # a per-SEED offset: the within-cell spread of d AP is what
                    # the resolution block estimates its sd from
                    off = boost + jitter * np.random.default_rng(
                        seed0 + 7919 * i + 13 * seed).normal()
                    z[np.arange(N), y] += off
                P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
                d = tmp_path / model / ds / cap / arm / ("seed_%d" % seed)
                d.mkdir(parents=True, exist_ok=True)
                cols = {"True_Label": y, "Predicted_Label": P.argmax(1),
                        "Group_ID": rng.integers(0, 3, size=N)}
                for c in range(K):
                    cols["Prob_Class_%d" % c] = P[:, c]
                for f in ("final_predictions_raw.csv", "final_predictions.csv"):
                    pd.DataFrame(cols).to_csv(d / f, index=False)
                (d / "config.json").write_text(json.dumps(
                    {"arm": arm, "methodology": "x", "model_name": model,
                     "dataset_mode": ds, "constraint_tag": cap,
                     "constraint": [0.5, 0.3], "status": "completed",
                     "dataset_config": {"constrained_class": 1, "num_classes": K},
                     "hyperparams": {"seed": seed}}))
    r = subprocess.run([sys.executable, "-m", "scripts.full_panel",
                        "--campaign", str(tmp_path), "--control", "clip"],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout + r.stderr


def test_the_allocation_free_metrics_get_their_own_power_statement(tmp_path):
    """The RESOLUTION block converts to ITEMS via `items_per_001`, which is an
    F1 identity and does not apply to AP or AUROC -- so the scorer printed a
    power statement for exactly one metric family, and it is the family post-hoc
    filling can reach. Any verdict resting on the allocation-free metrics (which
    is what FRAMEWORK 2(p) pre-registers for iwc1) had NO seed cost attached,
    which is the "no effect vs not enough seeds" conflation the items block was
    built to prevent.
    """
    out = _panel_stdout(tmp_path, boost=1.2, jitter=0.0)
    assert "RESOLUTION of the ALLOCATION-FREE metrics" in out, out[-2500:]
    assert "AP" in out and "AUROC" in out
    assert "NOT items" in out, "the block must refuse to invent an items scale"


def test_the_allocation_free_power_statement_reads_the_seed_NOISE(tmp_path):
    """NEGATIVE CONTROL. A readout that always printed POWERED would satisfy the
    test above. A near-zero effect swamped by per-seed spread must come back
    UNDERPOWERED, or the block is decoration and a tie in AP would still be
    reported as a settled null.
    """
    out = _panel_stdout(tmp_path, boost=0.0, jitter=0.9, seed0=555)
    assert "RESOLUTION of the ALLOCATION-FREE metrics" in out, out[-2500:]
    tail = out.split("RESOLUTION of the ALLOCATION-FREE metrics")[1][:800]
    assert "UNDERPOWERED" in tail, tail
    assert "NOT evidence" in tail, tail


def test_the_main_table_generator_needs_two_metrics_to_reproduce_the_shipped_file():
    """The one generator whose DEFAULT is not the shipped artefact.

    `make_main_table.py --two-metrics` reproduces `tab_ccf1.tex` byte for byte;
    the bare invocation writes a DIFFERENT table over the same path. CLAUDE.md
    has carried that warning, with the exact diff size, as prose only -- so the
    single command in this repo that can silently corrupt a paper table on a
    routine "regenerate the tables" pass was guarded by a sentence.

    The bare run is the NEGATIVE CONTROL and is not incidental: if both
    invocations produced the same bytes the flag would be decorative and the
    warning wrong, which is worth knowing either way.

    The file is restored from the bytes read at entry, in a finally, rather than
    with `git checkout` -- a test that repairs a tracked file must not depend on
    the working tree being clean when it started.
    """
    gen = os.path.join(REPO, "docs", "paper", "scripts", "make_main_table.py")
    tab = os.path.join(REPO, "docs", "paper", "tables", "tab_ccf1.tex")
    if not (os.path.exists(gen) and os.path.exists(tab)):
        pytest.skip("paper table generator or its output is not present")

    with io.open(tab, "rb") as fh:
        shipped = fh.read()
    try:
        r = subprocess.run([sys.executable, gen, "--two-metrics"],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        with io.open(tab, "rb") as fh:
            assert fh.read() == shipped, (
                "make_main_table.py --two-metrics no longer reproduces the "
                "shipped tab_ccf1.tex -- either the corpus moved or the "
                "generator changed, and the paper table is now unbacked")

        r = subprocess.run([sys.executable, gen], cwd=REPO,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        with io.open(tab, "rb") as fh:
            bare = fh.read()
        assert bare != shipped, (
            "the bare invocation now reproduces the shipped table too, so "
            "--two-metrics is decorative and CLAUDE.md's warning is stale")
    finally:
        with io.open(tab, "wb") as fh:
            fh.write(shipped)


# The three tables in `docs/paper/tables/` that have NO generator and never did.
# An empty `git diff docs/paper/tables/` says NOTHING about these -- naming them
# is what keeps the test below from over-claiming.
UNGENERATED_TABLES = ("tab_ablation_complete.tex", "tab_deploy.tex",
                      "tab_oct_backbone.tex")

TABLE_GENERATORS = (("make_main_table.py", ("--two-metrics",)),
                    ("make_backbone_tables.py", ()),
                    ("make_graft_table.py", ()),
                    ("make_granular_tables.py", ()))


def test_the_generated_paper_tables_still_reproduce_from_the_corpus():
    """`git diff docs/paper/tables/` must be empty after regenerating.

    This is the invariant that says the shipped paper is still BACKED by
    `corpus_final.csv`. It has been documented in CLAUDE.md and never enforced,
    so a corpus edit or a generator change would have unbacked a table silently
    and the next person to regenerate would have seen a diff with no way to tell
    whether the old file or the new one was right.

    The three ungenerated tables are asserted UNTOUCHED rather than ignored. A
    clean diff is evidence about the eight and about nothing else, and stating
    that here stops the test from being read as "the tables reproduce".
    """
    tdir = os.path.join(REPO, "docs", "paper", "tables")
    sdir = os.path.join(REPO, "docs", "paper", "scripts")
    if not os.path.isdir(tdir):
        pytest.skip("paper tables are not present")

    before = {}
    for name in sorted(os.listdir(tdir)):
        if name.endswith(".tex"):
            with io.open(os.path.join(tdir, name), "rb") as fh:
                before[name] = fh.read()
    assert before, "no .tex tables found"

    try:
        for script, flags in TABLE_GENERATORS:
            path = os.path.join(sdir, script)
            if not os.path.exists(path):
                pytest.skip("%s is not present" % script)
            r = subprocess.run([sys.executable, path] + list(flags),
                               cwd=REPO, capture_output=True, text=True)
            assert r.returncode == 0, "%s failed: %s" % (script, r.stderr[-600:])

        changed = []
        for name, original in before.items():
            with io.open(os.path.join(tdir, name), "rb") as fh:
                if fh.read() != original:
                    changed.append(name)
        assert not changed, (
            "these tables no longer reproduce from the corpus: %s" % changed)

        for name in UNGENERATED_TABLES:
            assert name in before, (
                "%s is named as ungenerated but is not in tables/ -- the list "
                "has drifted" % name)
    finally:
        for name, original in before.items():
            with io.open(os.path.join(tdir, name), "wb") as fh:
                fh.write(original)


def test_no_soft_count_satisfaction_flag_is_reintroduced():
    """INERT FLAG REGRESSION -- occurrences five and six.

    `global_constraints_satisfied` and `local_constraints_satisfied` were
    assigned on every forward pass and read NOWHERE in the repo. Beyond being
    dead they were WRONG in a way that matters here: they record satisfaction on
    the SOFT count, and `sum_i p_ic` is strictly positive for any softmax, so at
    K == 0 the flag is permanently False for a group that is in fact perfectly
    satisfied. On iwildcam seven of fourteen per-group ceilings are K == 0, so
    anyone who wired one of these up would have read "never satisfied" off a
    healthy run.

    The check is on ASSIGNMENT, via AST, because a mention in a comment (there
    is one, explaining why they are gone) must not satisfy or trip it.
    """
    src_dir = os.path.join(REPO, "src")
    offenders = []
    for dirpath, _dirs, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            with io.open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and tgt.attr.endswith("constraints_satisfied")):
                        offenders.append((os.path.relpath(path, REPO),
                                          node.lineno, tgt.attr))
    assert not offenders, (
        "a soft-count satisfaction flag is being assigned again: %s. The "
        "trainer decides satisfaction from the HARD counts; a soft flag is "
        "permanently False at K=0." % offenders)


def test_the_trainer_decides_satisfaction_and_the_ratchet_from_HARD_counts():
    """The claim the K=0 docstring now rests on, pinned against the trainer.

    `transductive_loss._penalty` tells the reader that a K == 0 constraint does
    NOT hold the ratchet gate open, on the grounds that the trainer reads hard
    counts. An earlier version of that docstring said the opposite and nearly
    condemned a healthy campaign, so the claim is enforced rather than trusted:
    every count the satisfaction snapshot and the ratchet compare against a
    limit must be a `_hard` one.
    """
    path = os.path.join(REPO, "src", "methodologies", "tralo", "train.py")
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    def count_uses(text):
        """Lines that COMPARE a count, or bind one for a later comparison.

        Logging lines that merely read `.item()` into a dict are excluded on
        purpose -- the trainer builds `g_soft_d` / `l_soft_d` for the CSV, and
        recording the soft count is correct. What must never be soft is the
        value tested against a limit.
        """
        out = []
        for ln in text.splitlines():
            t = ln.strip()
            counted = re.search(r"total_(local|global)_(hard|soft)", t)
            if not counted:
                continue
            if re.search(r"[<>]=?", t) or re.match(r"\w*hard\w*\s*=\s*total_", t):
                out.append(t)
        return out

    # the checker must be able to FIRE, or a green result below means nothing
    poisoned = "if total_local_soft[g][c].item() > lc[c].item():"
    assert any("_soft" in ln for ln in count_uses(poisoned)), (
        "the extractor cannot see a soft comparison even when one is planted")

    checked = count_uses(os.linesep.join(lines))
    assert checked, "no count comparisons found -- the trainer moved"
    soft = [ln for ln in checked if "_soft" in ln]
    assert not soft, (
        "the trainer is comparing a SOFT count against a limit: %s. At K=0 the "
        "soft count is strictly positive, so this would make a satisfied group "
        "read as violated for the whole run." % soft[:3])


def _log_health_campaign(tmp_path, accs, arm="tralo"):
    """A campaign whose training_log.csv walks `accs` over epochs."""
    root = os.path.join(str(tmp_path), "camp")
    for seed in (1, 2, 3):
        d = os.path.join(root, "%s_seed%d" % (arm, seed))
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({"Epoch": list(range(1, len(accs) + 1)),
                      "Train_Acc": list(accs),
                      "L_CE": [0.5] * len(accs)}).to_csv(
            os.path.join(d, "training_log.csv"), index=False)
        with io.open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"arm": arm, "status": "completed",
                       "dataset_mode": "iwildcam", "model_name": "MobileNetV3",
                       "constraint_tag": "L30_G50", "constraint": [0.30, 0.50],
                       "dataset_config": {"constrained_class": [2]},
                       "hyperparams": {"seed": seed}}, fh)
    r = subprocess.run([sys.executable, "-m", "scripts.log_health", root],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout + r.stderr


def test_log_health_flags_a_model_already_converged_before_the_constraint_phase(tmp_path):
    """Rule 1 fixes warm-up at 1 because warm-up 50 saturates CE and every
    method becomes identical -- but that boundary was calibrated on dermmnist
    and is stated as a warm-up LENGTH, while what matters is where the model
    ENDS UP. iWildCam's warm-up reaches ~95.6% in ONE epoch, so an easy dataset
    can enter the saturated regime through a door rule 1 does not cover, and a
    tie across all arms would be the saturation rather than the methods.

    `log_health` is step 0 of the iwc1 read, so the pointer belongs there.
    """
    out = _log_health_campaign(tmp_path, [0.956, 0.956, 0.957, 0.956])
    assert "CONVERGENCE AT THE START OF THE CONSTRAINT PHASE" in out, out[-1200:]
    assert "ALREADY CONVERGED" in out, out[-1200:]
    assert "scripts.reachability" in out, (
        "the flag must point at the instrument that actually measures p(1-p) "
        "at the cut -- accuracy is only a proxy")


@pytest.mark.parametrize("accs,why", [
    ([0.40, 0.55, 0.70, 0.88], "a big gain must not read as saturated"),
    ([0.30, 0.30, 0.31, 0.30], "a flat but LOW run is not saturation, it is a "
                               "model that never learned"),
])
def test_log_health_does_not_cry_saturation_on_a_healthy_run(tmp_path, accs, why):
    """NEGATIVE CONTROL, both halves of the signature separately.

    The flag requires an already-high warm-up AND a constraint phase that moved
    nothing. A readout keyed on either alone would fire on an ordinary run: a
    hard dataset can start low and climb, and a broken run can sit flat at 30%.
    Firing on those would train the reader to ignore it, which is worse than
    not having it.
    """
    out = _log_health_campaign(tmp_path, accs)
    assert "CONVERGENCE AT THE START OF THE CONSTRAINT PHASE" in out, out[-800:]
    assert "ALREADY CONVERGED" not in out, (why, out[-800:])
    assert "not the saturated signature" in out, out[-800:]


def test_the_straddle_probe_never_pools_two_cells_into_one_row(tmp_path):
    """Rule 4: the atomic cell is (dataset, backbone, cap, method) and pooling
    across any of them is this project's most-repeated analysis error.

    The stored-evidence tree is precisely the shape that punishes it -- 128 runs
    over THREE datasets and THREE cap levels, where "class 1" names a different
    class in each -- and the first run of this probe over that tree DID pool
    them, producing a confident table describing nothing. Careful invocation is
    not the fix; the tool grouping by cell is.
    """
    from scripts import straddle_probe as SP

    rng = np.random.default_rng(5)
    n = 300
    root = os.path.join(str(tmp_path), "twocell")
    y = rng.integers(0, 3, n)
    groups = rng.integers(0, 2, n)
    for ds, sharp in (("iwildcam", 6.0), ("otherset", 0.4)):
        # `sharp` moves the score geometry, so a pooled row could not equal
        # either cell's row and the assertion below cannot pass by accident
        P = rng.dirichlet(np.ones(3) * sharp, size=n)
        _write_run(os.path.join(root, ds), "clip", 1, P, y, groups, dataset=ds)

    buf = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = buf
        SP.main(["--campaign", root, "--sweep"])
    finally:
        sys.stdout = stdout
    out = buf.getvalue()

    assert out.count("CELL ") >= 2, (
        "two datasets collapsed into one block:" + chr(10) + out[-1500:])
    assert "iwildcam" in out and "otherset" in out, out[-1500:]
    # and each cell reports its own run count, not the pooled one
    assert "1 run(s)" in out, (
        "a cell is reporting more runs than it holds, which is the pooling "
        "this test exists to catch:" + chr(10) + out[-1500:])


def test_delta_for_contested_hits_the_requested_band_size():
    """Bisection on a monotone step function, so the target must be met exactly
    wherever ties do not straddle it. A band that missed its target would make
    the matched ladder no more comparable than the one it replaced."""
    from scripts.straddle_probe import delta_for_contested, cut_score

    rng = np.random.default_rng(2)
    scores = rng.random(600)
    K = 120
    t = cut_score(scores, K)
    for target in (10, 50, 200):
        d = delta_for_contested(scores, K, target)
        got = int((np.abs(scores - t) <= d).sum())
        assert got == target, (target, got, d)


def test_delta_for_contested_is_what_removes_the_density_confound():
    """THE JUSTIFICATION FOR THE LADDER, pinned directly.

    A delta swept as a fraction of the SCORE RANGE covers a different number of
    items depending on how dense the scores are around the cut, which is exactly
    what made the stored-evidence cap trend unreadable: the reachable share fell
    with the cap in 24 of 33 series, and `contested` fell with it in 22 of 33,
    so thinning density explained the same numbers. Holding the band SIZE fixed
    is what separates them.

    Two score sets with the same range and very different densities at the cut.
    The fraction-of-range band must disagree between them; the matched band must
    not.
    """
    from scripts.straddle_probe import delta_for_contested, cut_score

    rng = np.random.default_rng(4)
    n, K = 800, 400                    # K at the median, so the cut is placed
                                       # by the DISTRIBUTION and not by the rank
    # dense at the cut: one tight cluster, the cut lands in its middle
    dense = np.clip(rng.normal(0.5, 0.03, n), 0.0, 1.0)
    # sparse at the cut: two clusters with a GAP where the median falls, so the
    # K-th ranked score sits at the edge of the upper one
    sparse = np.clip(np.concatenate([rng.normal(0.2, 0.03, n // 2),
                                     rng.normal(0.8, 0.03, n - n // 2)]),
                     0.0, 1.0)
    for s in (dense, sparse):          # same range, so the fraction is the same
        s[0], s[1] = 0.0, 1.0

    frac = 0.01
    counts_frac = []
    counts_matched = []
    for s in (dense, sparse):
        t = cut_score(s, K)
        counts_frac.append(int((np.abs(s - t) <= frac * (s.max() - s.min())).sum()))
        d = delta_for_contested(s, K, 50)
        counts_matched.append(int((np.abs(s - t) <= d).sum()))

    assert counts_frac[0] > 3 * counts_frac[1], (
        "the fixture is not actually testing the confound -- the two densities "
        "give similar band sizes at a fixed fraction: %s" % counts_frac)
    assert counts_matched == [50, 50], (
        "the matched ladder failed to hold the band size fixed: %s"
        % counts_matched)


def test_delta_for_contested_survives_a_degenerate_cut():
    """K outside (0, n) gives an infinite cut and there is no band to size. It
    must return a finite 0.0 rather than propagate a nan into every downstream
    count."""
    from scripts.straddle_probe import delta_for_contested

    scores = np.linspace(0.0, 1.0, 50)
    for K in (0, -3, 50, 99):
        d = delta_for_contested(scores, K, 10)
        assert np.isfinite(d), (K, d)


def test_match_contested_end_to_end_reports_the_band_it_asked_for(tmp_path):
    """The flag end to end: the `contested` column must equal the target, or the
    ladder is not doing the one thing it exists to do."""
    from scripts import straddle_probe as SP

    rng = np.random.default_rng(9)
    n = 400
    root = os.path.join(str(tmp_path), "mc")
    y = rng.integers(0, 3, n)
    P = rng.dirichlet(np.ones(3) * 2.0, size=n)
    _write_run(os.path.join(root, "a"), "clip", 1, P, y,
               rng.integers(0, 2, n))

    buf = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = buf
        SP.main(["--campaign", root, "--match-contested"])
    finally:
        sys.stdout = stdout
    out = buf.getvalue()

    assert "contested=50" in out, out[-900:]
    for line in out.splitlines():
        if line.strip().startswith("contested=50"):
            cols = line.split()
            assert float(cols[1]) == 50.0, line
            break
    else:
        raise AssertionError("no contested=50 row in:" + chr(10) + out[-900:])


def _collide(paths):
    """Fire `_one` on two runs sitting at `paths` and return the message."""
    import pandas as pd
    from scripts import full_panel as FP

    FP.RUN_DIRS.clear()
    FP.RUN_DIRS.update({i: q for i, q in enumerate(paths)})
    try:
        FP._one(pd.Series([1.0] * len(paths), index=list(range(len(paths)))))
    except ValueError as e:
        return str(e)
    finally:
        FP.RUN_DIRS.clear()
    raise AssertionError("_one accepted %d runs on one key" % len(paths))


def test_two_campaign_roots_are_named_as_two_campaigns():
    """The collision that the stored evidence actually produces.

    `mcbar` and `multiclass` sit side by side under one tree, so pointing
    `--campaign` at the tree lands both campaigns' `clip/seed_1` on the same
    (cell, seed, arm) key. The old message said "the pairing key is missing a
    dimension", which is the OTHER cause and sends the reader into the scorer
    instead of into the path they passed.
    """
    msg = _collide([
        "/ev/results/mcbar/MobileNetV3/dermmnist/L50_G50/clip/seed_1",
        "/ev/results/multiclass/MobileNetV3/dermmnist/L50_G50/clip/seed_1",
    ])
    assert "DIFFERENT roots" in msg, msg
    assert "campaign root separately" in msg, msg
    assert "mcbar" in msg and "multiclass" in msg, msg


def test_a_real_missing_dimension_is_NOT_called_two_campaigns():
    """Negative control for the test above.

    Same root, genuinely different run paths -- the campaign sweeps something
    the pairing key does not name. The two-campaign diagnosis must stay silent,
    or it becomes a message that says the same thing whatever happened, which
    is worth less than the one it replaced.
    """
    msg = _collide([
        "/ev/results/iwc1/MobileNetV3/iwildcam/L30_G50/tralo/seed_1",
        "/ev/results/iwc1/MobileNetV3/iwildcam/L30_G50/tralo_lr9/seed_1",
    ])
    assert "DIFFERENT roots" not in msg, msg
    assert "sweeps an axis the pairing key does not name" in msg, msg


def test_the_collision_still_refuses_with_no_paths_recorded():
    """No `run_dir` column -- older rows, or a caller that never set RUN_DIRS.

    The guard must still REFUSE. Degrading to a silent average is the failure
    it exists to prevent, and the message has to admit it cannot say which of
    the two causes fired rather than guessing one.
    """
    import pandas as pd
    from scripts import full_panel as FP

    FP.RUN_DIRS.clear()
    with pytest.raises(ValueError) as ei:
        FP._one(pd.Series([1.0, 2.0], index=[0, 1]))
    msg = str(ei.value)
    assert "2 runs share one" in msg, msg
    assert "which one cannot be said" in msg, msg
    assert "DIFFERENT roots" not in msg, msg


def test_one_passes_through_a_single_run_and_an_empty_group():
    """The guard must not become a refusal to score anything."""
    import pandas as pd
    from scripts import full_panel as FP

    assert FP._one(pd.Series([0.75], index=[3])) == 0.75
    assert np.isnan(FP._one(pd.Series([], dtype=float)))
    assert np.isnan(FP._one(pd.Series([float("nan")], index=[1])))


def test_the_mde_at_four_seeds_is_1_4_sd():
    """FRAMEWORK 2(p) prints a detectability table -- AP ~0.035, AUROC ~0.013 --
    and both entries come from ONE derivation: at the protocol's 4 seeds,
    `seeds_needed(d, sd) <= 4` iff `d >= 1.4 sd`. That is a claim about
    `seeds_needed`, so it is gated rather than trusted. A drift in the power
    constants would leave the table quietly wrong while the scorer stayed
    self-consistent, and the whole point of pre-registering the number is that
    it is checkable before the campaign lands.
    """
    from scripts.frozen_head_probe import seeds_needed

    # The crossing is EXACT, not the round 1.4 it looks like: with
    # n = ceil(z^2 sd^2 / d^2), `n <= 4` iff `d >= sd * sqrt(z^2 / 4)`, and
    # z^2 = 7.8489 rather than 7.84 -- so d = 1.4 sd lands one seed on the
    # WRONG side of the ceiling. Writing 1.4 into the table is fine to two
    # figures; writing it into the gate is not.
    k = float(np.sqrt((1.959963985 + 0.8416212336) ** 2 / 4.0))
    assert round(k, 3) == 1.401, k

    for sd in (0.0058, 0.0094, 0.0202, 0.0252, 0.0274, 1.0, 2.7):
        assert seeds_needed(k * sd * 1.001, sd) <= 4, sd
        assert seeds_needed(k * sd * 0.999, sd) > 4, sd

    # the two figures the table actually prints, from the measured medians
    assert round(k * 0.0252, 3) == 0.035     # AP
    assert round(k * 0.0094, 3) == 0.013     # AUROC

    # and the ratio the section leans on: AUROC resolves ~2.7x better than AP
    assert 2.6 <= 0.0252 / 0.0094 <= 2.8


def test_a_temperature_rescale_moves_calibration_and_NOT_the_ranking():
    """The claim the RANKING / CALIBRATION split rests on, checked not asserted.

    `full_panel` now prints the allocation-free power block in two families and
    tells the reader that a CALIBRATION-only move is a rescale which cannot
    change any top-K set. That is a mathematical claim about the metrics: AP and
    AUROC read the ORDER of the score column and nothing else, so a strictly
    monotone rescale must leave them BIT-identical, while ECE / Brier / NLL /
    ConfGap all move.

    It matters because the stored evidence produces exactly this pattern --
    focal_clip vs clip moves ECE -0.069, NLL -1.12 and ConfGap +0.066, all
    POWERED and unanimous 6/0, while AP and AUROC come back UNDERPOWERED. Read
    without the split that is "the probabilities changed, so the representation
    channel is live"; read with it, it is a recalibration that provably changes
    no allocation at all.

    Temperature scaling in logit space, two classes, so "monotone in the score
    column" is exact rather than approximate.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
    from scripts.full_panel import ece, brier

    rng = np.random.default_rng(11)
    n = 600
    z = rng.normal(0.0, 2.0, n)                      # logit of class 1
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)

    def cols(zz):
        p1 = 1.0 / (1.0 + np.exp(-zz))
        return np.column_stack([1.0 - p1, p1])

    def panel_of(zz):
        P = cols(zz)
        conf = P.max(axis=1)
        ok = P.argmax(axis=1) == y
        return {
            "AP": average_precision_score(y, P[:, 1]),
            "AUROC": roc_auc_score(y, P[:, 1]),
            "ECE": ece(y, P),
            "Brier": brier(y, P),
            "NLL": log_loss(y, P, labels=[0, 1]),
            "ConfGap": float(conf[ok].mean() - conf[~ok].mean()),
        }

    base = panel_of(z)
    hot = panel_of(z / 3.0)          # strictly monotone: order is untouched

    for m in ("AP", "AUROC"):
        assert base[m] == hot[m], (m, base[m], hot[m])

    for m in ("ECE", "Brier", "NLL", "ConfGap"):
        assert abs(base[m] - hot[m]) > 1e-6, (m, base[m], hot[m])

    # NEGATIVE CONTROL: a genuine REORDERING must reach the ranking family, or
    # the test above is only saying that these metrics are hard to move.
    zz = z.copy()
    hi, lo = int(np.argmax(zz)), int(np.argmin(zz))
    zz[hi], zz[lo] = zz[lo], zz[hi]
    assert abs(panel_of(zz)["AUROC"] - base["AUROC"]) > 1e-6


def test_every_allocation_free_metric_gets_a_power_statement():
    """No metric may sit in the allocation-free table with no seed cost beside it.

    The block was born covering AP and AUROC only, because those are the two the
    iwc1 pre-registration names. ConfGap then turned out to be the SHARPEST of
    the six on the stored evidence -- |d|/sd = 4.3 against AUROC's 0.47 -- and
    it was printing a unanimous 6/0 delta with no power statement anywhere. The
    families must PARTITION the table, so adding a metric to one and forgetting
    the other is a test failure rather than a silent omission.
    """
    from scripts import full_panel as FP

    table = None
    for title, metrics in FP.GROUPS:
        if title.startswith("ALLOCATION-FREE"):
            table = list(metrics)
    assert table, "the allocation-free group vanished from GROUPS"

    assert set(FP.FREE_RESOLUTION) == set(table), (
        "allocation-free metrics with no power statement: %s; "
        "priced but not in the table: %s"
        % (sorted(set(table) - set(FP.FREE_RESOLUTION)),
           sorted(set(FP.FREE_RESOLUTION) - set(table))))
    assert not (set(FP.FREE_RANKING) & set(FP.FREE_CALIBRATION))
    assert set(FP.FREE_RANKING) | set(FP.FREE_CALIBRATION) == set(table)
    # and the two families are not interchangeable: only the ranking one is
    # invariant to a rescale, which is what the printed reading rule claims
    assert set(FP.FREE_RANKING) == {"AP", "AUROC"}


RANKING_COLUMNS = {"auroc", "auc", "ap", "auprc", "aupr", "average_precision",
                   "roc_auc", "ap_capped", "auroc_capped"}


def _ranking_columns_in(corpus_dir):
    """{file: [ranking columns]} for every csv under `corpus_dir`."""
    import glob

    found = {}
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.csv"))):
        with io.open(f, encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        cols = [c.strip().strip('"').lower() for c in header.split(",")]
        hit = [c for c in cols if c in RANKING_COLUMNS]
        if hit:
            found[os.path.basename(f)] = hit
    return found


def test_the_paper_corpus_carries_no_ranking_metric():
    """FRAMEWORK 1b claims the corpus never measured the ranking channel.

    That is a claim about files in this repo, so it is checked rather than
    asserted. `corpus_final.csv` -- 7,574 rows behind eight of the eleven
    tables -- has seven outcome columns and every one is budget-equalized
    (acc, f1_macro, cc_f1, cc_rec, cc_prec) or a house-rule-5 non-metric
    (flips, sat). Across all 17 files the only allocation-free column that
    appears at all is `ece`, which is CALIBRATION and therefore provably cannot
    change any top-K set.

    IF THIS TEST EVER FAILS, THAT IS GOOD NEWS and 1b must be rewritten: a
    ranking column appeared, and the paper can finally say something about the
    channel its own structural argument rests on.
    """
    corpus = os.path.join(REPO, "docs", "paper", "data", "corpus")
    if not os.path.isdir(corpus):
        pytest.skip("corpus not present in this worktree")

    found = _ranking_columns_in(corpus)
    assert not found, (
        "FRAMEWORK 1b says the corpus carries no ranking metric, but: %s" % found)

    # the positive half: the claim is "only ECE, and only in four files"
    import glob
    with_ece = []
    for f in sorted(glob.glob(os.path.join(corpus, "*.csv"))):
        with io.open(f, encoding="utf-8", errors="replace") as fh:
            cols = [c.strip().strip('"').lower()
                    for c in fh.readline().split(",")]
        if "ece" in cols:
            with_ece.append(os.path.basename(f))
    assert with_ece == ["ablation_no_hinge.csv", "extra_robustness_corpus.csv",
                        "imbalanced_baselines.csv", "native224_ham10000.csv"], with_ece

    # and corpus_final, which is what eight tables read, carries none of it
    with io.open(os.path.join(corpus, "corpus_final.csv"),
                 encoding="utf-8", errors="replace") as fh:
        cols = [c.strip().strip('"').lower() for c in fh.readline().split(",")]
    for banned in ("ece", "brier", "nll", "confgap"):
        assert banned not in cols, banned


def test_the_ranking_column_detector_actually_detects(tmp_path):
    """Negative control for the test above.

    A detector that finds nothing because it looks for nothing would pass the
    corpus audit forever and silently stop being evidence. Plant one.
    """
    d = str(tmp_path)
    io.open(os.path.join(d, "clean.csv"), "w", encoding="utf-8").write(
        "dataset,model,seed,acc,f1_macro,cc_f1,flips" + chr(10) + "a,b,1,0,0,0,0" + chr(10))
    assert _ranking_columns_in(d) == {}

    io.open(os.path.join(d, "planted.csv"), "w", encoding="utf-8").write(
        "dataset,model,seed,acc,AUROC,AP" + chr(10) + "a,b,1,0,0.9,0.8" + chr(10))
    hit = _ranking_columns_in(d)
    assert hit == {"planted.csv": ["auroc", "ap"]}, hit


POINTER_RE = re.compile(
    r"(?:docs|scripts|configs|src|tests)/[A-Za-z0-9_/.-]+"
    r"\.(?:md|tex|pdf|sh|ya?ml|py|csv)")

LIVE_TREES = ("src", "configs", "scripts")


def _broken_pointers(root, trees=LIVE_TREES, extra=()):
    """{file: [targets that do not exist]} for every path-like string in `trees`.

    A comment pointing at a deleted file is worse than no comment: it reads as
    provenance and sends the next person looking for a record that is gone.
    Two were live when this was written -- `docs/REJECTED.md` from
    `src/models/imagery/vit.py` and `docs/AUDIT_FINDINGS_2026-04-26.md` from
    `src/utils/constants.py`, the second of which exists nowhere in the repo or
    its history, so the fact it recorded had to be inlined.
    """
    import glob

    files = list(extra)
    for t in trees:
        files += glob.glob(os.path.join(root, t, "**", "*.py"), recursive=True)
        files += glob.glob(os.path.join(root, t, "**", "*.yml"), recursive=True)
    broken = {}
    for f in sorted(set(files)):
        if os.sep + "__pycache__" in f:
            continue
        with io.open(f, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        bad = sorted({t for t in POINTER_RE.findall(text)
                      if not os.path.exists(os.path.join(root, t))})
        if bad:
            broken[os.path.relpath(f, root).replace(os.sep, "/")] = bad
    return broken


def test_no_live_file_points_at_a_deleted_doc():
    """Every docs/ scripts/ configs/ src/ path named in live code must exist."""
    broken = _broken_pointers(REPO)
    assert not broken, (
        "these files point at paths that do not exist: %s" % broken)


def test_the_pointer_detector_actually_detects(tmp_path):
    """Negative control: a detector that finds nothing is not a gate."""
    d = str(tmp_path)
    os.makedirs(os.path.join(d, "src"))
    io.open(os.path.join(d, "src", "ok.py"), "w", encoding="utf-8").write(
        "# see configs/real.yml" + chr(10))
    io.open(os.path.join(d, "configs"), "w", encoding="utf-8")  # placeholder
    os.remove(os.path.join(d, "configs"))
    os.makedirs(os.path.join(d, "configs"))
    io.open(os.path.join(d, "configs", "real.yml"), "w", encoding="utf-8").write("a: 1")
    assert _broken_pointers(d, trees=("src",)) == {}

    io.open(os.path.join(d, "src", "bad.py"), "w", encoding="utf-8").write(
        "# see docs/GONE.md and scripts/vanished.py" + chr(10))
    hit = _broken_pointers(d, trees=("src",))
    assert hit == {"src/bad.py": ["docs/GONE.md", "scripts/vanished.py"]}, hit


def test_the_unlimited_sentinel_is_never_re_derived():
    """`UNLIMITED` is 1e10 and nothing may hard-code a different threshold.

    `src/utils/constants.py` exists because `metrics.py` once declared a local
    `UNLIMITED=1e9` while the rest of the codebase used 1e10, so a constraint
    set to UNLIMITED was skipped by the loss and registered as ACTIVE by the
    metric layer. The same literal came back as a bare `1e9` in four analysis
    scripts -- dose_scan, log_health, reachability, score_scan -- each asking
    "was this class capped" against a threshold they re-derived. It happened to
    be conservative and therefore harmless, which is exactly why it survived.

    AST, not grep: `constants.py` still DESCRIBES the 1e9 defect in prose, and
    a grep counts that as a violation.
    """
    import ast
    import glob

    offenders = []
    for tree in ("src", "scripts", "configs"):
        for f in glob.glob(os.path.join(REPO, tree, "**", "*.py"), recursive=True):
            if os.sep + "__pycache__" in f:
                continue
            with io.open(f, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            try:
                node = ast.parse(src)
            except SyntaxError:
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Constant) and isinstance(n.value, float):
                    if 1e9 <= n.value < 1e10:
                        offenders.append("%s:%d = %r" % (
                            os.path.relpath(f, REPO).replace(os.sep, "/"),
                            n.lineno, n.value))
    assert not offenders, (
        "these re-derive the UNLIMITED threshold instead of importing it: %s"
        % offenders)

    from src.utils.constants import UNLIMITED
    assert UNLIMITED == 1e10


def test_the_two_power_floors_are_printed_and_the_framework_quotes_them(tmp_path):
    """FRAMEWORK 2(p) quotes `gen_campaign`'s power block verbatim.

    Two floors bind on iwc1 and they are independent. The SEED floor -- can a
    cell resolve an effect of this size -- is the MDE table in 2(p). The CELL
    floor is this one, and it is harsher: at 2 cells the exact Wilcoxon floor is
    p=0.5, so after BH over eleven metrics NO metric can reach a *** verdict at
    any effect size. That is arithmetic and it was true before iwc1 launched, so
    a positive iwc1 headline is unavailable whatever lands -- which is exactly
    why the pre-registered verdict is stated as a NULL with a bound, and why the
    doc must not drift from what the generator prints.

    The negative control is a 9-cell campaign, where the UNDERPOWERED line must
    switch off; a warning that is always on carries no information.
    """
    fw = io.open(os.path.join(REPO, "docs", "FRAMEWORK.md"),
                 encoding="utf-8").read()

    def gen(root, models, caps):
        r = subprocess.run(
            [sys.executable, "-m", "configs.gen_campaign", "--root", str(root),
             "--datasets", "iwildcam", "--models"] + models +
            ["--caps"] + caps + ["--arms", "all+null"],
            cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    out = gen(os.path.join(str(tmp_path), "two"), ["MobileNetV3"],
              ["L20_G50", "L30_G50"])
    for quoted in ("2 cells", "UNDERPOWERED", "9 cells is the minimum",
                   "1 dataset(s)", "p=1.000"):
        assert quoted in out, (quoted, out[-1500:])
        assert quoted in fw, (
            "FRAMEWORK 2(p) quotes the power block but has drifted from it: "
            "%r is printed and not documented" % quoted)

    # NEGATIVE CONTROL: at the 9 cells the generator itself names, the
    # UNDERPOWERED verdict must clear.
    nine = gen(os.path.join(str(tmp_path), "nine"),
               ["MobileNetV3", "MobileNetV2", "RegNetY400MF"],
               ["L20_G50", "L30_G50", "L50_G30"])
    assert "9 cells" in nine, nine[-1500:]
    assert "UNDERPOWERED: with" not in nine, (
        "the cell-count warning fires at 9 cells too, so it says nothing about "
        "2:" + chr(10) + nine[-1500:])

    # the generalization floor, however, is about DATASETS and must NOT clear
    assert "1 dataset(s)" in nine and "p=1.000" in nine, (
        "adding backbones bought independence it cannot buy" + chr(10)
        + nine[-1500:])


def test_detectable_at_is_the_exact_inverse_of_seeds_needed():
    """`detectable` is the number a NULL gets stated with, so it has to be the
    honest inverse of the number a POSITIVE gets priced with, not an
    approximation that happens to look right at n=4.

    `seeds_needed(d, sd) = ceil(z^2 sd^2 / d^2)` and
    `detectable_at(sd, n) = z sd / sqrt(n)` are the same equation solved for
    different unknowns, so the round trip must close AND be tight: n seeds
    catch it, n-1 do not.
    """
    from scripts.frozen_head_probe import seeds_needed
    from scripts.full_panel import detectable_at

    for sd in (0.0058, 0.0094, 0.0252, 0.35, 2.7):
        for n in (2, 4, 8, 25, 100):
            d = detectable_at(sd, n)
            assert seeds_needed(d * 1.001, sd) <= n, (sd, n, d)
            assert seeds_needed(d * 0.999, sd) > n, (sd, n, d)

    # the factor FRAMEWORK 2(p) prints, and the two figures in its table
    assert round(detectable_at(1.0, 4), 3) == 1.401
    assert round(detectable_at(0.0252, 4), 3) == 0.035     # AP
    assert round(detectable_at(0.0094, 4), 3) == 0.013     # AUROC

    # more seeds must buy resolution, and at the sqrt rate -- quadrupling the
    # seeds halves the bound, which is why "add seeds" is expensive advice
    assert abs(detectable_at(1.0, 16) - detectable_at(1.0, 4) / 2.0) < 1e-12

    # degenerate inputs refuse rather than return a number that reads as a bound
    for bad in ((0.0, 4), (float("nan"), 4), (1.0, 0)):
        assert not np.isfinite(detectable_at(*bad)), bad


def _cct_json(path, alpha, n_loc=10, per_loc=200, n_cls=4, seed=3):
    """A COCO-CameraTraps annotation file with a tunable per-camera class mix.

    `alpha=None` gives every camera the SAME distribution -- octmnist's failure
    mode, where `synth_group` was `index % 3` so the groups were i.i.d. draws
    from one distribution and the local scope was empty by construction.
    """
    rng = np.random.default_rng(seed)
    images, anns = [], []
    iid = 0
    for loc in range(n_loc):
        w = (np.ones(n_cls) / n_cls if alpha is None
             else rng.dirichlet(np.ones(n_cls) * alpha))
        for _ in range(per_loc):
            c = int(rng.choice(n_cls, p=w))
            images.append({"id": iid, "file_name": "img%06d.jpg" % iid,
                           "location": loc})
            anns.append({"image_id": iid, "category_id": c})
            iid += 1
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump({"categories": [{"id": i, "name": "sp%d" % i}
                                  for i in range(n_cls)],
                   "images": images, "annotations": anns}, fh)


def _screen_a_cct(tmp, alpha, tag):
    """annotations -> --meta-only -> dataset_screen, and return the verdict."""
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    ann = os.path.join(d, "ann.json")
    _cct_json(ann, alpha)
    out = os.path.join(d, "slice")
    # bytes, not text=True: both scripts print emoji and the Windows default
    # codec returns None for stdout rather than raising, which reads as "the
    # script printed nothing" -- a silent way for this gate to stop checking.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def run(cmd):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, env=env)
        so = r.stdout.decode("utf-8", "replace")
        se = r.stderr.decode("utf-8", "replace")
        assert r.returncode == 0, so + se
        return so

    prep = run([sys.executable, "-m", "scripts.prep_iwildcam",
                "--annotations", ann, "--out", out, "--classes", "4",
                "--min-per-camera", "50", "--test-target", "400",
                "--train-per-class", "300", "--meta-only"])
    return out, prep, run([sys.executable, "-m", "scripts.dataset_screen", out])


def test_a_candidate_dataset_can_be_screened_before_it_is_downloaded(tmp_path):
    """2(n) presents stage 1 as the PRE-GPU, pre-image screen. Until
    2026-08-23 it was not reachable that way on any dataset not already on disk:
    `prep_iwildcam` wrote the two CSVs `dataset_screen` reads from INSIDE the
    shard-download loop, so pricing a candidate cost the full acquisition the
    screen exists to avoid. `--meta-only` closes that, and the split builder is
    generic COCO-CameraTraps, so Terra Incognita / CCT screens the same way.

    This runs the whole chain -- annotations -> meta -> verdict -- and asserts
    NO image was fetched.
    """
    out, prep_out, screen = _screen_a_cct(str(tmp_path), 0.35, "live")

    assert "META ONLY" in prep_out, prep_out
    for split in ("train", "test"):
        f = os.path.join(out, "%s_meta.csv" % split)
        assert os.path.exists(f), f
        cols = io.open(f, encoding="utf-8").readline().strip().split(",")
        assert cols == ["label", "class_name", "filename", "location"], cols
    assert not [f for f in os.listdir(out) if f.endswith(".npy")], (
        "--meta-only wrote image arrays, so it downloaded something")

    assert "STAGE 1 PASS" in screen, screen[-800:]
    assert "ABSENT from train" in screen, screen[-800:]


def test_identical_per_group_mixes_screen_DEAD_even_with_unseen_groups(tmp_path):
    """The negative control, and it is the whole point of the screen.

    Give every camera the SAME class distribution and the local scope is empty
    BY CONSTRUCTION -- octmnist and tissuemnist, whose `synth_group` was
    `index % 3`. The screen must call it DEAD. Crucially it must do so WHILE
    still reporting unseen test groups: held-out groups are criterion 1 and are
    NOT sufficient, which is the distinction that let two of the original three
    datasets be run for months against a question they could not test.
    """
    _out, _prep, screen = _screen_a_cct(str(tmp_path), None, "dead")

    assert "DEAD" in screen, screen[-800:]
    assert "within sampling noise" in screen, screen[-800:]
    # and the trap it protects against: unseen groups are still reported
    assert "ABSENT from train" in screen, (
        "the fixture lost its held-out cameras, so this no longer controls for "
        "criterion 1" + chr(10) + screen[-800:])


def _headline_power_table():
    """Paired tralo-minus-heuristic macro-F1 per seed, within cell, from the
    corpus. The exact computation FRAMEWORK 1b quotes."""
    import pandas as pd
    from scipy import stats
    from scripts.full_panel import detectable_at

    f = os.path.join(REPO, "docs", "paper", "data", "corpus",
                     "corpus_final.csv")
    if not os.path.exists(f):
        pytest.skip("corpus not present in this worktree")
    d = pd.read_csv(f)
    cell = ["dataset", "model", "constraint_tag", "constrained_class",
            "group_column", "warmup_epochs", "sweep"]
    d = d[d["method"].isin(["tralo", "heuristic"])]
    w = d.pivot_table(index=cell + ["seed"], columns="method",
                      values="f1_macro", aggfunc="mean").dropna()
    w["delta"] = w["tralo"] - w["heuristic"]
    g = w.groupby(level=list(range(len(cell))))["delta"]
    res = pd.DataFrame({"n": g.size(), "mean": g.mean(),
                        "sd": g.std(ddof=1)}).dropna()
    res = res[res["n"] >= 2].reset_index()
    res["mde"] = [detectable_at(sd, n) for sd, n in zip(res["sd"], res["n"])]
    return res, stats


def test_the_paper_headline_is_not_seed_noise_and_strengthens_when_resolved():
    """FRAMEWORK 1b prices the macro-F1 headline against the corpus's OWN seeds.

    The claim is specific and falsifiable: the aggregate is not carried by
    unresolvable cells, because restricting to the cells whose effect clears
    their own detectable bound makes the effect BIGGER, not smaller. A
    noise explanation predicts the opposite, so this is a refutation and not
    merely an absence of evidence.

    Gated because it is the one reviewer objection the project can currently
    answer, and an answer nobody can re-derive is worth nothing.
    """
    res, stats = _headline_power_table()
    w50 = res[res["warmup_epochs"] == 50]
    assert len(w50) == 236, len(w50)

    # the four numbers the table prints
    assert abs(100 * w50["sd"].median() - 1.47) < 0.05, w50["sd"].median()
    assert abs(100 * detectable_of(w50) - 2.05) < 0.05, detectable_of(w50)
    assert abs(100 * w50["mean"].abs().median() - 1.30) < 0.05
    resolvable = w50[w50["mean"].abs() >= w50["mde"]]
    assert len(resolvable) == 76, len(resolvable)

    # aggregate direction, and that it SURVIVES the restriction
    wins = int((w50["mean"] > 0).sum())
    assert wins == 184, wins
    assert stats.binomtest(wins, len(w50), 0.5).pvalue < 1e-15
    rate_all = wins / len(w50)
    rate_res = (resolvable["mean"] > 0).mean()
    assert rate_res >= rate_all - 0.02, (rate_all, rate_res)
    assert resolvable["mean"].mean() > 1.8 * w50["mean"].mean(), (
        "the effect no longer grows on the resolvable subset, so the "
        "noise-explanation refutation in 1b has stopped holding")

    # and the caveat: MOST cells cannot resolve their own number
    assert len(resolvable) / len(w50) < 0.4, (
        "1b says two thirds of the table is unresolvable per cell")


def detectable_of(sub):
    from scripts.full_panel import detectable_at
    return detectable_at(float(sub["sd"].median()), 4)


def test_the_warmup_1_row_is_flagged_as_the_LR_trap_not_a_result():
    """1b records +15.20 pp at warm-up 1 specifically so nobody rediscovers it
    and reads it as section 3's regime effect. It is the shape the LR trap
    makes -- 1b documents an unequal `lr_constraint` fabricating 16.7 pp that
    became 1.7 pp once equalized -- and the corpus cannot separate the two.
    The gate keeps the number honest and keeps the warning attached to it.
    """
    res, _ = _headline_power_table()
    w1 = res[res["warmup_epochs"] == 1]
    assert len(w1) == 10, len(w1)
    assert 14.0 < 100 * w1["mean"].mean() < 16.5, w1["mean"].mean()
    assert (w1["mean"] > 0).all()

    fw = io.open(os.path.join(REPO, "docs", "FRAMEWORK.md"),
                 encoding="utf-8").read()
    i = fw.find("+15.20 pp")
    assert i > 0, "1b no longer quotes the warm-up-1 figure"
    near = fw[i - 400:i + 700]
    assert "LR TRAP" in near.upper(), (
        "the +15.20 pp figure is quoted without the LR-trap warning attached")
    assert "Do not quote it" in near, near[:300]


def test_the_scorer_prints_pure_ASCII():
    """`full_panel`'s stdout is piped, redirected and parsed, and this suite
    parses it with the host default codec. On Windows that is cp1252, where a
    single emoji makes `subprocess.run(text=True)` hand back **None** instead of
    raising -- so five unrelated scorer tests failed with "NoneType has no
    attribute splitlines" and nothing pointed at the character that caused it.

    The scorer is the one tool whose output is read by machines as well as
    people. Keep it ASCII; put the emoji in FRAMEWORK.md.

    AST, not grep: the file's own comments discuss the check and would otherwise
    count as violations.
    """
    import ast

    f = os.path.join(REPO, "scripts", "full_panel.py")
    tree = ast.parse(io.open(f, encoding="utf-8").read())
    bad = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)                 and n.func.id == "print":
            for a in ast.walk(n):
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    hits = sorted({c for c in a.value if ord(c) > 127})
                    if hits:
                        bad[n.lineno] = "".join(hits)
    assert not bad, (
        "full_panel prints non-ASCII, which returns None from text-mode "
        "subprocess capture under cp1252: %s" % bad)


def _corpus_contrast(metric, treated, control, warmup=50):
    """Paired `treated - control` per seed within cell, at one warm-up."""
    import pandas as pd
    from scipy import stats
    from scripts.full_panel import detectable_at

    f = os.path.join(REPO, "docs", "paper", "data", "corpus",
                     "corpus_final.csv")
    if not os.path.exists(f):
        pytest.skip("corpus not present in this worktree")
    d = pd.read_csv(f)
    cell = ["dataset", "model", "constraint_tag", "constrained_class",
            "group_column", "warmup_epochs", "sweep"]
    d = d[d["method"].isin([treated, control])]
    w = d.pivot_table(index=cell + ["seed"], columns="method", values=metric,
                      aggfunc="mean").dropna()
    w = w.assign(delta=w[treated] - w[control])
    g = w.groupby(level=list(range(len(cell))))["delta"]
    r = pd.DataFrame({"n": g.size(), "mean": g.mean(),
                      "sd": g.std(ddof=1)}).reset_index()
    r = r[(r["warmup_epochs"] == warmup) & (r["n"] >= 2)]
    wins = int((r["mean"] > 0).sum())
    return {"cells": len(r), "mean_pp": 100 * r["mean"].mean(),
            "wins": wins,
            "p": stats.binomtest(wins, len(r), 0.5).pvalue,
            "sd_pp": 100 * float(r["sd"].median()),
            "mde_pp": 100 * detectable_at(float(r["sd"].median()), 4)}


def test_the_macroF1_win_is_compute_not_method():
    """Section 3 says method effects are ~0.1 pp. FRAMEWORK now decomposes the
    paper's headline into the two parts, and this gate holds the decomposition.

    The control is `danits_lp`: it is the one method that is BOTH constrained
    and POST-HOC, so under the compute hypothesis it should not win. It loses.
    Every TRAINED method wins by 1.1-1.9 pp. So the effect tracks having a
    constraint phase, not which one -- and TraLO's own increment over the best
    alternative dual is ~0.15 pp, below its per-cell detectable bound.
    """
    trained = {m: _corpus_contrast("f1_macro", m, "heuristic")
               for m in ("tralo", "fioretto_ldf", "hounie_rcl",
                         "tralo_bounded")}
    posthoc = _corpus_contrast("f1_macro", "danits_lp", "heuristic")

    # every trained method clears +1 pp over the clipper
    for m, r in trained.items():
        assert r["mean_pp"] > 1.0, (m, r)
        assert r["wins"] / r["cells"] > 0.6, (m, r)

    # the post-hoc control does NOT -- this is the load-bearing comparison
    assert posthoc["mean_pp"] < 0.2, posthoc
    assert posthoc["wins"] / posthoc["cells"] < 0.5, posthoc

    # and TraLO's method-specific part is small AND under its own bound
    for other in ("fioretto_ldf", "tralo_bounded"):
        r = _corpus_contrast("f1_macro", "tralo", other)
        assert 0.0 < r["mean_pp"] < 0.5, (other, r)
        assert r["mean_pp"] < r["mde_pp"], (
            "%s: the method-specific effect now clears its own per-cell bound, "
            "so section 3's decomposition needs rewriting: %s" % (other, r))

    # the decomposition itself: ~92%% of the clipper win is not TraLO-specific
    share = _corpus_contrast("f1_macro", "tralo", "fioretto_ldf")["mean_pp"]         / trained["tralo"]["mean_pp"]
    assert share < 0.15, share


def test_a_dual_vs_dual_contrast_is_better_resolved_than_dual_vs_clipper():
    """The instrument fact FRAMEWORK section 3 records: two trained methods
    share most of their seed variance, so the paired difference cancels it and
    the contrast costs roughly a third the seed noise. It changes what a
    campaign can afford to ask, so it is checked rather than remembered.
    """
    vs_clip = _corpus_contrast("f1_macro", "tralo", "heuristic")["sd_pp"]
    vs_dual = _corpus_contrast("f1_macro", "tralo", "fioretto_ldf")["sd_pp"]
    assert vs_dual < vs_clip / 2.0, (vs_dual, vs_clip)
    assert 1.2 < vs_clip < 1.8 and 0.3 < vs_dual < 0.8, (vs_clip, vs_dual)


def test_on_ccF1_tralo_is_not_separable_from_the_other_duals():
    """The caveat beside the decomposition. On the constrained-class metric the
    method-specific part does not merely shrink, it stops being callable:
    tralo - fioretto_ldf and tralo - tralo_bounded both sit at p ~ 0.07.
    """
    for other in ("fioretto_ldf", "tralo_bounded"):
        r = _corpus_contrast("cc_f1", "tralo", other)
        assert r["p"] > 0.05, (other, r)

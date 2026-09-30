import torch

from tralo.fmow_local import ALM_PILOT, RECIPE, snapshot_side_steps, validate


def test_named_alm_config_and_seed_boundary():
    config = dict(RECIPE, seed=ALM_PILOT, snapshot_steps=True,
                  study="local_alm_direction_v1", step_radius=0.1, alm_rho=0.5)
    validate(config)
    for key, bad in (("alm_rho", 0.6), ("step_radius", 0.05), ("seed", 6200)):
        changed = dict(config, **{key: bad})
        try:
            validate(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(key + " bypassed the named protocol")


def test_runner_side_step_preserves_pto_and_advances_phr(tmp_path):
    model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 2.0, 0.0])
    pool = [torch.tensor([[0.2, 0.4], [-0.6, 0.8]]),
            torch.tensor([[0.3, -0.5], [0.1, 0.7]])]
    groups = ["A", "A", "B", "B"]
    quota = {"global_cap": 1, "local_caps": {"A": 0, "B": 1}}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    state = {"dual": torch.zeros(3)}
    steps = snapshot_side_steps(model, pool, groups, quota, 6300, 1, tmp_path,
                                fixed_radius=0.1, phr_state=state, phr_rho=0.5)
    assert set(steps) == {"joint", "global_dose", "sham", "phr_local"}
    assert steps["phr_local"]["applied"]
    assert abs(steps["phr_local"]["displacement"] - 0.1) < 1e-5
    assert torch.any(state["dual"] > 0)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert (tmp_path / "epoch01_phr_local.pt").is_file()
    assert steps["phr_local"]["probability_sha256"]


def test_old_fixed_runner_does_not_emit_phr(tmp_path):
    model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 2.0, 0.0])
    pool = [torch.zeros(2, 2)]
    quota = {"global_cap": 0, "local_caps": {"A": 0}}
    steps = snapshot_side_steps(model, pool, ["A", "A"], quota, 6199, 1, tmp_path,
                                fixed_radius=0.1)
    assert set(steps) == {"joint", "global_dose", "sham"}


def test_phr_duals_persist_by_cap_without_changing_common_model(tmp_path):
    model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 2.0, 0.0])
    pool = [torch.tensor([[0.2, 0.4], [-0.6, 0.8],
                          [0.3, -0.5], [0.1, 0.7]])]
    groups = ["A", "A", "B", "B"]
    quotas = [{"global_cap": 1, "local_caps": {"A": 0, "B": 1}},
              {"global_cap": 0, "local_caps": {"A": 0, "B": 0}}]
    states = [{"dual": torch.zeros(3)}, {"dual": torch.zeros(3)}]
    original = {key: value.clone() for key, value in model.state_dict().items()}
    for epoch in (1, 2):
        for index, quota in enumerate(quotas):
            directory = tmp_path / f"cap{index}_epoch{epoch}"
            directory.mkdir()
            previous = states[index]["dual"].clone()
            steps = snapshot_side_steps(model, pool, groups, quota, 6300, epoch,
                                        directory, fixed_radius=0.1,
                                        phr_state=states[index], phr_rho=0.5)
            assert steps["phr_local"]["dual_before"] == previous.tolist()
            assert steps["phr_local"]["dual_after"] == states[index]["dual"].tolist()
        assert all(torch.equal(model.state_dict()[key], value)
                   for key, value in original.items())
    assert not torch.equal(states[0]["dual"], states[1]["dual"])

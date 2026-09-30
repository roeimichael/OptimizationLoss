"""Independent integrity examples for the local PHR snapshot-direction study."""

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from analysis import score_fmow_local_alm as score
from tralo.knee_end_to_end import infer
from tralo.local_alm import snapshot_phr_step


def _matrix(rows):
    return torch.tensor([row + [0.] * 5 for row in rows])


def test_phr_recounts_signed_residuals_dual_and_inactive_snapshot():
    pto = _matrix([[.1, .8, .1], [.7, .3, .0]])
    groups = ["A", "B"]
    quota = {"global_cap": 1, "local_caps": {"A": 1, "B": 0}}
    soft_a, soft_b = float(pto[0, 1]), float(pto[1, 1])
    pooled = soft_a + soft_b
    residuals = [pooled - 1, soft_a - 1, soft_b]
    penalty = .25 * (max(0, residuals[0]) ** 2 + max(0, residuals[1]) ** 2 +
                     max(0, residuals[2]) ** 2)
    dual_after = [max(0, .5 * g) for g in residuals]
    record = {"applied": False, "displacement": 0.0, "rho": .5,
              "activation_reason": "zero_phr_gradient", "gradient_norm": 0.0,
              "scope_directional_derivatives": {}, "dual_before": [0., 0., 0.],
              "dual_after": dual_after, "residuals_before": residuals,
              "residuals_after": residuals, "penalty_before": penalty,
              "penalty_after": penalty, "hard_before_global": 1,
              "hard_after_global": 1, "hard_before_local": {"A": 1, "B": 0},
              "hard_after_local": {"A": 1, "B": 0},
              "soft_before_global": pooled, "soft_after_global": pooled,
              "soft_before_local": {"A": soft_a, "B": soft_b},
              "soft_after_local": {"A": soft_a, "B": soft_b}}
    assert score._audit_phr(record, pto, pto.clone(), groups, quota, [0., 0., 0.]) == pytest.approx(dual_after)
    wrong = dict(record, dual_after=[0., 0., 0.])
    with pytest.raises(RuntimeError, match="projected dual"):
        score._audit_phr(wrong, pto, pto, groups, quota, [0., 0., 0.])
    wrong = dict(record, residuals_before=[-g for g in residuals])
    with pytest.raises(RuntimeError, match="signed residual"):
        score._audit_phr(wrong, pto, pto, groups, quota, [0., 0., 0.])
    with pytest.raises(RuntimeError, match="continuity"):
        score._audit_phr(record, pto, pto, groups, quota, [0.1, 0., 0.])


def test_phr_applied_dose_and_direction_identity():
    pto = _matrix([[.1, .8, .1], [.7, .3, .0]])
    groups = ["A", "B"]
    quota = {"global_cap": 1, "local_caps": {"A": 1, "B": 0}}
    counts, residuals = score._scope_values(pto, groups, quota)
    dual_after = [max(0, .5 * g) for g in residuals]
    penalty = sum(max(0, .5 * g) ** 2 for g in residuals)
    penalty /= 1.0  # denominator 2*rho is one
    record = {"applied": True, "radius": .1, "displacement": .1,
              "tensor_displacement_norms": [.1], "rho": .5,
              "activation_reason": "finite_nonzero_phr_gradient", "gradient_norm": 2.,
              "scope_directional_derivatives": {"global": -10., "A": 0., "B": -10.},
              "dual_before": [0., 0., 0.], "dual_after": dual_after,
              "residuals_before": residuals, "residuals_after": residuals,
              "penalty_before": penalty, "penalty_after": penalty,
              "hard_before_global": 1, "hard_after_global": 1,
              "hard_before_local": {"A": 1, "B": 0}, "hard_after_local": {"A": 1, "B": 0},
              "soft_before_global": counts["global"], "soft_after_global": counts["global"],
              "soft_before_local": {"A": counts["A"], "B": counts["B"]},
              "soft_after_local": {"A": counts["A"], "B": counts["B"]}}
    # Identity weights are .5*g for active scopes, giving -2 here.
    record["scope_directional_derivatives"]["B"] = (
        -2 - .5 * residuals[0] * -10) / (.5 * residuals[2])
    score._audit_phr(record, pto, pto, groups, quota, [0., 0., 0.])
    wrong = json.loads(json.dumps(record))
    wrong["scope_directional_derivatives"]["global"] = 0.
    with pytest.raises(RuntimeError, match="directional gradient identity"):
        score._audit_phr(wrong, pto, pto, groups, quota, [0., 0., 0.])
    wrong = dict(record, displacement=.05)
    with pytest.raises(RuntimeError, match="displacement"):
        score._audit_phr(wrong, pto, pto, groups, quota, [0., 0., 0.])


def test_actual_cpu_runner_record_passes_independent_phr_gate():
    model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0., 2., 0.])
    chunks = [torch.tensor([[.2, .4], [-.6, .8]]),
              torch.tensor([[.3, -.5], [.1, .7]])]
    groups = ["A", "A", "B", "B"]
    quota = {"global_cap": 1, "local_caps": {"A": 0, "B": 1}}
    before = infer(model, chunks)
    record, next_dual = snapshot_phr_step(model, chunks, groups, 1, 1,
                                          quota["local_caps"], torch.zeros(3),
                                          rho=.5, radius=.1)
    after = infer(model, chunks)
    assert record["applied"]
    assert score._audit_phr(record, before, after, groups, quota,
                           [0., 0., 0.]) == pytest.approx(next_dual.tolist())


def test_complete_block_reports_all_primary_and_negative_secondary(tmp_path, monkeypatch):
    rows = {}
    for seed in score.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
        caps = {}
        for divisor in score.prior.DIVISORS:
            arms = {}
            for arm in score.ARMS:
                value = .4 + (seed - score.SEEDS[0]) * .001
                if arm == "ens_phr_local":
                    value -= .005 + (seed % 3) * .001
                arms[arm] = {"allocated": {metric: value for metric in score.prior.METRICS},
                             "prediction_sha256": f"{seed}-{divisor}-{arm}"}
            caps[str(divisor)] = {"arms": arms}
        rows[seed] = {"seed": seed, "manifest_sha256": "same", "caps": caps}
    monkeypatch.setattr(score, "load_seed", lambda path, data_root: rows[int(path.name[4:])])
    monkeypatch.setattr(score.prior, "source", lambda: {"source": "hash"})
    path = tmp_path / "result.json"
    report = score.main(tmp_path, tmp_path, path)
    assert len(report["seeds"]) == 12
    assert len(report["primary_family"]) == 4
    assert all(report["contrasts"]["cc_f1"][name]["mean"] < 0
               for name in report["primary_family"])
    assert not any(row["meets_registered_exploratory_signal"]
                   for row in report["exploratory_signal_by_cap"].values())
    assert set(report["seeds"][0]["caps"]["10"]["arms"]) == set(score.ARMS)
    assert path.is_file()
    (tmp_path / "seed6312").rename(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="incomplete"):
        score.main(tmp_path, tmp_path)


def _make_runner_shaped_alm_pilot(tmp_path, monkeypatch, *, active=True):
    path = Path(__file__).with_name("test_score_fmow_local.py")
    spec = importlib.util.spec_from_file_location("score_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot_root, ref_root, pilot, ref, identities, quotas = module.make_runner_shaped_pilot(
        tmp_path, monkeypatch)
    configs = tmp_path / "configs"
    configs.mkdir()
    monkeypatch.setattr(score, "CONFIGS", configs)
    for directory in (pilot, ref):
        config = json.loads((directory / "config.json").read_text())
        config.update(seed=6300, study=score.STUDY, step_radius=.1, alm_rho=.5)
        (directory / "config.json").write_text(json.dumps(config))
        input_path = configs / f"fmow_local_6300_{'step' if directory == pilot else 'ref'}.json"
        input_path.write_text(json.dumps(config, indent=2) + "\n")
        top = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        top[0]["config_sha256"] = score.prior.sha256(input_path)
        top[0]["device"] = "synthetic-cpu"
        top[0]["precision"] = "fp32"
        top[-1]["seconds"] = 60.
        (directory / "events.jsonl").write_text("\n".join(json.dumps(row) for row in top) + "\n")
        job = "6300_step" if directory == pilot else "6300_ref"
        final_dir = directory.parent / ("seed6300" if directory == pilot else "seed6300_ref")
        common = {"job": job, "seed": 6300, "run_root": str(directory.parent.resolve()),
                  "output_dir": str(final_dir.resolve()), "release_commit": "a" * 40,
                  "host": "dsisco02", "gpu_uuid": "GPU-synthetic"}
        launch = {**common, "gpu_index": 2, "precision": "fp32",
                  "source_sha256": {"x.py": "hash"},
                  "config_sha256": score.prior.sha256(input_path),
                  "data_root": "/synthetic/data", "started_utc": "2026-09-30T00:00:00Z"}
        complete = {**common, "exit_code": 0, "ended_utc": "2026-09-30T00:01:00Z"}
        (directory.parent / f"seed{job}.launch.json").write_text(json.dumps(launch))
        (directory.parent / f"seed{job}.complete.json").write_text(json.dumps(complete))
        summary = json.loads((directory / "summary.json").read_text())
        summary["seed"] = 6300
        if directory == pilot:
            events = [json.loads(line) for line in
                      (directory / "retrain1" / "events.jsonl").read_text().splitlines()]
            duals = {divisor: [0.] * 6 for divisor in score.prior.DIVISORS}
            for epoch in range(1, 7):
                for divisor in score.prior.DIVISORS:
                    quota = quotas[str(divisor)]
                    pto = score.prior._probabilities(
                        directory / "retrain1" / f"epoch{epoch:02d}.pt", len(identities))
                    groups = [row["location"] for row in identities]
                    counts, residuals = score._scope_values(pto, groups, quota)
                    old = duals[divisor]
                    projected = [max(0., lam + .5 * g) for lam, g in zip(old, residuals)]
                    penalty = sum((max(0., lam + .5 * g) ** 2 - lam ** 2)
                                  for lam, g in zip(old, residuals))
                    hard, local = score.prior._count(pto.argmax(1).tolist(), groups)
                    record = {"applied": False, "displacement": 0., "rho": .5,
                              "activation_reason": "zero_phr_gradient", "gradient_norm": 0.,
                              "scope_directional_derivatives": {}, "dual_before": old,
                              "dual_after": projected, "residuals_before": residuals,
                              "residuals_after": residuals, "penalty_before": penalty,
                              "penalty_after": penalty, "hard_before_global": hard,
                              "hard_after_global": hard, "hard_before_local": local,
                              "hard_after_local": local, "soft_before_global": counts["global"],
                              "soft_after_global": counts["global"],
                              "soft_before_local": {g: counts[g] for g in quota["local_caps"]},
                              "soft_after_local": {g: counts[g] for g in quota["local_caps"]}}
                    if active and epoch == 1:
                        assert residuals[0] > 0
                        record.update(applied=True, displacement=.1, radius=.1,
                                      tensor_displacement_norms=[.1], gradient_norm=1.,
                                      activation_reason="finite_nonzero_phr_gradient",
                                      scope_directional_derivatives={
                                          "global": -1 / (.5 * residuals[0]),
                                          **{g: 0. for g in quota["local_caps"]}})
                    side = directory / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_phr_local.pt"
                    torch.save(pto, side)
                    record["probability_sha256"] = score.prior.sha256(side)
                    summary["steps"][str(epoch)][str(divisor)]["phr_local"] = record
                    event = next(row for row in events if row["event"] == "snapshot_cap" and
                                 row["epoch"] == epoch and row["divisor"] == divisor)
                    event["steps"]["phr_local"] = record
                    duals[divisor] = projected
            (directory / "retrain1" / "events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events) + "\n")
        (directory / "summary.json").write_text(json.dumps(summary))
    pilot.rename(pilot_root / "seed6300")
    ref.rename(ref_root / "seed6300_ref")
    return pilot_root, ref_root, identities, quotas


def test_label_blind_pilot_gate_recounts_dual_across_all_epochs(tmp_path, monkeypatch):
    pilot_root, ref_root, _, _ = _make_runner_shaped_alm_pilot(tmp_path, monkeypatch)
    result = score.gate(pilot_root, ref_root)
    assert result["pto_equal"] is True
    assert result["development_labels_accessed"] is False
    assert result["projected_total_gpu_hours"] == pytest.approx(14 * 60 / 3600)
    assert result["phr_active_epochs_by_cap"] == {"10": 1, "20": 1}
    # Changing a later multiplier must fail even though the first epoch is intact.
    directory = pilot_root / "seed6300"
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["steps"]["2"]["10"]["phr_local"]["dual_before"][0] += .1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="step event/summary/quota disagreement"):
        score.gate(pilot_root, ref_root)


def test_pilot_rejects_all_noop_phr_despite_positive_residuals(tmp_path, monkeypatch):
    pilot_root, ref_root, _, _ = _make_runner_shaped_alm_pilot(tmp_path, monkeypatch,
                                                                 active=False)
    with pytest.raises(RuntimeError, match="no meaningful active PHR step"):
        score.gate(pilot_root, ref_root)


def test_pilot_requires_same_host_device_and_fp32(tmp_path, monkeypatch):
    pilot_root, ref_root, _, _ = _make_runner_shaped_alm_pilot(tmp_path, monkeypatch)
    launch_path = ref_root / "seed6300_ref.launch.json"
    launch = json.loads(launch_path.read_text())
    launch["host"] = "dsisco01"
    launch_path.write_text(json.dumps(launch))
    with pytest.raises(RuntimeError, match="launch/completion receipt"):
        score.gate(pilot_root, ref_root)
    complete_path = ref_root / "seed6300_ref.complete.json"
    complete = json.loads(complete_path.read_text())
    complete["host"] = "dsisco01"
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(RuntimeError, match="pilot/reference setup"):
        score.gate(pilot_root, ref_root)
    complete["host"] = "dsisco02"
    complete_path.write_text(json.dumps(complete))
    launch["host"] = "dsisco02"
    launch_path.write_text(json.dumps(launch))
    top_path = ref_root / "seed6300_ref" / "events.jsonl"
    rows = [json.loads(line) for line in top_path.read_text().splitlines()]
    rows[0]["precision"] = "bf16"
    top_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="launch/completion receipt"):
        score.gate(pilot_root, ref_root)
    rows[0]["precision"] = "fp32"
    rows[0]["device"] = "different GPU model"
    top_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="pilot/reference setup"):
        score.gate(pilot_root, ref_root)


def test_runner_shaped_full_scorer_uses_labels_ensemble_and_allocator(tmp_path, monkeypatch):
    pilot_root, _, identities, quotas = _make_runner_shaped_alm_pilot(tmp_path, monkeypatch)
    template = pilot_root / "seed6300"
    root = tmp_path / "full"
    root.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    labels = np.zeros(3442, dtype=np.int64)
    labels[:1673:7] = 1
    np.save(data / "test_labels.npy", labels)
    files = {"f": "data", "test_labels.npy": score.prior.sha256(data / "test_labels.npy")}
    monkeypatch.setattr(score.prior, "FILES", files)
    groups = [row["location"] for row in identities]
    ids = [row["sample_id"] for row in identities]
    swap = None
    for seed in score.SEEDS:
        directory = root / f"seed{seed}"
        shutil.copytree(template, directory)
        config = json.loads((directory / "config.json").read_text())
        config["seed"] = seed
        (directory / "config.json").write_text(json.dumps(config))
        input_path = score.CONFIGS / f"fmow_local_{seed}_step.json"
        input_path.write_text(json.dumps(config, indent=2) + "\n")
        manifest = {"files": files, "counts": {"train": 15841, "stop": 1829, "dev": 1673},
                    "quotas": quotas, "stop_countries": [],
                    "dev_countries": ["IRQ", "NLD", "DZA", "PHL", "TUR"],
                    "reserved_countries": sorted(score.prior.RESERVED),
                    "rows": [dict(row, split="val") for row in identities]}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        top_path = directory / "events.jsonl"
        top = [json.loads(line) for line in top_path.read_text().splitlines()]
        top[0].update(config_sha256=score.prior.sha256(input_path), data_files=files,
                      manifest_sha256=score.prior.sha256(directory / "manifest.json"))
        top_path.write_text("\n".join(json.dumps(row) for row in top) + "\n")
        common = {"job": f"{seed}_step", "seed": seed, "host": "dsisco02",
                  "gpu_uuid": "GPU-synthetic", "release_commit": "a" * 40,
                  "run_root": str(root.resolve()), "output_dir": str(directory.resolve())}
        (root / f"seed{seed}_step.launch.json").write_text(json.dumps({
            **common, "gpu_index": 2, "precision": "fp32", "source_sha256": {"x.py": "hash"},
            "config_sha256": score.prior.sha256(input_path), "data_root": str(data),
            "started_utc": "2026-09-30T00:00:00Z"}))
        (root / f"seed{seed}_step.complete.json").write_text(json.dumps({
            **common, "exit_code": 0, "ended_utc": "2026-09-30T00:01:00Z"}))
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["seed"] = seed
        events_path = directory / "retrain1" / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        duals = {divisor: [0.] * 6 for divisor in score.prior.DIVISORS}
        for epoch in range(1, 7):
            q = .1 + .3 * ((np.arange(1673) + seed * 137) % 1673) / 1673
            q += (epoch - 3) * .0001
            pto = torch.zeros(1673, score.prior.CLASSES)
            pto[:, 1] = torch.tensor(q, dtype=torch.float32)
            pto[:, 0] = 1 - pto[:, 1]
            snapshot = directory / "retrain1" / f"epoch{epoch:02d}.pt"
            torch.save(pto, snapshot)
            summary["pto_snapshot_sha256"][str(epoch)] = score.prior.sha256(snapshot)
            for divisor in score.prior.DIVISORS:
                quota = quotas[str(divisor)]
                if seed == score.SEEDS[0] and divisor == 10 and epoch == 1:
                    # Independent top-within-country, then global-top recount.
                    eligible = []
                    for group in sorted(quota["local_caps"]):
                        members = [i for i, country in enumerate(groups) if country == group]
                        eligible.extend(sorted(members, key=lambda i: (-q[i], ids[i]))
                                        [:quota["local_caps"][group]])
                    selected = set(sorted(eligible, key=lambda i: (-q[i], ids[i]))
                                   [:quota["global_cap"]])
                    for group in sorted(quota["local_caps"]):
                        evictions = [i for i in sorted(selected) if groups[i] == group and labels[i] == 0]
                        entries = [i for i in range(1673) if i not in selected and
                                   groups[i] == group and labels[i] == 1]
                        if evictions and entries:
                            swap = (entries[0], evictions[0], selected)
                            break
                    assert swap is not None
                records = summary["steps"][str(epoch)][str(divisor)]
                counts, residuals = score._scope_values(pto, groups, quota)
                local_soft = {g: counts[g] for g in quota["local_caps"]}
                joint = records["joint"]
                joint["soft_before_global"] = counts["global"]
                joint["soft_before_local"] = local_soft
                dual = duals[divisor]
                penalty = sum((max(0., lam + .5 * g) ** 2 - lam ** 2)
                              for lam, g in zip(dual, residuals))
                hard, local = score.prior._count(pto.argmax(1).tolist(), groups)
                phr_side = pto.clone()
                if seed == score.SEEDS[0] and divisor == 10:
                    target, evict, _ = swap
                    phr_side[target, 1], phr_side[target, 0] = .49, .51
                    phr_side[evict, 1], phr_side[evict, 0] = .01, .99
                after_counts, after_residuals = score._scope_values(phr_side, groups, quota)
                after_hard, after_local = score.prior._count(
                    phr_side.argmax(1).tolist(), groups)
                projected = [max(0., lam + .5 * g) for lam, g in zip(dual, after_residuals)]
                penalty_after = sum((max(0., lam + .5 * g) ** 2 - lam ** 2)
                                    for lam, g in zip(dual, after_residuals))
                applied = seed == score.SEEDS[0] and divisor == 10
                records["phr_local"] = {
                    "applied": applied, "displacement": .1 if applied else 0., "rho": .5,
                    "activation_reason": ("finite_nonzero_phr_gradient" if applied
                                          else "zero_phr_gradient"),
                    "gradient_norm": 1. if applied else 0.,
                    "scope_directional_derivatives": (
                        {"global": -1 / (dual[0] + .5 * residuals[0]),
                         **{g: 0. for g in quota["local_caps"]}} if applied else {}),
                    "dual_before": dual,
                    "dual_after": projected, "residuals_before": residuals,
                    "residuals_after": after_residuals, "penalty_before": penalty,
                    "penalty_after": penalty_after, "hard_before_global": hard,
                    "hard_after_global": after_hard, "hard_before_local": local,
                    "hard_after_local": after_local, "soft_before_global": counts["global"],
                    "soft_after_global": after_counts["global"],
                    "soft_before_local": local_soft,
                    "soft_after_local": {g: after_counts[g] for g in quota["local_caps"]}}
                if applied:
                    records["phr_local"].update(radius=.1, tensor_displacement_norms=[.1])
                for arm in ("joint", "global_dose", "sham", "phr_local"):
                    artifact = directory / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_{arm}.pt"
                    torch.save(phr_side if arm == "phr_local" else pto, artifact)
                    records[arm]["probability_sha256"] = score.prior.sha256(artifact)
                event = next(row for row in events if row["event"] == "snapshot_cap" and
                             row["epoch"] == epoch and row["divisor"] == divisor)
                event["steps"] = records
                duals[divisor] = projected
        final = directory / "retrain1" / "final_probabilities.pt"
        shutil.copyfile(directory / "retrain1" / "epoch01.pt", final)
        summary["final_probability_sha256"] = score.prior.sha256(final)
        summary_path.write_text(json.dumps(summary))
        events_path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    report = score.main(root, data)
    assert len(report["seeds"]) == 12
    assert report["seeds"][0]["window"] == [1, 2, 3, 4, 5, 6]
    first = report["seeds"][0]["caps"]["10"]["arms"]["ens_pto"]
    phr = report["seeds"][0]["caps"]["10"]["arms"]["ens_phr_local"]
    confusion = first["class1_confusion"]
    assert first["allocated"]["cc_f1"] == pytest.approx(
        2 * confusion["tp"] / (2 * confusion["tp"] + confusion["fp"] + confusion["fn"]))
    assert len(first["selected_ids"]) == quotas["10"]["global_cap"]
    assert first["raw"]["cc_f1"] == 0
    assert first["allocated"]["cc_f1"] > 0
    target, evict, expected_pto = swap
    assert set(first["selected_ids"]) == {ids[i] for i in expected_pto}
    assert set(phr["selected_ids"]) == {ids[i] for i in (expected_pto - {evict}) | {target}}
    assert phr["class1_confusion"] == {"tp": confusion["tp"] + 1,
                                      "fp": confusion["fp"] - 1,
                                      "fn": confusion["fn"] - 1}
    assert phr["allocated"]["cc_f1"] == pytest.approx(
        2 * (confusion["tp"] + 1) /
        (2 * (confusion["tp"] + 1) + confusion["fp"] - 1 + confusion["fn"] - 1))
    assert phr["allocated"]["cc_f1"] > first["allocated"]["cc_f1"]
    assert len({row["caps"]["10"]["arms"]["ens_pto"]["prediction_sha256"]
                for row in report["seeds"]}) == 12
    labels[0] = 1 - labels[0]
    np.save(data / "test_labels.npy", labels)
    with pytest.raises(RuntimeError, match="label file hash"):
        score.load_seed(root / "seed6301", data)
    monkeypatch.setattr(score.prior, "allocate_local_capped_first",
                        lambda *args, **kwargs: [0] * 1673)
    # Restore label-file hash before checking independent allocator agreement.
    labels[0] = 1 - labels[0]
    np.save(data / "test_labels.npy", labels)
    with pytest.raises(RuntimeError, match="independent joint allocation"):
        score.load_seed(root / "seed6301", data)

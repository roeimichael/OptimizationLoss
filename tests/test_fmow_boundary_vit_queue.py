"""CPU-only tests for the separate exclusive fmow2 ViT launcher."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shlex
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fmow_local_boundary_vit_queue.sh"
SMOKE_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fmow_local_boundary_vit_smoke.py"
REAL_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fmow_local_boundary_vit_real_preflight.py"


def linux(path):
    path = Path(path).resolve()
    if not path.drive:
        return path.as_posix()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}"


def wsl(*args):
    return subprocess.run(["bash", "-lc", shlex.join(args)], text=True,
                          capture_output=True, check=True).stdout.strip()


@pytest.fixture
def queue(tmp_path):
    release = tmp_path / "release"
    configs = release / "experiments" / "configs" / "fmow_local_boundary_vit_20260930"
    configs.mkdir(parents=True)
    package = release / "tralo"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "knee_experiment.py").write_text(
        "import hashlib\n"
        "from pathlib import Path\n"
        "def source():\n    return {'fmow_local.py': 'test-source-hash'}\n"
        "def digest(p):\n    return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n")
    (package / "fmow_local.py").write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "VIT_WEIGHT_SHA256 = 'c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9'\n"
        "def vit_weight_provenance():\n    return {'sha256': VIT_WEIGHT_SHA256}\n"
        "def validate(c):\n"
        "    if c.get('study') != 'local_boundary_vit_v1':\n"
        "        raise ValueError('wrong study')\n"
        "if __name__ == '__main__':\n"
        "    output = Path(sys.argv[-1])\n"
        "    output.mkdir(exist_ok=False)\n"
        "    time.sleep(float(os.environ.get('MOCK_RUNNER_SLEEP', '0')))\n"
        "    (output / 'observed.json').write_text(json.dumps({\n"
        "        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "        'config': Path(sys.argv[-2]).name}))\n"
        "    if os.environ.get('MOCK_RUNNER_FAIL') == '1':\n"
        "        raise SystemExit(7)\n")
    (package / "fmow_yuval.py").write_text("FILES = {'test_images.npy': 'abc'}\n")
    tools = release / "tools"
    tools.mkdir()
    mock_entry = '''
def main(argv=None):
    import hashlib
    import subprocess
    from tralo.knee_experiment import source
    argv = sys.argv[1:] if argv is None else argv
    commit, data, gpu_index, receipt_path = argv
    if os.environ.get("FMOW_VIT_INHERITED_LOCK_FD") != "9":
        raise RuntimeError("queue did not pass shared GPU lock")
    os.fstat(9)
    if os.environ.get("MOCK_SMOKE_EXIT_BEFORE_RECEIPT") == "1":
        raise SystemExit(7)
    started = time.time()
    time.sleep(0.02)
    phase = dict(completed=True, seconds=0.01, peak_allocated_bytes=100)
    cap = dict(joint_gradient_norm=1.0, phr_gradient_norm=1.0,
               joint_applied=True, phr_applied=True, all_four_arms=True,
               scope_derivatives_finite=True, pto_unchanged=True)
    receipt = dict(
        release_commit=commit, host=subprocess.run(["hostname", "-f"],
            check=True, capture_output=True, text=True).stdout.strip(),
        gpu_uuid="GPU-TEST-UUID", gpu_index=int(gpu_index),
        backbone="vit_b_16", batch_size=16, development_batch_size=8,
        weight_sha256=WEIGHT_SHA, precision="fp32", label_free=True,
        memory_smoke_passed=os.environ.get("MOCK_SMOKE_FAIL") != "1",
        peak_allocated_bytes=100, total_memory_bytes=1000,
        source_sha256=source(),
        smoke_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_files={}, device_name="Test GPU", development_pool_count=1673,
        started_utc=started, ended_utc=time.time(),
        phases=dict(
            train_backward=dict(phase, input_shape=[16, 3, 224, 224],
                                loss=1.0, gradient_norm=1.0, optimizer_step=True),
            development_inference=dict(phase, probabilities_shape=[1673, 8],
                                       finite_rows=1673, row_sum_max_error=0.0),
            side_copy_constraint_gradient=dict(phase, pto_unchanged=True,
                                               caps={"10": cap, "20": cap})))
    field = os.environ.get("MOCK_SMOKE_MUTATE_FIELD")
    if field:
        receipt[field] = json.loads(os.environ["MOCK_SMOKE_MUTATE_JSON"])
    with open(receipt_path, "x", encoding="utf-8") as stream:
        json.dump(receipt, stream)
    if not receipt["memory_smoke_passed"]:
        raise SystemExit(7)

if __name__ == "__main__":
    main()
'''
    real_smoke = SMOKE_SCRIPT.read_text()
    (tools / SMOKE_SCRIPT.name).write_text(
        real_smoke.split('if __name__ == "__main__":')[0] + mock_entry)
    mock_real_entry = '''
def main(argv=None):
    import hashlib
    import subprocess
    from tralo.fmow_yuval import FILES
    from tralo.knee_experiment import source
    argv = sys.argv[1:] if argv is None else argv
    commit, data, gpu_index, receipt_path = argv
    if os.environ.get("FMOW_VIT_INHERITED_LOCK_FD") != "9":
        raise RuntimeError("queue did not pass GPU lock to numerical preflight")
    os.fstat(9)
    if os.environ.get("MOCK_REAL_EXIT_BEFORE_RECEIPT") == "1":
        raise SystemExit(7)
    started = time.time()
    time.sleep(0.02)
    artifacts = Path(receipt_path).with_suffix(".artifacts")
    artifacts.mkdir()
    artifact_sha = {}
    for arm in ARMS:
        path = artifacts / f"epoch01_{arm}.pt"
        path.write_bytes(arm.encode())
        artifact_sha[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(
        release_commit=commit, host=subprocess.run(["hostname", "-f"],
            check=True, capture_output=True, text=True).stdout.strip(),
        gpu_uuid="GPU-TEST-UUID", gpu_index=int(gpu_index),
        precision="fp32", backbone="vit_b_16", weight_sha256=WEIGHT_SHA,
        development_batch_size=8, chunk_sizes=[8, 7],
        pretrained_weight={'sha256': WEIGHT_SHA},
        development_labels_accessed=False,
        preflight_passed=os.environ.get("MOCK_REAL_FAIL") != "1",
        images_count=15, country_counts=dict.fromkeys(COUNTRIES, 3),
        pto_unchanged=True, arms_audited=list(ARMS), artifact_sha256=artifact_sha,
        max_probability_difference=1e-7,
        gradient_relative_errors={scope: 0.001 for scope in SCOPES},
        finite_differences={f"{scope}@{epsilon}": dict(analytic=-0.2,
            numeric=-0.2, error=0.0, tolerance=0.013)
            for scope in SCOPES for epsilon in EPSILONS},
        source_sha256=source(), data_file_sha256=FILES, device_name="Test GPU",
        preflight_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_utc=started, ended_utc=time.time())
    field = os.environ.get("MOCK_REAL_MUTATE_FIELD")
    if field:
        receipt[field] = json.loads(os.environ["MOCK_REAL_MUTATE_JSON"])
    with open(receipt_path, "x", encoding="utf-8") as stream:
        json.dump(receipt, stream)
    if not receipt["preflight_passed"]:
        raise SystemExit(7)

if __name__ == "__main__":
    main()
'''
    real_preflight = REAL_SCRIPT.read_text()
    (tools / REAL_SCRIPT.name).write_text(
        real_preflight.split('if __name__ == "__main__":')[0] + mock_real_entry)
    analysis = release / "analysis"
    analysis.mkdir()
    (analysis / "score_fmow_boundary_vit.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] != '--gate': raise SystemExit(2)\n"
        "step, ref, output = sys.argv[2:]\n"
        "with open(output, 'x', encoding='utf-8') as stream:\n"
        "    json.dump({'status': 'vit_pilot_integrity_pass',\n"
        "               'pilot_step_root': step, 'pilot_ref_root': ref}, stream)\n")
    for job in ("6500_step", "6500_ref", *(f"{seed}_step" for seed in range(6501, 6513))):
        seed, arm = job.split("_")
        (configs / f"fmow_local_{job}.json").write_text(json.dumps({
            "study": "local_boundary_vit_v1", "seed": int(seed),
            "snapshot_steps": arm == "step"}))
    wsl("git", "-C", linux(release), "init", "-q")
    wsl("git", "-C", linux(release), "add", ".")
    wsl("git", "-C", linux(release), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-qm", "fixture")
    sha = wsl("git", "-C", linux(release), "rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", sha)

    runs = tmp_path / "runs"
    runs.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_bytes(("#!/usr/bin/env bash\n"
                   "if [[ $* = *--query-gpu=uuid* ]]; then echo GPU-TEST-UUID; exit 0; fi\n"
                   "if [[ $* = *--query-compute-apps=pid* ]]; then\n"
                   "  count=$(cat \"$MOCK_COUNT_FILE\" 2>/dev/null || echo 0)\n"
                   "  count=$((count + 1))\n"
                   "  echo \"$count\" > \"$MOCK_COUNT_FILE\"\n"
                   "  if [[ $count = ${MOCK_BUSY_CALL:-0} ]]; then echo 4321; fi\n"
                   "  exit 0\n"
                   "fi\n"
                   "exit 2\n").encode())
    wsl("chmod", "+x", linux(smi))

    script = tmp_path / "queue.sh"
    text = SCRIPT.read_text().replace(
        "REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA", f"REL={linux(release)}"
    ).replace(
        "RUNS=/home/dsi/michaer8/tralo-rebuild/runs", f"RUNS={linux(runs)}"
    ).replace(
        "DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice", f"DATA={linux(data)}"
    ).replace(
        "PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python", "PY=/usr/bin/python3"
    ).replace(
        "nvidia-smi", linux(smi)
    )
    script.write_bytes(text.encode())
    env = {"MOCK_COUNT_FILE": linux(tmp_path / "compute_queries")}
    return script, sha, tmp_path, env


def run_queue(queue, mode="pilot-step", root=None, extra_env=None):
    script, sha, parent, env = queue
    root = root or parent / "runs" / mode
    env = dict(env, **(extra_env or {}))
    command = ["env", *(f"{key}={value}" for key, value in env.items()),
               "bash", linux(script), sha, linux(root), "0", mode]
    return subprocess.run(["bash", "-lc", shlex.join(command)],
                          text=True, capture_output=True), root


def prepare_pilots(queue):
    step_result, step_root = run_queue(queue, mode="pilot-step")
    assert step_result.returncode == 0, step_result.stderr
    ref_result, ref_root = run_queue(queue, mode="pilot-ref")
    assert ref_result.returncode == 0, ref_result.stderr
    gate = queue[2] / "pilot-gate.json"
    gate.write_text(json.dumps({"status": "vit_pilot_integrity_pass",
                                "pilot_step_root": linux(step_root),
                                "pilot_ref_root": linux(ref_root)}))
    env = {"FMOW_VIT_PILOT_STEP_ROOT": linux(step_root),
           "FMOW_VIT_PILOT_REF_ROOT": linux(ref_root),
           "FMOW_VIT_PILOT_GATE_RECEIPT": linux(gate)}
    (queue[2] / "compute_queries").write_text("0")
    return env, step_root, ref_root, gate


def test_shell_syntax():
    subprocess.run(["bash", "-n", linux(SCRIPT)], check=True)


def test_separate_pilot_receipts_and_new_root_refusal(queue):
    for mode, job, output in (("pilot-step", "6500_step", "seed6500"),
                              ("pilot-ref", "6500_ref", "seed6500_ref")):
        result, root = run_queue(queue, mode=mode)
        assert result.returncode == 0, result.stderr
        launch = json.loads((root / f"seed{job}.launch.json").read_text())
        complete = json.loads((root / f"seed{job}.complete.json").read_text())
        observed = json.loads((root / output / "observed.json").read_text())
        assert launch["output_dir"] == linux(root / output)
        assert launch["source_sha256"] == {"fmow_local.py": "test-source-hash"}
        assert launch["host"] and launch["gpu_uuid"] == "GPU-TEST-UUID"
        assert launch["precision"] == "fp32" and launch["release_commit"] == queue[1]
        assert re.fullmatch(r"[0-9a-f]{64}", launch["config_sha256"])
        smoke = root / "vit_memory_smoke.json"
        smoke_launch = root / "vit_memory_smoke.launch.json"
        smoke_complete = root / "vit_memory_smoke.complete.json"
        assert launch["memory_smoke_receipt_path"] == linux(smoke)
        assert launch["memory_smoke_receipt_sha256"] == hashlib.sha256(smoke.read_bytes()).hexdigest()
        assert launch["memory_smoke_launch_path"] == linux(smoke_launch)
        assert launch["memory_smoke_launch_sha256"] == hashlib.sha256(smoke_launch.read_bytes()).hexdigest()
        assert launch["memory_smoke_complete_path"] == linux(smoke_complete)
        assert launch["memory_smoke_complete_sha256"] == hashlib.sha256(smoke_complete.read_bytes()).hexdigest()
        assert launch["memory_smoke_execution"] == "queue_executed"
        real = root / "vit_real_preflight.json"
        real_launch = root / "vit_real_preflight.launch.json"
        real_complete = root / "vit_real_preflight.complete.json"
        assert launch["real_preflight_receipt_path"] == linux(real)
        assert launch["real_preflight_receipt_sha256"] == hashlib.sha256(real.read_bytes()).hexdigest()
        assert launch["real_preflight_launch_path"] == linux(real_launch)
        assert launch["real_preflight_launch_sha256"] == hashlib.sha256(real_launch.read_bytes()).hexdigest()
        assert launch["real_preflight_complete_path"] == linux(real_complete)
        assert launch["real_preflight_complete_sha256"] == hashlib.sha256(real_complete.read_bytes()).hexdigest()
        assert launch["real_preflight_execution"] == "queue_executed"
        assert json.loads(real_complete.read_text())["exit_code"] == 0
        assert json.loads(smoke_complete.read_text())["exit_code"] == 0
        assert complete["exit_code"] == 0
        assert observed["cuda_visible_devices"] == "GPU-TEST-UUID"
        assert len(list(root.glob("seed*.launch.json"))) == 1
    again, _ = run_queue(queue, mode="pilot-ref", root=root)
    assert again.returncode == 2 and "must be new" in again.stderr


def test_foreign_compute_pid_stops_before_next_seed(queue):
    env, *_ = prepare_pilots(queue)
    result, root = run_queue(queue, mode="full", extra_env=dict(env, MOCK_BUSY_CALL="6"))
    assert result.returncode == 2
    assert "compute PIDs before 6502_step: 4321" in result.stderr
    assert (root / "seed6501_step.complete.json").is_file()
    assert not (root / "seed6502_step.launch.json").exists()
    assert not (root / "seed6502").exists()


def test_foreign_compute_pid_after_receipt_refuses_cuda_launch(queue):
    result, root = run_queue(queue, extra_env={"MOCK_BUSY_CALL": "5"})
    assert result.returncode == 2
    assert "compute PIDs before 6500_step: 4321" in result.stderr
    assert (root / "seed6500_step.launch.json").is_file()
    assert not (root / "seed6500").exists()
    assert not (root / "seed6500_step.complete.json").exists()


def test_second_new_root_cannot_rerun_claimed_pilot_seed(queue):
    first, first_root = run_queue(queue, mode="pilot-step")
    assert first.returncode == 0, first.stderr
    second_root = queue[2] / "runs" / "another-pilot-step"
    second, _ = run_queue(queue, mode="pilot-step", root=second_root)
    assert second.returncode == 2 and "already claimed: 6500_step" in second.stderr
    assert not second_root.exists()
    claim = queue[2] / "runs" / ".fmow-local-boundary-vit-claims" / "6500_step" / "owner.json"
    assert json.loads(claim.read_text())["run_root"] == linux(first_root)


def test_concurrent_claim_allows_only_one_clean_root(queue):
    roots = [queue[2] / "runs" / "runa", queue[2] / "runs" / "runb"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_queue, queue, "pilot-step", root,
                                   {"MOCK_RUNNER_SLEEP": "1"}) for root in roots]
        results = [future.result()[0] for future in futures]
    assert sorted(result.returncode for result in results) == [0, 2]
    assert sum((root / "seed6500_step.launch.json").is_file() for root in roots) == 1
    assert sum(root.exists() for root in roots) == 1


def test_full_mode_uses_only_the_registered_twelve_seeds(queue):
    env, *_ = prepare_pilots(queue)
    result, root = run_queue(queue, mode="full", extra_env=env)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in root.glob("seed*.launch.json")} == {
        f"seed{seed}_step.launch.json" for seed in range(6501, 6513)}
    assert len(list(root.glob("seed*.complete.json"))) == 12
    cost = json.loads((root / "vit_cost_gate.json").read_text())
    assert cost["gate_passed"] and cost["projected_gpu_hours"] < 24
    assert set(cost["preflight_seconds"]) == {"pilot_step", "pilot_ref", "full"}
    launch = json.loads((root / "seed6501_step.launch.json").read_text())
    assert launch["cost_gate_receipt_path"] == linux(root / "vit_cost_gate.json")
    assert launch["pilot_gate_receipt_path"] == linux(root / "vit_pilot_gate_recheck.json")


def test_failed_fit_preserves_receipt_and_does_not_run_reference(queue):
    result, root = run_queue(queue, mode="pilot-step", extra_env={"MOCK_RUNNER_FAIL": "1"})
    assert result.returncode == 7
    assert json.loads((root / "seed6500_step.complete.json").read_text())["exit_code"] == 7
    assert (root / "seed6500_step.launch.json").is_file()
    assert not (root / "seed6500_ref.launch.json").exists()


def test_run_root_outside_owned_runs_directory_is_refused(queue):
    outside = queue[2] / "outside"
    result, _ = run_queue(queue, mode="pilot-step", root=outside)
    assert result.returncode == 2 and "must be below" in result.stderr
    assert not outside.exists()


@pytest.mark.parametrize("field,bad_value", [
    ("release_commit", "0" * 40),
    ("host", "wrong-host"),
    ("gpu_uuid", "GPU-OTHER"),
    ("batch_size", 32),
    ("development_batch_size", 16),
    ("weight_sha256", "0" * 64),
    ("label_free", False),
    ("memory_smoke_passed", False),
    ("peak_allocated_bytes", 1000),
])
def test_mismatched_memory_smoke_refuses_before_claim(queue, field, bad_value):
    result, root = run_queue(queue, extra_env={
        "MOCK_SMOKE_MUTATE_FIELD": field,
        "MOCK_SMOKE_MUTATE_JSON": json.dumps(bad_value)})
    assert result.returncode == 2
    assert "memory smoke failed" in result.stderr or "memory-smoke provenance failed" in result.stderr
    assert root.exists()
    assert (root / "vit_memory_smoke.json").is_file()
    assert not (root / "seed6500_step.launch.json").exists()
    assert not (queue[2] / "runs" / ".fmow-local-boundary-vit-claims" / "6500_step").exists()


def test_smoke_failure_preserves_negative_receipt_and_never_claims_seed(queue):
    result, root = run_queue(queue, extra_env={"MOCK_SMOKE_FAIL": "1"})
    assert result.returncode == 2
    assert json.loads((root / "vit_memory_smoke.json").read_text())["memory_smoke_passed"] is False
    assert json.loads((root / "vit_memory_smoke.complete.json").read_text())["exit_code"] == 7
    assert (root / "vit_memory_smoke.log").is_file()
    assert not (root / "seed6500_step.launch.json").exists()


def test_smoke_early_exit_still_preserves_negative_receipt(queue):
    result, root = run_queue(queue, extra_env={"MOCK_SMOKE_EXIT_BEFORE_RECEIPT": "1"})
    assert result.returncode == 2
    receipt = json.loads((root / "vit_memory_smoke.json").read_text())
    assert receipt["memory_smoke_passed"] is False
    assert receipt["exit_code"] == 7
    assert not (root / "seed6500_step.launch.json").exists()


@pytest.mark.parametrize("field,bad_value", [
    ("release_commit", "0" * 40),
    ("host", "wrong-host"),
    ("gpu_uuid", "GPU-OTHER"),
    ("weight_sha256", "0" * 64),
    ("development_batch_size", 3),
    ("chunk_sizes", [3, 3, 3, 3, 3]),
    ("development_labels_accessed", True),
    ("preflight_passed", False),
    ("max_probability_difference", 0.1),
    ("gradient_relative_errors", {"pooled": 0.0}),
])
def test_mismatched_real_preflight_refuses_before_claim(queue, field, bad_value):
    result, root = run_queue(queue, extra_env={
        "MOCK_REAL_MUTATE_FIELD": field,
        "MOCK_REAL_MUTATE_JSON": json.dumps(bad_value)})
    assert result.returncode == 2
    assert "real-image preflight" in result.stderr
    assert (root / "vit_real_preflight.json").is_file()
    assert not (root / "seed6500_step.launch.json").exists()
    assert not (queue[2] / "runs" / ".fmow-local-boundary-vit-claims" / "6500_step").exists()


def test_real_preflight_failure_preserves_negative_receipt(queue):
    result, root = run_queue(queue, extra_env={"MOCK_REAL_FAIL": "1"})
    assert result.returncode == 2
    assert json.loads((root / "vit_real_preflight.json").read_text())["preflight_passed"] is False
    assert json.loads((root / "vit_real_preflight.complete.json").read_text())["exit_code"] == 7
    assert (root / "vit_real_preflight.log").is_file()
    assert not (root / "seed6500_step.launch.json").exists()


def test_real_preflight_early_exit_still_preserves_negative_receipt(queue):
    result, root = run_queue(queue, extra_env={"MOCK_REAL_EXIT_BEFORE_RECEIPT": "1"})
    assert result.returncode == 2
    receipt = json.loads((root / "vit_real_preflight.json").read_text())
    assert receipt["preflight_passed"] is False and receipt["exit_code"] == 7
    assert not (root / "seed6500_step.launch.json").exists()


def test_full_queue_refuses_without_independent_pilot_gate(queue):
    result, root = run_queue(queue, mode="full")
    assert result.returncode == 2
    assert "requires both pilot roots" in result.stderr
    assert root.exists()
    assert not (root / "vit_memory_smoke.json").exists()
    assert not (root / "seed6501_step.launch.json").exists()


def test_full_queue_refuses_forged_pilot_gate_before_smoke(queue):
    env, _step, _ref, gate = prepare_pilots(queue)
    gate.write_text(json.dumps({"status": "vit_pilot_integrity_pass", "forged": True}))
    result, root = run_queue(queue, mode="full", extra_env=env)
    assert result.returncode == 2
    assert "differs from independent recomputation" in result.stderr
    assert (root / "vit_pilot_gate_recheck.json").is_file()
    assert not (root / "vit_memory_smoke.json").exists()
    assert not (root / "seed6501_step.launch.json").exists()


def test_full_queue_stops_before_seed_when_projected_cost_exceeds_ceiling(queue):
    env, step_root, _ref, _gate = prepare_pilots(queue)
    launch_path = step_root / "seed6500_step.launch.json"
    launch = json.loads(launch_path.read_text())
    launch["started_utc"] = "2020-01-01T00:00:00+00:00"
    launch_path.write_text(json.dumps(launch))
    result, root = run_queue(queue, mode="full", extra_env=env)
    assert result.returncode == 2
    gate = json.loads((root / "vit_cost_gate.json").read_text())
    assert gate["gate_passed"] is False
    assert gate["projected_gpu_hours"] > 24
    assert (root / "vit_memory_smoke.json").is_file()
    assert not (root / "seed6501_step.launch.json").exists()

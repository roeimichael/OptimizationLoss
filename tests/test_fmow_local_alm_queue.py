"""CPU-only boundary tests for the exclusive fmow2 ALM launcher."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shlex
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fmow_local_alm_queue.sh"


def linux(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}"


def wsl(*args):
    return subprocess.run(["bash", "-lc", shlex.join(args)], text=True,
                          capture_output=True, check=True).stdout.strip()


@pytest.fixture
def queue(tmp_path):
    release = tmp_path / "release"
    configs = release / "experiments" / "configs" / "fmow_local_alm_20260930"
    configs.mkdir(parents=True)
    package = release / "tralo"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "knee_experiment.py").write_text(
        "def source():\n    return {'fmow_local.py': 'test-source-hash'}\n")
    (package / "fmow_local.py").write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "def validate(c):\n"
        "    if c.get('study') != 'local_alm_direction_v1':\n"
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
    for job in ("6300_step", "6300_ref", *(f"{seed}_step" for seed in range(6301, 6313))):
        seed, arm = job.split("_")
        (configs / f"fmow_local_{job}.json").write_text(json.dumps({
            "study": "local_alm_direction_v1", "seed": int(seed),
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


def test_shell_syntax():
    subprocess.run(["bash", "-n", linux(SCRIPT)], check=True)


def test_separate_pilot_receipts_and_new_root_refusal(queue):
    for mode, job, output in (("pilot-step", "6300_step", "seed6300"),
                              ("pilot-ref", "6300_ref", "seed6300_ref")):
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
        assert complete["exit_code"] == 0
        assert observed["cuda_visible_devices"] == "GPU-TEST-UUID"
        assert len(list(root.glob("seed*.launch.json"))) == 1
    again, _ = run_queue(queue, mode="pilot-ref", root=root)
    assert again.returncode == 2 and "must be new" in again.stderr


def test_foreign_compute_pid_stops_before_next_seed(queue):
    result, root = run_queue(queue, mode="full", extra_env={"MOCK_BUSY_CALL": "2"})
    assert result.returncode == 2
    assert "compute PIDs before 6302_step: 4321" in result.stderr
    assert (root / "seed6301_step.complete.json").is_file()
    assert not (root / "seed6302_step.launch.json").exists()
    assert not (root / "seed6302").exists()


def test_second_new_root_cannot_rerun_claimed_pilot_seed(queue):
    first, first_root = run_queue(queue, mode="pilot-step")
    assert first.returncode == 0, first.stderr
    second_root = queue[2] / "runs" / "another-pilot-step"
    second, _ = run_queue(queue, mode="pilot-step", root=second_root)
    assert second.returncode == 2 and "already claimed: 6300_step" in second.stderr
    assert not second_root.exists()
    claim = queue[2] / "runs" / ".fmow-local-alm-claims" / "6300_step" / "owner.json"
    assert json.loads(claim.read_text())["run_root"] == linux(first_root)


def test_concurrent_claim_allows_only_one_clean_root(queue):
    roots = [queue[2] / "runs" / "runa", queue[2] / "runs" / "runb"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_queue, queue, "pilot-step", root,
                                   {"MOCK_RUNNER_SLEEP": "1"}) for root in roots]
        results = [future.result()[0] for future in futures]
    assert sorted(result.returncode for result in results) == [0, 2]
    assert sum((root / "seed6300_step.launch.json").is_file() for root in roots) == 1
    assert sum(root.exists() for root in roots) == 1


def test_full_mode_uses_only_the_registered_twelve_seeds(queue):
    result, root = run_queue(queue, mode="full")
    assert result.returncode == 0, result.stderr
    assert {path.name for path in root.glob("*.launch.json")} == {
        f"seed{seed}_step.launch.json" for seed in range(6301, 6313)}
    assert len(list(root.glob("*.complete.json"))) == 12


def test_failed_fit_preserves_receipt_and_does_not_run_reference(queue):
    result, root = run_queue(queue, mode="pilot-step", extra_env={"MOCK_RUNNER_FAIL": "1"})
    assert result.returncode == 7
    assert json.loads((root / "seed6300_step.complete.json").read_text())["exit_code"] == 7
    assert (root / "seed6300_step.launch.json").is_file()
    assert not (root / "seed6300_ref.launch.json").exists()


def test_run_root_outside_owned_runs_directory_is_refused(queue):
    outside = queue[2] / "outside"
    result, _ = run_queue(queue, mode="pilot-step", root=outside)
    assert result.returncode == 2 and "must be below" in result.stderr
    assert not outside.exists()

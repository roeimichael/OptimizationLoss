"""Static and syntax checks for the immutable, exclusive server queue.

GPU launch is deliberately not exercised by local tests.
"""

import shutil
import subprocess
import os
import shlex
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fmow_persistent_local_queue.sh"


def _wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def _executable(path, contents):
    path.write_text(contents, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _queue_harness(tmp_path, mode):
    """Run the actual queue control flow against a disposable WSL server fixture."""
    if os.name != "nt" or not shutil.which("bash"):
        pytest.skip("behavioral queue fixture uses WSL paths")
    release, runs, data, binaries = (tmp_path / name for name in
                                    ("release", "runs", "data", "bin"))
    for path in (release, runs, data, binaries):
        path.mkdir()
    configs = release / "experiments" / "configs" / "fmow_persistent_local_20261001"
    configs.mkdir(parents=True)
    jobs = ("6700_step",) if mode == "pilot-step" else (
        f"{seed}_step" for seed in range(6701, 6713))
    for job in jobs:
        (configs / f"fmow_persistent_{job}.json").write_text("{}")
    sha = "a" * 40
    _executable(binaries / "git", "#!/bin/sh\ncase \"$*\" in\n"
                f"  *\"rev-parse HEAD\"*) echo {sha};;\n"
                "  *) :;;\nesac\n")
    _executable(binaries / "nvidia-smi", "#!/bin/sh\ncase \"$*\" in\n"
                "  *query-gpu=uuid*) echo GPU-test;;\n"
                "  *query-compute-apps=pid*) [ \"${TEST_OCCUPIED:-0}\" = 1 ] && echo 999;;\n"
                "esac\nexit 0\n")
    _executable(binaries / "hostname", "#!/bin/sh\necho dsisco02.test\n")
    fake_python = binaries / "python"
    _executable(fake_python, '''#!/usr/bin/python3
import json, os, pathlib, sys
args = sys.argv[1:]
if args[:1] == ['-']:
    code = sys.stdin.read()
    sys.argv = ['-', *args[1:]]
    if 'from tralo.fmow_persistent_local import validate_config' in code:
        raise SystemExit(0)
    if 'gate = json.loads(gate_file.read_text())' in code:
        raise SystemExit(0)
    if args[1].endswith('/preflight.json'):
        pathlib.Path(args[1]).write_text(json.dumps({'run_root': args[2],
            'passed': True, 'data_files': {}, 'source_sha256': {}}))
        raise SystemExit(0)
    if args[1].endswith('/smoke.json'):
        pathlib.Path(args[1]).write_text(json.dumps({'run_root': args[2],
            'gpu_uuid': args[3], 'passed': True, 'elapsed_seconds': 1}))
        raise SystemExit(0)
    exec(compile(code, '<queue-heredoc>', 'exec'), {'__name__': '__main__'})
elif args[:1] == ['analysis/score_fmow_persistent_local.py']:
    raise SystemExit(7)
elif args[:2] == ['-u', '-m']:
    pathlib.Path(os.environ['TEST_LAUNCHED']).write_text('launched')
else:
    raise SystemExit(9)
''')
    text = SCRIPT.read_text()
    for old, new in (
        ("REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA", f"REL={_wsl_path(release)}"),
        ("RUNS=/tmp/tralo-weekend-michaer8-20261001/runs", f"RUNS={_wsl_path(runs)}"),
        ("REGISTRY=/home/dsi/michaer8/tralo-rebuild/runs", f"REGISTRY={_wsl_path(runs)}"),
        ("DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice", f"DATA={_wsl_path(data)}"),
        ("PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python", f"PY={_wsl_path(fake_python)}")):
        assert old in text
        text = text.replace(old, new)
    fixture_script = tmp_path / "queue.sh"
    fixture_script.write_text(text, encoding="utf-8", newline="\n")
    fixture_script.chmod(0o755)
    env = {"FAKE_BIN": _wsl_path(binaries),
           "TEST_LAUNCHED": _wsl_path(tmp_path / "launched")}
    return fixture_script, runs, sha, env


@pytest.mark.parametrize("case", ["occupied", "claim_collision", "failed_gate"])
def test_queue_behavior_refuses_occupied_claimed_or_failed_gate(tmp_path, case):
    mode = "full" if case == "failed_gate" else "pilot-step"
    script, runs, sha, env = _queue_harness(tmp_path, mode)
    root = runs / "fresh"
    args = [sha, _wsl_path(root), "0", mode]
    if case == "occupied":
        env["TEST_OCCUPIED"] = "1"
    elif case == "claim_collision":
        (runs / ".fmow-persistent-local-claims" / "6700_step").mkdir(parents=True)
    else:
        pilot, reference, gate = (runs / name for name in ("pilot", "reference", "gate.json"))
        pilot.mkdir(); reference.mkdir(); gate.write_text("{}")
        args += [_wsl_path(gate), _wsl_path(pilot), _wsl_path(reference)]
    command = (f"export PATH={shlex.quote(env['FAKE_BIN'])}:/usr/bin:/bin "
               f"TEST_OCCUPIED={shlex.quote(env.get('TEST_OCCUPIED', '0'))} "
               f"TEST_LAUNCHED={shlex.quote(env['TEST_LAUNCHED'])}; "
               "exec bash " + " ".join(shlex.quote(x) for x in [_wsl_path(script), *args]))
    result = subprocess.run(["bash", "-c", command],
                            capture_output=True, text=True, timeout=30)
    expected = {"occupied": "occupied", "claim_collision": "already claimed",
                "failed_gate": "fresh pilot artifact/replay/cost gate failed"}[case]
    assert result.returncode != 0, result.stdout + result.stderr
    assert expected in result.stderr, result.stdout + result.stderr
    assert not (tmp_path / "launched").exists()
    assert not list(root.glob("seed*"))


def test_queue_shell_syntax_and_fixed_modes():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    script = SCRIPT.read_text()
    subprocess.run([bash, "-n"], input=script.encode(), check=True)
    bad = subprocess.run([bash, "-s", "--", "a" * 40, "/tmp/new", "0", "unknown"],
                         input=script.encode(), capture_output=True)
    assert bad.returncode != 0 and b"unknown fixed mode" in bad.stderr
    assert "JOBS=(6700_step)" in script and "JOBS=(6700_ref)" in script
    assert "{6701..6712}" in script
    assert "tralo.fmow_persistent_local" in script
    assert "--fresh-full-gate" in script and "fixed ceiling" in script
    assert "_training_input_fingerprints" in script
    assert "pilot/full release bytes differ" in script
    assert script.index("preflight.start.json") < script.index("smoke.start.json")
    assert script.index("smoke.start.json") < script.index("--fresh-full-gate")
    assert script.index("--fresh-full-gate") < script.index('mkdir "$CLAIM"')


def test_queue_has_irreversible_seed_claim_and_physical_gpu_rechecks():
    script = SCRIPT.read_text()
    assert "mkdir \"$CLAIM\"" in script
    assert "flock -n 9" in script
    assert "--query-gpu=uuid" in script
    assert "--query-compute-apps=pid" in script
    assert "REGISTRY_CANON/.fmow-persistent-local-claims" in script
    assert "REGISTRY_CANON/.fmow-persistent-local-gpu-locks" in script
    assert "REGISTRY_CANON/.fmow-persistent-local-cost" in script
    assert "restricted to dsisco02" in script
    assert script.count('check_gpu_free "$JOB"') >= 2
    assert "CUDA_VISIBLE_DEVICES=\"$UUID\"" in script
    assert "config_sha256" in script and "source_sha256" in script
    assert "open(path, 'x'" in script
    assert "exit_code=int(rc)" in script

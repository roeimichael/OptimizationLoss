"""Non-interactive 2-GPU dispatcher for the overnight sweep.

Splits all currently-pending experiments under a given root round-robin
across GPU 0 and GPU 1, and runs them in parallel via subprocess. Use
inside tmux for resilience.

Usage:
    python scripts/dispatch_sweep.py [--root results/pending_runs] [--gpus 0,1]
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.filesystem_manager import get_experiments_by_status

RUNNER = "src.experiments.runner"


def run_worker(gpu_id, experiments, prefix):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    completed = failed = 0
    t0 = time.time()
    for i, (exp_path, config) in enumerate(experiments, 1):
        cfg_path = Path(exp_path) / "config.json"
        name = config.get("exp_name", Path(exp_path).name)
        elapsed = time.time() - t0
        print(f"{prefix} [{i}/{len(experiments)}] {name}  (elapsed {elapsed/60:.1f}m, "
              f"completed {completed}, failed {failed})", flush=True)
        rc = subprocess.run(
            [sys.executable, "-u", "-m", RUNNER, str(cfg_path)],
            env=env,
        ).returncode
        if rc == 0:
            completed += 1
        else:
            failed += 1
            print(f"{prefix} FAILED rc={rc}: {name}", flush=True)
    print(f"{prefix} DONE: completed={completed} failed={failed} "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    return completed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pending_runs")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--filter", default="",
                    help="optional substring; only experiments whose path "
                         "contains this string are dispatched")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    pending = get_experiments_by_status(args.root)["pending"]
    if args.filter:
        pending = [(p, c) for p, c in pending if args.filter in p]

    if not pending:
        print("No pending experiments")
        return

    print(f"Dispatching {len(pending)} experiments across GPUs {gpus}")
    assignments = {g: [] for g in gpus}
    for idx, item in enumerate(pending):
        assignments[gpus[idx % len(gpus)]].append(item)
    for g in gpus:
        print(f"  GPU {g}: {len(assignments[g])} experiments")

    threads = []
    for g in gpus:
        if not assignments[g]:
            continue
        prefix = f"[GPU {g}]"
        t = threading.Thread(
            target=run_worker, args=(g, assignments[g], prefix),
            daemon=True, name=f"gpu-{g}",
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("All workers complete")


if __name__ == "__main__":
    main()

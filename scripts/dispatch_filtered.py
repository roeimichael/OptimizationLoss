"""Filtered dispatcher: run only pending experiments whose methodology is in a whitelist.

Usage:
    python scripts/dispatch_filtered.py --methods heuristic,danits_lp --gpus 0,1

Dispatches the matching pending experiments in parallel across the listed GPUs
using the same subprocess-runner pattern as main.py, but bypasses input()
prompts.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status

EXPERIMENT_DIR = os.environ.get("EXPERIMENT_DIR", "results/pending_runs")
RUNNER_MODULE = "src.experiments.runner"
_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def fmt_dur(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


def run_worker(gpu_id, experiments, label):
    total = len(experiments)
    completed, failed = 0, 0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    prefix = f"[GPU{gpu_id}|{label}] "
    overall_start = time.time()
    for i, (exp_path, cfg) in enumerate(experiments, 1):
        name = cfg.get("exp_name", Path(exp_path).name)
        meth = cfg.get("methodology", "?")
        mdl = cfg.get("model_name", "?")
        ds = cfg.get("dataset_mode", "?")
        tag = cfg.get("constraint_tag", "?")
        _safe_print(f"{prefix}[{i}/{total}] {meth} {mdl} {ds} {tag}")
        exp_start = time.time()
        try:
            res = subprocess.run(
                [sys.executable, "-u", "-m", RUNNER_MODULE,
                 str(Path(exp_path) / "config.json")],
                env=env, capture_output=True, text=True, timeout=3600,
            )
            elapsed = time.time() - exp_start
            if res.returncode == 0:
                completed += 1
                tag = "OK"
            else:
                failed += 1
                tag = "FAIL"
                _safe_print(f"{prefix}  STDERR (last 8 lines):")
                for line in (res.stderr or "").splitlines()[-8:]:
                    _safe_print(f"{prefix}    {line}")
            _safe_print(f"{prefix}  -> {tag} in {fmt_dur(elapsed)}  "
                        f"(done={completed} fail={failed} / {total})")
        except subprocess.TimeoutExpired:
            failed += 1
            _safe_print(f"{prefix}  TIMEOUT (>1h)")
        except Exception as e:
            failed += 1
            _safe_print(f"{prefix}  EXCEPTION: {e}")
    return completed, failed, time.time() - overall_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True,
                    help="Comma-separated methodologies to dispatch")
    ap.add_argument("--gpus", required=True,
                    help="Comma-separated GPU ids (e.g. 0,1)")
    ap.add_argument("--max", type=int, default=None,
                    help="Optional cap on number of cells")
    ap.add_argument("--sweep-substr", default=None,
                    help="Only run cells whose path contains this substring")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model whitelist (skip others)")
    args = ap.parse_args()

    methods = set(args.methods.split(","))
    gpus = [int(g) for g in args.gpus.split(",")]
    models = set(args.models.split(",")) if args.models else None

    print(f"Methods filter: {methods}")
    print(f"GPUs: {gpus}")

    r = get_experiments_by_status(EXPERIMENT_DIR)
    pending = r["pending"]
    print(f"Total pending: {len(pending)}")

    substrs = args.sweep_substr.split(",") if args.sweep_substr else None
    filtered = [(p, c) for p, c in pending
                if c.get("methodology") in methods
                and (substrs is None or any(s in p for s in substrs))
                and (models is None or c.get("model_name") in models)]
    print(f"Filtered: {len(filtered)}")
    if args.max is not None:
        filtered = filtered[:args.max]
        print(f"Capped to: {len(filtered)}")

    if not filtered:
        print("Nothing to do.")
        return

    # Round-robin over GPUs
    by_gpu = defaultdict(list)
    for i, item in enumerate(filtered):
        by_gpu[gpus[i % len(gpus)]].append(item)
    for g, lst in by_gpu.items():
        print(f"  GPU {g}: {len(lst)} cells")

    threads = []
    results = {}
    for g in gpus:
        lst = by_gpu.get(g, [])
        if not lst:
            continue

        def _w(g=g, lst=lst):
            label = "+".join(sorted({c.get("methodology", "?")
                                     for _, c in lst}))[:10]
            results[g] = run_worker(g, lst, label)

        t = threading.Thread(target=_w, daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_done = sum(r[0] for r in results.values())
    total_fail = sum(r[1] for r in results.values())
    print(f"\nDONE: completed={total_done} failed={total_fail}")


if __name__ == "__main__":
    main()

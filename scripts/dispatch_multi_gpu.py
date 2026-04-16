"""Round-robin dispatcher across N GPUs, ignoring model-name grouping.

Usage:
    python -m scripts.dispatch_multi_gpu --gpus 0,1,2,3 [--dir results/pending_runs]

Unlike main.py, this round-robins individual experiments across GPUs so a
single-model sweep actually uses all available GPUs.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

OPTIMIZATION_MODULE = 'src.experiments.run_experiment'
HEURISTIC_MODULE = 'src.experiments.run_heuristic'

_print_lock = threading.Lock()
log = logging.getLogger(__name__)


def _p(*a, **kw):
    with _print_lock:
        print(*a, **kw)
        sys.stdout.flush()


def _fmt(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f}m"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"{h}h{m:02d}m"


def run_one(exp_path, config, gpu_id, prefix):
    methodology = config.get('methodology', 'our_approach')
    runner = (HEURISTIC_MODULE if methodology in ('heuristic', 'po_lp', 'danits_lp')
              else OPTIMIZATION_MODULE)
    name = config.get('exp_name', Path(exp_path).name)
    cfg_path = Path(exp_path) / 'config.json'
    env = {**os.environ,
           'CUDA_VISIBLE_DEVICES': str(gpu_id),
           'CUDA_MODULE_LOADING': 'EAGER',
           'TORCH_COMPILE_DISABLE': '1',
           }
    start = time.time()
    _p(f"{prefix}START {name} -> {runner}")
    proc = subprocess.Popen(
        [sys.executable, '-u', '-m', runner, str(cfg_path)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    for line in proc.stdout:
        _p(f"{prefix}  {line.rstrip()}")
    rc = proc.wait()
    dur = time.time() - start
    status = "DONE" if rc == 0 else f"FAIL({rc})"
    _p(f"{prefix}{status} {name} in {_fmt(dur)}")
    return rc == 0, dur


def worker(gpu_id, experiments, stop_evt):
    prefix = f"[GPU {gpu_id}] "
    done, fail, times = 0, 0, []
    for exp_path, config in experiments:
        if stop_evt.is_set():
            break
        ok, dur = run_one(exp_path, config, gpu_id, prefix)
        if ok:
            done += 1
        else:
            fail += 1
        times.append(dur)
    return done, fail, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0', help='comma-separated gpu ids, e.g. 0,1,2,3')
    ap.add_argument('--dir', default='results/pending_runs')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    gpu_ids = [int(x) for x in args.gpus.split(',')]

    print_status_summary(args.dir)
    pending = get_experiments_by_status(args.dir)['pending']
    if not pending:
        _p("No pending experiments")
        return
    _p(f"Dispatching {len(pending)} experiments across GPUs: {gpu_ids}")

    # Round-robin experiment -> gpu
    assignments = {g: [] for g in gpu_ids}
    for idx, item in enumerate(pending):
        assignments[gpu_ids[idx % len(gpu_ids)]].append(item)

    for g in gpu_ids:
        _p(f"  GPU {g}: {len(assignments[g])} experiments")

    stop_evt = threading.Event()
    results = {}
    threads = []
    for g in gpu_ids:
        if not assignments[g]:
            continue
        def _run(gid=g, exps=assignments[g]):
            results[gid] = worker(gid, exps, stop_evt)
        t = threading.Thread(target=_run, name=f"gpu-{g}", daemon=True)
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        _p("\nINTERRUPT -- signaling stop")
        stop_evt.set()
        for t in threads:
            t.join(timeout=10)

    total = time.time() - t0
    done = sum(r[0] for r in results.values())
    fail = sum(r[1] for r in results.values())
    all_times = [x for r in results.values() for x in r[2]]
    _p(f"\n{'='*60}\nALL DONE: {done} ok, {fail} fail, wall {_fmt(total)}")
    if all_times:
        _p(f"  avg {_fmt(sum(all_times)/len(all_times))} "
           f"min {_fmt(min(all_times))} max {_fmt(max(all_times))}")
    print_status_summary(args.dir)


if __name__ == '__main__':
    main()

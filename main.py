# Main experiment orchestrator: runs all pending experiments via subprocess.
# Supports parallel GPU execution with live output streaming.
# Safe for tmux on remote servers -- detach and reattach freely.

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

DEFAULT_EXPERIMENT_DIR = 'results/pending_runs'
RUNNER_MODULE = 'src.experiments.runner'

log = logging.getLogger(__name__)
_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def select_gpu():
    if not torch.cuda.is_available():
        print("\nWARNING: No CUDA GPUs detected -- will run on CPU")
        return ['cpu']
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  GPU Selection -- {n_gpus} GPU(s) detected")
    print(f"{'='*60}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
        print(f"  [{i}]  {name}  ({mem / (1024**3):.1f} GB)")
    print(f"{'='*60}")
    while True:
        choice = input(f"\nSelect GPU (0-{n_gpus-1}), comma-separated (e.g. 1,2), or 'all': ").strip().lower()
        if choice == 'all':
            gpu_ids = list(range(n_gpus))
            print(f"  -> Using all {n_gpus} GPUs: {gpu_ids}")
            return gpu_ids
        try:
            gpu_ids = [int(x.strip()) for x in choice.split(',')]
            if all(0 <= g < n_gpus for g in gpu_ids):
                names = [torch.cuda.get_device_name(g) for g in gpu_ids]
                if len(gpu_ids) == 1:
                    print(f"  -> Using GPU {gpu_ids[0]}: {names[0]}")
                else:
                    print(f"  -> Using {len(gpu_ids)} GPUs: {list(zip(gpu_ids, names))}")
                return gpu_ids
            else:
                print(f"  Invalid: all IDs must be 0-{n_gpus-1}")
        except ValueError:
            print(f"  Invalid input. Enter number(s) (0-{n_gpus-1}), e.g. '1,2', or 'all'")


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def print_experiment_header(index, total, exp_path, config, completed, failed, prefix=''):
    name = config.get('exp_name', Path(exp_path).name)
    methodology = config.get('methodology', 'our_approach')
    constraint = config.get('constraint', [])
    hp = config.get('hyperparams', {})
    _safe_print(f"\n{prefix}{'='*70}")
    _safe_print(f"{prefix}  [{index}/{total}]  {name}")
    _safe_print(f"{prefix}  Method: {methodology}  |  Constraint: {constraint}")
    if methodology == 'our_approach':
        _safe_print(f"{prefix}  rho={hp.get('initial_rho', 1.0)}  kl={hp.get('alpha_kl', 0.0)}  "
                     f"lr_con={hp.get('lr_constraint', 5e-6):.0e}  "
                     f"pretrained={hp.get('pretrained', False)}  "
                     f"kl_temp={hp.get('kl_temperature', 1.0)}  "
                     f"weighted_ce={hp.get('class_weighted_ce', False)}")
    _safe_print(f"{prefix}  Progress so far: {completed} done, {failed} failed, "
                f"{total - index + 1} remaining (including this one)")
    _safe_print(f"{prefix}{'='*70}")


def print_experiment_result(name, returncode, elapsed, completed, failed, total,
                            total_elapsed, times_list, prefix=''):
    status = "DONE" if returncode == 0 else "FAIL"
    marker = "  [OK]" if returncode == 0 else "  [!!]"
    if times_list:
        avg_time = sum(times_list) / len(times_list)
        remaining = total - (completed + failed)
        eta_seconds = avg_time * remaining
        eta_str = f"  ETA: ~{format_duration(eta_seconds)}"
    else:
        eta_str = ""
    _safe_print(f"\n{prefix}{marker} {name}: {status} in {format_duration(elapsed)}  "
                f"({completed}/{total} done, {failed} failed){eta_str}")


def run_worker(gpu_id, experiments, worker_name, stop_event):
    total = len(experiments)
    completed, failed = 0, 0
    experiment_times = []
    prefix = f"[GPU {gpu_id} | {worker_name:<12s}]  "
    worker_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    overall_start = time.time()
    for i, (exp_path, config) in enumerate(experiments, 1):
        if stop_event.is_set():
            _safe_print(f"\n{prefix}Stopping (interrupt received) -- "
                        f"{total - i + 1} experiments skipped")
            break
        config_path = Path(exp_path) / 'config.json'
        name = config.get('exp_name', Path(exp_path).name)
        print_experiment_header(i, total, exp_path, config, completed, failed, prefix=prefix)
        exp_start = time.time()
        try:
            proc = subprocess.Popen(
                [sys.executable, '-u', '-m', RUNNER_MODULE, str(config_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=worker_env, text=True, bufsize=1)
            for line in proc.stdout:
                if stop_event.is_set():
                    proc.terminate()
                    proc.wait(timeout=5)
                    break
                _safe_print(f"{prefix}{line}", end='')
            proc.wait()
            elapsed = time.time() - exp_start
            experiment_times.append(elapsed)
            if proc.returncode == 0:
                completed += 1
            else:
                failed += 1
            print_experiment_result(name, proc.returncode, elapsed,
                                    completed, failed, total,
                                    time.time() - overall_start, experiment_times,
                                    prefix=prefix)
        except Exception as exc:
            elapsed = time.time() - exp_start
            failed += 1
            _safe_print(f"\n{prefix}ERROR on {name}: {exc} (after {format_duration(elapsed)})")
    return completed, failed, experiment_times


def run_sequential(pending, gpu_id=None):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    total = len(pending)
    completed, failed = 0, 0
    experiment_times = []
    overall_start = time.time()
    for i, (exp_path, config) in enumerate(pending, 1):
        config_path = Path(exp_path) / 'config.json'
        name = config.get('exp_name', Path(exp_path).name)
        print_experiment_header(i, total, exp_path, config, completed, failed)
        exp_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, '-u', '-m', RUNNER_MODULE, str(config_path)],
                stdout=sys.stdout, stderr=sys.stderr)
            elapsed = time.time() - exp_start
            experiment_times.append(elapsed)
            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
            print_experiment_result(name, result.returncode, elapsed,
                                    completed, failed, total,
                                    time.time() - overall_start, experiment_times)
        except KeyboardInterrupt:
            elapsed = time.time() - exp_start
            print(f"\n\n{'!'*70}")
            print(f"  INTERRUPTED after {format_duration(elapsed)} on: {name}")
            print(f"  Completed: {completed}  |  Failed: {failed}  |  "
                  f"Remaining: {total - i}")
            print(f"  Total time: {format_duration(time.time() - overall_start)}")
            print(f"{'!'*70}")
            break
    return completed, failed, experiment_times, time.time() - overall_start


def run_parallel(pending, gpu_ids):
    by_model = defaultdict(list)
    for exp_path, config in pending:
        model = config.get('model_name', 'unknown')
        by_model[model].append((exp_path, config))
    model_names = sorted(by_model.keys())
    if len(model_names) > len(gpu_ids):
        print(f"\n  WARNING: {len(model_names)} models but only {len(gpu_ids)} GPUs.")
        print(f"  Models will be distributed round-robin across GPUs.\n")
    gpu_assignments = {}
    for gpu in gpu_ids:
        gpu_assignments[gpu] = []
    for idx, model in enumerate(model_names):
        gpu = gpu_ids[idx % len(gpu_ids)]
        gpu_assignments[gpu].append((model, by_model[model]))
    print(f"\n{'='*60}")
    print(f"  Parallel GPU Assignment")
    print(f"{'='*60}")
    for gpu in gpu_ids:
        assignments = gpu_assignments[gpu]
        if assignments:
            for model, exps in assignments:
                print(f"  GPU {gpu}  <-  {model} ({len(exps)} experiments)")
        else:
            print(f"  GPU {gpu}  <-  (idle)")
    print(f"{'='*60}\n")
    stop_event = threading.Event()
    threads = []
    results = {}
    for gpu in gpu_ids:
        assignments = gpu_assignments[gpu]
        if not assignments:
            continue
        combined_exps = []
        label_parts = []
        for model, exps in assignments:
            combined_exps.extend(exps)
            label_parts.append(model)
        worker_name = '+'.join(label_parts)
        def _worker(g=gpu, exps=combined_exps, wn=worker_name):
            results[g] = run_worker(g, exps, wn, stop_event)
        t = threading.Thread(target=_worker, name=f"gpu-{gpu}-worker", daemon=True)
        threads.append(t)
    overall_start = time.time()
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print(f"\n\n{'!'*70}")
        print(f"  INTERRUPT received -- signaling all workers to stop...")
        print(f"{'!'*70}")
        stop_event.set()
        for t in threads:
            t.join(timeout=10)
    total_time = time.time() - overall_start
    total_completed = sum(r[0] for r in results.values())
    total_failed = sum(r[1] for r in results.values())
    all_times = []
    for r in results.values():
        all_times.extend(r[2])
    return total_completed, total_failed, all_times, total_time


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    gpu_ids = select_gpu()
    is_cpu = gpu_ids == ['cpu']
    is_parallel = not is_cpu and len(gpu_ids) > 1
    if is_cpu:
        gpu_info = 'CPU'
    elif is_parallel:
        gpu_info = f'{len(gpu_ids)} GPUs: {gpu_ids}'
    else:
        gpu_info = f'GPU {gpu_ids[0]}'
    log.info("Device: %s", gpu_info)
    experiment_dir = os.environ.get('EXPERIMENT_DIR', DEFAULT_EXPERIMENT_DIR)
    log.info("Experiment directory: %s", experiment_dir)
    print_status_summary(experiment_dir)
    pending = get_experiments_by_status(experiment_dir)['pending']
    if not pending:
        log.info("No pending experiments")
        return
    total = len(pending)
    log.info("Running %d pending experiments on %s", total, gpu_info)
    if is_parallel:
        completed, failed, experiment_times, total_time = run_parallel(pending, gpu_ids)
    else:
        single_gpu = None if is_cpu else gpu_ids[0]
        completed, failed, experiment_times, total_time = run_sequential(pending, gpu_id=single_gpu)
    print(f"\n{'='*70}")
    print(f"  ALL DONE")
    print(f"  Completed: {completed}  |  Failed: {failed}  |  "
          f"Total time: {format_duration(total_time)}")
    if experiment_times:
        print(f"  Avg per experiment: {format_duration(sum(experiment_times)/len(experiment_times))}")
    print(f"{'='*70}\n")
    print_status_summary(experiment_dir)


if __name__ == "__main__":
    main()

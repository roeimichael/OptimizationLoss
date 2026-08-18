# Main experiment orchestrator: runs all pending experiments via subprocess.
# One GPU per process; partition a campaign by EXPERIMENT_DIR, not by threads.
# Safe for tmux on remote servers -- detach and reattach freely.

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

DEFAULT_EXPERIMENT_DIR = 'results/pending_runs'
RUNNER_MODULE = 'src.experiments.runner'

# Capture CVD from the launching shell. If set, the child process's "GPU index"
# is already remapped; overriding CVD again would re-resolve to a physical
# index and route to the wrong GPU (host-crash risk under shared use).
PARENT_CVD = os.environ.get('CUDA_VISIBLE_DEVICES')

log = logging.getLogger(__name__)


def select_gpu():
    """Pick ONE gpu. Campaigns are partitioned by separate EXPERIMENT_DIR roots
    and launched once per card, so in-process multi-GPU was removed: it shared a
    single model_cache across workers, which races on the warm-up write."""
    if not torch.cuda.is_available():
        print("\nWARNING: No CUDA GPUs detected -- will run on CPU")
        return None
    n = torch.cuda.device_count()
    print("\n" + "=" * 60)
    print("  GPU Selection -- %d GPU(s) detected" % n)
    print("=" * 60)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
        print("  [%d]  %s  (%.1f GB)"
              % (i, torch.cuda.get_device_name(i), mem / 1024 ** 3))
    print("=" * 60)
    while True:
        choice = input("\nSelect GPU (0-%d): " % (n - 1)).strip()
        try:
            g = int(choice)
            if 0 <= g < n:
                print("  -> Using GPU %d: %s" % (g, torch.cuda.get_device_name(g)))
                return g
        except ValueError:
            pass
        print("  Invalid -- enter a single index 0-%d" % (n - 1))


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
    methodology = config.get('methodology', 'tralo')
    constraint = config.get('constraint', [])
    hp = config.get('hyperparams', {})
    print(f"\n{prefix}{'='*70}")
    print(f"{prefix}  [{index}/{total}]  {name}")
    print(f"{prefix}  Method: {methodology}  |  Constraint: {constraint}")
    if methodology == 'tralo':
        print(f"{prefix}  rho={hp.get('initial_rho', 1.0)}  "
                     f"lr_con={hp.get('lr_constraint', 5e-6):.0e}  "
                     f"pretrained={hp.get('pretrained', False)}  "
                     f"weighted_ce={hp.get('class_weighted_ce', False)}")
    print(f"{prefix}  Progress so far: {completed} done, {failed} failed, "
                f"{total - index + 1} remaining (including this one)")
    print(f"{prefix}{'='*70}")


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
    print(f"\n{prefix}{marker} {name}: {status} in {format_duration(elapsed)}  "
                f"({completed}/{total} done, {failed} failed){eta_str}")


def run_sequential(pending, gpu_id=None):
    if gpu_id is not None and PARENT_CVD is None:
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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    gpu_id = select_gpu()
    gpu_info = 'CPU' if gpu_id is None else 'GPU %d' % gpu_id
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
    completed, failed, experiment_times, total_time = run_sequential(pending, gpu_id=gpu_id)
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

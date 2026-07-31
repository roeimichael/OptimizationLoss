"""Minimal per-lane dispatcher for the TMLR imbalanced campaign (Track B / B1).

Runs every pending config under $EXPERIMENT_DIR sequentially on the pinned GPU
(CUDA_VISIBLE_DEVICES is set by the launcher), in SORTED path order so priority
tiers run first (t1 core before t2/t3) and both caps of a cell run back-to-back
(the first trains the warmup, the second reuses the cache -- no duplicate train,
no cross-GPU cache write-race). Idempotent + resumable: completed configs are
skipped, so re-launching after an interruption continues where it left off.

    CUDA_VISIBLE_DEVICES=0 \
    EXPERIMENT_DIR=results/tmlr_track_b/imbalanced_2026-07/lane_gpu0 \
        python -m src.config_generators.run_lane
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = os.environ.get("EXPERIMENT_DIR")
    if not root:
        raise SystemExit("set EXPERIMENT_DIR")
    # Optional phase gate: only run configs whose path contains ALL of the
    # comma-separated substrings in LANE_FILTER (e.g. "/t1/,/L30_G30/" = the
    # professor's minimal B1-108). Empty = run everything.
    filt = [p for p in os.environ.get("LANE_FILTER", "").split(",") if p]
    configs = sorted(str(p) for p in Path(root).rglob("config.json"))
    pending = []
    for c in configs:
        cn = c.replace("\\", "/")
        if filt and not all(p in cn for p in filt):
            continue
        try:
            if json.load(open(c)).get("status") != "completed":
                pending.append(c)
        except Exception:
            pending.append(c)
    print(f"[run_lane] {root}: {len(pending)} pending"
          + (f" (LANE_FILTER={filt})" if filt else ""), flush=True)
    done = fail = 0
    t0 = time.time()
    for i, cp in enumerate(pending, 1):
        print(f"[run_lane] {i}/{len(pending)} start {cp}", flush=True)
        r = subprocess.run([sys.executable, "-u", "-m", "src.experiments.runner", cp])
        if r.returncode == 0:
            done += 1
        else:
            fail += 1
            print(f"[run_lane] FAIL rc={r.returncode} {cp}", flush=True)
        print(f"[run_lane] {done} done / {fail} fail / {len(pending) - i} left / "
              f"{(time.time() - t0) / 60:.1f}min elapsed", flush=True)
    print(f"[run_lane] DONE {root}: {done} done, {fail} fail, "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()

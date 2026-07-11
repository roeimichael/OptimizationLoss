#!/bin/bash
# OctMNIST asymmetric campaign (480 runs) -- fills the missing Oct rows of the
# regime table (both asym directions). Run ON THE SERVER from ~/OptimizationLoss:
#
#     bash run_oct_asym.sh <GPU_A> <GPU_B>
#
# GPU_A dispatches g2_asym_oct_gl (G<L, global tighter, 240 runs)
# GPU_B dispatches g2_asym_oct_lg (G>L, local tighter,  240 runs)
# Pick two GPUs that are FREE (checked below; never share with another user).
set -euo pipefail

PY=~/anaconda3/envs/optloss/bin/python
GPU_A=${1:?usage: bash run_oct_asym.sh GPU_A GPU_B}
GPU_B=${2:?usage: bash run_oct_asym.sh GPU_A GPU_B}

echo "== GPU state =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
for g in "$GPU_A" "$GPU_B"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1024 ]; then
    echo "ABORT: GPU $g has ${used} MiB in use -- someone else is on it." >&2
    exit 1
  fi
done

echo "== CUDA sanity (optloss env) =="
$PY -c 'import torch; assert torch.cuda.is_available(), "CPU torch -- wrong env"; print("cuda OK,", torch.cuda.get_device_name(0))'

echo "== Generating configs (idempotent: completed cells are skipped) =="
$PY -m src.config_generators.gen_g2_asym_oct

mkdir -p logs
echo "== Dispatching detached (survives SSH disconnect) =="
setsid bash -c "echo all | CUDA_VISIBLE_DEVICES=$GPU_A EXPERIMENT_DIR=results/pending_runs/g2_asym_oct_gl $PY -u main.py > logs/oct_asym_gl.log 2>&1" < /dev/null & disown
setsid bash -c "echo all | CUDA_VISIBLE_DEVICES=$GPU_B EXPERIMENT_DIR=results/pending_runs/g2_asym_oct_lg $PY -u main.py > logs/oct_asym_lg.log 2>&1" < /dev/null & disown
sleep 5
echo "== Dispatcher processes =="
pgrep -af "main.py" || true
echo
echo "Progress:  find results/pending_runs/g2_asym_oct_* -name evaluation_metrics.csv | wc -l   (target 480)"
echo "Logs:      tail -f logs/oct_asym_gl.log logs/oct_asym_lg.log"

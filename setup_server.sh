#!/bin/bash
# Server setup: creates conda env with CUDA-enabled PyTorch.
# Usage: ./setup_server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="optloss"
PYTHON_VER="3.10"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

echo "============================================================"
echo "  OptimizationLoss — Server Setup"
echo "============================================================"
echo ""

# ── 1. Check conda ────────────────────────────────────────────────────
echo "[1/6] Checking conda..."
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    exit 1
fi
echo "  Found: $(conda --version)"

# ── 2. Check GPU drivers ──────────────────────────────────────────────
echo ""
echo "[2/6] Checking GPU drivers..."
if command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  Driver CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        echo "    $line"
    done
    echo ""
else
    echo "  WARNING: nvidia-smi not found — no GPU drivers?"
    echo "  Training will fall back to CPU (very slow)."
fi

# ── 3. Create conda environment ───────────────────────────────────────
echo "[3/6] Setting up conda environment '$ENV_NAME'..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists."
    read -p "  Recreate from scratch? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "  Removing old environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
        echo "  Creating fresh environment..."
        conda create -n "$ENV_NAME" python="$PYTHON_VER" -y -q
    fi
else
    echo "  Creating environment with Python $PYTHON_VER..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y -q
fi

# Activate (works in both bash and script contexts)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Activated: $(python --version) at $(which python)"

# ── 4. Install PyTorch with CUDA ──────────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch with CUDA support..."
echo "  Index: $CUDA_INDEX"
pip install torch torchvision --index-url "$CUDA_INDEX" -q
echo "  Done."

# ── 5. Install remaining dependencies ─────────────────────────────────
echo ""
echo "[5/6] Installing project dependencies..."
pip install numpy pandas scikit-learn scipy tqdm matplotlib medmnist Pillow seaborn -q
echo "  Done."

# ── 6. Verify CUDA ────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying PyTorch CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA compiled: {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
n = torch.cuda.device_count()
if n == 0:
    print('')
    print('  *** CUDA NOT AVAILABLE ***')
    print('  Possible causes:')
    print('    - Driver too old for this CUDA toolkit')
    print('    - CUDA_VISIBLE_DEVICES set incorrectly')
    print('    - GPU in exclusive mode for another user')
else:
    print(f'  GPUs visible: {n}')
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f'    [{i}] {name} ({mem:.1f} GB)')
    # Quick sanity test
    x = torch.randn(256, 256, device='cuda')
    y = torch.mm(x, x)
    print(f'  GPU compute test: PASSED')
"

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Activate:    conda activate $ENV_NAME"
echo "  Run:         ./run_experiments.sh"
echo "  Diagnose:    python diagnose_server.py"
echo ""
echo "  tmux:"
echo "    tmux attach -t optloss       # Reattach"
echo "    Ctrl+B, D                    # Detach"
echo "    tmux kill-session -t optloss # Stop"
echo ""

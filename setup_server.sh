#!/bin/bash
# ============================================================================
# OptimizationLoss — Server Setup Script
# ============================================================================
# Run this once on the university server to set up the environment.
# Usage:  ./setup_server.sh
# ============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  OptimizationLoss — Server Setup"
echo "============================================================"
echo ""

# ── 1. Check Python ──────────────────────────────────────────────────────
echo "[1/5] Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Install Python 3.8+ first."
    exit 1
fi

PYVER=$($PYTHON --version 2>&1)
echo "  Found: $PYVER"

# ── 2. Create virtual environment ────────────────────────────────────────
echo ""
echo "[2/5] Setting up virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation"
else
    $PYTHON -m venv .venv
    echo "  Created .venv"
fi

source .venv/bin/activate
echo "  Activated .venv ($(which python))"

# ── 3. Install dependencies ──────────────────────────────────────────────
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed"

# ── 4. Check GPU access ─────────────────────────────────────────────────
echo ""
echo "[4/5] Checking GPU access..."

# nvidia-smi check
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
else
    echo "  WARNING: nvidia-smi not found (GPUs may still work via PyTorch)"
fi

# PyTorch CUDA check
python -c "
import torch
n = torch.cuda.device_count()
if n == 0:
    print('  WARNING: PyTorch cannot see any CUDA GPUs')
    print('  Make sure CUDA drivers are installed and accessible')
else:
    print(f'  PyTorch sees {n} GPU(s):')
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
        print(f'    [{i}] {name} ({mem / (1024**3):.1f} GB)')
"

# ── 5. Check tmux ────────────────────────────────────────────────────────
echo ""
echo "[5/5] Checking tmux..."
if command -v tmux &>/dev/null; then
    echo "  tmux is available ($(tmux -V))"
else
    echo "  WARNING: tmux not found."
    echo "  Install it with:  sudo apt install tmux  (Ubuntu/Debian)"
    echo "  Or:               sudo yum install tmux  (CentOS/RHEL)"
    echo "  tmux keeps experiments running after you disconnect SSH."
fi

# ── Done ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Quick start:"
echo "    ./run_experiments.sh              # Run pending_runs (default)"
echo "    ./run_experiments.sh test_runs    # Run test_runs only"
echo ""
echo "  Or manually:"
echo "    source .venv/bin/activate"
echo "    EXPERIMENT_DIR=results/test_runs python main.py"
echo ""
echo "  Tmux commands:"
echo "    tmux attach -t optloss       # Reattach to running session"
echo "    Ctrl+B, D                    # Detach (experiments keep running)"
echo "    tmux kill-session -t optloss # Stop session"
echo ""

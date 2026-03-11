#!/bin/bash
# ============================================================================
# OptimizationLoss — Run Experiments in tmux
# ============================================================================
# Launches experiments in a persistent tmux session called "optloss".
# Safe to disconnect SSH — experiments continue running.
#
# Usage:
#   ./run_experiments.sh                    # Run pending_runs (default)
#   ./run_experiments.sh test_runs          # Run test_runs
#   ./run_experiments.sh pending_runs       # Run pending_runs (explicit)
#   tmux attach -t optloss                  # Reattach later
#   Ctrl+B, D                               # Detach (keeps running)
#   tmux kill-session -t optloss            # Stop everything
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="optloss"

# ── Configurable experiment directory ────────────────────────────────────
# Pass as first argument or defaults to pending_runs
RUN_DIR="${1:-pending_runs}"
export EXPERIMENT_DIR="results/${RUN_DIR}"

echo "Experiment directory: $EXPERIMENT_DIR"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: $EXPERIMENT_DIR does not exist."
    echo "Available:"
    ls -d results/*/ 2>/dev/null
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &>/dev/null; then
    echo "tmux is not installed. Running directly (NOT persistent)..."
    echo "Install tmux for session persistence: sudo apt install tmux"
    echo ""
    source .venv/bin/activate
    EXPERIMENT_DIR="$EXPERIMENT_DIR" python main.py
    exit $?
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "============================================================"
    echo "  Existing tmux session '$SESSION_NAME' found!"
    echo "============================================================"
    echo ""
    echo "  Attaching to it now..."
    echo "  (Ctrl+B, D to detach without stopping)"
    echo ""
    sleep 1
    tmux attach -t "$SESSION_NAME"
else
    echo "============================================================"
    echo "  Starting new tmux session: $SESSION_NAME"
    echo "  Experiment dir: $EXPERIMENT_DIR"
    echo "============================================================"
    echo ""
    echo "  Experiments will keep running even if you disconnect SSH."
    echo ""
    echo "  Commands:"
    echo "    Ctrl+B, D                    Detach (keeps running)"
    echo "    tmux attach -t $SESSION_NAME    Reattach later"
    echo "    tmux kill-session -t $SESSION_NAME  Stop everything"
    echo ""
    sleep 2

    # Create new tmux session running the experiments
    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$SCRIPT_DIR' && source .venv/bin/activate && EXPERIMENT_DIR='$EXPERIMENT_DIR' python main.py; echo ''; echo 'Experiments finished. Press Enter to close.'; read"

    # Attach to it
    tmux attach -t "$SESSION_NAME"
fi

#!/bin/bash
# Run experiments in a persistent tmux session.
# Usage:
#   ./run_experiments.sh                    # Run pending_runs (default)
#   ./run_experiments.sh test_runs          # Run specific directory
#   tmux attach -t optloss                  # Reattach later

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="optloss"
ENV_NAME="optloss"
RUN_DIR="${1:-pending_runs}"
export EXPERIMENT_DIR="results/${RUN_DIR}"

echo "Experiment directory: $EXPERIMENT_DIR"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: $EXPERIMENT_DIR does not exist."
    echo "Available:"
    ls -d results/*/ 2>/dev/null
    exit 1
fi

ACTIVATE_CMD="eval \"\$(conda shell.bash hook)\" && conda activate $ENV_NAME"

# Verify the environment exists
eval "$(conda shell.bash hook)" 2>/dev/null
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: conda environment '$ENV_NAME' not found."
    echo "Run ./setup_server.sh first."
    exit 1
fi

if ! command -v tmux &>/dev/null; then
    echo "tmux not installed. Running directly (NOT persistent)..."
    eval "$ACTIVATE_CMD"
    EXPERIMENT_DIR="$EXPERIMENT_DIR" python main.py
    exit $?

fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "============================================================"
    echo "  Existing tmux session '$SESSION_NAME' found!"
    echo "  Attaching... (Ctrl+B, D to detach without stopping)"
    echo "============================================================"
    sleep 1
    tmux attach -t "$SESSION_NAME"
else
    echo "============================================================"
    echo "  Starting new tmux session: $SESSION_NAME"
    echo "  Experiment dir: $EXPERIMENT_DIR"
    echo "  Conda env: $ENV_NAME"
    echo "============================================================"
    echo ""
    echo "  Ctrl+B, D                         Detach (keeps running)"
    echo "  tmux attach -t $SESSION_NAME      Reattach later"
    echo "  tmux kill-session -t $SESSION_NAME  Stop everything"
    echo ""
    sleep 2

    tmux new-session -d -s "$SESSION_NAME" \
        "cd '$SCRIPT_DIR' && $ACTIVATE_CMD && EXPERIMENT_DIR='$EXPERIMENT_DIR' python main.py; echo ''; echo 'Experiments finished. Press Enter to close.'; read"

    tmux attach -t "$SESSION_NAME"
fi

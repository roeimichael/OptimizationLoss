"""Thin shim: kept for back-compat after Stage C.

The fioretto_ldf logic moved to src/methodologies/fioretto_ldf/train.py.
The single entry point is src.experiments.runner. This shim makes
existing call sites (scripts/run_anchor.sh, scripts/validate_anchor.py)
that still target fioretto_research.run_fioretto keep working.
"""

from src.experiments.runner import main, run_experiment as run_fioretto

__all__ = ["main", "run_fioretto"]


if __name__ == "__main__":
    main()

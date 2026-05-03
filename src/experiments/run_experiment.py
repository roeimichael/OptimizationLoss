"""Thin shim: keep src.experiments.run_experiment as a back-compat entry point.

The real implementation moved to src.experiments.runner in Stage C step 8.
Existing scripts (main.py dispatch, run_anchor.sh, dispatch_multi_gpu.py)
keep working without changes.
"""

from src.experiments.runner import main, run_experiment

__all__ = ["main", "run_experiment"]


if __name__ == "__main__":
    main()

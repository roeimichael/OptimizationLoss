"""Thin shim: kept for back-compat after Stage C step 7 + 8.

run_heuristic was the heuristic+danits_lp entry. Its logic is now part of
the methodology dispatch in src.experiments.runner. This shim makes
existing call sites that still target the old module path work.
"""

from src.experiments.runner import main, run_experiment

__all__ = ["main", "run_experiment"]


if __name__ == "__main__":
    main()

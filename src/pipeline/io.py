"""Shared IO: write config['results'] block + final status.

Re-exports save_final_predictions / save_evaluation_metrics so that
methodology runners can import everything from pipeline.io.
"""

from src.training.logging import save_evaluation_metrics, save_final_predictions
from src.utils.filesystem_manager import save_config_to_path

__all__ = [
    "save_evaluation_metrics",
    "save_final_predictions",
    "save_results_to_config",
]


def save_results_to_config(config, experiment_path, results):
    """Write the results block, mark completed, persist config.json."""
    config["results"] = results
    config["status"] = "completed"
    save_config_to_path(config, experiment_path)

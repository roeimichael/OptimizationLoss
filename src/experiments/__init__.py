"""Experiment runners for different training methodologies.

This module contains runner scripts for different experimental methodologies:
- run_experiment: Adaptive lambda methodology

Each runner handles:
- Loading experiment configuration
- Data preprocessing and splitting
- Model training with methodology-specific trainer
- Evaluation and metrics computation
- Saving results and marking experiments complete
"""

from .run_experiment import run_experiment

__all__ = [
    'run_experiment',
]

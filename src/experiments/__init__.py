"""Experiment runners for different training methodologies.

This module contains runner scripts for different experimental methodologies:
- run_experiment: Original adaptive lambda methodology
- run_static_lambda_experiment: Static lambda experiments
- run_loss_proportional_experiment: Loss-proportional adaptive lambda
- run_scheduled_growth_experiment: Scheduled growth with loss gates

Each runner handles:
- Loading experiment configuration
- Data preprocessing and splitting
- Model training with methodology-specific trainer
- Evaluation and metrics computation
- Saving results and marking experiments complete
"""

from .run_experiment import run_experiment
from .run_static_lambda_experiment import run_static_lambda_experiment
from .run_loss_proportional_experiment import run_loss_proportional_experiment
from .run_scheduled_growth_experiment import run_scheduled_growth_experiment

__all__ = [
    'run_experiment',
    'run_static_lambda_experiment',
    'run_loss_proportional_experiment',
    'run_scheduled_growth_experiment',
]

from .trainer import ConstraintTrainer
from .static_lambda_trainer import StaticLambdaTrainer, ConstraintsNotMetError
from .loss_proportional_trainer import LossProportionalTrainer
from .scheduled_growth_trainer import ScheduledGrowthTrainer
from .metrics import (
    compute_train_accuracy,
    get_predictions_with_probabilities,
    compute_metrics,
    compute_prediction_statistics
)
from .constraints import (
    compute_global_constraints,
    compute_local_constraints
)
from .logging import (
    log_progress_to_csv,
    print_progress,
    save_final_predictions,
    save_evaluation_metrics
)

__all__ = [
    # Trainers
    'ConstraintTrainer',  # Adaptive lambda methodology
    'StaticLambdaTrainer',  # Static lambda methodology
    'LossProportionalTrainer',  # Loss-proportional adaptive lambda
    'ScheduledGrowthTrainer',  # Scheduled growth with loss gates
    'ConstraintsNotMetError',  # Exception for static lambda failures
    # Metrics
    'compute_train_accuracy',
    'get_predictions_with_probabilities',
    'compute_metrics',
    'compute_prediction_statistics',
    # Constraints
    'compute_global_constraints',
    'compute_local_constraints',
    # Logging
    'log_progress_to_csv',
    'print_progress',
    'save_final_predictions',
    'save_evaluation_metrics'
]

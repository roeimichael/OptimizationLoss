from typing import Tuple


def adjust_lambdas(lambda_global: float, lambda_local: float,
                   global_loss: float, local_loss: float,
                   threshold: float, lambda_step: float = 0.005,
                   lambda_max: float = 50.0) -> Tuple[float, float]:
    """Linearly increase lambda when constraint loss exceeds threshold."""
    new_lambda_global = lambda_global
    new_lambda_local = lambda_local

    if global_loss > threshold:
        new_lambda_global = min(lambda_global + lambda_step, lambda_max)

    if local_loss > threshold:
        new_lambda_local = min(lambda_local + lambda_step, lambda_max)

    return new_lambda_global, new_lambda_local

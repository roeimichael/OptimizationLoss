"""
Lambda Adjustment Strategies for Constraint Optimization

This module provides three different strategies for adjusting lambda weights
during constraint-based training:

1. Linear: Simple linear increase when constraints not satisfied (baseline)
2. Transfer: Transfer lambda steps from satisfied constraints to unsatisfied ones
3. Balanced: Initialize lambdas based on initial loss magnitudes, then linear adjustment
"""

from typing import Tuple, Dict, Any


class LambdaAdjuster:
    """Base class for lambda adjustment strategies."""

    def __init__(self, lambda_step: float = 0.005, lambda_max: float = 50.0):
        """
        Args:
            lambda_step: Step size for lambda increases
            lambda_max: Maximum allowed lambda value
        """
        self.lambda_step = lambda_step
        self.lambda_max = lambda_max
        self.strategy_name = "base"

    def initialize_lambdas(self, initial_global_loss: float, initial_local_loss: float,
                          lambda_global: float, lambda_local: float) -> Tuple[float, float]:
        """
        Initialize lambda values based on strategy.

        Args:
            initial_global_loss: Initial global constraint loss
            initial_local_loss: Initial local constraint loss
            lambda_global: Default global lambda
            lambda_local: Default local lambda

        Returns:
            Tuple of (lambda_global, lambda_local)
        """
        return lambda_global, lambda_local

    def adjust_lambdas(self, lambda_global: float, lambda_local: float,
                       global_satisfied: bool, local_satisfied: bool,
                       global_loss: float, local_loss: float,
                       threshold: float) -> Tuple[float, float]:
        """
        Adjust lambda values based on constraint satisfaction.

        Args:
            lambda_global: Current global lambda
            lambda_local: Current local lambda
            global_satisfied: Whether global constraints are satisfied
            local_satisfied: Whether local constraints are satisfied
            global_loss: Current global constraint loss
            local_loss: Current local constraint loss
            threshold: Loss threshold for satisfaction

        Returns:
            Tuple of (new_lambda_global, new_lambda_local)
        """
        raise NotImplementedError("Subclasses must implement adjust_lambdas")

    def get_info(self) -> Dict[str, Any]:
        """Return strategy information for logging."""
        return {
            'strategy': self.strategy_name,
            'lambda_step': self.lambda_step,
            'lambda_max': self.lambda_max
        }


class LinearLambdaAdjuster(LambdaAdjuster):
    """
    Linear lambda adjustment strategy (baseline).

    Simply increases lambda by lambda_step when constraint is not satisfied.
    This is the current methodology used in the codebase.
    """

    def __init__(self, lambda_step: float = 0.005, lambda_max: float = 50.0):
        super().__init__(lambda_step, lambda_max)
        self.strategy_name = "linear"

    def adjust_lambdas(self, lambda_global: float, lambda_local: float,
                       global_satisfied: bool, local_satisfied: bool,
                       global_loss: float, local_loss: float,
                       threshold: float) -> Tuple[float, float]:
        """Linear adjustment: increase lambda if constraint not satisfied."""
        new_lambda_global = lambda_global
        new_lambda_local = lambda_local

        # Increase global lambda if not satisfied
        if global_loss > threshold:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)

        # Increase local lambda if not satisfied
        if local_loss > threshold:
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)

        return new_lambda_global, new_lambda_local


class TransferLambdaAdjuster(LambdaAdjuster):
    """
    Transfer lambda adjustment strategy.

    When one constraint is satisfied, its lambda_step is transferred to
    accelerate the other constraint's satisfaction. This allows faster
    convergence by focusing optimization power on the remaining constraint.

    Example:
        - Global satisfied, local not satisfied
        - Next step: lambda_local increases by 2 * lambda_step
        - Next step: lambda_global stays constant
    """

    def __init__(self, lambda_step: float = 0.005, lambda_max: float = 50.0):
        super().__init__(lambda_step, lambda_max)
        self.strategy_name = "transfer"

    def adjust_lambdas(self, lambda_global: float, lambda_local: float,
                       global_satisfied: bool, local_satisfied: bool,
                       global_loss: float, local_loss: float,
                       threshold: float) -> Tuple[float, float]:
        """Transfer adjustment: move step from satisfied to unsatisfied constraint."""
        new_lambda_global = lambda_global
        new_lambda_local = lambda_local

        global_needs_increase = global_loss > threshold
        local_needs_increase = local_loss > threshold

        # Case 1: Both need increase - linear increase
        if global_needs_increase and local_needs_increase:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)

        # Case 2: Only global needs increase - transfer local's step to global
        elif global_needs_increase and not local_needs_increase:
            # Double the step for global (own step + transferred step)
            new_lambda_global = min(lambda_global + 2 * self.lambda_step, self.lambda_max)
            # Local stays constant

        # Case 3: Only local needs increase - transfer global's step to local
        elif local_needs_increase and not global_needs_increase:
            # Global stays constant
            # Double the step for local (own step + transferred step)
            new_lambda_local = min(lambda_local + 2 * self.lambda_step, self.lambda_max)

        # Case 4: Both satisfied - no changes

        return new_lambda_global, new_lambda_local


class BalancedLambdaAdjuster(LambdaAdjuster):
    """
    Balanced lambda initialization strategy.

    Initializes lambdas proportionally to balance the contribution of global
    and local losses to the total loss. This prevents one constraint from
    dominating early training due to magnitude differences.

    After initialization, uses linear adjustment.

    Example:
        - Initial global loss: 0.5
        - Initial local loss: 1.5 (3x larger)
        - Base lambda: 0.1
        - Result: lambda_global = 0.3, lambda_local = 0.1
        - This balances: 0.3 * 0.5 ≈ 0.1 * 1.5
    """

    def __init__(self, lambda_step: float = 0.005, lambda_max: float = 50.0,
                 base_lambda: float = 0.1, epsilon: float = 1e-6):
        super().__init__(lambda_step, lambda_max)
        self.strategy_name = "balanced"
        self.base_lambda = base_lambda
        self.epsilon = epsilon
        self.initialized = False

    def initialize_lambdas(self, initial_global_loss: float, initial_local_loss: float,
                          lambda_global: float, lambda_local: float) -> Tuple[float, float]:
        """
        Initialize lambdas to balance loss contributions.

        The lambda with smaller loss gets scaled up proportionally to the ratio
        of losses, so both constraints contribute equally initially.
        """
        self.initialized = True

        # Avoid division by zero
        if initial_global_loss < self.epsilon and initial_local_loss < self.epsilon:
            return lambda_global, lambda_local

        # Calculate the ratio to balance contributions
        if initial_global_loss > initial_local_loss:
            # Global loss is larger, scale up local lambda
            ratio = initial_global_loss / (initial_local_loss + self.epsilon)
            balanced_lambda_global = self.base_lambda
            balanced_lambda_local = min(self.base_lambda * ratio, self.lambda_max)
        else:
            # Local loss is larger, scale up global lambda
            ratio = initial_local_loss / (initial_global_loss + self.epsilon)
            balanced_lambda_global = min(self.base_lambda * ratio, self.lambda_max)
            balanced_lambda_local = self.base_lambda

        return balanced_lambda_global, balanced_lambda_local

    def adjust_lambdas(self, lambda_global: float, lambda_local: float,
                       global_satisfied: bool, local_satisfied: bool,
                       global_loss: float, local_loss: float,
                       threshold: float) -> Tuple[float, float]:
        """After initialization, use linear adjustment."""
        # Use linear adjustment after initialization
        new_lambda_global = lambda_global
        new_lambda_local = lambda_local

        if global_loss > threshold:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)

        if local_loss > threshold:
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)

        return new_lambda_global, new_lambda_local

    def get_info(self) -> Dict[str, Any]:
        """Return strategy information for logging."""
        info = super().get_info()
        info['base_lambda'] = self.base_lambda
        info['initialized'] = self.initialized
        return info


class CombinedLambdaAdjuster(LambdaAdjuster):
    """
    Combined lambda adjustment strategy.

    Combines the best of both balanced and transfer strategies:
    1. Balanced initialization: Initialize lambdas to equalize loss contributions
    2. Transfer adjustment: Transfer lambda step from satisfied to unsatisfied constraint

    This should provide both stable initialization and fast convergence.

    Example:
        - Initial global loss: 0.5, local loss: 1.5
        - Initialization: lambda_global = 0.3, lambda_local = 0.1
        - If global satisfied, local not: lambda_local increases by 2 * lambda_step
    """

    def __init__(self, lambda_step: float = 0.005, lambda_max: float = 50.0,
                 base_lambda: float = 0.1, epsilon: float = 1e-6):
        super().__init__(lambda_step, lambda_max)
        self.strategy_name = "combined"
        self.base_lambda = base_lambda
        self.epsilon = epsilon
        self.initialized = False

    def initialize_lambdas(self, initial_global_loss: float, initial_local_loss: float,
                          lambda_global: float, lambda_local: float) -> Tuple[float, float]:
        """
        Initialize lambdas to balance loss contributions (same as BalancedLambdaAdjuster).
        """
        self.initialized = True

        # Avoid division by zero
        if initial_global_loss < self.epsilon and initial_local_loss < self.epsilon:
            return lambda_global, lambda_local

        # Calculate the ratio to balance contributions
        if initial_global_loss > initial_local_loss:
            # Global loss is larger, scale up local lambda
            ratio = initial_global_loss / (initial_local_loss + self.epsilon)
            balanced_lambda_global = self.base_lambda
            balanced_lambda_local = min(self.base_lambda * ratio, self.lambda_max)
        else:
            # Local loss is larger, scale up global lambda
            ratio = initial_local_loss / (initial_global_loss + self.epsilon)
            balanced_lambda_global = min(self.base_lambda * ratio, self.lambda_max)
            balanced_lambda_local = self.base_lambda

        return balanced_lambda_global, balanced_lambda_local

    def adjust_lambdas(self, lambda_global: float, lambda_local: float,
                       global_satisfied: bool, local_satisfied: bool,
                       global_loss: float, local_loss: float,
                       threshold: float) -> Tuple[float, float]:
        """
        After initialization, use transfer adjustment (same as TransferLambdaAdjuster).
        """
        new_lambda_global = lambda_global
        new_lambda_local = lambda_local

        global_needs_increase = global_loss > threshold
        local_needs_increase = local_loss > threshold

        # Case 1: Both need increase - linear increase
        if global_needs_increase and local_needs_increase:
            new_lambda_global = min(lambda_global + self.lambda_step, self.lambda_max)
            new_lambda_local = min(lambda_local + self.lambda_step, self.lambda_max)

        # Case 2: Only global needs increase - transfer local's step to global
        elif global_needs_increase and not local_needs_increase:
            # Double the step for global (own step + transferred step)
            new_lambda_global = min(lambda_global + 2 * self.lambda_step, self.lambda_max)
            # Local stays constant

        # Case 3: Only local needs increase - transfer global's step to local
        elif local_needs_increase and not global_needs_increase:
            # Global stays constant
            # Double the step for local (own step + transferred step)
            new_lambda_local = min(lambda_local + 2 * self.lambda_step, self.lambda_max)

        # Case 4: Both satisfied - no changes

        return new_lambda_global, new_lambda_local

    def get_info(self) -> Dict[str, Any]:
        """Return strategy information for logging."""
        info = super().get_info()
        info['base_lambda'] = self.base_lambda
        info['initialized'] = self.initialized
        return info


def create_lambda_adjuster(strategy: str, lambda_step: float = 0.005,
                           lambda_max: float = 50.0) -> LambdaAdjuster:
    """
    Factory function to create lambda adjusters.

    Args:
        strategy: One of 'linear', 'transfer', 'balanced', 'combined'
        lambda_step: Step size for lambda increases
        lambda_max: Maximum allowed lambda value

    Returns:
        LambdaAdjuster instance

    Raises:
        ValueError: If strategy is not recognized
    """
    strategy = strategy.lower()

    if strategy == 'linear':
        return LinearLambdaAdjuster(lambda_step, lambda_max)
    elif strategy == 'transfer':
        return TransferLambdaAdjuster(lambda_step, lambda_max)
    elif strategy == 'balanced':
        return BalancedLambdaAdjuster(lambda_step, lambda_max)
    elif strategy == 'combined':
        return CombinedLambdaAdjuster(lambda_step, lambda_max)
    else:
        raise ValueError(f"Unknown lambda adjustment strategy: {strategy}. "
                        f"Choose from: 'linear', 'transfer', 'balanced', 'combined'")

"""
Improved convergence criterion with sustained satisfaction requirement.

PROBLEM:
Current code stops immediately when constraints are satisfied once.
This causes under-utilization (e.g., 39/43 budget used).

SOLUTION:
Require sustained satisfaction over multiple recent epochs.
This allows the model to oscillate around constraint boundaries
and find the optimal point that fully utilizes the budget.

Example:
- Constraint: 43 dropouts allowed
- Epochs 45-50: [41, 43, 42, 43, 44, 43] predictions
- Constraint satisfied in 4/6 epochs → keep training
- Epochs 51-60: [43, 43, 44, 43, 43, 42, 43, 43, 43, 43]
- Constraint satisfied in 9/10 epochs → CONVERGE (sustained)
"""

from collections import deque
from typing import Tuple


class SustainedConvergenceChecker:
    """
    Checks for sustained constraint satisfaction over a rolling window.

    Requires constraints to be satisfied for N out of M recent epochs
    before declaring convergence.
    """

    def __init__(self, window_size: int = 20, required_satisfied: int = 15):
        """
        Args:
            window_size: Number of recent epochs to track
            required_satisfied: Number of epochs that must be satisfied within window
        """
        self.window_size = window_size
        self.required_satisfied = required_satisfied
        self.history = deque(maxlen=window_size)

    def update(self, global_satisfied: bool, local_satisfied: bool) -> Tuple[bool, str]:
        """
        Update with current epoch's constraint status.

        Returns:
            (should_stop, reason): Whether to stop training and why
        """
        both_satisfied = global_satisfied and local_satisfied
        self.history.append(both_satisfied)

        if len(self.history) < self.window_size:
            # Not enough history yet
            satisfied_count = sum(self.history)
            return False, f"Building history: {satisfied_count}/{len(self.history)} epochs satisfied"

        # Check sustained satisfaction
        satisfied_count = sum(self.history)

        if satisfied_count >= self.required_satisfied:
            return True, f"Sustained convergence: {satisfied_count}/{self.window_size} recent epochs satisfied"
        else:
            return False, f"Not sustained: only {satisfied_count}/{self.window_size} recent epochs satisfied"

    def get_satisfaction_rate(self) -> float:
        """Get current satisfaction rate in the window."""
        if len(self.history) == 0:
            return 0.0
        return sum(self.history) / len(self.history)

    def reset(self):
        """Reset the checker (e.g., if constraint threshold changes)."""
        self.history.clear()


# Example usage in trainer.py:
"""
# At the top of train_constraints method:
convergence_checker = SustainedConvergenceChecker(
    window_size=20,      # Look at last 20 epochs
    required_satisfied=15 # Need 15/20 satisfied (75%)
)

# In the training loop, replace lines 196-224 with:
should_stop, reason = convergence_checker.update(
    criterion_constraint.global_constraints_satisfied,
    criterion_constraint.local_constraints_satisfied
)

if should_stop:
    print(f"\n[CONVERGED] {reason}")
    print(f"  Epoch {epoch + 1}")
    print(f"  Final loss: Global={avg_global:.6f}, Local={avg_local:.6f}")
    print(f"  Lambda values: Global={criterion_constraint.lambda_global:.2f}, Local={criterion_constraint.lambda_local:.2f}")

    # Save converged status
    save_run_status(...)
    break
else:
    # Optionally print progress every N epochs
    if (epoch + 1) % 10 == 0:
        rate = convergence_checker.get_satisfaction_rate()
        print(f"  [PROGRESS] Satisfaction rate: {rate*100:.1f}% ({sum(convergence_checker.history)}/{len(convergence_checker.history)})")
"""

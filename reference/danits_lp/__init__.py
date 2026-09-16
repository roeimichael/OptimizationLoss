"""danits_lp: in-house implementation of the two-phase LP framework from

    Shifman, Margolin, Halfi, Singer (2025).
    "Classification Tasks with Local and Global Resource Allocation Constraints."
    IFAC PapersOnLine 59-1: 61-66.

Phase 2 only: given frozen probabilities and per-class / per-group resource
budgets, solve the LP (totally-unimodular constraint matrix, so LP relaxation
yields integer solutions) for the cost-optimal assignment.

Clean-room reimplementation; fixes a transposed cost-matrix axis found while
auditing a colleague's notebook.

Only the LP is kept. That paper's Algorithm 1 greedy, the general cost-matrix
builder and a second cap->K derivation lived here and were reachable only
through this file's re-exports, which is why the AST reachability pass reported
them as live. The manuscript claims two post-hoc clippers -- a greedy threshold
(that is src/methodologies/heuristic, the `clip` arm) and LP-LG -- and says
LP-LG uses "an identity misclassification cost rather than the general cost
matrix", so none of the three was ever in scope.
"""

from .lp_solver import solve_lp_assignment, LPResult

__all__ = ["solve_lp_assignment", "LPResult"]

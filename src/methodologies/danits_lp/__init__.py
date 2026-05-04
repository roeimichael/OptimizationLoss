"""danits_lp: in-house implementation of the two-phase LP framework from

    Shifman, Margolin, Halfi, Singer (2025).
    "Classification Tasks with Local and Global Resource Allocation Constraints."
    IFAC PapersOnLine 59-1: 61-66.

Phase 2 only: given frozen probabilities and per-class / per-group resource
budgets, solve the LP (totally-unimodular constraint matrix, so LP relaxation
yields integer solutions) for the cost-optimal assignment. Also provides the
paper's Algorithm 1 greedy heuristic as a baseline.

Clean-room reimplementation. Fixes three bugs found while auditing a colleague's
notebook: (1) LP objective transposed cost matrix axes; (2) greedy heuristic
sorted descending instead of ascending; (3) heuristic had an extra "argmin fit"
gate absent from Alg. 1.
"""

from .lp_solver import solve_lp_assignment, LPResult
from .heuristic import solve_greedy_assignment, HeuristicResult
from .constraints_builder import build_psi_phi_from_percentages
from .cost_matrices import build_priority_cost_matrix, describe_cost_matrix

__all__ = [
    "solve_lp_assignment",
    "LPResult",
    "solve_greedy_assignment",
    "HeuristicResult",
    "build_psi_phi_from_percentages",
    "build_priority_cost_matrix",
    "describe_cost_matrix",
]

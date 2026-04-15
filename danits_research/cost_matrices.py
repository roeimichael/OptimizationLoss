"""
Cost-matrix presets for the Phase-2 LP, adapted from the three matrices
used in the colleague's student-dropout notebook.

Paper [5] convention:
    Omega[i, j] = cost of predicting class i when the true class is j.
    Diagonal = 0 (correct prediction is free).

The student notebook uses three matrices, all centered on class 0
("Dropout") being the most important class to not miss:

    MATRIX_A (62% of rows)  miss_top:over_top = 4:2  -> 2.00x
        [[0 1 2]
         [2 0 1]
         [4 2 0]]

    MATRIX_B (19% of rows)  miss_top:over_top = 5:3  -> 1.67x
        [[0 2 3]
         [4 0 2]
         [5 3 0]]

    MATRIX_C (19% of rows)  symmetric (|i - j|)
        [[0 1 2]
         [1 0 1]
         [2 1 0]]

We port that philosophy to DermMNIST, where the important class is
MEL (class 4) — missing a melanoma is the clinically dangerous mistake.

Builder semantics
-----------------
`build_priority_cost_matrix` produces a (C, C) matrix with three knobs:

    cost_miss   : cost of predicting NOT-important when true IS important
                  (false negative on the priority class)
    cost_over   : cost of predicting important when true is NOT important
                  (false positive: unnecessary biopsy / tutor / etc.)
    cost_other  : cost of any other misclassification
                  (a benign-for-benign confusion, no priority-class involvement)

so that for `priority_idx = 4` and `n_classes = 7`:

    Omega[i, 4] = cost_miss        for all i != 4       (row i, col 4)
    Omega[4, j] = cost_over        for all j != 4       (row 4, col j)
    Omega[i, j] = cost_other       for all i != j, i,j != 4
    Omega[i, i] = 0
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# generic builder
# ----------------------------------------------------------------------

def build_priority_cost_matrix(
    n_classes: int,
    priority_idx: int,
    cost_miss: float,
    cost_over: float,
    cost_other: float,
) -> np.ndarray:
    """
    Build a square cost matrix centered on one priority class.

    See module docstring for the semantics of cost_miss / cost_over / cost_other.
    """
    assert 0 <= priority_idx < n_classes
    assert cost_miss >= 0 and cost_over >= 0 and cost_other >= 0

    omega = np.full((n_classes, n_classes), float(cost_other))
    np.fill_diagonal(omega, 0.0)
    # Column = priority class -> the "miss it" false-negative costs
    omega[:, priority_idx] = cost_miss
    # Row = priority class -> the "over-call it" false-positive costs
    omega[priority_idx, :] = cost_over
    omega[priority_idx, priority_idx] = 0.0
    return omega


def build_symmetric_distance_cost_matrix(n_classes: int) -> np.ndarray:
    """Matrix C of the notebook: |i - j| — used when classes have a natural
    ordinal ordering. Kept mostly for reference; DermMNIST classes are not
    ordinal, so use one of the presets below instead."""
    idx = np.arange(n_classes)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


# ----------------------------------------------------------------------
# DermMNIST presets (priority = MEL = class 4)
# ----------------------------------------------------------------------
#
# AKIEC=0, BCC=1, BKL=2, DF=3, MEL=4, NV=5, VASC=6
#
# Ratios are chosen to bracket the notebook's A/B matrices (~1.67x-2x miss:over).

DERMMNIST_NUM_CLASSES = 7
DERMMNIST_MEL_IDX = 4

#: Zero/one cost (baseline, equivalent to minimising error rate under constraints).
#: Use this to get a LP result that is directly comparable to accuracy.
DERMMNIST_IDENTITY = build_priority_cost_matrix(
    DERMMNIST_NUM_CLASSES, DERMMNIST_MEL_IDX,
    cost_miss=1.0, cost_over=1.0, cost_other=1.0,
)

#: Analog of notebook Matrix A (miss/over = 2.0x). "Moderate MEL priority".
#: Missing a melanoma is 2x more costly than over-calling one.
DERMMNIST_MEL_PRIORITY_MODERATE = build_priority_cost_matrix(
    DERMMNIST_NUM_CLASSES, DERMMNIST_MEL_IDX,
    cost_miss=4.0, cost_over=2.0, cost_other=1.0,
)

#: Analog of notebook Matrix B (miss/over = 1.67x but absolute costs higher).
#: "Strong MEL priority" — all mistakes hurt more, MEL misses still worst.
DERMMNIST_MEL_PRIORITY_STRONG = build_priority_cost_matrix(
    DERMMNIST_NUM_CLASSES, DERMMNIST_MEL_IDX,
    cost_miss=5.0, cost_over=3.0, cost_other=2.0,
)

#: Clinically-motivated extreme (miss/over = 5x). Cited ratios in oncology
#: triage literature typically run 5x-10x; 5x is the low end. Treat as the
#: "aggressive" sensitivity ablation.
DERMMNIST_MEL_PRIORITY_CLINICAL = build_priority_cost_matrix(
    DERMMNIST_NUM_CLASSES, DERMMNIST_MEL_IDX,
    cost_miss=10.0, cost_over=2.0, cost_other=1.0,
)


DERMMNIST_PRESETS: dict[str, np.ndarray] = {
    "identity":  DERMMNIST_IDENTITY,
    "moderate":  DERMMNIST_MEL_PRIORITY_MODERATE,
    "strong":    DERMMNIST_MEL_PRIORITY_STRONG,
    "clinical":  DERMMNIST_MEL_PRIORITY_CLINICAL,
}


def describe_cost_matrix(omega: np.ndarray, priority_idx: int | None = None) -> str:
    """Pretty-print a cost matrix for logs / smoke tests."""
    lines = []
    header = "         " + " ".join(f"j={j:>4d}" for j in range(omega.shape[1]))
    lines.append(header)
    for i in range(omega.shape[0]):
        row = " ".join(f"{omega[i, j]:>5.1f}" for j in range(omega.shape[1]))
        marker = " <- priority pred" if priority_idx is not None and i == priority_idx else ""
        lines.append(f"i={i:>4d}  {row}{marker}")
    if priority_idx is not None:
        miss_vals = [omega[i, priority_idx] for i in range(omega.shape[0]) if i != priority_idx]
        over_vals = [omega[priority_idx, j] for j in range(omega.shape[1]) if j != priority_idx]
        other_vals = [
            omega[i, j]
            for i in range(omega.shape[0]) for j in range(omega.shape[1])
            if i != j and i != priority_idx and j != priority_idx
        ]
        lines.append(
            f"miss={miss_vals[0]:g}  over={over_vals[0]:g}  other={other_vals[0]:g}  "
            f"miss/over={miss_vals[0] / over_vals[0]:.2f}x"
        )
    return "\n".join(lines)

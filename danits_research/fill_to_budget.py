"""
Gap-closing post-hoc correction.

Takes ANY `y_pred` (not argmax) and adjusts it to hit the budget targets
Psi (global) and Phi (local) exactly, flipping the minimum number of
samples to close each gap, while respecting all constraints during the
process. Never overrides the input method's decisions beyond what is
strictly needed to close a gap.

Two directions of correction are applied:
    - REDUCE over-limit: if a constrained class has count > bound, flip
      the (count - bound) least-confident samples out of that class.
    - FILL under-limit: if a constrained class has count < bound, flip
      the (bound - count) most-confident non-class samples into it,
      without pulling from other classes that are themselves at or
      below their bounds.

The order of phases is:
    Phase 1 - reduce global over-limit  (Psi)
    Phase 2 - reduce local over-limit   (Phi)
    Phase 3 - fill global under-limit   (Psi) respecting Phi
    Phase 4 - fill local under-limit    (Phi) respecting Psi

We run reduce-phases first because they are strictly required for
feasibility; fill-phases only apply slack once feasibility holds.

Parameters
----------
y_pred_in : (N,) int array -- starting predictions
y_proba   : (N, C) float array -- model probabilities (used only for flip ordering)
groups    : (N,) array -- group ids per sample
psi       : list of length C, int bound or None per class
phi       : dict[group_id -> list of length C, int bound or None per class]
constrained_classes : iterable of ints; classes for which Psi/Phi are defined

Returns
-------
(y_pred_out, info) where info is a dict:
    {
      'phase1_reduce_global_flips' : int,
      'phase2_reduce_local_flips'  : int,
      'phase3_fill_global_flips'   : int,
      'phase4_fill_local_flips'    : int,
      'total_flips'                : int,
      'final_feasible'             : bool,
      'final_violations'           : list[str],
    }
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def posthoc_fill_gap(
    y_pred_in: np.ndarray,
    y_proba: np.ndarray,
    groups: np.ndarray,
    psi: list,
    phi: Optional[dict],
    constrained_classes: Iterable[int],
) -> tuple[np.ndarray, dict]:
    y_pred = np.asarray(y_pred_in, dtype=np.int64).copy()
    y_proba = np.asarray(y_proba, dtype=np.float64)
    groups = np.asarray(groups)

    n_samples, n_classes = y_proba.shape
    constrained = list(constrained_classes)
    phi_safe: dict = phi or {}

    info = {
        "phase1_reduce_global_flips": 0,
        "phase2_reduce_local_flips": 0,
        "phase3_fill_global_flips": 0,
        "phase4_fill_local_flips": 0,
        "total_flips": 0,
        "final_feasible": False,
        "final_violations": [],
    }

    # ----- running counts ---------------------------------------------
    def global_count(c: int) -> int:
        return int((y_pred == c).sum())

    def local_count(g, c: int) -> int:
        return int(((y_pred == c) & (groups == g)).sum())

    # A destination class is legal if it does not push Psi or Phi over
    # the bound. Filling and reducing share this check.
    def _legal_destination(idx: int, dest: int) -> bool:
        if dest in constrained:
            psi_dest = psi[dest] if dest < len(psi) else None
            if psi_dest is not None and global_count(dest) >= psi_dest:
                return False
            gid = groups[idx]
            if gid in phi_safe:
                bounds = phi_safe[gid]
                if dest < len(bounds) and bounds[dest] is not None:
                    if local_count(gid, dest) >= bounds[dest]:
                        return False
        return True

    def _pick_destination(idx: int, exclude: int) -> Optional[int]:
        """Best (highest P) class other than `exclude` that is legal."""
        # Rank classes by P(k) descending; pick first legal one.
        order = np.argsort(-y_proba[idx])
        for k in order:
            k = int(k)
            if k == exclude:
                continue
            if _legal_destination(idx, k):
                return k
        return None

    # ========== Phase 1: reduce global over-limit (Psi) ================
    for c in constrained:
        bound = psi[c] if c < len(psi) else None
        if bound is None:
            continue
        over = global_count(c) - bound
        if over <= 0:
            continue
        idx_c = np.where(y_pred == c)[0]
        # flip the LEAST confident-in-c samples first
        sort = idx_c[np.argsort(y_proba[idx_c, c])]
        flipped = 0
        for idx in sort:
            if flipped >= over:
                break
            dest = _pick_destination(int(idx), c)
            if dest is None:
                continue
            y_pred[idx] = dest
            flipped += 1
            info["phase1_reduce_global_flips"] += 1

    # ========== Phase 2: reduce local over-limit (Phi) =================
    for gid, bounds in phi_safe.items():
        for c in constrained:
            if c >= len(bounds) or bounds[c] is None:
                continue
            bound = bounds[c]
            over = local_count(gid, c) - bound
            if over <= 0:
                continue
            idx_gc = np.where((y_pred == c) & (groups == gid))[0]
            sort = idx_gc[np.argsort(y_proba[idx_gc, c])]
            flipped = 0
            for idx in sort:
                if flipped >= over:
                    break
                dest = _pick_destination(int(idx), c)
                if dest is None:
                    continue
                y_pred[idx] = dest
                flipped += 1
                info["phase2_reduce_local_flips"] += 1

    # ========== Phase 3: fill global under-limit (Psi) =================
    for c in constrained:
        bound = psi[c] if c < len(psi) else None
        if bound is None:
            continue
        under = bound - global_count(c)
        if under <= 0:
            continue
        # pick samples not currently in c, ranked by P(c) descending
        idx_not_c = np.where(y_pred != c)[0]
        sort = idx_not_c[np.argsort(-y_proba[idx_not_c, c])]
        filled = 0
        for idx in sort:
            if filled >= under:
                break
            # must not pull from an at-or-below-bound constrained class
            old = int(y_pred[idx])
            if old in constrained:
                old_bound = psi[old] if old < len(psi) else None
                if old_bound is not None and global_count(old) <= old_bound:
                    continue
            # must not violate Phi for destination c
            if not _legal_destination(int(idx), c):
                continue
            y_pred[idx] = c
            filled += 1
            info["phase3_fill_global_flips"] += 1

    # ========== Phase 4: fill local under-limit (Phi) ==================
    for gid, bounds in phi_safe.items():
        for c in constrained:
            if c >= len(bounds) or bounds[c] is None:
                continue
            bound = bounds[c]
            under = bound - local_count(gid, c)
            if under <= 0:
                continue
            # remaining global headroom for c
            global_bound = psi[c] if c < len(psi) else None
            if global_bound is not None:
                under = min(under, global_bound - global_count(c))
            if under <= 0:
                continue
            idx_g_not_c = np.where((y_pred != c) & (groups == gid))[0]
            sort = idx_g_not_c[np.argsort(-y_proba[idx_g_not_c, c])]
            filled = 0
            for idx in sort:
                if filled >= under:
                    break
                old = int(y_pred[idx])
                if old in constrained:
                    old_bound = psi[old] if old < len(psi) else None
                    if old_bound is not None and global_count(old) <= old_bound:
                        continue
                y_pred[idx] = c
                filled += 1
                info["phase4_fill_local_flips"] += 1

    info["total_flips"] = (
        info["phase1_reduce_global_flips"]
        + info["phase2_reduce_local_flips"]
        + info["phase3_fill_global_flips"]
        + info["phase4_fill_local_flips"]
    )

    # ----- final feasibility sanity check -----------------------------
    viol: list[str] = []
    for c in constrained:
        bound = psi[c] if c < len(psi) else None
        if bound is None:
            continue
        cnt = global_count(c)
        if cnt > bound:
            viol.append(f"psi[c={c}]: count={cnt} > bound={bound}")
    for gid, bounds in phi_safe.items():
        for c in constrained:
            if c >= len(bounds) or bounds[c] is None:
                continue
            bound = bounds[c]
            cnt = local_count(gid, c)
            if cnt > bound:
                viol.append(f"phi[g={gid},c={c}]: count={cnt} > bound={bound}")
    info["final_feasible"] = not viol
    info["final_violations"] = viol

    return y_pred, info

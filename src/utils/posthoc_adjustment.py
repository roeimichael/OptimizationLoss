# Post-hoc constraint enforcement.
# Two strategies:
#   1. minimal_correction: starts from model's argmax predictions, only flips
#      the minimum samples needed to satisfy constraints. Preserves the model's
#      learned decisions as much as possible.
#   2. lp_assignment: re-assigns all samples from scratch via LP to maximize
#      total confidence under constraints. Ignores model's argmax entirely.

import logging

import numpy as np
from scipy.optimize import linprog

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)

MAX_ITERATIONS = 10


# ================================================================
# Shared utilities
# ================================================================

def _check_all_satisfied(y_pred, global_con, local_con, group_ids, constrained_classes):
    for c in constrained_classes:
        if global_con[c] >= UNLIMITED:
            continue
        if (y_pred == c).sum() > int(global_con[c]):
            return False
    if local_con and group_ids is not None:
        for gid, group_limits in local_con.items():
            g_mask = (group_ids == gid)
            for c in constrained_classes:
                if group_limits[c] >= UNLIMITED:
                    continue
                if (y_pred[g_mask] == c).sum() > int(group_limits[c]):
                    return False
    return True


# ================================================================
# Strategy 1: Minimal correction (default)
# Only flips the minimum number of predictions to satisfy constraints.
# Per-flip dynamic blocking prevents flipping into other at-limit classes.
# ================================================================

def _compute_global_counts(y_pred, constrained_classes):
    return {c: int((y_pred == c).sum()) for c in constrained_classes}


def _compute_local_counts(y_pred, group_mask, constrained_classes):
    return {c: int((y_pred[group_mask] == c).sum()) for c in constrained_classes}


def _blocked_from_counts(counts, limits, constrained_classes, exclude_class):
    blocked = set()
    for c in constrained_classes:
        if c == exclude_class:
            continue
        limit = limits.get(c, UNLIMITED) if isinstance(limits, dict) else limits[c]
        if limit >= UNLIMITED:
            continue
        if counts.get(c, 0) >= int(limit):
            blocked.add(c)
    return blocked


def _flip_to_best_allowed(probs, blocked_classes):
    masked = probs.copy()
    for c in blocked_classes:
        masked[c] = -1
    return np.argmax(masked)


def _adjust_global(y_pred, y_proba, global_con, constrained_classes, target_class):
    limit = int(global_con[target_class])
    current = int((y_pred == target_class).sum())
    delta = current - limit
    if delta <= 0:
        return y_pred, 0

    counts = _compute_global_counts(y_pred, constrained_classes)
    indices = np.where(y_pred == target_class)[0]
    sorted_order = np.argsort(y_proba[indices, target_class])
    adjusted = 0
    for idx in indices[sorted_order[:delta]]:
        blocked = _blocked_from_counts(counts, global_con, constrained_classes, target_class)
        mask_classes = {target_class} | blocked
        new_class = _flip_to_best_allowed(y_proba[idx], mask_classes)
        counts[target_class] -= 1
        counts[new_class] = counts.get(new_class, 0) + 1
        y_pred[idx] = new_class
        adjusted += 1
    return y_pred, adjusted


def _adjust_local(y_pred, y_proba, group_ids, local_con, constrained_class,
                  global_con, all_constrained_classes):
    global_counts = _compute_global_counts(y_pred, all_constrained_classes)
    total_adjusted = 0
    for gid, group_limits in local_con.items():
        g_limit = group_limits[constrained_class]
        if g_limit >= UNLIMITED:
            continue
        g_limit = int(g_limit)
        g_mask = (group_ids == gid)
        g_count = int(((y_pred == constrained_class) & g_mask).sum())
        if g_count <= g_limit:
            continue
        local_counts = _compute_local_counts(y_pred, g_mask, all_constrained_classes)
        g_constrained = np.where(g_mask & (y_pred == constrained_class))[0]
        sorted_order = np.argsort(y_proba[g_constrained, constrained_class])
        n_to_flip = g_count - g_limit
        for idx in g_constrained[sorted_order[:n_to_flip]]:
            local_blocked = _blocked_from_counts(
                local_counts, group_limits, all_constrained_classes, constrained_class)
            global_blocked = _blocked_from_counts(
                global_counts, global_con, all_constrained_classes, constrained_class)
            mask_classes = {constrained_class} | local_blocked | global_blocked
            new_class = _flip_to_best_allowed(y_proba[idx], mask_classes)
            local_counts[constrained_class] -= 1
            local_counts[new_class] = local_counts.get(new_class, 0) + 1
            global_counts[constrained_class] -= 1
            global_counts[new_class] = global_counts.get(new_class, 0) + 1
            y_pred[idx] = new_class
        total_adjusted += n_to_flip
    return y_pred, total_adjusted


def minimal_correction(y_proba, group_ids, global_con, local_con,
                       constrained_classes):
    y_pred = np.argmax(y_proba, axis=1)

    if _check_all_satisfied(y_pred, global_con, local_con, group_ids,
                            constrained_classes):
        log.info("Minimal correction: argmax already satisfies all constraints")
        return y_pred, 0

    total_adjusted = 0
    for iteration in range(MAX_ITERATIONS):
        adjusted = 0
        for cc in constrained_classes:
            y_pred, n = _adjust_global(y_pred, y_proba, global_con,
                                       constrained_classes, cc)
            adjusted += n
        if local_con and group_ids is not None:
            for cc in constrained_classes:
                y_pred, n = _adjust_local(y_pred, y_proba, group_ids, local_con,
                                          cc, global_con, constrained_classes)
                adjusted += n
        total_adjusted += adjusted
        if _check_all_satisfied(y_pred, global_con, local_con, group_ids,
                                constrained_classes):
            log.info("Minimal correction: converged in %d iteration(s), %d flips",
                     iteration + 1, total_adjusted)
            return y_pred, total_adjusted
        if adjusted == 0:
            log.warning("Minimal correction: stalled at iteration %d", iteration + 1)
            break
    log.warning("Minimal correction: max iterations reached, %d flips", total_adjusted)
    return y_pred, total_adjusted


# ================================================================
# Strategy 2: LP-based full re-assignment (kept as option)
# ================================================================

def lp_constrained_assignment(y_proba, group_ids, global_con, local_con,
                              constrained_classes):
    n_samples, n_classes = y_proba.shape
    argmax_preds = np.argmax(y_proba, axis=1)

    if _check_all_satisfied(argmax_preds, global_con, local_con, group_ids,
                            constrained_classes):
        return argmax_preds, 0

    c_obj = -y_proba.flatten()

    from scipy.sparse import csr_matrix
    A_eq_rows, A_eq_cols, A_eq_vals = [], [], []
    for i in range(n_samples):
        for c in range(n_classes):
            A_eq_rows.append(i)
            A_eq_cols.append(i * n_classes + c)
            A_eq_vals.append(1.0)
    A_eq = csr_matrix((A_eq_vals, (A_eq_rows, A_eq_cols)),
                      shape=(n_samples, n_samples * n_classes))
    b_eq = np.ones(n_samples)

    A_ub_rows, A_ub_cols, A_ub_vals, b_ub_list = [], [], [], []
    row_idx = 0
    for cc in constrained_classes:
        if global_con[cc] >= UNLIMITED:
            continue
        for i in range(n_samples):
            A_ub_rows.append(row_idx)
            A_ub_cols.append(i * n_classes + cc)
            A_ub_vals.append(1.0)
        b_ub_list.append(int(global_con[cc]))
        row_idx += 1

    if local_con and group_ids is not None:
        for gid in np.unique(group_ids):
            if gid not in local_con:
                continue
            g_indices = np.where(group_ids == gid)[0]
            for cc in constrained_classes:
                k_local = local_con[gid][cc]
                if k_local >= UNLIMITED:
                    continue
                for i in g_indices:
                    A_ub_rows.append(row_idx)
                    A_ub_cols.append(i * n_classes + cc)
                    A_ub_vals.append(1.0)
                b_ub_list.append(int(k_local))
                row_idx += 1

    n_vars = n_samples * n_classes
    A_ub = csr_matrix((A_ub_vals, (A_ub_rows, A_ub_cols)),
                      shape=(row_idx, n_vars)) if A_ub_rows else None
    b_ub = np.array(b_ub_list, dtype=float) if b_ub_list else None

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, 1)] * n_vars, method='highs',
                     options={'presolve': True})

    if not result.success:
        log.warning("LP failed (status=%d), falling back to argmax", result.status)
        return argmax_preds, 0

    y_pred = np.argmax(result.x.reshape(n_samples, n_classes), axis=1)
    n_changed = int((y_pred != argmax_preds).sum())
    return y_pred, n_changed


# ================================================================
# Strategy 3: Targeted bidirectional correction (default)
# Handles both over-limit AND under-limit cases.
# Falls back to small-scope LP when greedy phases can't resolve all violations.
# ================================================================

def _build_gap_ledger(y_pred, global_con, local_con, group_ids, constrained_classes):
    """Compute gap = count - limit for each constraint. Positive=over, negative=under."""
    global_gap = {}
    for c in constrained_classes:
        if global_con[c] >= UNLIMITED:
            continue
        global_gap[c] = int((y_pred == c).sum()) - int(global_con[c])

    local_gap = {}
    if local_con and group_ids is not None:
        for gid, group_limits in local_con.items():
            local_gap[gid] = {}
            g_mask = (group_ids == gid)
            for c in constrained_classes:
                if group_limits[c] >= UNLIMITED:
                    continue
                local_gap[gid][c] = int((y_pred[g_mask] == c).sum()) - int(group_limits[c])

    return global_gap, local_gap


def _update_gaps(global_gap, local_gap, group_ids, idx, old_class, new_class):
    """Update gap ledgers after a flip."""
    if old_class in global_gap:
        global_gap[old_class] -= 1
    if new_class in global_gap:
        global_gap[new_class] += 1
    if group_ids is not None and local_gap:
        gid = int(group_ids[idx])
        if gid in local_gap:
            if old_class in local_gap[gid]:
                local_gap[gid][old_class] -= 1
            if new_class in local_gap[gid]:
                local_gap[gid][new_class] += 1


def _find_flip_target(idx, from_class, y_proba, global_gap, local_gap,
                      group_ids, constrained_classes, n_classes):
    """Find best target class for flipping a sample away from from_class.

    Prefers under-limit constrained classes, then non-constrained classes.
    Blocks constrained classes at/over limit and classes violating local limits.
    """
    constrained_set = set(constrained_classes)
    blocked = {from_class}

    # Block constrained classes at or over limit
    for c in constrained_classes:
        if c == from_class:
            continue
        if global_gap.get(c, 0) >= 0:
            blocked.add(c)

    # Block classes that would violate local limits for this sample's group
    if group_ids is not None and local_gap:
        gid = int(group_ids[idx])
        if gid in local_gap:
            for c in range(n_classes):
                if c in blocked:
                    continue
                if c in local_gap[gid] and local_gap[gid][c] >= 0:
                    blocked.add(c)

    # Priority 1: under-limit constrained classes
    under_limit = [c for c in constrained_classes
                   if c not in blocked and global_gap.get(c, 0) < 0]
    if under_limit:
        return max(under_limit, key=lambda c: y_proba[idx, c])

    # Priority 2: non-constrained classes
    non_constrained = [c for c in range(n_classes)
                       if c not in blocked and c not in constrained_set]
    if non_constrained:
        return max(non_constrained, key=lambda c: y_proba[idx, c])

    return None


def _fallback_lp(y_pred, y_proba, group_ids, global_con, local_con,
                 constrained_classes, global_gap, local_gap):
    """Small-scope LP on candidate samples only."""
    n_samples, n_classes = y_proba.shape

    # Identify candidate samples
    candidate_set = set()
    for c in constrained_classes:
        gap = global_gap.get(c, 0)
        if gap > 0:
            candidate_set.update(np.where(y_pred == c)[0])
        elif gap < 0:
            not_c = np.where(y_pred != c)[0]
            top_k = min(abs(gap) * 3, len(not_c))
            if top_k > 0:
                top_idx = not_c[np.argsort(-y_proba[not_c, c])[:top_k]]
                candidate_set.update(top_idx)

    if local_gap:
        for gid in local_gap:
            for c in constrained_classes:
                if c not in local_gap[gid]:
                    continue
                lgap = local_gap[gid][c]
                g_mask = (group_ids == gid)
                if lgap > 0:
                    candidate_set.update(np.where(g_mask & (y_pred == c))[0])
                elif lgap < 0:
                    g_not_c = np.where(g_mask & (y_pred != c))[0]
                    top_k = min(abs(lgap) * 3, len(g_not_c))
                    if top_k > 0:
                        top_idx = g_not_c[np.argsort(-y_proba[g_not_c, c])[:top_k]]
                        candidate_set.update(top_idx)

    candidate_indices = sorted(candidate_set)
    n_candidates = len(candidate_indices)

    local_violations = sum(
        1 for gid in local_gap for c in local_gap[gid] if local_gap[gid][c] > 0
    ) if local_gap else 0
    log.warning("LP fallback triggered: %d candidate samples, global_gaps=%s, local_violations=%d",
                n_candidates,
                {c: global_gap[c] for c in global_gap if global_gap[c] != 0},
                local_violations)

    if n_candidates == 0:
        log.warning("LP fallback: no candidates identified")
        return y_pred, 0, n_candidates

    from scipy.sparse import csr_matrix

    # Fixed counts from non-candidate samples
    fixed_global = np.zeros(n_classes)
    for i in range(n_samples):
        if i not in candidate_set:
            fixed_global[y_pred[i]] += 1

    fixed_local = {}
    if local_con and group_ids is not None:
        for gid in local_con:
            fixed_local[gid] = np.zeros(n_classes)
            g_mask = (group_ids == gid)
            for i in range(n_samples):
                if g_mask[i] and i not in candidate_set:
                    fixed_local[gid][y_pred[i]] += 1

    n_vars = n_candidates * n_classes
    c_obj = np.zeros(n_vars)
    for ci, si in enumerate(candidate_indices):
        for k in range(n_classes):
            c_obj[ci * n_classes + k] = -y_proba[si, k]

    # Each candidate assigned to exactly one class
    A_eq_rows, A_eq_cols, A_eq_vals = [], [], []
    for ci in range(n_candidates):
        for k in range(n_classes):
            A_eq_rows.append(ci)
            A_eq_cols.append(ci * n_classes + k)
            A_eq_vals.append(1.0)
    A_eq = csr_matrix((A_eq_vals, (A_eq_rows, A_eq_cols)),
                      shape=(n_candidates, n_vars))
    b_eq = np.ones(n_candidates)

    # Global + local upper bounds
    A_ub_rows, A_ub_cols, A_ub_vals, b_ub_list = [], [], [], []
    row_idx = 0
    for cc in constrained_classes:
        if global_con[cc] >= UNLIMITED:
            continue
        for ci in range(n_candidates):
            A_ub_rows.append(row_idx)
            A_ub_cols.append(ci * n_classes + cc)
            A_ub_vals.append(1.0)
        b_ub_list.append(int(global_con[cc]) - fixed_global[cc])
        row_idx += 1

    if local_con and group_ids is not None:
        for gid, group_limits in local_con.items():
            g_candidates = [ci for ci, si in enumerate(candidate_indices)
                            if group_ids[si] == gid]
            for cc in constrained_classes:
                if group_limits[cc] >= UNLIMITED:
                    continue
                for ci in g_candidates:
                    A_ub_rows.append(row_idx)
                    A_ub_cols.append(ci * n_classes + cc)
                    A_ub_vals.append(1.0)
                b_ub_list.append(
                    int(group_limits[cc]) -
                    fixed_local.get(gid, np.zeros(n_classes))[cc])
                row_idx += 1

    if not A_ub_rows:
        return y_pred, 0, n_candidates

    A_ub = csr_matrix((A_ub_vals, (A_ub_rows, A_ub_cols)),
                      shape=(row_idx, n_vars))
    b_ub = np.array(b_ub_list, dtype=float)

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, 1)] * n_vars, method='highs',
                     options={'presolve': True})

    if not result.success:
        log.warning("LP fallback infeasible (status=%d), returning best greedy result",
                     result.status)
        return y_pred, 0, n_candidates

    lp_assignments = result.x.reshape(n_candidates, n_classes)
    lp_flips = 0
    for ci, si in enumerate(candidate_indices):
        new_class = int(np.argmax(lp_assignments[ci]))
        if y_pred[si] != new_class:
            y_pred[si] = new_class
            lp_flips += 1

    log.info("LP fallback: %d additional flips from %d candidates",
             lp_flips, n_candidates)
    return y_pred, lp_flips, n_candidates


def targeted_correction(y_proba, group_ids, global_con, local_con,
                        constrained_classes):
    """Bidirectional post-hoc correction handling over-limit AND under-limit cases.

    Three phases:
      1. Reduce over-limit globally
      2. Fill under-limit globally
      3. Local enforcement (bidirectional per group)

    Falls back to small-scope LP if greedy phases leave violations.

    Returns: (y_pred, total_flips, metadata_dict)
    """
    y_pred = np.argmax(y_proba, axis=1)
    n_samples, n_classes = y_proba.shape

    if _check_all_satisfied(y_pred, global_con, local_con, group_ids,
                            constrained_classes):
        log.info("Targeted correction: argmax already satisfies all constraints")
        return y_pred, 0, {'lp_fallback_used': False, 'lp_fallback_candidates': 0}

    global_gap, local_gap = _build_gap_ledger(
        y_pred, global_con, local_con, group_ids, constrained_classes)
    constrained_set = set(constrained_classes)
    total_flips = 0

    # Phase 1: Reduce over-limit (global)
    for c in constrained_classes:
        gap = global_gap.get(c, 0)
        if gap <= 0:
            continue
        indices = np.where(y_pred == c)[0]
        sorted_idx = indices[np.argsort(y_proba[indices, c])]
        flipped = 0
        for idx in sorted_idx:
            if flipped >= gap:
                break
            target = _find_flip_target(idx, c, y_proba, global_gap, local_gap,
                                       group_ids, constrained_classes, n_classes)
            if target is None:
                continue
            y_pred[idx] = target
            _update_gaps(global_gap, local_gap, group_ids, idx, c, target)
            total_flips += 1
            flipped += 1

    # Phase 2: Fill under-limit (global)
    for c in constrained_classes:
        gap = global_gap.get(c, 0)
        if gap >= 0:
            continue
        n_fill = abs(gap)
        candidates = np.where(y_pred != c)[0]
        sorted_idx = candidates[np.argsort(-y_proba[candidates, c])]
        filled = 0
        for idx in sorted_idx:
            if filled >= n_fill:
                break
            old_class = y_pred[idx]
            # Only pull from non-constrained or over-limit constrained
            if old_class in constrained_set and global_gap.get(old_class, 0) <= 0:
                continue
            # Check local: would adding c to this group violate local limit?
            if group_ids is not None and local_gap:
                gid = int(group_ids[idx])
                if gid in local_gap and c in local_gap[gid] and local_gap[gid][c] >= 0:
                    continue
            y_pred[idx] = c
            _update_gaps(global_gap, local_gap, group_ids, idx, old_class, c)
            total_flips += 1
            filled += 1

    # Phase 3: Local enforcement (bidirectional per group)
    if local_con and group_ids is not None:
        for gid, group_limits in local_con.items():
            g_mask = (group_ids == gid)
            g_indices = np.where(g_mask)[0]

            # 3a: Reduce over-limit locally
            for c in constrained_classes:
                if gid not in local_gap or c not in local_gap[gid]:
                    continue
                lgap = local_gap[gid].get(c, 0)
                if lgap <= 0:
                    continue
                local_c = g_indices[y_pred[g_indices] == c]
                sorted_idx = local_c[np.argsort(y_proba[local_c, c])]
                flipped = 0
                for idx in sorted_idx:
                    if flipped >= lgap:
                        break
                    target = _find_flip_target(
                        idx, c, y_proba, global_gap, local_gap,
                        group_ids, constrained_classes, n_classes)
                    if target is None:
                        continue
                    y_pred[idx] = target
                    _update_gaps(global_gap, local_gap, group_ids, idx, c, target)
                    total_flips += 1
                    flipped += 1

            # 3b: Fill under-limit locally
            for c in constrained_classes:
                if gid not in local_gap or c not in local_gap[gid]:
                    continue
                lgap = local_gap[gid].get(c, 0)
                if lgap >= 0:
                    continue
                n_fill = abs(lgap)
                local_not_c = g_indices[y_pred[g_indices] != c]
                sorted_idx = local_not_c[np.argsort(-y_proba[local_not_c, c])]
                filled = 0
                for idx in sorted_idx:
                    if filled >= n_fill:
                        break
                    old_class = y_pred[idx]
                    if old_class in constrained_set and global_gap.get(old_class, 0) <= 0:
                        continue
                    y_pred[idx] = c
                    _update_gaps(global_gap, local_gap, group_ids, idx, old_class, c)
                    total_flips += 1
                    filled += 1

    # Verify
    if _check_all_satisfied(y_pred, global_con, local_con, group_ids,
                            constrained_classes):
        log.info("Targeted correction: %d flips, all constraints satisfied",
                 total_flips)
        return y_pred, total_flips, {
            'lp_fallback_used': False, 'lp_fallback_candidates': 0}

    # Fallback LP
    y_pred, lp_flips, n_lp_candidates = _fallback_lp(
        y_pred, y_proba, group_ids, global_con, local_con,
        constrained_classes, global_gap, local_gap)
    total_flips += lp_flips

    satisfied = _check_all_satisfied(
        y_pred, global_con, local_con, group_ids, constrained_classes)
    if not satisfied:
        log.warning("Targeted correction: constraints STILL not satisfied after "
                    "LP fallback (%d total flips)", total_flips)
    else:
        log.info("Targeted correction: %d total flips "
                 "(LP resolved remaining violations)", total_flips)

    return y_pred, total_flips, {
        'lp_fallback_used': True, 'lp_fallback_candidates': n_lp_candidates}

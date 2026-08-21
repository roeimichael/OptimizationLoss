"""Transductive prediction-count constraint penalty.

For each capped (class, scope) with soft count `s` and budget `K`, with excess
E = relu(s - K) and e = E/K:

    penalty(s, K) = E/(E+K)  +  rho * e^2 / (1 + e^2)

Rational saturation plus a bounded quadratic, so the whole term stays in
[0, 1+rho) no matter how far over budget the model is. `rho` ramps over the
constraint phase; `lambda` ratchets per class. The total is

    L_constraint = sum over capped (class, scope) of  lambda * penalty(s, K)

with soft (differentiable) counts here and hard (argmax) counts used only for
verification.

Scope note -- read `docs/FRAMEWORK.md` section 2a before adding a shape here.
Roughly thirteen arms varied this penalty (rational vs quadratic vs linear, rho
schedules, lambda schedules, finer granularity) and every one of them tied. The
reason is structural and is not about the shape: the penalty is a function of the
AGGREGATE COUNT, and post-hoc allocation scores only the RANKING, which an
aggregate-count gradient cannot reorder. `_penalty` is deliberately one small
method so a genuinely different idea is cheap to try -- but a new *shape* is a
repeat of a closed experiment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import UNLIMITED, EPSILON


def threshold_soft_count(p, K, temp):
    """A soft count whose per-item gradient lands ON the cut, not at p = 0.5.

    The shipped soft count is `sum_i p_i`, so `d s / d logit_i = p_i(1 - p_i)`:
    it peaks where the model is UNSURE and vanishes as `p -> 1`. The penalty
    shape only rescales `d penalty / d s` -- for `linear` it is the constant
    `1/K` -- so no shape moves WHERE the gradient lands. Measured on dermmnist
    x ViTB16, the cut (the K-th largest probability) sits at p = 0.94 on a
    4-epoch model and p = 0.999 once converged, where `p(1-p)` is 0.055 and
    0.0009. That is why the penalty reshuffles the middle of the ranking --
    Jaccard 0.29-0.42 against its own control -- and leaves prec@K unmoved.

    This keeps the count and moves its weight:

        s_c = sum_i sigmoid((p_i - tau) / temp),   tau = the K-th largest p_i

    whose per-item derivative peaks exactly at `tau`. It remains a transductive
    count -- in fact a closer approximation to the HARD count above the cut than
    `sum_i p_i` is -- so this stays inside the paper's formulation instead of
    bolting a per-item objective alongside it.

    `tau` is detached: it is an order statistic of the model's own probabilities,
    used to place the window, not a quantity to backpropagate through. No label
    is involved, so the transductive setting is preserved.

    `temp` is a real dose knob WITH A HARD UPPER BOUND, found while testing this:
    it must be small relative to the SPREAD OF PROBABILITIES NEAR THE CUT, not
    merely small in absolute terms. The per-item gradient is

        sigmoid'((p_i - tau)/temp) / temp   *   p_i(1 - p_i)

    and if `temp` exceeds the local spacing the first factor is flat across the
    whole window, leaving `p(1-p)` to dominate again -- which reproduces exactly
    the defect this exists to fix. Concretely: on a converged model the top 20
    probabilities span about 0.02, so `temp = 0.02` is already too coarse there
    and `temp = 0.002` is not. As `temp -> 0` it approaches the hard count and
    its gradient approaches a delta at the cut, which is useless from the other
    direction. Sweep it, and sweep it against the measured spacing.
    """
    k = int(K)
    if k <= 0 or k >= p.numel():
        # No cut to centre on. Fall back to the plain sum so the caller still
        # gets a differentiable count rather than a silent zero.
        return p.sum()
    tau = torch.topk(p.detach(), k, largest=True).values[-1]
    return torch.sigmoid((p - tau) / max(float(temp), EPSILON)).sum()


class MulticlassTransductiveLoss(nn.Module):

    def __init__(self, global_constraints, local_constraints,
                 num_classes=7, initial_rho=0.5,
                 penalty_shape="rational_bounded"):
        super().__init__()
        self.num_classes = num_classes
        self.penalty_shape = penalty_shape
        self.register_buffer('rho', torch.tensor(float(initial_rho)))
        self.global_constraints_satisfied = False
        self.local_constraints_satisfied = False
        # lambda keys: class idx for global, (group_id, class) for local
        self.lambda_global_per_class = {}
        self.lambda_local_per_key = {}

        if global_constraints is not None:
            assert len(global_constraints) == num_classes
            self.register_buffer('global_constraints',
                                 torch.tensor(global_constraints, dtype=torch.float32))
        else:
            self.register_buffer('global_constraints', torch.tensor([]))

        self.local_groups = {}
        for group_id, constraints in (local_constraints or {}).items():
            name = 'local_%d' % int(group_id)
            self.register_buffer(name, torch.tensor(constraints, dtype=torch.float32))
            self.local_groups[group_id] = name

    # ---- the penalty shape: the one place to change it ----------------------
    def _penalty(self, soft, K):
        """Rational saturation plus a bounded quadratic, both in the excess E.

        The excess is measured against the TRUE K, but both terms are scaled by
        max(K, 1). At K == 0 the unscaled forms are pinned at their own bound --
        E/(E+0) == 1 and (E/0)^2/(1+(E/0)^2) == 1 -- so the penalty is a nonzero
        CONSTANT with exactly zero gradient. A group holding no true instances of
        the capped class gets K == 0 legitimately, and that constraint then sits
        permanently unsatisfied: it contributes nothing to the model, but it
        holds the ratchet gate open and blocks the satisfaction freeze for every
        OTHER constraint, for the whole run.

        max(K, 1) is the identity for every K >= 1, so this is bit-identical to
        the previous form on every run made so far.
        """
        E = F.relu(soft - K)
        scale = K if K >= 1 else 1.0
        e = E / (scale + EPSILON)
        if self.penalty_shape == "linear":
            return e
        if self.penalty_shape == "squared":
            return e ** 2
        return (E / (E + scale + EPSILON)
                + self.rho * (e ** 2) / (1 + e ** 2 + EPSILON))

    # ---- why `linear` and `squared` exist -------------------------------
    # The shipped shape is bounded, so its gradient d(pen)/d(soft) is
    # NON-MONOTONE in the violation above rho ~ 1: near zero at the boundary,
    # peaking around 53-58% over, and decaying toward zero for anything worse.
    # A scope violated by 8x its budget gets 167x LESS pull than one violated
    # by 58% (FRAMEWORK 2a2, reproduced independently to four decimals).
    #
    # With a SINGLE term that is harmless: the constraint gradient is clipped
    # alone, so the shape is a scalar times a fixed direction and divides out.
    # With several terms it sets their RELATIVE weights, and it sets them
    # backwards. Measured on a real multi-class run (dermmnist, classes 2+4
    # capped, L30_G20, 2026-08-20): class 4 at 1.3x its budget was pulled to
    # 57 against K=45, while class 2 at 9.3x its budget ROSE to 410 against
    # K=44 and was never touched. The deepest violator is the one the shape
    # starves. Single-class runs never showed this because their spread across
    # scopes has median 1.5x; here it is ~30x.
    #
    # `linear` (e) and `squared` (e**2) have constant and growing pull with
    # depth respectively, which is what a penalty is supposed to do.
    # ⚠️ The default stays `rational_bounded`: it is the manuscript's Eq. 4,
    # and changing the default would silently reinterpret every stored result.

    def _sum(self, entries, device):
        """entries: (soft_count, K, lambda) for every capped (class, scope).
        Returns (total, all_satisfied, n_capped). Satisfaction is judged on the
        SOFT count here; hard counts verify separately in the trainer."""
        total = torch.tensor(0.0, device=device)
        all_satisfied, n = True, 0
        for soft, K, lam in entries:
            if (soft > K).item():
                all_satisfied = False
            total = total + lam * self._penalty(soft, K)
            n += 1
        return total, all_satisfied, n

    def _capped(self, constraints, num_classes):
        return [c for c in range(num_classes)
                if c < len(constraints) and constraints[c] < UNLIMITED]

    def compute_global_from_counts(self, soft_counts):
        device = soft_counts.device
        if len(self.global_constraints) == 0:
            self.global_constraints_satisfied = True
            return soft_counts.sum() * 0.0          # keeps the autograd graph alive
        con = self.global_constraints.to(device)
        entries = [(soft_counts[c], con[c], self.lambda_global_per_class.get(c, 0.0))
                   for c in self._capped(con, self.num_classes)]
        total, satisfied, n = self._sum(entries, device)
        self.global_constraints_satisfied = satisfied
        return total if n else soft_counts.sum() * 0.0

    def _zero(self, device=None):
        """A zero on the module's own device.

        A bare torch.tensor(0.0) is always on CPU, so adding it to a CUDA term
        raises. It reached the caller only when there was no local count tensor
        to hang the graph on, which is why it survived: tralo always adds a
        connected global term alongside it.
        """
        if device is None:
            device = self.global_constraints.device
        return torch.zeros((), device=device)

    def compute_local_from_counts(self, local_soft_counts):
        if not self.local_groups or not local_soft_counts:
            self.local_constraints_satisfied = True
            for v in local_soft_counts.values():
                return v.sum() * 0.0
            return self._zero()
        device = next(iter(local_soft_counts.values())).device
        entries = []
        for gid, buffer_name in self.local_groups.items():
            if gid not in local_soft_counts:
                continue
            soft = local_soft_counts[gid]
            con = getattr(self, buffer_name).to(device)
            entries += [(soft[c], con[c], self.lambda_local_per_key.get((gid, c), 0.0))
                        for c in self._capped(con, self.num_classes)]
        total, satisfied, n = self._sum(entries, device)
        self.local_constraints_satisfied = satisfied
        if n:
            return total
        for v in local_soft_counts.values():
            return v.sum() * 0.0
        return self._zero(device)

    # ---- lambda / rho -------------------------------------------------------
    def set_lambda_per_class(self, class_idx, value, scope='global', group_id=None):
        if scope == 'global':
            self.lambda_global_per_class[class_idx] = float(value)
        elif scope == 'local' and group_id is not None:
            self.lambda_local_per_key[(group_id, class_idx)] = float(value)

    def get_lambda_per_class(self, class_idx, scope='global', group_id=None):
        if scope == 'global':
            return self.lambda_global_per_class.get(class_idx, 0.0)
        return self.lambda_local_per_key.get((group_id, class_idx), 0.0)

    def increment_rho(self, step):
        self.rho.add_(step)

    def get_rho(self):
        return self.rho.item()

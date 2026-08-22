"""Did the constraint phase REORDER the capped class, or only shift it?

The scorer thresholds the ranking at the budget, so a monotone shift of the
capped score column is invisible to 9 of its 13 metrics while moving the soft
count freely. Without this, "the constraint phase did nothing useful" and "the
constraint phase moved a bias the scorer cannot see" are the same observation.

This lives here, not in a trainer, because it was written inside
`tralo/train.py` and reached only TraLO -- the exact shape of the CE-skip
asymmetry that produced a 0.22 cc-F1 artifact (FRAMEWORK section 3b). Every
trained arm imports these two functions and calls them identically.

Both are measured on the model that is actually SCORED, i.e. after any
checkpoint restore, since the restored model is the one whose predictions the
scorer reads.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def capped_scores(model, X_test, classes, chunk_size):
    """Softmax probability of each capped class, over the whole test set."""
    if not classes:
        return None
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X_test), chunk_size):
            logits = model(X_test[start:start + chunk_size])
            out.append(F.softmax(logits, dim=1)[:, classes].detach().cpu())
    model.train(was_training)
    return torch.cat(out, dim=0) if out else torch.zeros((0, len(classes)))


def reordering_report(model, X_test, before, classes, chunk_size):
    """Compare the capped class's test ranking before and after the constraint phase.

    tau = 1.0 with a large bias_shift means the constraint phase changed the
    count without changing a single decision the scorer can see.
    """
    if before is None or not classes:
        return {}
    after = capped_scores(model, X_test, classes, chunk_size)
    if after is None:
        return {}
    rep = {}
    for j, c in enumerate(classes):
        b, a = before[:, j].numpy(), after[:, j].numpy()
        if len(b) != len(a) or len(b) < 2:
            continue
        try:
            from scipy.stats import kendalltau, spearmanr
            tau = float(kendalltau(b, a).statistic)
            rho = float(spearmanr(b, a).statistic)
        except Exception:
            tau = rho = float("nan")
        # the single logit shift that best explains the soft-count change, on
        # the log-odds scale where a uniform bias IS an additive constant
        #
        # ⚠️ `eps` CLIPS A LOT OF REAL DATA HERE, and anyone reading
        # `shift_residual_sd` will and should worry about it: on
        # `results/dualbar2`, 35-73% of capped-class probabilities fall BELOW
        # 1e-6, so for most items the log-odds below is the constant -13.8
        # rather than a score. Treated runs are also more clipped than their
        # nulls (mean 0.590 vs 0.473), which is exactly the direction that
        # would manufacture a lower dispersion for treated runs out of nothing.
        #
        # MEASURED, 2026-08-22, and it does NOT explain the separation:
        # clipped-fraction vs `shift_residual_sd` over 29 runs is r = -0.360
        # (p = 0.055, spearman -0.250), about 13% of the variance, against
        # r = -0.942 for the constraint's actual soft-count displacement. And
        # the clipped fraction does NOT separate the groups on its own --
        # zero-dose 0.268-0.580 against treated 0.381-0.730, overlapping --
        # while `shift_residual_sd` does. So the clip is a real limitation of
        # the SCALE and not the cause of the effect.
        #
        # It is left at 1e-6 deliberately rather than tightened: a smaller eps
        # does not recover information, it hands the dispersion to the extreme
        # tail of a float32 softmax, and changing it would silently break
        # comparability with every stored run.
        eps = 1e-6
        lb = np.log(np.clip(b, eps, 1 - eps) / np.clip(1 - b, eps, 1 - eps))
        la = np.log(np.clip(a, eps, 1 - eps) / np.clip(1 - a, eps, 1 - eps))
        delta = float(np.mean(la - lb))
        resid = float(np.std(la - lb))
        rep["class_%d" % c] = {
            "kendall_tau": round(tau, 6), "spearman": round(rho, 6),
            "bias_shift": round(delta, 6), "shift_residual_sd": round(resid, 6),
            "soft_before": round(float(b.sum()), 3),
            "soft_after": round(float(a.sum()), 3),
        }
        log.info("reordering, class %d: tau=%.4f bias_shift=%+.4f (resid sd %.4f), "
                 "soft count %.1f -> %.1f. tau near 1.0 with a large shift "
                 "means the count moved but the RANKING did not, and the "
                 "scorer cannot see a ranking-preserving change.",
                 c, tau, delta, resid, b.sum(), a.sum())
    return rep

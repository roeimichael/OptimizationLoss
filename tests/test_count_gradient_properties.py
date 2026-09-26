"""Executable statements of WHY a count penalty cannot choose WHICH items fill the capped slots.

Each test is one property of the published bounded count penalty (tralo.global_constraint),
checked against autograd and against the analytic streamed gradient. Together they are the
mechanism behind LEDGER settled #9: at the right dose the constraint evicts what the post-hoc
cut evicts.
"""
import torch

from tralo.global_constraint import bounded_count_penalty
from tralo.streamed_constraint import count_logit_gradient

C = 3
CAPS = [None, None, None, 20, None]


def logits_over_cap(seed=0, n=120, lift=1.5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 5, generator=g, dtype=torch.float64)
    z[:, C] += lift                       # soft count of class 3 well above 20
    return z


def grad(z, lam=0.3, rho=0.5, caps=CAPS):
    z = z.clone().requires_grad_(True)
    bounded_count_penalty(z, caps, torch.full((5,), lam, dtype=torch.float64), rho).backward()
    return z.grad


def log_odds(z):
    other = torch.cat((z[:, :C], z[:, C + 1:]), 1)
    return z[:, C] - torch.logsumexp(other, 1)


def test_the_analytic_streamed_gradient_is_the_autograd_gradient():
    z = logits_over_cap()
    p = z.softmax(1)
    analytic = count_logit_gradient(p, CAPS, torch.full((5,), 0.3, dtype=torch.float64), 0.5)
    assert torch.allclose(analytic, grad(z), atol=1e-12)


def test_descent_lowers_every_items_capped_log_odds_and_raises_none():
    """Sign structure: a descent step can only DEMOTE; it can never lift an item into the slots."""
    z = logits_over_cap()
    step = -1e-4 * grad(z)
    change = log_odds(z + step) - log_odds(z)
    assert (change <= 1e-15).all()
    assert (change < 0).sum() == len(z)


def test_the_demotion_of_each_item_is_a_fixed_function_of_its_own_probability_vector():
    """Per item, the first-order change in capped log-odds along -grad is
    -k * p3 * [(1 - p3) + sum_j q_j p_j], k > 0 shared by all items, q = softmax over the other
    classes. It depends on the item's own probability vector and nothing else, so the
    constraint's choice of whom to demote uses no information the allocator lacks."""
    z = logits_over_cap(seed=1)
    g = grad(z)
    p = z.softmax(1)
    q = torch.softmax(torch.cat((z[:, :C], torch.full((len(z), 1), -1e300, dtype=z.dtype), z[:, C + 1:]), 1), 1)
    rate = ((torch.eye(5, dtype=z.dtype)[C] - q) * (-g)).sum(1)   # d(log-odds_3) per unit step along -grad
    shape = p[:, C] * ((1 - p[:, C]) + (q * p).sum(1))
    k = rate / shape
    assert torch.allclose(k, k.mean().expand_as(k), rtol=1e-9)
    assert k.mean() < 0


def test_the_direction_does_not_depend_on_lambda_or_rho_with_one_capped_class():
    z = logits_over_cap(seed=2)
    a, b = grad(z, lam=0.01, rho=0.0), grad(z, lam=5.0, rho=3.0)
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    assert abs(cos - 1.0) < 1e-12


def test_exactly_inactive_when_the_soft_count_meets_the_cap_even_if_the_hard_count_does_not():
    """Audit D2: a soft count under the cap gives a zero gradient although the argmax count exceeds it."""
    z = torch.zeros(60, 5, dtype=torch.float64)
    z[:30, C] = 0.2                                    # argmax 3 on 30 items, but each p3 ~ 0.24
    soft, hard = z.softmax(1)[:, C].sum(), int((z.argmax(1) == C).sum())
    assert hard > 20 and soft < 20
    assert grad(z).abs().max() == 0


def test_the_gradient_vanishes_on_confident_items():
    """A saturated model (p3 -> 1 on its grade-3 calls) gets almost no push on exactly those items."""
    z = logits_over_cap(seed=3)
    z[:10, C] += 20.0                                  # ten extremely confident grade-3 calls
    g = grad(z).norm(dim=1)
    assert g[:10].max() < 1e-6 * g[10:].median()


def test_the_induced_order_of_demotion_is_ordered_by_p3_near_the_cut():
    """Among items with equal other-class structure, demotion strength is monotone in p3(1 - p3):
    the push concentrates on UNCERTAIN items, whatever their true class."""
    z = torch.zeros(9, 5, dtype=torch.float64)
    z[:, C] = torch.linspace(-3.0, 5.0, 9, dtype=torch.float64)
    z = torch.cat((z, logits_over_cap(seed=4, n=60, lift=3.0)))
    g = grad(z)[:9]
    p3 = z[:9].softmax(1)[:, C]
    strength = g[:, C]
    expected = p3 * (1 - p3)
    assert torch.allclose(strength / strength.sum(), expected / expected.sum(), atol=1e-9)

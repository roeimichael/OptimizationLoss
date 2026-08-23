"""How much of the "constraint step" is actually the constraint?

The claim under test
--------------------
`optimizer` is built once and serves BOTH phases: 126 CE steps, then ONE
constraint step, per epoch, through the same Adam buffers. So when the
constraint step runs, Adam's state is whatever the CE steps left there:

    m_new = b1 * m_CE + (1 - b1) * g_con          b1 = 0.9
    v_new = b2 * v_CE + (1 - b2) * g_con^2        b2 = 0.999
    update = -lr * m_hat / (sqrt(v_hat) + eps)

Whether that ruins the step is NOT decidable from the code alone: it depends on
the relative magnitudes of m_CE and g_con, and those move during training (the
unit-norm clip pins |g_con| to 1 whenever it binds, while CE gradients shrink as
CE converges). So this measures it instead of assuming it.

The number that decides it is cos(actual_update, g_con): the alignment between
the step Adam actually takes and the direction the constraint asked for.

    ~1.0  -> the shared state is harmless, the step IS the constraint step
    ~0.0  -> the step is orthogonal to the constraint; the phase is theatre

Reported at several points in training, because CE gradient magnitude decays and
the answer may differ early vs late. A fresh-Adam reference is computed at each
point as the counterfactual the `separate_constraint_optimizer` arm implements.
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from src.losses.transductive_loss import MulticlassTransductiveLoss  # noqa: E402
from src.pipeline.data import load_data  # noqa: E402
from src.pipeline.setup import seed_all  # noqa: E402
from src.pipeline.warmup import make_dataloader, make_optimizer, run_warmup  # noqa: E402
from src.utils.constants import UNLIMITED  # noqa: E402
from src.utils.filesystem_manager import load_config_from_path  # noqa: E402


def flat_grad(model):
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                      if p.grad is not None])


def flat_state(opt, model, key):
    out = []
    for p in model.parameters():
        st = opt.state.get(p, {})
        v = st.get(key)
        out.append(v.detach().reshape(-1) if v is not None
                   else torch.zeros(p.numel(), device=p.device))
    return torch.cat(out)


def constraint_grad(model, X_test, gids, crit, chunk):
    """Exactly the pipeline's pass 1 + pass 2, leaving .grad populated."""
    model.eval()
    n = len(X_test)
    nch = (n + chunk - 1) // chunk
    with torch.no_grad():
        tg = torch.zeros(crit.num_classes, device=X_test.device)
        tl = {g: torch.zeros(crit.num_classes, device=X_test.device)
              for g in crit.local_groups}
        for ci in range(nch):
            s, e = ci * chunk, min((ci + 1) * chunk, n)
            p = F.softmax(model(X_test[s:e]).float(), dim=1)
            tg += p.sum(0)
            for g in crit.local_groups:
                m = gids[s:e] == g
                if m.any():
                    tl[g] += p[m].sum(0)
    for ci in range(nch):
        s, e = ci * chunk, min((ci + 1) * chunk, n)
        p = F.softmax(model(X_test[s:e]).float(), dim=1)
        cg = p.sum(0)
        cl = {}
        for g in crit.local_groups:
            m = gids[s:e] == g
            cl[g] = p[m].sum(0) if m.any() else torch.zeros_like(cg)
        g_soft = tg.detach() - cg.detach() + cg
        l_soft = {g: tl[g].detach() - cl[g].detach() + cl[g] for g in crit.local_groups}
        loss = (crit.compute_global_from_counts(g_soft)
                + crit.compute_local_from_counts(l_soft)) / nch
        loss.backward()
    return tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    # The escalation the log shows: lambda ratchets 0.01 -> ~1.43 and rho ramps
    # 0.5 -> rho_target (default 100) over the phase. Running the diagnostic at
    # the INITIAL values understates the real gradient by ~3500x, which is
    # exactly the regime where the unit-norm clip decides everything.
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--rho", type=float, default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config_from_path(a.config)
    hp = cfg["hyperparams"]
    seed_all(hp.get("seed"))
    d = load_data(cfg)
    Xte = d.X_test.to(dev)
    gids = torch.LongTensor(d.groups_test).to(dev)
    model, _ = run_warmup(cfg, d.num_classes, d.X_train, d.y_train, dev)

    crit = MulticlassTransductiveLoss(
        global_constraints=d.global_con, local_constraints=d.local_con,
        num_classes=d.num_classes, initial_rho=hp.get("initial_rho", 0.5),
        alpha_kl=0.0, penalty_mode=hp.get("penalty_mode", "both")).to(dev)
    ccs = [c for c in range(d.num_classes) if d.global_con[c] < UNLIMITED]
    for c in ccs:
        crit.set_lambda_per_class(c, a.lam, scope="global")
    for g, b in d.local_con.items():
        for c in ccs:
            if b[c] < UNLIMITED:
                crit.set_lambda_per_class(c, a.lam, scope="local", group_id=g)
    if a.rho is not None:
        crit.increment_rho(a.rho - crit.get_rho())
    print("lambda=%.4f  rho=%.2f" % (a.lam, crit.get_rho()))

    lr = hp.get("lr_constraint", 1e-4)
    opt = make_optimizer(model.parameters(), lr, dev)
    loader = make_dataloader(d.X_train, d.y_train, hp["batch_size"])
    ce = torch.nn.CrossEntropyLoss()

    print("caps=%s  constrained=%s  n_test=%d  batches/epoch=%d"
          % ([d.global_con[c] for c in ccs], ccs, len(Xte), len(loader)))
    print("\n%-6s %10s %10s %10s | %14s %14s | %s"
          % ("epoch", "|m_CE|", "|g_con|", "|sqrt v|", "cos(upd,g_con)",
             "cos(upd,m_CE)", "counts vs caps"))

    for ep in range(a.epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad(set_to_none=True)
            ce(model(bx), by).backward()
            opt.step()

        m_ce = flat_state(opt, model, "exp_avg").clone()
        v_ce = flat_state(opt, model, "exp_avg_sq").clone()

        opt.zero_grad(set_to_none=True)
        tg = constraint_grad(model, Xte, gids, crit, hp.get("constraint_chunk_size", 256))
        raw_norm = flat_grad(model).norm().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        g_con = flat_grad(model).clone()

        # what the SHARED optimizer will actually do on this step
        m_new = 0.9 * m_ce + 0.1 * g_con
        v_new = 0.999 * v_ce + 0.001 * g_con.pow(2)
        upd = m_new / (v_new.sqrt() + 1e-8)

        cos = torch.nn.functional.cosine_similarity
        print("%-6d %10.3e %10.3e(raw %8.2e) %10.3e | %14.4f %14.4f | %s"
              % (ep + 1, m_ce.norm(), g_con.norm(), raw_norm, v_ce.sqrt().norm(),
                 cos(upd, g_con, dim=0), cos(upd, m_ce, dim=0),
                 " ".join("%d/%d" % (tg[c].item(), d.global_con[c]) for c in ccs)))

        # counterfactual: a dedicated optimizer sees only g_con
        if ep == a.epochs - 1:
            fresh = make_optimizer(model.parameters(), lr, dev)
            m_f = 0.1 * g_con
            v_f = 0.001 * g_con.pow(2)
            upd_f = m_f / (v_f.sqrt() + 1e-8)
            print("\nfresh (dedicated) optimizer, same g_con: cos(upd,g_con) = %.4f"
                  % cos(upd_f, g_con, dim=0))
            del fresh


if __name__ == "__main__":
    main()

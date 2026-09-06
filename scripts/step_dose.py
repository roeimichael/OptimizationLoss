"""HOW BIG IS THE CONSTRAINT STEP, IN WEIGHTS, UNDER EACH DELIVERY RULE?

`constraint_step_rule` picks how the constraint gradient reaches the weights:

  shared  `optimizer.step()` on the SAME Adam the CE pass just took ~126 steps
          with, so the update is `lr * m_hat / (sqrt(v_hat) + eps)` and the
          constraint contributes only `(1 - beta1) = 0.1` of the numerator.
  sgd     `p.add_(p.grad, alpha=-lr)`, so the update is exactly `lr * ||g||`,
          which under `constraint_grad_mode: normalize` is exactly `lr * clip`.

This project has already measured what `shared` does to the DIRECTION:
`cos(parameter update, constraint gradient)` is 0.009-0.017, i.e. the
"constraint step" is ~98% a 127th CE step (`src/training/constraint_step.py`).

IT HAS NEVER MEASURED THE MAGNITUDE, and without it a null from `tralo_sgd` is
uninterpretable. Two failure modes point opposite ways and both are plausible
a priori:

  * `sgd` UNDER-delivers. Its step norm is `lr * clip` = 1e-4 flat, while an
    Adam step is roughly `lr * sqrt(N) * E|m_hat/sqrt(v_hat)|`. At MobileNetV2's
    ~3.5M parameters `sqrt(N)` alone is ~1871.
  * `sgd` OVER-delivers where it counts. Only the component ALONG the
    constraint direction can enforce the constraint, and `shared` keeps just
    ~1.3% of its large step there.

So the quantity that decides it is neither norm alone nor cosine alone. It is
the CONSTRAINT-ALIGNED DISPLACEMENT, `||dw|| * cos(dw, g_c)` -- how far the
weights actually move along the direction the constraint asked for. This script
measures that for both rules on a REAL backbone with REAL Adam state.

WHY REAL STATE AND NOT A TOY. `scripts/hp_liveness_real` exists because the
smoke net inverts verdicts: on it the clip never engages, so `lambda` reads LIVE
and `constraint_grad_clip` reads INERT, and on ViTB16 both flip. Adam's `v` is
the whole question here, and `v` is a property of the real loss surface. A
random-init net with synthetic gradients answers a different question.

READ THE RATIO, NOT EITHER COLUMN. The verdict is
`aligned_sgd / aligned_shared`:

  >> 1   `sgd` delivers more constraint-aligned movement. A null from
         `tralo_sgd` is then about the MECHANISM, not the dose.
  ~= 1   the two rules are dose-matched and the contrast is clean.
  << 1   `sgd` is under-dosed. A null from `tralo_sgd` says nothing about
         whether delivering the direction helps, and the honest fix is to
         report the dose gap rather than the null.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _flat(vecs):
    import torch
    return torch.cat([v.reshape(-1) for v in vecs])


def _grads(model):
    import torch
    return [p.grad.detach().clone() if p.grad is not None
            else torch.zeros_like(p) for p in model.parameters()]


def _weights(model):
    return [p.detach().clone() for p in model.parameters()]


def measure(model, optimizer, g_c, lr, clip, device):
    """Constraint-aligned displacement under each rule, from ONE shared state.

    Both rules are measured from the SAME Adam state and the SAME constraint
    gradient, and the optimizer state is restored between them, so the only
    difference is the rule. Measuring them on two separately-trained models
    would put the loss surface in the contrast.
    """
    import copy
    import torch

    g_flat = _flat(g_c)
    gn = float(g_flat.norm())
    assert gn > 0, "constraint gradient is exactly zero; nothing to measure"
    # THE DESCENT DIRECTION, not the gradient. A step is `-lr * g`, so
    # measuring against `+g` reports cos = -1 for a correct step and inverts
    # the whole verdict. Positive here means "the weights moved where the
    # constraint asked".
    unit = -g_flat / gn

    before = _weights(model)
    opt_state = copy.deepcopy(optimizer.state_dict())

    out = {}
    for rule in ("shared", "sgd"):
        # restore
        with torch.no_grad():
            for p, w in zip(model.parameters(), before):
                p.copy_(w)
        optimizer.load_state_dict(copy.deepcopy(opt_state))
        # install the constraint gradient, normalised exactly as
        # finish_constraint_step does under mode="normalize"
        with torch.no_grad():
            for p, g in zip(model.parameters(), g_c):
                p.grad = g.clone() * (clip / (gn + 1e-12))

        if rule == "sgd":
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
        else:
            optimizer.step()

        dw = _flat([p.detach() - w for p, w in zip(model.parameters(), before)])
        norm = float(dw.norm())
        cos = float(torch.dot(dw, unit) / (norm + 1e-30))
        out[rule] = {"norm": norm, "cos": cos, "aligned": norm * cos}

    # restore the model so the caller can reuse it
    with torch.no_grad():
        for p, w in zip(model.parameters(), before):
            p.copy_(w)
    optimizer.load_state_dict(opt_state)
    return out


def warm_adam(model, optimizer, loader, steps, device):
    """Take real CE steps so Adam's m and v are the ones a constraint step meets.

    This is the whole point of the probe: `v` is a property of the real loss
    surface, and the per-coordinate 1/sqrt(v) is what rescales the constraint
    gradient under `shared`.
    """
    import torch
    import torch.nn as nn
    crit = nn.CrossEntropyLoss()
    model.train()
    n = 0
    for xb, yb in loader:
        if n >= steps:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        crit(model(xb), yb).backward()
        optimizer.step()
        n += 1
    return n


def constraint_grad(model, loader, classes, device):
    """A gradient of the same SHAPE as the real constraint's: d(soft count)/dw.

    `S_c = sum_i p_ic`, differentiated through the network. Its DIRECTION is not
    the thing under test here -- both rules receive the same vector -- but using
    the real functional form keeps its interaction with `v` honest, which a
    random vector would not.
    """
    import torch
    model.zero_grad(set_to_none=True)
    model.eval()
    total = None
    for xb, _yb in loader:
        xb = xb.to(device)
        p = torch.softmax(model(xb), dim=1)
        s = p[:, classes].sum()
        total = s if total is None else total + s
        break                       # one batch is enough for a direction
    total.backward()
    return _grads(model)


def report(res, lr, clip, n_params, steps, out=sys.stdout):
    w = out.write
    w("\n%s\n" % ("=" * 74))
    w("CONSTRAINT STEP DOSE -- weight displacement per rule\n")
    w("  parameters %d   sqrt(N) %.0f   lr %.3g   clip %.3g   CE steps %d\n"
      % (n_params, math.sqrt(n_params), lr, clip, steps))
    w("%s\n" % ("=" * 74))
    w("  %-8s %14s %10s %16s\n" % ("rule", "||dw||", "cos", "aligned |dw|"))
    for rule in ("shared", "sgd"):
        r = res[rule]
        w("  %-8s %14.6g %10.4f %16.6g\n"
          % (rule, r["norm"], r["cos"], r["aligned"]))
    a_s, a_g = res["shared"]["aligned"], res["sgd"]["aligned"]
    ratio = a_g / a_s if a_s else float("inf")
    w("\n  aligned_sgd / aligned_shared = %.3g\n" % ratio)
    if ratio > 3:
        w("  VERDICT: `sgd` delivers MORE constraint-aligned movement (%.1fx).\n"
          "           A null from tralo_sgd is then about the MECHANISM.\n"
          % ratio)
    elif ratio > 1 / 3.0:
        w("  VERDICT: DOSE-MATCHED within 3x. The contrast is clean; a null\n"
          "           is about the mechanism, not the step size.\n")
    else:
        w("  VERDICT: `sgd` is UNDER-DOSED %.0fx. A null from tralo_sgd would\n"
          "           NOT be evidence that delivering the direction fails --\n"
          "           report the dose gap instead of the null.\n" % (1 / ratio))
    w("\n  Note `shared`'s cos is the fraction of a LARGE step that points\n"
      "  where the constraint asked; `sgd`'s is 1.0 by construction. Neither\n"
      "  column decides anything alone -- the product does.\n")
    return ratio


def self_test(out=sys.stdout):
    """Gate the two analytic identities the measurement rests on."""
    import torch
    checks = []

    torch.manual_seed(0)
    lin = torch.nn.Linear(50, 3, bias=False)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    lr, clip = 1e-3, 1.0

    g = [torch.randn_like(p) for p in lin.parameters()]

    # 1. sgd's step norm is EXACTLY lr*clip, by construction. If this drifts the
    #    probe is not measuring the rule the trainer implements.
    r = measure(lin, opt, g, lr, clip, "cpu")
    # RELATIVE tolerance: ||dw|| is accumulated in float32, whose relative
    # precision is ~1e-7 and which reaches ~4e-6 over a few hundred elements.
    # An absolute 1e-9 bar fails on arithmetic that is exactly right.
    checks.append(("sgd step norm is lr*clip to float32 precision",
                   abs(r["sgd"]["norm"] - lr * clip) / (lr * clip) < 1e-5))
    checks.append(("  and it is perfectly aligned by construction",
                   abs(r["sgd"]["cos"] - 1.0) < 1e-6))

    # 2. NEGATIVE CONTROL: with Adam state EMPTY, its first step is
    #    lr*sign(g) elementwise (m_hat = g, sqrt(v_hat) = |g|), so the norm is
    #    lr*sqrt(N) and the cos is NOT 1. A probe that reported cos 1.0 here
    #    would be silently running the sgd branch for both rules.
    n = sum(p.numel() for p in lin.parameters())
    exp = lr * math.sqrt(n)
    checks.append(("NEGATIVE CONTROL: empty-state Adam steps lr*sqrt(N) "
                   "(%.4g vs %.4g)" % (r["shared"]["norm"], exp),
                   abs(r["shared"]["norm"] - exp) / exp < 0.02))
    checks.append(("  and its cos is NOT 1, so the two rules really differ",
                   r["shared"]["cos"] < 0.999))

    # 3. the measurement must not MUTATE the model or the optimizer, or the
    #    second rule would be measured from a state the first one moved.
    w0 = _flat(_weights(lin)).clone()
    measure(lin, opt, g, lr, clip, "cpu")
    checks.append(("measuring twice leaves the weights untouched",
                   float((_flat(_weights(lin)) - w0).abs().max()) < 1e-12))

    print("", file=out)
    for label, good in checks:
        print("  %-64s %s" % (label[:64], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g_ in checks if not g_]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--config", help="a config.json from the campaign to price")
    a.add_argument("--ce-steps", type=int, default=60,
                   help="real CE steps taken before measuring, so Adam's v is "
                        "the real one. The trainer runs ~126 per epoch.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.config:
        a.error("give --config <path to a campaign config.json> (or --self-test)")

    import torch
    from src.models.model_factory import get_model
    from src.pipeline.data import load_data
    from src.pipeline.warmup import make_dataloader, make_optimizer

    cfg = json.load(open(args.config))
    hp = cfg["hyperparams"]
    # A torch.device, not a string: `src.pipeline.warmup.make_optimizer`
    # reads `device.type` to decide on fused Adam, and a str has no `.type`.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(hp["seed"]))

    data = load_data(cfg)
    model = get_model(cfg["model_name"], data.num_classes, **hp).to(device)
    loader = make_dataloader(data.X_train, data.y_train, int(hp["batch_size"]))
    opt = make_optimizer(model.parameters(), float(hp["lr"]), device)

    took = warm_adam(model, opt, loader, args.ce_steps, device)
    g_c = constraint_grad(model, loader, data.constrained_classes, device)

    res = measure(model, opt, g_c, float(hp["lr_constraint"]),
                  float(hp["constraint_grad_clip"]), device)
    n = sum(p.numel() for p in model.parameters())
    report(res, float(hp["lr_constraint"]),
           float(hp["constraint_grad_clip"]), n, took)
    return 0


if __name__ == "__main__":
    sys.exit(main())

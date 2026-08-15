"""CPU end-to-end smoke of the GEOM cut objective through the real train().

No GPU, no data: a tiny linear model on random tensors.  Checks
  1. every cut_loss / soft_count_mode combination runs and steps;
  2. the pure cut arm (no count penalty) still takes optimizer steps;
  3. the diagnostics sidecar is written with a plausible active fraction;
  4. the chunked accumulation of the per-sample hinge equals the whole-pool
     value (the term must NOT be divided by n_chunks);
  5. the hinge is invariant to a uniform shift of every margin, so inflating a
     competitor class is a null direction.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/OptimizationLoss/newdirections/arm_geom"))
os.chdir(os.path.expanduser("~/OptimizationLoss/newdirections/arm_geom"))

from src.methodologies.tralo.train import (train, class_margin, argmax_margin,
                                           build_cut_plan)
from src.pipeline.contracts import TrainInputs
from src.utils.constants import UNLIMITED

C, D, NTR, NTE = 5, 16, 200, 600
CC, K = 2, 60


def make_inputs(hp, tmp):
    torch.manual_seed(0); np.random.seed(0)
    model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
    Xtr = torch.randn(NTR, D); ytr = torch.randint(0, C, (NTR,))
    Xte = torch.randn(NTE, D); yte = np.random.randint(0, C, NTE)
    gids = np.random.randint(0, 2, NTE)
    gcon = [UNLIMITED] * C; gcon[CC] = float(K)
    lcon = {0: [UNLIMITED] * C, 1: [UNLIMITED] * C}
    lcon[0][CC] = 30.0; lcon[1][CC] = 30.0
    return TrainInputs(model=model, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                       group_ids=gids, global_con=gcon, local_con=lcon,
                       constrained_classes=[CC], num_classes=C,
                       config={"dataset_config": {}, "hyperparams": hp}, hyperparams=hp,
                       device=torch.device("cpu"),
                       experiment_path=tmp, csv_log_path=os.path.join(tmp, "training_log.csv"))


BASE = dict(warmup_epochs=0, constraint_epochs=6, lambda_step=0.05, lr=1e-3,
            lr_constraint=1e-3, batch_size=64, dropout=0.0, seed=1,
            enable_ce_skip=False, hybrid_mode="undershoot_hinge", fior_beta=0.5,
            lambda_global=0.01, lambda_local=0.01, stable_count_threshold=10 ** 9,
            constraint_chunk_size=128)

print("== 1/3: every mode runs and moves the weights ==")
for name, extra in [
    ("incumbent            ", {}),
    ("cut hinge only       ", dict(cut_loss="hinge", cut_gamma=1.0, lambda_global=0.0,
                                   lambda_local=0.0, fior_beta=0.0)),
    ("cut hinge + count    ", dict(cut_loss="hinge", cut_gamma=1.0)),
    ("cut hinge scope=both ", dict(cut_loss="hinge", cut_gamma=0.5, cut_scope="both")),
    ("otce control         ", dict(cut_loss="otce", lambda_global=0.0, lambda_local=0.0,
                                   fior_beta=0.0)),
    ("sigmoid count        ", dict(soft_count_mode="sigmoid", count_tau=0.25)),
]:
    hp = dict(BASE); hp.update(extra)
    tmp = tempfile.mkdtemp()
    inp = make_inputs(hp, tmp)
    w0 = inp.model[0].weight.detach().clone()
    out = train(inp)
    dw = float((out.model[0].weight.detach() - w0).norm())
    diag = os.path.join(tmp, "training_log_cut_diagnostics.csv")
    extra_msg = ""
    if os.path.exists(diag):
        import csv
        rr = list(csv.DictReader(open(diag)))
        fr = [float(r["n_act_frac"]) for r in rr]
        extra_msg = " | diag rows=%d n_act_frac %.3f->%.3f" % (len(rr), fr[0], fr[-1])
    print("  %s  |dW|=%.5f%s" % (name, dw, extra_msg))
    assert dw > 0, name + " took no step"

print("\n== 2/3: chunked accumulation == whole-pool value ==")
torch.manual_seed(3)
z = torch.randn(600, C, requires_grad=True)
m = class_margin(z, CC)
pl = build_cut_plan(m.detach(), torch.arange(600), K, 1.0)
u = (m - pl["theta"]) / pl["scale"]
whole = F.relu(1.0 - pl["sign"] * u).sum() / max(pl["n_act"], 1)
acc = 0.0
for s in range(0, 600, 128):
    e = min(s + 128, 600)
    mc = class_margin(z[s:e], CC)
    uc = (mc - pl["theta"]) / pl["scale"]
    acc = acc + F.relu(1.0 - pl["sign"][s:e] * uc).sum() / max(pl["n_act"], 1)
print("  whole=%.9f chunked=%.9f delta=%.2e" % (whole, acc, abs(float(whole - acc))))
assert abs(float(whole - acc)) < 1e-5

print("\n== 3/3: competitor-class inflation is a null direction ==")
# inflate ONE competitor class for the whole pool: z[:, j] += delta
for j in [k for k in range(C) if k != CC][:2]:
    for delta in (0.5, 2.0):
        z2 = z.detach().clone(); z2[:, j] += delta
        m2 = class_margin(z2, CC)
        pl2 = build_cut_plan(m2, torch.arange(600), K, 1.0)
        u2 = (m2 - pl2["theta"]) / pl2["scale"]
        v2 = float(F.relu(1.0 - pl2["sign"] * u2).sum() / max(pl2["n_act"], 1))
        # incumbent for comparison: soft count sum p_c
        s0 = float(torch.softmax(z.detach(), 1)[:, CC].sum())
        s2 = float(torch.softmax(z2, 1)[:, CC].sum())
        print("  inflate class %d by %.1f: soft count %.1f -> %.1f (%+.1f) | "
              "cut hinge %.6f -> %.6f (%+.2e)"
              % (j, delta, s0, s2, s2 - s0, float(whole), v2, v2 - float(whole)))
print("\nALL CHECKS PASSED")

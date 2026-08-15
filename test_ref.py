"""Regression guard: with every new hyperparameter left at its default, the
GEOM worktree's train() must reproduce the frozen reference bit for bit.

Runs the identical tiny CPU problem against whichever tree is passed on argv
and prints the resulting weight-delta norm to 12 digits.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

TREE = sys.argv[1]
sys.path.insert(0, TREE)
os.chdir(TREE)

from src.methodologies.tralo.train import train
from src.pipeline.contracts import TrainInputs
from src.utils.constants import UNLIMITED

C, D, NTR, NTE = 5, 16, 200, 600
CC, K = 2, 60

hp = dict(warmup_epochs=0, constraint_epochs=6, lambda_step=0.05, lr=1e-3,
          lr_constraint=1e-3, batch_size=64, dropout=0.0, seed=1,
          enable_ce_skip=False, hybrid_mode="undershoot_hinge", fior_beta=0.5,
          lambda_global=0.01, lambda_local=0.01, stable_count_threshold=10 ** 9,
          constraint_chunk_size=128)

torch.manual_seed(0); np.random.seed(0)
model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
Xtr = torch.randn(NTR, D); ytr = torch.randint(0, C, (NTR,))
Xte = torch.randn(NTE, D); yte = np.random.randint(0, C, NTE)
gids = np.random.randint(0, 2, NTE)
gcon = [UNLIMITED] * C; gcon[CC] = float(K)
lcon = {0: [UNLIMITED] * C, 1: [UNLIMITED] * C}
lcon[0][CC] = 30.0; lcon[1][CC] = 30.0
tmp = tempfile.mkdtemp()
inp = TrainInputs(model=model, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                  group_ids=gids, global_con=gcon, local_con=lcon,
                  constrained_classes=[CC], num_classes=C,
                  config={"dataset_config": {}, "hyperparams": hp}, hyperparams=hp,
                  device=torch.device("cpu"),
                  experiment_path=tmp, csv_log_path=os.path.join(tmp, "training_log.csv"))
w0 = [p.detach().clone() for p in model.parameters()]
out = train(inp)
d = sum(float(((p.detach() - q) ** 2).sum()) for p, q in zip(out.model.parameters(), w0))
print("TREE=%s  |dW|=%.12f" % (TREE, d ** 0.5))

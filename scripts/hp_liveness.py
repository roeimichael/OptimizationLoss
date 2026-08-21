"""Which hyperparameters actually change what TraLO does?

Not which ones the code READS -- audit_config already proves every emitted key
has a reader. This asks the different and more useful question: does changing
the value change the OUTPUT. A key that is read, used in an expression, and then
cancelled downstream is indistinguishable from a live knob in every static
check, and this project has shipped four of those.

The specific reason to doubt TraLO's knobs: the constraint gradient is passed
through `clip_grad_norm_(max_norm=constraint_grad_clip)`. Any UNIFORM scaling of
the penalty is therefore divided straight back out, so lambda and rho should be
inert for MAGNITUDE. What they can still do is change the MIX between scopes --
the total gradient is a weighted sum over the global scope and each bound local
group, and re-weighting that sum rotates the direction, which survives
normalisation.

So the prediction this script tests, per knob:

    lambda_global x lambda_local scaled TOGETHER   -> INERT (pure magnitude)
    lambda_local alone (changes the scope mix)     -> LIVE  (direction)
    initial_rho                                    -> LIVE  (shape -> mix)
    constraint_grad_clip                           -> LIVE  (the real dose axis)
    lambda_step                                    -> depends on whether the
                                                      ratchet outruns the clip

!! READ THE `clip binds` COLUMN. On this tiny smoke net the raw gradient norm
may sit BELOW the clip, in which case the clip is not normalising anything and a
magnitude knob will look live here while being dead on a real backbone where the
clip binds every epoch. A verdict from this script is only transferable when the
clip binds in both runs, and the column says whether it did.

    python -m scripts.hp_liveness
"""

import argparse
import hashlib
import sys
import tempfile

import numpy as np
import torch

from configs.gen_campaign import load_protocol
from scripts.smoke_arms import make_inputs, TRAIN_FNS

# (label, block, {key: value}) -- each entry is ONE knob moved, against baseline
PROBES = [
    ("lambda_g+l x10 (uniform)", "tralo",
     {"lambda_global": 0.1, "lambda_local": 0.1}),
    ("lambda_local x10 (mix)", "tralo", {"lambda_local": 0.1}),
    ("lambda_global x10 (mix)", "tralo", {"lambda_global": 0.1}),
    ("lambda_step 0.05->0.5", "tralo", {"lambda_step": 0.5}),
    ("initial_rho 0.5->3.0", "tralo", {"initial_rho": 3.0}),
    ("rho_target 100->10", "tralo", {"rho_target": 10.0}),
    ("grad_clip 1.0->3.0", "constraint_phase", {"constraint_grad_clip": 3.0}),
    ("grad_clip 1.0->0.3", "constraint_phase", {"constraint_grad_clip": 0.3}),
    ("lr_constraint x10", "constraint_phase", {"lr_constraint": 1e-3}),
]


def run(arm, overrides, epochs):
    seed = 1
    """One smoke run with `overrides` applied; returns (md5, max raw grad norm)."""
    P = load_protocol()
    for block, kv in overrides:
        target = (P["constraint_phase"] if block == "constraint_phase"
                  else P["blocks"][block])
        target.update(kv)
    torch.manual_seed(seed)
    np.random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        inputs, _g, _l = make_inputs(P, arm, tmp, seed=seed)
        inputs.hyperparams["constraint_epochs"] = epochs
        out = TRAIN_FNS[P["arms"][arm]["methodology"]](inputs)
        model = getattr(out, "model", None) or inputs.model
        model.eval()
        with torch.no_grad():
            logits = model(inputs.X_test).cpu().numpy().astype(np.float64)
        h = hashlib.md5(np.ascontiguousarray(logits).tobytes()).hexdigest()[:12]
        gn = out.summary.get("last_grad_norm")
        try:
            import pandas as pd
            df = pd.read_csv(inputs.csv_log_path)
            col = "Grad_Norm" if "Grad_Norm" in df.columns else "grad_norm"
            gn = float(pd.to_numeric(df[col], errors="coerce").max())
        except Exception:
            pass
    return h, gn


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--arm", default="tralo")
    a.add_argument("--epochs", type=int, default=4)
    args = a.parse_args()

    base_h, base_gn = run(args.arm, [], args.epochs)
    P = load_protocol()
    clip = P["constraint_phase"]["constraint_grad_clip"]
    print("arm=%s  baseline md5=%s  max raw grad norm=%s  clip=%s"
          % (args.arm, base_h, ("%.4g" % base_gn) if base_gn else "?", clip))
    print("clip BOUND in baseline: %s\n"
          % ("yes" if (base_gn or 0) >= clip else
             "NO -- magnitude verdicts below do NOT transfer to a real backbone"))

    print("%-28s %-14s %-12s %s" % ("knob moved", "md5", "max |g|", "verdict"))
    print("-" * 72)
    for label, block, kv in PROBES:
        try:
            h, gn = run(args.arm, [(block, kv)], args.epochs)
        except Exception as exc:
            print("%-28s %-14s %-12s %s"
                  % (label, "ERROR", "-", type(exc).__name__ + ": " + str(exc)[:32]))
            continue
        verdict = "LIVE" if h != base_h else "*** INERT (bit-identical)"
        print("%-28s %-14s %-12s %s"
              % (label, h, ("%.4g" % gn) if gn else "?", verdict))

    print()
    print("An INERT row means the value cannot affect a result, so a campaign")
    print("that sweeps it is running the same experiment repeatedly -- which")
    print("manufactures significance by duplicating n.")


if __name__ == "__main__":
    sys.argv = sys.argv or ["hp_liveness"]
    main()

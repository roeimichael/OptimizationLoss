"""Which TraLO knobs can change a RESULT, measured on the real backbone.

`scripts/hp_liveness.py` asks the same question on the smoke net and prints a
warning that its own answer does not transfer:

    clip BOUND in baseline: NO -- magnitude verdicts do NOT transfer
    max raw grad norm 0.03224 against a clip of 1.0

That warning is the whole point. On the smoke net the clip never engages, so
lambda and rho pass straight through and read LIVE, while
`constraint_grad_clip` reads INERT because there is nothing to clip. On ViTB16
the raw constraint-gradient norm is three to four orders of magnitude larger,
the clip binds every epoch, and both verdicts should inverT. A knob sweep
justified by the smoke net would be sweeping cancelled quantities.

WHY THIS IS NEWLY POSSIBLE. Until 2026-08-20 the pipeline's run-to-run noise
floor was 0.0358 macro-F1, so "did this knob change anything" was a statistical
question that no affordable number of seeds could answer -- the effects being
chased are ~0.002. With the determinism fix the floor is EXACTLY 0.0000: three
identical runs produced bit-identical predictions (md5 71aba83c x3) and
bit-identical weights (df387dd2 x3). So liveness is now a HASH COMPARISON, not
a test. Identical md5 means the knob provably cannot affect a result -- not
"the effect was small", but zero, with n=1 per setting.

WHAT IT DOES. Copies a completed TraLO run's config, moves ONE knob, and runs
the real pipeline through the same entry point a campaign uses. Every variant
shares one warm-up: none of the probed keys is in `warmup_identity_keys`
(lr, dropout, batch_size, warmup_epochs, pretrained, class_weighted_ce, seed,
warmup_loss, focal_alpha, focal_gamma, cb_beta, logit_adjust_tau), so they are
constraint-phase-only by construction and the shared cache is correct rather
than a confound. The assertion below enforces that; a probe on a warm-up key
would silently compare a model against itself.

    python -m scripts.hp_liveness_real <a completed tralo run dir> --epochs 5

Read the `clip bound` column before believing any magnitude verdict, exactly as
on the smoke net -- the instrument states its own validity.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

RUNNER = "src.experiments.runner"

# (label, {hyperparam: value}) -- ONE knob moved per row, against the baseline.
# Constraint-phase only; asserted against warmup_identity_keys below.
PROBES = [
    ("lambda_g+l x10 (uniform)", {"lambda_global": 0.1, "lambda_local": 0.1}),
    ("lambda_local x10 (mix)", {"lambda_local": 0.1}),
    ("lambda_global x10 (mix)", {"lambda_global": 0.1}),
    ("lambda_step 0.05->0.5", {"lambda_step": 0.5}),
    ("initial_rho 0.5->3.0", {"initial_rho": 3.0}),
    ("rho_target 100->10", {"rho_target": 10.0}),
    ("grad_clip 1.0->3.0", {"constraint_grad_clip": 3.0}),
    ("grad_clip 1.0->0.3", {"constraint_grad_clip": 0.3}),
    ("lr_constraint x10", {"lr_constraint": 1e-3}),
]


def _warmup_keys():
    p = Path(__file__).resolve().parents[1] / "configs" / "protocol.yml"
    return set(yaml.safe_load(p.read_text(encoding="utf-8"))["warmup_identity_keys"])


def one_run(base_cfg, dest, gpu, epochs, overrides, cache_dir):
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(base_cfg))
    cfg["status"] = "pending"
    cfg["hyperparams"]["constraint_epochs"] = epochs
    cfg["hyperparams"].update(overrides)
    out = dest / "config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # SHARED across every variant on purpose: the warm-up must be the same
    # model for all of them, or a knob's md5 would differ because it retrained.
    env["OPTLOSS_MODEL_CACHE"] = str(cache_dir.resolve())
    r = subprocess.run([sys.executable, "-m", RUNNER, str(out)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("run failed in %s" % dest)
    return dest


def read_result(dest, clip_value):
    """md5 of the predictions, plus whether the clip actually bound."""
    preds = dest / "final_predictions_raw.csv"
    md5 = (hashlib.md5(preds.read_bytes()).hexdigest()[:12]
           if preds.exists() else "NO-PREDS")

    max_gn, bound, sat_ever, last_hard = 0.0, 0, 0, None
    log = dest / "training_log.csv"
    if log.exists():
        with open(log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    gn = float(row.get("Grad_Norm") or 0.0)
                except ValueError:
                    continue
                if gn > max_gn:
                    max_gn = gn
                if gn >= clip_value:
                    bound += 1
                if str(row.get("Global_Satisfied", "")).strip() in ("1", "1.0"):
                    sat_ever = 1
                for k in row:
                    if k.startswith("Hard_Class") and row[k] not in ("", None):
                        try:
                            v = int(float(row[k]))
                        except ValueError:
                            continue
                        if v > 0:
                            last_hard = v
    return md5, max_gn, bound, sat_ever, last_hard


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("run_dir", help="a completed tralo run to base the probe on")
    a.add_argument("--epochs", type=int, default=5,
                   help="constraint epochs per variant (same for all)")
    a.add_argument("--gpu", default="0")
    a.add_argument("--out", default=None)
    args = a.parse_args()

    src = Path(args.run_dir) / "config.json"
    if not src.exists():
        raise SystemExit("no config.json in %s" % args.run_dir)
    base_cfg = json.loads(src.read_text(encoding="utf-8"))

    # A probe on a warm-up key would compare a model with itself, because every
    # variant deliberately shares one cache entry.
    wk = _warmup_keys()
    for label, ov in PROBES:
        bad = wk.intersection(ov)
        if bad:
            raise SystemExit(
                "probe %r moves %s, which is in warmup_identity_keys. Every "
                "variant shares one cached warm-up, so this would report INERT "
                "for a knob that in a real campaign retrains the backbone."
                % (label, sorted(bad)))

    root = Path(args.out or (Path(args.run_dir).parent / "_hp_liveness"))
    if root.exists():
        shutil.rmtree(root)
    cache = root / "_shared_cache"
    clip_value = float(base_cfg["hyperparams"]["constraint_grad_clip"])

    print("probing %s  arm=%s seed=%s  constraint_epochs=%d"
          % (args.run_dir, base_cfg.get("arm"),
             base_cfg.get("hyperparams", {}).get("seed"), args.epochs))
    print("baseline clip=%s\n" % clip_value)

    d = one_run(base_cfg, root / "baseline", args.gpu, args.epochs, {}, cache)
    b_md5, b_gn, b_bound, b_sat, b_hard = read_result(d, clip_value)
    print("baseline md5=%s  max|g|=%.4g  clip bound %d/%d epochs  sat_ever=%d  hard=%s"
          % (b_md5, b_gn, b_bound, args.epochs, b_sat, b_hard))
    print("CLIP BINDS: %s\n" % (
        "YES -- magnitude knobs should read INERT below" if b_bound else
        "NO -- magnitude verdicts below do NOT transfer to a full campaign"))

    print("%-28s %-14s %-11s %-9s %-7s %s"
          % ("knob moved", "md5", "max |g|", "clip bnd", "sat", "verdict"))
    print("-" * 88)
    rows = []
    for label, ov in PROBES:
        slug = label.split()[0].replace("/", "_") + "_" + str(len(rows))
        try:
            d = one_run(base_cfg, root / slug, args.gpu, args.epochs, ov, cache)
            md5, gn, bound, sat, hard = read_result(d, clip_value)
        except SystemExit as exc:
            print("%-28s %-14s %s" % (label, "ERROR", exc))
            continue
        verdict = "LIVE" if md5 != b_md5 else "*** INERT (bit-identical)"
        rows.append((label, verdict))
        print("%-28s %-14s %-11.4g %-9s %-7d %s"
              % (label, md5, gn, "%d/%d" % (bound, args.epochs), sat, verdict))

    inert = [l for l, v in rows if v.startswith("***")]
    print()
    if inert:
        print("INERT, so a campaign that sweeps them duplicates n rather than")
        print("testing anything: %s" % ", ".join(inert))
    else:
        print("every probed knob changed the predictions.")
    print()
    print("The floor is 0.0000 (three identical runs -> one md5), so an")
    print("identical hash is not 'a small effect'. It is no effect.")


if __name__ == "__main__":
    main()

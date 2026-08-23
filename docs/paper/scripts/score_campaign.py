"""Score ONE new-directions campaign against ITS OWN control arm, seed-paired.

Why this exists rather than score_arm.py --arm: score_arm groups by `method`,
but every arm in newdirections/ runs method=tralo and differs only by a flag, so
the arms all collapse into one row. It also compares against a reference corpus
built from a DIFFERENT campaign, which reintroduces the 0.027 cross-campaign
drift. Both problems disappear if the control lives inside the campaign and the
comparison is paired on seed, which is how these campaigns are generated.

Metric definitions are IMPORTED from score_arm.py, not re-implemented, so the
two scorers cannot drift apart.

LIVENESS GATE (this project has produced three fake headlines from flags that
were declared in config.json and never consumed): before any metric is printed,
final_predictions_raw.csv is md5'd for every paired ON/OFF run. A pair whose
hashes match means the flag changed nothing at all, and it is reported as DEAD
rather than as a 0.000 delta -- those two look identical on a scorecard and mean
completely different things.

    python score_campaign.py --campaign results/joint
    python score_campaign.py --campaign results/joint --control joint_off
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

# Load score_arm from THIS script's own directory, by path. A package import
# would resolve against the cwd, and these campaigns are run from five different
# worktrees -- one of which would silently pick up a different copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_score_arm", os.path.join(_HERE, "score_arm.py"))
_sa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sa)
score_run = _sa.score_run

# An arm is defined by whichever hyperparameters actually VARY inside the
# campaign -- auto-detected rather than listed. A hardcoded flag list silently
# scores a campaign as single-arm the moment a branch names its knob something
# the list does not contain (steps5 varies `constraint_clip_norm`, not
# `clip_norm`), and "no arm to score" and "the arm did nothing" look the same.
# Only the keys that identify the CELL are excluded.
CELL_KEYS = {"seed", "base_model_id", "experiment_name", "run_name", "lane",
             "output_dir", "gpu", "device"}

# Which prediction artifacts to score. A one-element list so load()
# can see the parsed flag without threading it through every call.
PREFIX = ["final"]

PRIMARY = "AP"          # allocation-free: quota filling cannot manufacture it
ORDER = ["AP", "ccF1eq", "macroEq", "count_over_K", "sat"]


def md5(p):
    if not os.path.exists(p):
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def arm_of(hp, varying):
    return "|".join("%s=%s" % (k, hp.get(k)) for k in varying)


def load(campaign):
    rows = []
    for cfg_path in glob.glob(campaign + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        d = os.path.dirname(cfg_path)
        r = score_run(d, cfg, prefix=PREFIX[0])
        if not r:
            continue
        hp = cfg.get("hyperparams") or {}
        r["hp"] = hp
        r["dir"] = d
        r["campaign"] = campaign
        r["base_model_id"] = cfg.get("base_model_id")
        r["md5"] = md5(os.path.join(
            d, "%s_predictions_raw.csv" % PREFIX[0]))
        r["warmup"] = hp.get("warmup_epochs")
        r["ce_skip"] = hp.get("enable_ce_skip")
        r["lr"] = hp.get("lr")
        r["lr_con"] = hp.get("lr_constraint")
        rows.append(r)
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--campaign", required=True, nargs="+",
                   help="one or more campaign dirs. Several are allowed ONLY "
                        "when the warm-up checkpoints pair -- see the "
                        "base_model_id gate below, which enforces it.")
    a.add_argument("--warmup-is-the-treatment", action="store_true",
                   help="allow the arms to start from different warm-up "
                        "checkpoints. Correct ONLY for an arm whose treatment "
                        "changes the warm-up itself -- a base_loss arm is the "
                        "case: base_loss sits in the warm-up cache key, so a "
                        "focal warm-up must not load a CE one, and demanding "
                        "identical checkpoints would refuse the very comparison "
                        "the arm exists to make. For a constraint-phase arm the "
                        "same mismatch means contamination, so this stays off "
                        "by default and has to be asserted per run.")
    a.add_argument("--prerestore", action="store_true",
                   help="score prerestore_predictions*.csv, i.e. the model as "
                        "training left it, instead of the post-restore model. "
                        "Required for any arm whose effect the lowest-excess "
                        "checkpoint selector can discard -- ortho's projection "
                        "trades away exactly the quantity the selector ranks on, "
                        "so the restored files can be byte-identical between arms "
                        "while the pre-restore ones differ.")
    a.add_argument("--control", default=None,
                   help="arm string to use as control; default = the arm whose "
                        "flags are all falsy/absent")
    args = a.parse_args()

    if args.prerestore:
        PREFIX[0] = "prerestore"
    rows = []
    for c in args.campaign:
        rows += load(c)
    if not rows:
        sys.exit("no scorable runs under " + ", ".join(args.campaign))

    keys = sorted({k for r in rows for k in r["hp"]} - CELL_KEYS)
    varying = [k for k in keys
               if len({json.dumps(r["hp"].get(k), sort_keys=True) for r in rows}) > 1]
    if not varying:
        sys.exit("every run has identical hyperparameters -- there is no arm to "
                 "score (are both arms finished?)")
    multi = len(args.campaign) > 1
    for r in rows:
        r["arm"] = arm_of(r["hp"], varying)
        if multi:
            # Two campaigns can set the same flags, so the campaign has to be
            # part of the arm identity or their runs silently merge into one arm.
            r["arm"] = os.path.basename(r["campaign"].rstrip("/")) + ":" + r["arm"]

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "hp"} for r in rows])
    df["count_over_K"] = df["count"] / df["K"]

    # ---- contract check. A campaign that broke the regime contract is not
    # comparable to anything, and this project has already published one
    # result that was a learning-rate artifact.
    print("=" * 78)
    print("CONTRACT  (warm-up 1, ce_skip False, lr_constraint == lr)")
    print("=" * 78)
    for c in ["warmup", "ce_skip", "lr", "lr_con"]:
        print("  %-9s %s" % (c, sorted(set(df[c].dropna().astype(str)))))
    bad = df[(df.warmup != 1) | (df.ce_skip != False) | (df.lr != df.lr_con)]
    print("  VIOLATIONS: %d of %d runs" % (len(bad), len(df)))

    arms = sorted(df.arm.unique())
    ctrl = args.control
    if ctrl is None:
        def falsy(s):
            return all(v in ("False", "None", "0", "0.0", "ce", "")
                       for v in (p.split("=", 1)[1] for p in s.split("|")))
        cands = [s for s in arms if falsy(s)]
        ctrl = cands[0] if len(cands) == 1 else None
    if ctrl is None or ctrl not in arms:
        sys.exit("could not identify a unique control among %s -- pass --control" % arms)

    print()
    print("=" * 78)
    print("ARMS      control = %s" % ctrl)
    print("=" * 78)
    for s in arms:
        n = (df.arm == s).sum()
        print("  %-45s n=%d%s" % (s, n, "   <- control" if s == ctrl else ""))

    key = ["dataset", "model", "cap", "seed"]
    C = df[df.arm == ctrl].set_index(key)

    for s in arms:
        if s == ctrl:
            continue
        T = df[df.arm == s].set_index(key)
        common = C.index.intersection(T.index)
        print()
        print("=" * 78)
        print("%s   vs   %s" % (s, ctrl))
        print("=" * 78)
        if not len(common):
            print("  no seed-paired cells")
            continue

        # WARM-UP PAIRING GATE. Comparing across campaigns is the move that
        # produced the 0.027 cross-campaign drift, but it is SAFE exactly when
        # both sides start from the same warm-up checkpoint -- and that is
        # checkable, because base_model_id is a hash of everything the warm-up
        # depends on. Prove it rather than assume it in either direction.
        bad = [i for i in common
               if C.loc[i, "base_model_id"] != T.loc[i, "base_model_id"]]
        if bad and not args.warmup_is_the_treatment:
            print("  *** WARM-UP MISMATCH in %d/%d pairs: these runs did NOT start"
                  % (len(bad), len(common)))
            print("      from the same checkpoint, so the delta contains warm-up")
            print("      drift as well as the treatment. REFUSING to score.")
            for i in bad[:4]:
                print("        %s  %s != %s"
                      % (i, C.loc[i, "base_model_id"], T.loc[i, "base_model_id"]))
            print("      If the arm CHANGES the warm-up on purpose (a base_loss arm")
            print("      does: base_loss is in the cache key so a focal warm-up")
            print("      cannot load a CE one), pass --warmup-is-the-treatment.")
            continue
        if bad:
            print("  WARM-UP   %d/%d pairs start from DIFFERENT checkpoints, declared"
                  % (len(bad), len(common)))
            print("            intentional. The delta therefore includes the warm-up,")
            print("            which for this arm is part of what is being tested.")
        else:
            print("  WARM-UP   all %d pairs share a bit-identical warm-up checkpoint"
                  % len(common))

        dead = [i for i in common
                if C.loc[i, "md5"] is not None and C.loc[i, "md5"] == T.loc[i, "md5"]]
        print("  LIVENESS  %d/%d paired runs differ from control"
              % (len(common) - len(dead), len(common)))
        if dead:
            print("  *** DEAD: the flag changed NOTHING in %d pairs. A 0.000 delta"
                  % len(dead))
            print("      here is not a null result, it is an inert flag. Pairs:")
            for i in dead[:6]:
                print("        ", i)
            if len(dead) == len(common):
                print("  *** REFUSING to report metrics: every pair is inert.")
                continue

        live = [i for i in common if i not in dead]
        print()
        print("  %-14s %10s %10s %10s   %s" % ("metric", "control", s.split("|")[0],
                                               "delta", "cells won"))
        for m in ORDER:
            c = C.loc[live, m].astype(float)
            t = T.loc[live, m].astype(float)
            d = (t - c)
            won = int((d > 0).sum())
            star = "  <-- PRIMARY" if m == PRIMARY else ""
            print("  %-14s %10.4f %10.4f %+10.4f   %d/%d%s"
                  % (m, c.mean(), t.mean(), d.mean(), won, len(live), star))
        print()
        print("  per-seed %s delta: %s" % (PRIMARY, ", ".join(
            "%s=%+.4f" % (i[-1], T.loc[i, PRIMARY] - C.loc[i, PRIMARY]) for i in live)))


if __name__ == "__main__":
    main()

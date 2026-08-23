"""How far does the lambda ratchet actually travel, per regime?

The mechanism argument says lambda is divided out by the unit-norm clip. Every
gradient-norm measurement backing that comes from the warm-up-1 `steps` arm,
because that is the only code path patched to log the pre-clip norm. The paper's
results are warm-up 50. Having just been burned generalising a warm-up-1
trajectory to the paper's regime, the same question has to be asked here before
the mechanism claim is stated as a fact about the method.

gn is not in the CSVs for the warm-up-50 corpora, so it cannot be read directly.
Lambda can. And lambda is the input to the ratchet story, so its trajectory
answers a sharper version of the question:

  - If lambda ratchets far at warm-up 50 (0.05 -> tens), the escalation the
    paper describes is really happening and the clip question is live there too.
  - If lambda barely moves, escalation never gets going in the paper's regime at
    all -- because satisfaction arrives first and the ratchet freezes -- and the
    paper's own mechanism figure (lambda 53 vs 0.18, a 297x spread) describes a
    regime its results never use.

Either answer is worth knowing. The second would be the more serious one.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def rows(root):
    out = []
    for cfg_path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg_path)
        try:
            cfg = json.load(open(cfg_path))
        except ValueError:
            continue
        if cfg.get("methodology") != "tralo":
            continue
        log = os.path.join(d, "training_log.csv")
        if not os.path.exists(log):
            continue
        try:
            t = pd.read_csv(log)
        except Exception:
            continue
        if "Epoch" not in t.columns or "Lambda_Global" not in t.columns or t.empty:
            continue
        hp = cfg["hyperparams"]
        warm = hp.get("warmup_epochs", 0)
        c = t[t.Epoch >= warm]
        if c.empty:
            continue
        lg = pd.to_numeric(c.Lambda_Global, errors="coerce").dropna()
        ll = pd.to_numeric(c.get("Lambda_Local", 0), errors="coerce").dropna()
        if lg.empty:
            continue
        sat = pd.to_numeric(c.get("Global_Satisfied", 0), errors="coerce").fillna(0)
        first = c.Epoch[sat > 0]
        out.append({
            "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
            "cap": cfg["constraint_tag"], "warmup": warm,
            "seed": hp.get("seed"), "step": hp.get("lambda_step"),
            "lam_g_max": float(lg.max()), "lam_g_final": float(lg.iloc[-1]),
            "lam_l_max": float(ll.max()) if len(ll) else float("nan"),
            "ratchet_steps": float(lg.max()) / hp.get("lambda_step", 0.05),
            "first_sat_epoch": float(first.iloc[0] - warm) if len(first) else float("nan"),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()

    all_d = []
    for r in args.roots:
        d = rows(r)
        if d.empty:
            print("no usable tralo runs under", r)
            continue
        d["corpus"] = os.path.basename(r.rstrip("/"))
        all_d.append(d)
    if not all_d:
        return 1
    d = pd.concat(all_d, ignore_index=True)

    print("=" * 96)
    print("HOW FAR THE LAMBDA RATCHET TRAVELS, BY REGIME")
    print("=" * 96)
    t = d.groupby(["corpus", "warmup"]).agg(
        n=("seed", "size"),
        lam_g_max_median=("lam_g_max", "median"),
        lam_g_max_p90=("lam_g_max", lambda s: s.quantile(.9)),
        lam_g_max_max=("lam_g_max", "max"),
        ratchet_steps_median=("ratchet_steps", "median"),
        first_sat_median=("first_sat_epoch", "median"))
    print(t.round(3).to_string())

    print()
    print("=" * 96)
    print("THE QUESTION: does lambda escalate in the regime the PAPER runs?")
    print("=" * 96)
    for (corpus, warm), g in d.groupby(["corpus", "warmup"]):
        base = g.step.median()
        med = g.lam_g_max.median()
        print("  %-22s warmup %-3d  lambda starts at %.3f, peaks at a median of %.3f  (%.1fx)"
              % (corpus, warm, base, med, med / base if base else float("nan")))
        if med / base < 3 if base else False:
            print("      -> the ratchet barely moves; escalation is not the operative mechanism here")
        else:
            print("      -> the ratchet does travel; the clip question is live in this regime")

    print()
    print("=" * 96)
    print("BY CELL, warm-up 50 only  (the paper's regime)")
    print("=" * 96)
    w = d[d.warmup >= 50]
    if w.empty:
        print("  no warm-up-50 runs in these roots")
    else:
        print(w.groupby(["dataset", "model", "cap"]).agg(
            n=("seed", "size"), lam_g_max=("lam_g_max", "median"),
            lam_l_max=("lam_l_max", "median"),
            first_sat=("first_sat_epoch", "median")).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

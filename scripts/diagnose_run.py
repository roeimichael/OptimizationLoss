"""Full stage-by-stage read of ONE completed run's training log.

Not a summary. The point is to recompute what the log CLAIMS from the raw
columns beside it, so a number that cannot be reproduced from its own inputs is
visible rather than trusted.

    python -m scripts.diagnose_run results/vit_diag/ViTB16/dermmnist/L30_G30/tralo/seed_1

Answers, per run:
  1. Did the constraint gradient CLIP bind, and on what fraction of epochs?
     This is the apples-to-apples question: an arm whose raw norm never reaches
     the clip is not receiving the same treatment as one that is clipped every
     epoch, whatever the config says they share.
  2. Did lambda move, and did it ratchet or converge?
  3. Did the soft count actually fall toward K, and did the HARD count follow?
     Soft is what the gradient sees; hard is what the metric is computed from.
  4. Is CE saturated? A saturated CE gates the constraint phase into
     irrelevance -- the warm-up-50 failure this project already burned a
     regime on.
  5. Does the logged L_Global match the penalty formula recomputed from the
     logged soft count? If not, the loss column and the count column disagree
     and one of them is wrong.

Epoch-convention trap: tralo logs an ABSOLUTE Epoch and the duals log a
RELATIVE one, and tralo logs SPARSELY -- so never assume row count == epochs.
"""
import argparse
import json
import os
import sys

import pandas as pd


def _fmt(x, n=6):
    try:
        return ("%." + str(n) + "g") % float(x)
    except (TypeError, ValueError):
        return str(x)


def _series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None


def _trajectory(s, k=6):
    """First few, last few -- enough to see direction without dumping 29 rows."""
    if s is None:
        return "absent"
    v = s.dropna().tolist()
    if len(v) <= 2 * k:
        return " ".join(_fmt(x, 4) for x in v)
    return (" ".join(_fmt(x, 4) for x in v[:k]) + "  ...  "
            + " ".join(_fmt(x, 4) for x in v[-k:]))


def _read_log(log_path, cfg):
    """Read training_log.csv whether or not it carries a header row.

    write_csv_header was called by exactly one arm (tralo), so every other
    arm's log is HEADERLESS and pandas would take the first logged epoch as the
    column names. Runs produced before that was fixed still have to be
    readable, and the schema is deterministic from num_classes plus the bound
    local groups, so reconstruct it rather than refusing the file.
    """
    from src.training.logging import build_csv_header
    from src.utils.constants import UNLIMITED

    with open(log_path, encoding="utf-8") as f:
        first = f.readline().strip()
    has_header = first.split(",")[0] == "Epoch"
    if has_header:
        return pd.read_csv(log_path)

    ds = cfg.get("dataset_config", {})
    n_cls = int(ds.get("num_classes", 7))
    raw = pd.read_csv(log_path, header=None)
    # Rebuild the local-constraint shape from the column count: the fixed part
    # is 10 + 3*n_cls, and every bound (group, class) pair adds 3 more.
    fixed = 10 + 3 * n_cls
    extra = max(0, raw.shape[1] - fixed)
    n_bound = extra // 3
    local = {g: [UNLIMITED] * n_cls for g in range(n_bound)}
    capped = ds.get("constrained_class")
    capped = capped if isinstance(capped, (list, tuple)) else [capped]
    c0 = int(capped[0]) if capped and capped[0] is not None else 0
    for g in local:
        local[g][c0] = 1          # any finite value: only boundedness matters
    header = build_csv_header(n_cls, local if n_bound else None)
    if len(header) != raw.shape[1]:
        header = header[:raw.shape[1]] + [
            "col%d" % i for i in range(len(header), raw.shape[1])]
    raw.columns = header
    print("   NOTE: this log had NO header row; reconstructed %d columns from "
          "num_classes=%d and %d bound local group(s)."
          % (raw.shape[1], n_cls, n_bound))
    return raw


def diagnose(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    log_path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(cfg_path):
        sys.exit("no config.json in %s" % run_dir)
    cfg = json.load(open(cfg_path, encoding="utf-8"))

    hp = cfg.get("hyperparams", {})
    ds = cfg.get("dataset_config", {})
    print("=" * 78)
    print("RUN  %s" % run_dir)
    print("=" * 78)
    print("  arm=%s  status=%s" % (cfg.get("arm"), cfg.get("status")))
    print("  dataset=%s  model=%s  cap=%s  seed=%s" % (
        cfg.get("dataset_mode"), cfg.get("model_name"),
        cfg.get("constraint_tag"), hp.get("seed")))
    print("  capped_class=%s  num_classes=%s  group_column=%s" % (
        ds.get("constrained_class"), ds.get("num_classes"),
        ds.get("group_column")))
    rt = cfg.get("runtime", {})
    if rt:
        print("  AMP: %s  scaler=%s  gpu=%s" % (
            rt.get("amp_dtype"), rt.get("grad_scaler"), rt.get("gpu_name")))
    clip = hp.get("constraint_grad_clip")
    print("  warmup=%s  constraint_epochs=%s  lr_constraint=%s  clip=%s" % (
        hp.get("warmup_epochs"), hp.get("constraint_epochs"),
        hp.get("lr_constraint"), clip))

    if not os.path.exists(log_path):
        print("\n  (no training_log.csv -- a post-hoc arm with 0 constraint "
              "epochs may not write one)")
        _metrics(run_dir)
        return

    df = _read_log(log_path, cfg)
    ep = _series(df, "Epoch")
    print("\n  training_log.csv: %d rows, Epoch %s..%s"
          % (len(df),
             _fmt(ep.min(), 4) if ep is not None else "?",
             _fmt(ep.max(), 4) if ep is not None else "?"))

    # ---- 1. did the CLIP bind? the apples-to-apples question ---------------
    gn = _series(df, "Grad_Norm")
    print("\n-- 1. CONSTRAINT GRADIENT vs THE CLIP " + "-" * 38)
    if gn is None or gn.dropna().empty:
        print("   no Grad_Norm column")
    else:
        g = gn.dropna()
        nz = g[g > 0]
        print("   raw norm: min=%s  median=%s  max=%s   (nonzero rows %d/%d)"
              % (_fmt(g.min()), _fmt(g.median()), _fmt(g.max()), len(nz), len(g)))
        print("   trajectory: %s" % _trajectory(g))
        if clip:
            binds = int((g >= float(clip)).sum())
            pct = 100.0 * binds / max(1, len(g))
            print("   clip=%s BINDS on %d/%d epochs (%.1f%%)"
                  % (clip, binds, len(g), pct))
            if binds == 0:
                print("   *** THE CLIP NEVER BINDS. This arm's constraint step is")
                print("       whatever its raw gradient happens to be, while an arm")
                print("       clipped every epoch takes a fixed unit step. They are")
                print("       NOT receiving the same treatment.")
            elif pct < 95:
                print("   *** PARTIAL: on %.1f%% of epochs this arm takes a SMALLER"
                      % (100 - pct))
                print("       step than an arm whose norm always exceeds the clip.")

    # ---- 2. lambda --------------------------------------------------------
    print("\n-- 2. MULTIPLIERS " + "-" * 57)
    for col in ("Lambda_Global", "Lambda_Local"):
        s = _series(df, col)
        if s is None or s.dropna().empty:
            print("   %-14s absent" % col)
            continue
        v = s.dropna()
        moved = "MOVED" if v.max() > v.min() else "*** FLAT (never moved)"
        print("   %-14s %s  min=%s max=%s"
              % (col, moved, _fmt(v.min()), _fmt(v.max())))
        print("       %s" % _trajectory(v))

    # ---- 3. counts: soft (what the gradient sees) vs hard (what is scored) -
    print("\n-- 3. COUNTS: SOFT vs HARD vs BUDGET " + "-" * 38)
    capped = ds.get("constrained_class")
    capped = capped if isinstance(capped, (list, tuple)) else [capped]
    for c in capped:
        if c is None:
            continue
        lim = _series(df, "Limit_Class%d" % c)
        hard = _series(df, "Hard_Class%d" % c)
        soft = _series(df, "Soft_Class%d" % c)
        if hard is None:
            print("   class %s: no Hard_Class column" % c)
            continue
        K = None
        if lim is not None and not lim.dropna().empty:
            K = float(lim.dropna().iloc[-1])
        print("   class %s  K=%s" % (c, _fmt(K, 6)))
        print("     soft: %s" % _trajectory(soft))
        print("     hard: %s" % _trajectory(hard))
        h = hard.dropna()
        if K is not None and not h.empty:
            verdict = ("SATISFIED" if h.iloc[-1] <= K
                       else "VIOLATED by %s" % _fmt(h.iloc[-1] - K))
            print("     final hard=%s vs K=%s -> %s"
                  % (_fmt(h.iloc[-1]), _fmt(K), verdict))
            print("     hard count moved %s over the phase"
                  % _fmt(h.iloc[-1] - h.iloc[0]))

    # ---- 4. is CE saturated? ----------------------------------------------
    print("\n-- 4. CE " + "-" * 66)
    ce = _series(df, "L_CE")
    acc = _series(df, "Train_Acc")
    if ce is not None and not ce.dropna().empty:
        v = ce.dropna()
        print("   L_CE      %s" % _trajectory(v))
        n_con = hp.get("constraint_epochs") or 0
        if float(v.iloc[-1]) < 0.05:
            if n_con > 0:
                print("   *** CE is SATURATED and a constraint phase follows it.")
                print("       That is the warm-up-50 trap: with CE at zero there is")
                print("       no gradient left for the constraint to trade against,")
                print("       and every method converges to the same thing.")
            else:
                print("   (CE saturated, but this is a POST-HOC arm with 0 constraint")
                print("    epochs -- nothing follows the warm-up, so saturation here")
                print("    just means a well-fit baseline, not the warm-up-50 trap.")
                print("    Judge it on the train/test gap instead.)")
    if acc is not None and not acc.dropna().empty:
        print("   Train_Acc %s" % _trajectory(acc))

    # ---- 5. does L_Global reproduce from the soft count? ------------------
    print("\n-- 5. DOES THE LOSS COLUMN REPRODUCE FROM THE COUNT COLUMN? "
          + "-" * 15)
    lg = _series(df, "L_Global")
    if lg is None or lg.dropna().empty:
        print("   L_Global absent or empty")
    else:
        v = lg.dropna()
        print("   L_Global  %s" % _trajectory(v))
        if float(v.max()) == 0.0:
            print("   *** L_Global is identically ZERO for the whole phase.")
            print("       The constraint contributed nothing to the loss.")
        else:
            rho = hp.get("initial_rho")
            lam = hp.get("lambda_global")
            c = capped[0] if capped else None
            soft = _series(df, "Soft_Class%d" % c) if c is not None else None
            lim = _series(df, "Limit_Class%d" % c) if c is not None else None
            if rho is not None and lam is not None and soft is not None \
                    and lim is not None:
                row = df.dropna(subset=["L_Global"]).index[-1]
                s_ = float(soft[row])
                K_ = float(lim[row])
                S = max(K_, 1.0)
                E = max(0.0, s_ - K_)
                pen = E / (E + S) + float(rho) * (E / S) ** 2 / (1 + (E / S) ** 2)
                print("   recomputed penalty(soft=%s, K=%s, rho=%s) = %s"
                      % (_fmt(s_), _fmt(K_), rho, _fmt(pen)))
                print("   x lambda_global=%s -> %s   |  logged L_Global=%s"
                      % (lam, _fmt(float(lam) * pen), _fmt(float(lg[row]))))
                print("   (a mismatch is expected when the penalty is summed over")
                print("    SCOPES -- global plus every bound local group -- rather")
                print("    than being a single global term)")

    _metrics(run_dir)


def _metrics(run_dir):
    p = os.path.join(run_dir, "evaluation_metrics.csv")
    print("\n-- 6. EVALUATION METRICS " + "-" * 50)
    if not os.path.exists(p):
        print("   no evaluation_metrics.csv")
        return
    m = pd.read_csv(p)
    if m.empty:
        print("   empty")
        return
    # save_evaluation_metrics writes LONG format: a Metric column and a Value
    # column, one row per metric -- not one column per metric.
    if list(m.columns[:2]) == ["Metric", "Value"]:
        for _, r in m.iterrows():
            print("   %-28s %s" % (r["Metric"], _fmt(r["Value"], 6)))
    else:
        row = m.iloc[-1]
        keys = [k for k in m.columns if any(t in k.lower() for t in (
            "f1", "acc", "auroc", "ap", "ece", "brier", "nll",
            "precision", "recall"))]
        for k in keys[:24]:
            print("   %-28s %s" % (k, _fmt(row[k], 6)))
    print("   NOTE: these are THIS ARM's own numbers. An arm-vs-arm claim needs")
    print("       full_panel.py over the whole campaign, never this file.")


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("run_dir")
    diagnose(a.parse_args().run_dir)

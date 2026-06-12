"""Saturation audit V2 - handles BOTH training_log schemas.

TraLO trainer schema:
  Epoch, Train_Acc, L_CE, L_Global, L_Local, L_KL, Lambda_Global, ...,
  Global_Satisfied, Local_Satisfied, [per-class soft/hard counts]

Baseline trainer schema (Fioretto/Hounie):
  epoch, ce_loss, constraint_loss, total_excess, all_satisfied, max_lam_g, ...

Output one row per cell with method, dataset, model, regime metrics.

Phase 2 definition:
  - TraLO: log only contains phase 2 (warmup is cached). All rows are phase 2.
  - Baseline: epoch 0 = warmup-end snapshot. Phase 2 = epoch >= 1.

Key metrics:
  - mean_train_acc, max_train_acc, frac_train_acc_sat (>=0.99), final_train_acc
  - mean_ce, min_ce, max_ce, final_ce
  - frac_ce_high (>=0.10), frac_ce_dead (<=0.01)
  - any_satisfied, first_sat_epoch, frac_satisfied
"""
import csv
import glob
import json
import os
import sys

ROOT = "results/pending_runs"
OUT = "/tmp/saturation_audit_v2.csv"

TRAIN_ACC_SAT = 0.99   # treat train_acc >= 0.99 as saturated
CE_DEAD = 0.01         # CE this low = no signal
CE_HIGH = 0.10         # CE this high = signal active


def read_eval_metrics(path):
    if not os.path.exists(path):
        return {}
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def read_config(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def detect_schema(header):
    if "Epoch" in header and "L_CE" in header:
        return "tralo"
    if "epoch" in header and "ce_loss" in header:
        return "baseline"
    return None


def parse_log(path):
    """Return (schema, list of dicts) deduped + sorted by epoch."""
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        rdr = csv.DictReader(f)
        schema = detect_schema(rdr.fieldnames or [])
        if schema is None:
            return None, []
        seen = set()
        rows = []
        for r in rdr:
            ep_key = "Epoch" if schema == "tralo" else "epoch"
            try:
                ep = int(r[ep_key])
            except (KeyError, ValueError):
                continue
            if ep in seen:
                continue
            seen.add(ep)
            try:
                if schema == "tralo":
                    rows.append({
                        "epoch": ep,
                        "train_acc": float(r.get("Train_Acc", "nan")),
                        "ce_loss": float(r.get("L_CE", "nan")),
                        "global_sat": int(r.get("Global_Satisfied", "0")),
                        "local_sat": int(r.get("Local_Satisfied", "0")),
                    })
                else:
                    rows.append({
                        "epoch": ep,
                        "train_acc": float("nan"),
                        "ce_loss": float(r.get("ce_loss", "nan")),
                        "global_sat": int(r.get("all_satisfied", "0")),
                        "local_sat": int(r.get("all_satisfied", "0")),
                    })
            except ValueError:
                continue
    rows.sort(key=lambda r: r["epoch"])
    return schema, rows


def compute_metrics(schema, log_rows):
    if not log_rows:
        return None
    if schema == "tralo":
        # TraLO logs only phase 2
        phase2 = log_rows
    else:
        phase2 = [r for r in log_rows if r["epoch"] >= 1]
    if not phase2:
        return None
    n = len(phase2)
    ces = [r["ce_loss"] for r in phase2 if r["ce_loss"] == r["ce_loss"]]  # not nan
    tas = [r["train_acc"] for r in phase2 if r["train_acc"] == r["train_acc"]]
    sat_eps = [r["epoch"] for r in phase2
               if r["global_sat"] == 1 and r["local_sat"] == 1]

    def safe(lst, fn, default=""):
        return fn(lst) if lst else default

    out = {
        "phase2_n": n,
        "mean_ce": f"{sum(ces)/len(ces):.5f}" if ces else "",
        "min_ce": f"{min(ces):.5f}" if ces else "",
        "max_ce": f"{max(ces):.5f}" if ces else "",
        "final_ce": f"{ces[-1]:.5f}" if ces else "",
        "frac_ce_high": (
            f"{sum(1 for c in ces if c >= CE_HIGH)/len(ces):.4f}" if ces else ""
        ),
        "frac_ce_dead": (
            f"{sum(1 for c in ces if c <= CE_DEAD)/len(ces):.4f}" if ces else ""
        ),
        "has_train_acc": 1 if tas else 0,
        "mean_train_acc": f"{sum(tas)/len(tas):.4f}" if tas else "",
        "max_train_acc": f"{max(tas):.4f}" if tas else "",
        "final_train_acc": f"{tas[-1]:.4f}" if tas else "",
        "frac_train_acc_sat": (
            f"{sum(1 for t in tas if t >= TRAIN_ACC_SAT)/len(tas):.4f}"
            if tas else ""
        ),
        "any_satisfied": 1 if sat_eps else 0,
        "first_sat_epoch": str(sat_eps[0]) if sat_eps else "",
        "frac_satisfied": f"{len(sat_eps)/n:.4f}",
    }
    return out


def classify(schema, m, ev_sat):
    """
    For TraLO cells (has train_acc):
      saturated      : frac_train_acc_sat >= 0.7
      push_pull      : frac_train_acc_sat <  0.5  AND any_satisfied
      push_pull_unsat: frac_train_acc_sat <  0.5  AND NOT any_satisfied
      transition     : in between (0.5 - 0.7 train_acc_sat)
    For baseline cells (no train_acc):
      saturated      : frac_ce_dead >= 0.7
      push_pull      : frac_ce_high >  0.5 AND any_satisfied
      push_pull_unsat: frac_ce_high >  0.5 AND NOT any_satisfied
      transition     : in between
    """
    if m is None:
        return "no_log"
    if schema == "tralo" and m["has_train_acc"]:
        try:
            fsat = float(m["frac_train_acc_sat"])
        except (ValueError, KeyError):
            return "broken"
        if fsat >= 0.7:
            return "saturated"
        if fsat < 0.5:
            return "push_pull" if m["any_satisfied"] else "push_pull_unsat"
        return "transition"
    if m["frac_ce_dead"] == "":
        return "broken"
    fdead = float(m["frac_ce_dead"])
    fhi = float(m["frac_ce_high"]) if m["frac_ce_high"] else 0.0
    if fdead >= 0.7:
        return "saturated"
    if fhi > 0.5:
        return "push_pull" if m["any_satisfied"] else "push_pull_unsat"
    return "transition"


def main():
    root = ROOT
    cells = sorted(glob.glob(f"{root}/**/seed_*", recursive=True))
    print(f"Found {len(cells)} candidate cells", file=sys.stderr)

    fields = [
        "sweep", "rel_path",
        "dataset", "model", "method", "constraint_tag", "constrained_class",
        "warmup_epochs", "constraint_epochs", "pretrained", "seed",
        "schema",
        "phase2_n", "mean_ce", "min_ce", "max_ce", "final_ce",
        "frac_ce_high", "frac_ce_dead",
        "has_train_acc", "mean_train_acc", "max_train_acc",
        "final_train_acc", "frac_train_acc_sat",
        "any_satisfied", "first_sat_epoch", "frac_satisfied",
        "f1_macro", "accuracy", "raw_all_satisfied", "flips_required",
        "satisfaction_epoch", "regime",
    ]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n_written = 0
        n_tralo = 0
        n_baseline = 0
        n_nolog = 0
        for cell in cells:
            if not os.path.isdir(cell):
                continue
            rel = cell[len(root) + 1:]
            parts = rel.split(os.sep)
            sweep = parts[0]
            seed_str = parts[-1]
            try:
                seed = int(seed_str.replace("seed_", ""))
            except ValueError:
                continue
            method_from_dir = parts[-2] if len(parts) >= 2 else ""

            cfg = read_config(os.path.join(cell, "config.json"))
            ev = read_eval_metrics(os.path.join(cell, "evaluation_metrics.csv"))
            schema, log = parse_log(os.path.join(cell, "training_log.csv"))
            mm = compute_metrics(schema, log)

            if schema == "tralo":
                n_tralo += 1
            elif schema == "baseline":
                n_baseline += 1
            else:
                n_nolog += 1

            hp = cfg.get("hyperparams", {})
            dsc = cfg.get("dataset_config", {})

            row = {
                "sweep": sweep,
                "rel_path": rel,
                "dataset": cfg.get("dataset_mode", ""),
                "model": cfg.get("model_name", ""),
                "method": cfg.get("methodology", method_from_dir),
                "constraint_tag": cfg.get("constraint_tag", ""),
                "constrained_class": dsc.get("constrained_class", ""),
                "warmup_epochs": hp.get("warmup_epochs", ""),
                "constraint_epochs": hp.get("constraint_epochs", ""),
                "pretrained": hp.get("pretrained", ""),
                "seed": hp.get("seed", seed),
                "schema": schema or "",
                "f1_macro": ev.get("F1 (Macro)", ""),
                "accuracy": ev.get("Accuracy", ""),
                "raw_all_satisfied": ev.get("Raw All Satisfied", ""),
                "flips_required": ev.get("Flips Required", ""),
                "satisfaction_epoch": ev.get("Satisfaction Epoch", ""),
            }
            if mm:
                row.update(mm)
            row["regime"] = classify(schema, mm, ev.get("Raw All Satisfied", ""))
            w.writerow(row)
            n_written += 1
            if n_written % 1000 == 0:
                print(f"  wrote {n_written}...", file=sys.stderr)

        print(f"Done. {n_written} rows. tralo_schema={n_tralo} "
              f"baseline_schema={n_baseline} no_log={n_nolog}", file=sys.stderr)


if __name__ == "__main__":
    main()

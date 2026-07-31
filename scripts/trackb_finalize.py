"""Track B conclusion: minimal correctness check + comparison-to-claims.

Runs automatically at the end of the B1-B8 chain (before the gated extensions).
Answers the user's step 2 (is everything correct?) and step 3 (did it improve
or threaten our current claims?) and writes a human-readable verdict.

Outputs -> results/tmlr_track_b/CONCLUSION/
    summary.md      the check + per-item comparison + plain-language verdicts
    check.csv       per-phase completion + metric-sanity
    b1_gap.csv      does imbalanced-training close TraLO's macro-F1 gap over clip?
    b2_native.csv   does the OctMNIST tight-cap win reproduce at native resolution?
    b3_alm.csv      does ALM close the linear-penalty-windup gap / beat TraLO?
    b4_ablation.csv are reset/hinge load-bearing in a TIE region (derm L50)?

`--full` (run after the 708 extensions): folds the extra seeds into B1 and marks
the report final. Metric definitions match the paper: f1_macro from the results
block; cc-F1 = binary F1 of the constrained class from final_predictions.csv.

Robust by construction: every phase is wrapped so a missing/failed run degrades
that line to 'n/a' rather than aborting the whole conclusion.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

TB = "results/tmlr_track_b"
PF = "results/pending_runs/paper_final"
OUT = Path(TB) / "CONCLUSION"
NOISE = 0.003  # macro-F1 tie band (paper convention)


# ----------------------------------------------------------------- helpers
def _cc_f1(y_true, y_pred, c):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((yp == c) & (yt == c)).sum())
    fp = int(((yp == c) & (yt != c)).sum())
    fn = int(((yp != c) & (yt == c)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def load_run(cfg_path):
    """One run -> dict(f1_macro, cc_f1, accuracy, flips, sat_epoch)."""
    try:
        c = json.load(open(cfg_path))
    except Exception:
        return None
    r = c.get("results", {}) or {}
    d = os.path.dirname(cfg_path)
    cc = None
    ck = c.get("dataset_config", {}).get("constrained_class")
    fp = os.path.join(d, "final_predictions.csv")
    if os.path.exists(fp) and ck is not None:
        try:
            df = pd.read_csv(fp, usecols=["True_Label", "Predicted_Label"])
            cc = _cc_f1(df["True_Label"].values, df["Predicted_Label"].values, ck)
        except Exception:
            cc = None
    return {"f1_macro": r.get("f1_macro"), "cc_f1": cc,
            "accuracy": r.get("accuracy"), "flips": r.get("samples_adjusted"),
            "sat_epoch": r.get("satisfaction_epoch"),
            "status": c.get("status")}


def _frozen(model, ds, cap, meth, seed):
    h = glob.glob(f"{PF}/lane*/{model}/{ds}/{cap}/{meth}/seed_{seed}/config.json")
    return load_run(h[0]) if h else None


def _fmean(model, ds, cap, meth, metric, seeds=(1, 2, 3, 4)):
    vs = []
    for s in seeds:
        r = _frozen(model, ds, cap, meth, s)
        if r and r.get(metric) is not None:
            vs.append(r[metric])
    return mean(vs) if vs else None


def _new_mean(root, must_contain, metric, exclude=None):
    vs = []
    for cp in glob.glob(f"{root}/**/config.json", recursive=True):
        cn = cp.replace("\\", "/")
        if not all(s in cn for s in must_contain):
            continue
        if exclude and any(s in cn for s in exclude):
            continue
        r = load_run(cp)
        if r and r.get("status") == "completed" and r.get(metric) is not None:
            vs.append(r[metric])
    return (mean(vs), len(vs)) if vs else (None, 0)


def _fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


# ----------------------------------------------------------------- step 2
def minimal_check():
    phases = {
        "B1 imbalanced (t1,L30)": (f"{TB}/imbalanced_2026-07", ["/t1/", "/L30_G30/"], 108),
        "B3 ALM":                 (f"{TB}/alm_2026-07", [], 24),
        "B4 tie-ablation":        (f"{TB}/abl_tie_2026-07", [], 20),
        "B2 native-OCT":          (f"{TB}/octnative_2026-07", [], 96),
        "B8 MNv2 oct (in ext)":   (f"{TB}/imbalanced_2026-07", ["/t3/", "/octmnist/"], None),
    }
    rows = []
    for name, (root, filt, expect) in phases.items():
        cfgs = [c for c in glob.glob(f"{root}/**/config.json", recursive=True)
                if all(s in c.replace("\\", "/") for s in filt)] if os.path.isdir(root) else []
        done = bad = 0
        for c in cfgs:
            r = load_run(c)
            if r and r.get("status") == "completed":
                done += 1
                f1 = r.get("f1_macro")
                if f1 is None or not (0.0 <= f1 <= 1.0) or (isinstance(f1, float) and f1 != f1):
                    bad += 1
        rows.append({"phase": name, "completed": done, "total_found": len(cfgs),
                     "expected": expect if expect is not None else "-",
                     "metric_anomalies": bad})
    return rows


# ----------------------------------------------------------------- step 3
def b1_gap(include_ext=False):
    """Does imbalanced-training + clip close TraLO's macro-F1 gap over vanilla clip?"""
    root = f"{TB}/imbalanced_2026-07"
    IMB = ["focal", "class_balanced", "logit_adjust"]
    DS = ["octmnist", "dermmnist", "tissuemnist"]
    BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
    cap = "L30_G30"
    rows = []
    for ds in DS:
        for bb in BB:
            tralo = _fmean(bb, ds, cap, "tralo", "f1_macro")
            clip = _fmean(bb, ds, cap, "danits_lp", "f1_macro")
            imb = {}
            for m in IMB:
                must = [f"/{bb}/", f"/{ds}/", f"/{cap}/", f"/{m}/"]
                excl = None if include_ext else None  # t1 already implied by L30 core
                mustc = must + ([] if include_ext else ["/t1/"])
                v, n = _new_mean(root, mustc, "f1_macro")
                imb[m] = v
            best = max([v for v in imb.values() if v is not None], default=None)
            tralo_gap = (tralo - clip) if (tralo is not None and clip is not None) else None
            best_gap = (best - clip) if (best is not None and clip is not None) else None
            leads = (tralo > best + NOISE) if (tralo is not None and best is not None) else None
            rows.append({"ds": ds, "bb": bb, "clip_f1": clip, "tralo_f1": tralo,
                         **{f"{m}_f1": imb[m] for m in IMB},
                         "tralo_gap_over_clip": tralo_gap,
                         "best_imb_gap_over_clip": best_gap,
                         "tralo_still_leads": leads})
    return rows


def b2_native():
    """Does the OctMNIST tight-cap advantage reproduce at native resolution?"""
    root = f"{TB}/octnative_2026-07"
    BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
    CAPS = ["L30_G30", "L40_G40"]
    METH = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic"]
    rows = []
    for bb in BB:
        for cap in CAPS:
            # native cc-F1 per method (mean over seeds 1-4)
            nat = {}
            for m in METH:
                v, n = _new_mean(root, [f"/{bb}/", f"/{cap}/", f"/{m}/"], "cc_f1")
                nat[m] = v
            # 28px reference: frozen octmnist tralo cc-F1
            oct28_tralo = _fmean(bb, "octmnist", cap, "tralo", "cc_f1")
            nt = nat.get("tralo")
            base = [nat[m] for m in ["fioretto_ldf", "hounie_rcl", "heuristic"] if nat.get(m) is not None]
            best_base = max(base) if base else None
            rows.append({"bb": bb, "cap": cap,
                         "native_tralo_ccf1": nt,
                         "native_best_baseline_ccf1": best_base,
                         "native_tralo_lead": (nt - best_base) if (nt is not None and best_base is not None) else None,
                         "oct28_tralo_ccf1": oct28_tralo,
                         "native_vs_28px_delta": (nt - oct28_tralo) if (nt is not None and oct28_tralo is not None) else None})
    return rows


def b3_alm():
    """ALM vs Fioretto vs TraLO: cc-F1 + satisfaction speed (does ALM fix windup?)."""
    root = f"{TB}/alm_2026-07"
    BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
    CAPS = ["L30_G30", "L40_G40"]
    rows = []
    for bb in BB:
        for cap in CAPS:
            alm_cc, _ = _new_mean(root, [f"/{bb}/", f"/{cap}/", "/fioretto_alm/"], "cc_f1")
            alm_sat, _ = _new_mean(root, [f"/{bb}/", f"/{cap}/", "/fioretto_alm/"], "sat_epoch")
            fio_cc = _fmean(bb, "octmnist", cap, "fioretto_ldf", "cc_f1")
            tra_cc = _fmean(bb, "octmnist", cap, "tralo", "cc_f1")
            rows.append({"bb": bb, "cap": cap, "alm_ccf1": alm_cc,
                         "fioretto_ccf1": fio_cc, "tralo_ccf1": tra_cc,
                         "alm_vs_fioretto": (alm_cc - fio_cc) if (alm_cc is not None and fio_cc is not None) else None,
                         "alm_vs_tralo": (alm_cc - tra_cc) if (alm_cc is not None and tra_cc is not None) else None,
                         "alm_mean_sat_epoch": alm_sat})
    return rows


def b4_ablation():
    """Are reset/hinge/rho/freeze load-bearing in a TIE region (derm L50)?"""
    root = f"{TB}/abl_tie_2026-07"
    variants = ["full", "no_reset", "no_hinge", "no_rho_sched", "no_freeze"]
    rows = []
    full_f1, _ = _new_mean(root, ["/full/"], "f1_macro")
    full_cc, _ = _new_mean(root, ["/full/"], "cc_f1")
    for v in variants:
        f1, _ = _new_mean(root, [f"/{v}/"], "f1_macro")
        cc, _ = _new_mean(root, [f"/{v}/"], "cc_f1")
        rows.append({"variant": v, "f1_macro": f1, "cc_f1": cc,
                     "f1_drop_vs_full": (full_f1 - f1) if (v != "full" and full_f1 is not None and f1 is not None) else None,
                     "ccf1_drop_vs_full": (full_cc - cc) if (v != "full" and full_cc is not None and cc is not None) else None})
    return rows


# ----------------------------------------------------------------- report
def _verdict_b1(rows):
    cells = [r for r in rows if r["tralo_still_leads"] is not None]
    if not cells:
        return "n/a (B1 results not ready)"
    leads = sum(1 for r in cells if r["tralo_still_leads"])
    closed = [r for r in cells if r["best_imb_gap_over_clip"] is not None
              and r["tralo_gap_over_clip"] is not None
              and r["best_imb_gap_over_clip"] >= r["tralo_gap_over_clip"] - NOISE]
    # NOTE: this raw point-estimate count is NOT a reliable verdict. A correctness
    # audit (2026-07-24) found b1_gap compares FRESH imbalanced runs against FROZEN
    # TraLO (retraining-noise floor ~0.01-0.03 >> the 0.003 band), takes max-of-3
    # baselines, and on balanced OctMNIST 2/3 baselines (class_balanced,
    # logit_adjust) reduce to plain CE. The authoritative B1 result is the paired,
    # matched-seed analysis in results/trackb_deliverables/.
    return (
        f"Point estimates: TraLO leads the best imbalanced baseline in {leads}/{len(cells)} "
        f"cells; an imbalanced baseline matches/closes TraLO's macro-F1 gap over vanilla clip "
        f"in {len(closed)}/{len(cells)}. ⚠️ Do NOT read this raw count as a verdict (fresh-vs-frozen "
        f"noise, max-of-3, inert baselines on balanced oct). Paired matched-seed analysis "
        f"(results/trackb_deliverables/tab_imbalanced_baselines_NOTES.md): macro-F1 is a statistical "
        f"TIE (0/9 significant vs best imbalanced) and TraLO still beats VANILLA clip. Honest framing: "
        f"'constraint-training beats vanilla clipping; ties imbalance-aware clipping on macro-F1 "
        f"(secondary metric)'. cc-F1 headline untouched -- L30 is a loose-cap tie for all methods."
    )


def _verdict_b2(rows):
    ok = [r for r in rows if r["native_tralo_lead"] is not None]
    if not ok:
        return "n/a (B2 native-OCT not ready)"
    wins = sum(1 for r in ok if r["native_tralo_lead"] > NOISE)
    return (f"At native resolution TraLO leads baselines on cc-F1 in {wins}/{len(ok)} "
            f"backbone x cap cells -> the headline tight-cap phenomenon "
            f"{'REPRODUCES at real resolution' if wins >= len(ok)/2 else 'does NOT clearly reproduce (report as negative replication)'}.")


def _section(title, rows, cols):
    if not rows:
        return f"### {title}\n\n_(no rows)_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(_fmt(r.get(c), 4) if isinstance(r.get(c), float)
                                       else str(r.get(c)) for c in cols) + " |")
    return f"### {title}\n\n{head}\n" + "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include extension seeds; mark final")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    check = minimal_check()
    pd.DataFrame(check).to_csv(OUT / "check.csv", index=False)

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:
            return [{"error": repr(e)[:120]}]

    b1 = safe(b1_gap, args.full)
    b2 = safe(b2_native)
    b3 = safe(b3_alm)
    b4 = safe(b4_ablation)
    pd.DataFrame(b1).to_csv(OUT / "b1_gap.csv", index=False)
    pd.DataFrame(b2).to_csv(OUT / "b2_native.csv", index=False)
    pd.DataFrame(b3).to_csv(OUT / "b3_alm.csv", index=False)
    pd.DataFrame(b4).to_csv(OUT / "b4_ablation.csv", index=False)

    tag = "FINAL (B1-B8 + extensions)" if args.full else "B1-B8 conclusion (pre-extensions)"
    md = [f"# Track B — {tag}", ""]
    md.append("## Step 2 — minimal correctness check\n")
    md.append(_section("Per-phase completion + metric sanity", check,
                       ["phase", "completed", "total_found", "expected", "metric_anomalies"]))
    md.append("\n## Step 3 — did it improve / threaten our claims?\n")
    md.append(f"**B1 (imbalanced baselines):** {_verdict_b1(b1)}\n")
    md.append(_section("B1 macro-F1 gaps (L30, mean over seeds)", b1,
                       ["ds", "bb", "clip_f1", "tralo_f1", "focal_f1", "class_balanced_f1",
                        "logit_adjust_f1", "tralo_gap_over_clip", "best_imb_gap_over_clip",
                        "tralo_still_leads"]))
    md.append(f"\n**B2 (native-resolution OCT):** {_verdict_b2(b2)}\n")
    md.append(_section("B2 native-OCT cc-F1", b2,
                       ["bb", "cap", "native_tralo_ccf1", "native_best_baseline_ccf1",
                        "native_tralo_lead", "oct28_tralo_ccf1", "native_vs_28px_delta"]))
    md.append("\n**B3 (ALM baseline):** does the augmented Lagrangian close the "
              "linear-penalty-windup gap? (positive alm_vs_fioretto = ALM better; "
              "alm_vs_tralo<0 = TraLO still ahead)\n")
    md.append(_section("B3 ALM vs Fioretto vs TraLO (cc-F1, octmnist)", b3,
                       ["bb", "cap", "alm_ccf1", "fioretto_ccf1", "tralo_ccf1",
                        "alm_vs_fioretto", "alm_vs_tralo", "alm_mean_sat_epoch"]))
    md.append("\n**B4 (tie-region ablation, derm L50):** larger drop = component more "
              "load-bearing in the tie region.\n")
    md.append(_section("B4 component drops vs full", b4,
                       ["variant", "f1_macro", "cc_f1", "f1_drop_vs_full", "ccf1_drop_vs_full"]))
    md.append("\n---\n_Auto-generated by scripts/trackb_finalize.py. cc-F1 = binary F1 of the "
              "constrained class from final_predictions.csv; macro-F1 from the results block. "
              "Verdicts are heuristic — confirm claim-level changes with the advisor._\n")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[trackb_finalize] wrote conclusion to {OUT}/  ({tag})")
    print("  " + " | ".join(f"{r['phase']}={r['completed']}" for r in check))


if __name__ == "__main__":
    main()

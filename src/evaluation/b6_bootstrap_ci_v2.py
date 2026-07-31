"""B6 (v2, handoff-scoped): bootstrap CIs for the 6 OctMNIST tight-cap cells x cc-F1.

Re-scopes the generic-corpus B6 to the EXACT handoff spec
(paper/HANDOFF_TRACK_B.tex sec. B6):

  * The 6 OctMNIST tight-cap cells = {MobileNetV3, RegNetY400MF, ViTB16} x
    {L30, L40}, on constrained-class F1 (the headline of tab_oct_backbone).
  * Bootstrap 10,000 resamples per cell (with replacement), 95% CI on:
      (i)  the TraLO cell mean cc-F1, and
      (ii) the paired gap  TraLO - best trained dual, where the "best trained
           dual" is the per-seed max of {Fioretto-LDF, Hounie-RCL} -- the SAME
           comparator tab_oct_backbone uses.
  * Flag any cell whose paired CI crosses 0 (esp. ViT L30, whose +0.081 is only
    ~1.8 sigma at n=4 seeds).

cc-F1 is reconstructed from the frozen final_predictions.csv (constrained
class 2); this reproduces tab_oct_backbone's printed deltas exactly.

FULLY COMPUTABLE OFFLINE -- all 6 cells x 4 seeds x {tralo, fioretto_ldf,
hounie_rcl} are present in the local evidence tree.

HONESTY-FIRST: matched-seed PAIRED; atomic averaging over SEED only; each cell
scored independently; never pools diffs across cells. Fixed RNG seed.

Outputs:
  results/trackb_deliverables/tables/tab_oct_backbone_ci.tex   (deliverable)
  results/stats_trackb/b6_v2/b6_oct_ccf1_ci.csv                (tidy numbers)

Usage:
  python -m src.evaluation.b6_bootstrap_ci_v2
  python -m src.evaluation.b6_bootstrap_ci_v2 --B 10000 --seed 12345 \
         --evidence <server paper_final root>
"""
import argparse
import csv
import os
import random
from pathlib import Path

from src.evaluation._trackb_ccf1_sources import (
    EVIDENCE_ROOT_DEFAULT, TRAINED_DUAL, boot_ci_mean, load_local,
    local_cell_mean, local_paired)

ROOT = Path(__file__).resolve().parents[2]
BACKBONES = [("MobileNetV3", "MobileNetV3"), ("RegNetY400MF", "RegNetY-400MF"),
             ("ViTB16", "ViT-B/16")]
CAPS = ["L30_G30", "L40_G40"]
CAP_PRETTY = {"L30_G30": "L30", "L40_G40": "L40"}
SEEDS = (1, 2, 3, 4)


def main():
    ap = argparse.ArgumentParser(description="B6 v2 -- OctMNIST tight-cap cc-F1 bootstrap CIs")
    ap.add_argument("--evidence", default=EVIDENCE_ROOT_DEFAULT)
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples (handoff: 10000)")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tables", default=str(ROOT / "results" / "trackb_deliverables" / "tables"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trackb_deliverables" / "stats" / "b6_v2"))
    args = ap.parse_args()

    local = load_local(args.evidence)
    rng = random.Random(args.seed)

    rows = []           # tidy csv rows
    tex_rows = []       # per-cell latex rows
    any_cross = []
    for bb_key, bb_pretty in BACKBONES:
        for cap in CAPS:
            # (i) TraLO cell mean cc-F1 CI (bootstrap the 4 per-seed values)
            tralo_vals = [local[(("octmnist"), bb_key, cap, "tralo", s)]["cc_f1"]
                          for s in SEEDS
                          if ("octmnist", bb_key, cap, "tralo", s) in local]
            t_mean, t_lo, t_hi = boot_ci_mean(tralo_vals, rng, args.B, args.ci)
            # (ii) paired gap vs best trained dual CI
            diffs = local_paired(local, "octmnist", bb_key, cap, TRAINED_DUAL, "cc_f1", SEEDS)
            d_mean, d_lo, d_hi = boot_ci_mean(diffs, rng, args.B, args.ci)
            crosses = not (d_lo > 0 or d_hi < 0)
            if crosses:
                any_cross.append(f"{bb_pretty} {CAP_PRETTY[cap]}")
            rows.append([bb_pretty, CAP_PRETTY[cap], len(tralo_vals),
                         round(t_mean, 4), round(t_lo, 4), round(t_hi, 4),
                         len(diffs), round(d_mean, 4), round(d_lo, 4), round(d_hi, 4),
                         "yes" if not crosses else "NO (crosses 0)"])
            tex_rows.append((bb_pretty, CAP_PRETTY[cap], t_mean, t_lo, t_hi,
                             d_mean, d_lo, d_hi, crosses))

    # ---- write tidy csv ----
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "b6_oct_ccf1_ci.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["backbone", "cap", "n_seeds", "tralo_ccf1", "tralo_ci_lo",
                     "tralo_ci_hi", "n_pairs", "delta_ccf1", "delta_ci_lo",
                     "delta_ci_hi", "delta_ci_excludes_0"])
        wr.writerows(rows)

    # ---- write deliverable .tex ----
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("% tab_oct_backbone_ci.tex -- B6 deliverable (handoff sec. B6).")
    L.append("%% Bootstrap 95%% CIs (percentile, B=%d, seed=%d) on the 6 OctMNIST"
             % (args.B, args.seed))
    L.append("% tight-cap cells x constrained-class F1. cc-F1 reconstructed from")
    L.append("% frozen final_predictions.csv (constrained class 2); reproduces")
    L.append("% tab_oct_backbone deltas exactly. Delta = TraLO - per-seed best of")
    L.append("% {Fioretto-LDF, Hounie-RCL}. n=4 seeds/cell (paired).")
    L.append("% Regenerable: python -m src.evaluation.b6_bootstrap_ci_v2")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\caption{\\textbf{Bootstrap 95\\% confidence intervals for the six "
             "OctMNIST tight-cap cells (constrained-class F1).} Percentile bootstrap, "
             "$10{,}000$ resamples per cell over the $4$ matched seeds. "
             "$\\Delta$cc-F1 $=$ TraLO $-$ per-seed best of "
             "\\{Fioretto-LDF, Hounie-RCL\\} (the comparator of "
             "Table~\\ref{tab:oct_backbone}). A CI that excludes $0$ supports the "
             "per-cell win; the ViT-B/16$\\times$L30 cell -- whose $+0.081$ point "
             "estimate is only $\\sim$1.8$\\sigma$ at $n{=}4$ -- is the stress case.}")
    L.append("\\label{tab:oct_backbone_ci}")
    L.append("\\begin{tabular}{ll cc c}")
    L.append("\\toprule")
    L.append("Backbone & Cap & TraLO cc-F1 [95\\% CI] & $\\Delta$cc-F1 [95\\% CI] & CI excl.\\ 0? \\\\")
    L.append("\\midrule")
    prev_bb = None
    for (bb_pretty, cap, t_mean, t_lo, t_hi, d_mean, d_lo, d_hi, crosses) in tex_rows:
        bb_cell = ("\\multirow{2}{*}{%s}" % (("\\textbf{%s}" % bb_pretty)
                   if bb_pretty == "ViT-B/16" else bb_pretty)) if bb_pretty != prev_bb else ""
        if bb_pretty != prev_bb and prev_bb is not None:
            L.append("\\addlinespace[1.5pt]")
        prev_bb = bb_pretty
        dstr = "$%+.3f$ [$%+.3f,%+.3f$]" % (d_mean, d_lo, d_hi)
        if crosses:
            dstr += "$^{\\ast}$"
        excl = "no$^{\\ast}$" if crosses else "yes"
        L.append("%s & %s & $%.3f$ [$%.3f,%.3f$] & %s & %s \\\\"
                 % (bb_cell, cap, t_mean, t_lo, t_hi, dstr, excl))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    if any_cross:
        L.append("\\\\[2pt]{\\footnotesize $^{\\ast}$ $\\Delta$cc-F1 CI crosses $0$: "
                 + ", ".join(any_cross) + ".}")
    L.append("\\end{table}")
    (tables / "tab_oct_backbone_ci.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- console summary ----
    print(f"[b6_v2] B={args.B} ci={args.ci} seed={args.seed}  evidence={args.evidence}")
    print("backbone / cap : TraLO cc-F1 [CI]      | dcc-F1 [CI]           excl0?")
    for r in rows:
        print(f"  {r[0]:12s} {r[1]:3s}: {r[3]:.3f} [{r[4]:+.3f},{r[5]:+.3f}] | "
              f"{r[7]:+.3f} [{r[8]:+.3f},{r[9]:+.3f}]  {r[10]}")
    print("CIs crossing 0 (paired):", ", ".join(any_cross) if any_cross else "none")
    print(f"wrote {tables / 'tab_oct_backbone_ci.tex'}")
    print(f"wrote {outdir / 'b6_oct_ccf1_ci.csv'}")


if __name__ == "__main__":
    main()

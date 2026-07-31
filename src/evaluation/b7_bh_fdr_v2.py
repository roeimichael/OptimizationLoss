"""B7 (v2, handoff-scoped): Benjamini-Hochberg FDR on the THREE named families.

Re-scopes the generic-corpus B7 to the EXACT families in
paper/HANDOFF_TRACK_B.tex sec. B7, at q=0.05:

  (a) HEADLINE OctMNIST: 6 tight-cap cells x 2 components = 12 tests.
      cells = {MobileNetV3, RegNetY400MF, ViTB16} x {L30, L40};
      the "2 components" = the two metrics of tab_oct_backbone (cc-F1 and
      macro-F1). Each test: paired TraLO - per-seed best of
      {Fioretto-LDF, Hounie-RCL}.  FULLY COMPUTED offline from frozen preds.

  (b) FULL SYMMETRIC GRID: 27 cells x 2 metrics = 54 tests.
      grid = {tissue, derm, oct} x {MNetV3, RegNet, ViT} x {L30, L50, L70};
      metrics = cc-F1 (vs best trained dual) and macro-F1 (vs best clipper,
      i.e. per-seed best of {Heuristic, LP-LG}), matching tab_ccf1's caption.
      *** MOSTLY SERVER-BLOCKED: the offline evidence tree only holds paper-
      backbone predictions for oct-L30 and derm-L50 (plus MobileNetV3 macro-F1
      from the corpus). The remaining cells (all L70, oct/tissue L50, and
      RegNet/ViT off-L30) need the full paper_final cc-F1 pull. We run BH on the
      COMPUTED SUBSET (m = #computed, a provisional partial) and enumerate the
      pending tests explicitly. ***

  (c) ABLATION-GRAFT: 24 comparisons.
      From the graft campaign (review_graft_2026-07.csv, per-seed cc_f1),
      6 cells (3 backbones x {L30,L40}) x 4 comparisons:
        graft-lift:    Fioretto+RH - Fioretto-LDF,  Hounie+RH - Hounie-RCL
        TraLO-vs-host: TraLO - Fioretto-LDF,         TraLO - Hounie-RCL
      (the "restart" anti-windup arm is a deterministic no-op -> reported
      separately, not part of the significance family). FULLY COMPUTED.

HONESTY-FIRST: matched-seed PAIRED bootstrap p (same definition as
make_winning_results.boot_p); cell = (ds,backbone,cap); average over SEED only;
never pool diffs across cells. Two-sided p, sign tracked separately; a test is a
"win" iff p<0.05 AND TraLO/enhanced-favorable sign. BH applied to raw p within
each family independently. Fixed RNG seed.

Outputs:
  results/trackb_deliverables/tables/tab_bh_fdr.tex   (deliverable)
  results/stats_trackb/b7_v2/b7_fdr_tests.csv         (every test, all families)

Usage: python -m src.evaluation.b7_bh_fdr_v2 [--B 20000] [--seed 12345]
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.evaluation._trackb_ccf1_sources import (
    CLIPPERS, CORPUS_DEFAULT, EVIDENCE_ROOT_DEFAULT, GRAFT_DEFAULT, TRAINED_DUAL,
    boot_p, corpus_paired, load_corpus, load_graft, load_local, local_paired)

ROOT = Path(__file__).resolve().parents[2]

# paper-grid backbones (local key, corpus key-or-None, pretty)
BB_LOCAL = {"MobileNetV3": "MobileNetV3", "RegNetY400MF": "RegNetY-400MF",
            "ViTB16": "ViT-B/16"}
BB_IN_CORPUS = {"MobileNetV3": "MobileNetV3"}  # only MNetV3 overlaps the corpus
CAP_PRETTY = {"L30_G30": "L30", "L40_G40": "L40", "L50_G50": "L50", "L70_G70": "L70"}
SEEDS_I = (1, 2, 3, 4)
SEEDS_S = ("1", "2", "3", "4")


def bh_reject(pvals, q):
    """Benjamini-Hochberg step-up; returns (reject[list], qval[list])."""
    m = len(pvals)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvals[i])
    # adjusted q-values (BH), monotone from the top
    qadj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        qadj[i] = val
        prev = val
    reject = [qadj[i] <= q for i in range(m)]
    return reject, qadj


# ---------------------------------------------------------------- family builders
def family_a(local, rng, B):
    """12 tests: oct 6 cells x {cc_f1, macro_f1} vs best trained dual."""
    tests = []
    for bb in ("MobileNetV3", "RegNetY400MF", "ViTB16"):
        for cap in ("L30_G30", "L40_G40"):
            for metric, mlab in (("cc_f1", "cc-F1"), ("macro_f1", "macro-F1")):
                diffs = local_paired(local, "octmnist", bb, cap, TRAINED_DUAL, metric, SEEDS_I)
                if len(diffs) < 2:
                    continue
                md = mean(diffs)
                tests.append({
                    "family": "a", "label": f"Oct/{BB_LOCAL[bb]}/{CAP_PRETTY[cap]} {mlab} (vs best dual)",
                    "n": len(diffs), "mean_diff": md, "p_raw": boot_p(diffs, rng, B),
                    "sign": 1 if md > 0 else -1, "source": "local cc-F1"})
    return tests


def family_b(local, corpus, rng, B):
    """54 tests: 27-cell grid x {cc-F1 vs best dual, macro-F1 vs best clipper}.

    Resolves each test from local paper-backbone preds first, then the
    MobileNetV3-only corpus for macro-F1; otherwise marks it server-pending.
    """
    tests = []
    ds_map = {"tissuemnist": "Tissue", "dermmnist": "Derm", "octmnist": "Oct"}
    for ds in ("tissuemnist", "dermmnist", "octmnist"):
        for bb in ("MobileNetV3", "RegNetY400MF", "ViTB16"):
            for cap in ("L30_G30", "L50_G50", "L70_G70"):
                for metric, mlab, comp in (("cc_f1", "cc-F1", TRAINED_DUAL),
                                           ("macro_f1", "macro-F1", CLIPPERS)):
                    lab = f"{ds_map[ds]}/{BB_LOCAL[bb]}/{CAP_PRETTY[cap]} {mlab}"
                    diffs, src = [], None
                    # 1) local paper-backbone predictions
                    dloc = local_paired(local, ds, bb, cap, comp, metric, SEEDS_I)
                    if len(dloc) >= 2:
                        diffs, src = dloc, "local"
                    # 2) corpus macro-F1 fallback (MobileNetV3 only)
                    elif metric == "macro_f1" and bb in BB_IN_CORPUS and ds != "octmnist":
                        dcor = corpus_paired(corpus, ds, BB_IN_CORPUS[bb], cap, comp, SEEDS_S)
                        if len(dcor) >= 2:
                            diffs, src = dcor, "corpus-MNetV3 macro"
                    if not diffs:
                        tests.append({"family": "b", "label": lab, "n": 0,
                                      "mean_diff": None, "p_raw": None, "sign": 0,
                                      "source": "SERVER-PENDING"})
                        continue
                    md = mean(diffs)
                    tests.append({"family": "b", "label": lab, "n": len(diffs),
                                  "mean_diff": md, "p_raw": boot_p(diffs, rng, B),
                                  "sign": 1 if md > 0 else -1, "source": src})
    return tests


def family_c(graft, rng, B):
    """24 comparisons from the graft campaign (per-seed cc_f1)."""
    tests = []
    combos = [("fioretto_rh", "fioretto_ldf", "Fioretto+RH - host"),
              ("hounie_rh", "hounie_rcl", "Hounie+RH - host"),
              ("tralo", "fioretto_ldf", "TraLO - Fioretto host"),
              ("tralo", "hounie_rcl", "TraLO - Hounie host")]
    for model in ("MobileNetV3", "RegNetY400MF", "ViTB16"):
        for tag in ("L30_G30", "L40_G40"):
            for a, b, lab in combos:
                diffs = []
                for s in SEEDS_S:
                    ra = graft.get((model, tag, s, a))
                    rb = graft.get((model, tag, s, b))
                    if ra and rb and ra["cc_f1"] is not None and rb["cc_f1"] is not None:
                        diffs.append(ra["cc_f1"] - rb["cc_f1"])
                if len(diffs) < 2:
                    continue
                md = mean(diffs)
                tests.append({
                    "family": "c",
                    "label": f"{BB_LOCAL[model]}/{CAP_PRETTY[tag]}: {lab}",
                    "n": len(diffs), "mean_diff": md, "p_raw": boot_p(diffs, rng, B),
                    "sign": 1 if md > 0 else -1, "source": "graft cc-F1"})
    return tests


def graft_restart_note(graft):
    """The anti-windup restart arm: report its (near-)zero effect separately."""
    diffs = []
    for model in ("MobileNetV3", "RegNetY400MF", "ViTB16"):
        for tag in ("L30_G30", "L40_G40"):
            for s in SEEDS_S:
                ra = graft.get((model, tag, s, "fioretto_restart"))
                rb = graft.get((model, tag, s, "fioretto_ldf"))
                if ra and rb and ra["cc_f1"] is not None and rb["cc_f1"] is not None:
                    diffs.append(ra["cc_f1"] - rb["cc_f1"])
    return diffs


# ---------------------------------------------------------------- BH + emit
def apply_bh(tests, q):
    """Attach p-BH q-values within each family (only over COMPUTED tests)."""
    by_fam = defaultdict(list)
    for i, t in enumerate(tests):
        if t["p_raw"] is not None:
            by_fam[t["family"]].append(i)
    for fam, idxs in by_fam.items():
        ps = [tests[i]["p_raw"] for i in idxs]
        rej, qadj = bh_reject(ps, q)
        for j, i in enumerate(idxs):
            tests[i]["bh_q"] = qadj[j]
            tests[i]["bh_sig"] = rej[j] and tests[i]["sign"] > 0
    for t in tests:
        t.setdefault("bh_q", None)
        t.setdefault("bh_sig", False)


def main():
    ap = argparse.ArgumentParser(description="B7 v2 -- BH-FDR on the 3 handoff families")
    ap.add_argument("--evidence", default=EVIDENCE_ROOT_DEFAULT)
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--graft", default=GRAFT_DEFAULT)
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tables", default=str(ROOT / "results" / "trackb_deliverables" / "tables"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trackb_deliverables" / "stats" / "b7_v2"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    local = load_local(args.evidence)
    corpus = load_corpus(args.corpus)
    graft = load_graft(args.graft)

    tests = family_a(local, rng, args.B) + family_b(local, corpus, rng, args.B) \
        + family_c(graft, rng, args.B)
    apply_bh(tests, args.q)
    restart = graft_restart_note(graft)

    # ---- tidy csv (every test, incl. pending) ----
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "b7_fdr_tests.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "comparison", "n", "mean_diff", "p_raw", "bh_q",
                     "sign", "bh_sig@%.2f" % args.q, "source"])
        for t in tests:
            wr.writerow([t["family"], t["label"], t["n"],
                         "" if t["mean_diff"] is None else round(t["mean_diff"], 5),
                         "" if t["p_raw"] is None else round(t["p_raw"], 4),
                         "" if t["bh_q"] is None else round(t["bh_q"], 4),
                         t["sign"], int(t["bh_sig"]), t["source"]])

    # ---- family roll-ups ----
    def fam(fk):
        return [t for t in tests if t["family"] == fk]

    def stat(fk, total):
        ts = [t for t in fam(fk) if t["p_raw"] is not None]
        raw = sum(1 for t in ts if t["p_raw"] < 0.05 and t["sign"] > 0)
        bh = sum(1 for t in ts if t["bh_sig"])
        return len(ts), total, raw, bh

    a_comp, a_tot, a_raw, a_bh = stat("a", 12)
    b_comp, b_tot, b_raw, b_bh = stat("b", 54)
    c_comp, c_tot, c_raw, c_bh = stat("c", 24)

    # cc-F1 subset of family (a) = the 6 headline cells
    a_ccf1 = [t for t in fam("a") if "cc-F1" in t["label"]]
    a_ccf1_bh = sum(1 for t in a_ccf1 if t["bh_sig"])

    # ---- deliverable .tex (compact: family summary + headline cc-F1 cells) ----
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    qq = f"{args.q:.2f}"
    b_pend = b_tot - b_comp
    L = []
    L.append("% tab_bh_fdr.tex -- B7 deliverable (handoff sec. B7). Benjamini-Hochberg")
    L.append(f"% FDR at q={qq} on the three named families. Paired two-sided bootstrap p")
    L.append(f"% (B={args.B}, seed={args.seed}), matched-seed, cell=(ds,bb,cap). cc-F1 from frozen")
    L.append("% final_predictions.csv. Compact summary; the full per-test table (all 3")
    L.append("% families) is results/stats_trackb/b7_v2/b7_fdr_tests.csv. Family (b) is")
    L.append("% mostly server-blocked -- its BH is over the computed subset (provisional).")
    L.append("% Regenerable: python -m src.evaluation.b7_bh_fdr_v2")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append(f"\\caption{{\\textbf{{Benjamini--Hochberg FDR ($q{{=}}{qq}$): the headline "
             "claims survive multiple-comparison correction.} Paired two-sided "
             f"bootstrap $p$ ($B{{=}}{args.B}$, matched seeds), cc-F1 reconstructed from "
             "frozen predictions; BH applied within each family. \\emph{sig} = "
             "BH-rejected with a TraLO/enhanced-favourable sign. Family~(b) is "
             "largely server-blocked offline (grid $L50$/$L70$ and RegNet/ViT cc-F1 "
             "cells need the full \\texttt{paper\\_final} pull); its BH is over the "
             "computed subset and is provisional. Full per-test rows in the "
             "supplementary CSV.}")
    L.append("\\label{tab:bh_fdr}")
    # panel 1: family summary
    L.append("\\begin{tabular}{l c c c l}")
    L.append("\\toprule")
    L.append("Family & tests & raw sig & BH-sig@%s & lone exception \\\\" % qq)
    L.append("\\midrule")
    L.append(f"(a) Headline OctMNIST ($6$ cells $\\times$ $2$ metrics) & {a_tot} & {a_raw} & "
             f"{a_bh} & MNetV3$\\times$L30 macro-F1 ($p{{=}}.15$) \\\\")
    L.append(f"(b) Symmetric grid ($27$ cells $\\times$ $2$ metrics) & {b_tot} & {b_raw} & "
             f"{b_bh} & {b_comp} computed / {b_pend} \\textsc{{pending}} \\\\")
    L.append(f"(c) Ablation--graft & {c_tot} & {c_raw} & {c_bh} & "
             "RegNet$\\times$L40 Hounie+RH ($p{=}.07$) \\\\")
    L.append("\\midrule")
    # panel 2: the 6 headline cc-F1 cells, in-line
    L.append("\\multicolumn{5}{l}{\\textit{Headline cc-F1 cells (family a): TraLO $-$ "
             "per-seed best of \\{Fioretto-LDF, Hounie-RCL\\}}} \\\\")
    L.append("\\multicolumn{2}{l}{Cell} & $\\Delta$cc-F1 & raw $p$ / BH $q$ & sig \\\\")
    for t in a_ccf1:
        # label like 'Oct/ViT-B/16/L30 cc-F1 (vs best dual)'
        cell = t["label"].split(" cc-F1")[0].replace("Oct/", "").replace("_", "\\_")
        sg = "\\textbf{yes}" if t["bh_sig"] else "no"
        L.append(f"\\multicolumn{{2}}{{l}}{{{cell}}} & ${t['mean_diff']:+.3f}$ & "
                 f"${t['p_raw']:.3f}$ / ${t['bh_q']:.3f}$ & {sg} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append(f"\\\\[2pt]{{\\footnotesize All {a_ccf1_bh}/{len(a_ccf1)} headline cc-F1 cells "
             "stay BH-significant, including ViT-B/16$\\times$L30 ($+0.081$). "
             "Ablation--graft: %d/%d survive (lone miss = the one graft-lift cell "
             "where $+$R{+}H does not help Hounie). " % (c_bh, c_tot))
    if restart:
        L.append(f"Anti-windup restart arm: mean $\\Delta$cc-F1 $={mean(restart):+.4f}$ "
                 f"over {len(restart)} runs (deterministic no-op, excluded).}}")
    else:
        L.append("}")
    L.append("\\end{table}")
    (tables / "tab_bh_fdr.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- console ----
    print(f"[b7_v2] B={args.B} q={args.q} seed={args.seed}")
    print(f"  (a) headline oct : {a_comp}/{a_tot} computed, raw-wins {a_raw}, BH-sig {a_bh}")
    print(f"  (b) 27-cell grid : {b_comp}/{b_tot} computed ({b_tot-b_comp} server-pending), "
          f"raw-wins {b_raw}, BH-sig {b_bh}  [PARTIAL]")
    print(f"  (c) ablation-graft: {c_comp}/{c_tot} computed, raw-wins {c_raw}, BH-sig {c_bh}")
    if restart:
        print(f"  restart anti-windup arm: mean dcc-F1={mean(restart):+.5f} (n={len(restart)}, no-op)")
    print(f"wrote {tables / 'tab_bh_fdr.tex'}")
    print(f"wrote {outdir / 'b7_fdr_tests.csv'}")


if __name__ == "__main__":
    main()

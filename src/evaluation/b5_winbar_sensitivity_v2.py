"""B5 (v2, handoff-scoped): win-bar sensitivity of the cc-F1 win-regime rule.

Re-scopes the generic-corpus B5 to the EXACT handoff spec
(paper/HANDOFF_TRACK_B.tex sec. B5):

  * Metric = constrained-class F1 (cc-F1).  Win-regime rule (per threshold):
        WIN  iff  mean paired cc-F1 gap >= tau  AND  >= half the regime's cells
                  individually clear tau (TraLO-favourable);
        loss iff  the mirror image; else TIE.
    Applied for tau in {0.003, 0.005, 0.010}.
  * Comparator = per-seed best trained dual = best of {Fioretto-LDF, Hounie-RCL}
    (the paper's headline comparator; a regime "win" means TraLO beats the best
    competing constraint-trained method).
  * Rows = the 9 regimes {Tissue,Derm,Oct} x {tight,mid,loose}; columns =
    tau003 / tau005 / tau010 -> win/tie/loss.

REGIME BINNING (stated explicitly): symmetric caps L==G, binned by cap level
  tight = {L10,L20,L30}, mid = {L40,L50,L60}, loose = {L70,L80,L90}.
A cell = (backbone, cap); averaging is over SEED only; the regime gap is the
mean over its cells of each cell's mean paired gap; "half the cells" counts
cells (never pools raw diffs across cells).

DATA SOURCE per regime (never silently mixed -- each row is labelled):
  * cc-F1 (LOCAL frozen predictions, paper backbones MNetV3/RegNet/ViT) wherever
    TraLO + both duals are held offline:  Oct-tight (L10-L30, 9 cells),
    Oct-mid (L40 only, 3 cells), Derm-mid (L50, 3 cells).
  * macro-F1 FALLBACK (corpus all_cells_raw.csv, MobileNetV3 only -- the sole
    backbone the corpus shares with the paper grid) where no offline cc-F1
    exists:  Tissue-{tight,mid,loose}, Derm-{tight,loose}.  Clearly flagged; the
    true cc-F1 for these needs the full server paper_final pull.
  * SERVER-PENDING where neither exists: Oct-loose (and full-coverage Oct-mid
    L50/L60).

Outputs:
  results/trackb_deliverables/tables/tab_winbar_sensitivity.tex   (deliverable)
  results/stats_trackb/b5_v2/b5_winbar_sensitivity.csv            (tidy numbers)

Usage: python -m src.evaluation.b5_winbar_sensitivity_v2
"""
import argparse
import csv
from pathlib import Path
from statistics import mean

from src.evaluation._trackb_ccf1_sources import (
    CORPUS_DEFAULT, EVIDENCE_ROOT_DEFAULT, TRAINED_DUAL, corpus_paired,
    load_corpus, load_local, local_paired)

ROOT = Path(__file__).resolve().parents[2]
TAUS = [0.003, 0.005, 0.010]
BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
SEEDS_I = (1, 2, 3, 4)
SEEDS_S = ("1", "2", "3", "4")

# regime -> symmetric cap tokens (local cc-F1 uses L*_G*; corpus uses the same)
REGIME_CAPS = {"tight": ["L10_G10", "L20_G20", "L30_G30"],
               "mid": ["L40_G40", "L50_G50", "L60_G60"],
               "loose": ["L70_G70", "L80_G80", "L90_G90"],
               # extra: the paper's actual "tight-cap" headline = the BINDING
               # caps L30-L40 (tab_oct_backbone). Reported alongside the literal
               # L10-L30 tight bin to make the inverted-U explicit -- the win
               # peaks on the binding caps and vanishes at ultra-tight L10/L20.
               "bind": ["L30_G30", "L40_G40"]}
DS_PRETTY = {"tissuemnist": "Tissue", "dermmnist": "Derm", "octmnist": "Oct"}


def regime_label(ds, rg):
    if rg == "bind":
        return "Oct-bind (L30/L40)$^{\\dagger}$"
    return f"{DS_PRETTY[ds]}-{rg}"


def verdict(cell_means, tau):
    """Handoff win-regime rule -> WIN / tie / loss for one threshold."""
    n = len(cell_means)
    if n == 0:
        return "n/a"
    reg_mean = mean(cell_means)
    clear_pos = sum(1 for x in cell_means if x >= tau)
    clear_neg = sum(1 for x in cell_means if x <= -tau)
    if reg_mean >= tau and 2 * clear_pos >= n:
        return "WIN"
    if reg_mean <= -tau and 2 * clear_neg >= n:
        return "loss"
    return "tie"


def cc_cells(local, ds, caps):
    """Per-cell mean paired cc-F1 gap (TraLO - best dual) from local preds."""
    cms = []
    used = []
    for bb in BB:
        for cap in caps:
            diffs = local_paired(local, ds, bb, cap, TRAINED_DUAL, "cc_f1", SEEDS_I)
            if len(diffs) >= 2:
                cms.append(mean(diffs))
                used.append(cap.split("_")[0])
    return cms, sorted(set(used))


def macro_cells(corpus, ds, caps):
    """Per-cell mean paired macro-F1 gap (TraLO - best dual), corpus MNetV3."""
    cms = []
    used = []
    for cap in caps:
        diffs = corpus_paired(corpus, ds, "MobileNetV3", cap, TRAINED_DUAL, SEEDS_S)
        if len(diffs) >= 2:
            cms.append(mean(diffs))
            used.append(cap.split("_")[0])
    return cms, sorted(set(used))


def resolve(local, corpus, ds, regime):
    """Pick the best available data for (ds, regime).

    Returns (metric, source, cell_means, caps_used).  Prefers local cc-F1;
    falls back to corpus MobileNetV3 macro-F1; else empty (server-pending).
    """
    caps = REGIME_CAPS[regime]
    cms, used = cc_cells(local, ds, caps)
    if cms:
        return "cc-F1", "local", cms, used
    if ds != "octmnist":  # octmnist absent from the corpus
        cms, used = macro_cells(corpus, ds, caps)
        if cms:
            return "macro-F1*", "corpus MNetV3", cms, used
    return "cc-F1", "server-pending", [], []


def main():
    ap = argparse.ArgumentParser(description="B5 v2 -- cc-F1 win-bar sensitivity")
    ap.add_argument("--evidence", default=EVIDENCE_ROOT_DEFAULT)
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--tables", default=str(ROOT / "results" / "trackb_deliverables" / "tables"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trackb_deliverables" / "stats" / "b5_v2"))
    args = ap.parse_args()

    local = load_local(args.evidence)
    corpus = load_corpus(args.corpus)

    order = [(ds, rg) for ds in ("tissuemnist", "dermmnist", "octmnist")
             for rg in ("tight", "mid", "loose")]
    order.append(("octmnist", "bind"))   # paper's headline regime, highlighted

    rows = []      # tidy
    tex = []
    for ds, rg in order:
        metric, source, cms, used = resolve(local, corpus, ds, rg)
        regime_tex = regime_label(ds, rg)
        regime = regime_tex.replace("$^{\\dagger}$", "")
        cov = ("%d cells (%s)" % (len(cms), "/".join(used))) if cms else "--"
        if not cms:
            verdicts = ["--", "--", "--"]
            reg_mean = None
        else:
            verdicts = [verdict(cms, t) for t in TAUS]
            reg_mean = mean(cms)
        rows.append([regime, metric, source, cov,
                     "" if reg_mean is None else round(reg_mean, 4),
                     len(cms), *verdicts])
        tex.append((regime_tex, rg, metric, source, len(cms), used, reg_mean, verdicts))

    # ---- tidy csv ----
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "b5_winbar_sensitivity.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["regime", "metric", "source", "coverage", "regime_mean_gap",
                     "n_cells", "tau003", "tau005", "tau010"])
        wr.writerows(rows)

    # ---- deliverable .tex ----
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("% tab_winbar_sensitivity.tex -- B5 deliverable (handoff sec. B5).")
    L.append("% Win-regime rule on cc-F1: WIN iff mean paired gap >= tau AND >= half")
    L.append("% the regime cells clear tau; comparator = per-seed best of")
    L.append("% {Fioretto-LDF, Hounie-RCL}. cc-F1 from frozen preds (paper backbones);")
    L.append("% macro-F1* rows are the corpus MobileNetV3-only fallback (true cc-F1")
    L.append("% needs the server pull). Regenerable: python -m src.evaluation.b5_winbar_sensitivity_v2")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\caption{\\textbf{Sensitivity of the cc-F1 win-regime classification to "
             "the win bar $\\tau$.} A regime is a \\emph{win} iff the mean paired "
             "cc-F1 gap over TraLO's best trained rival (per-seed best of "
             "Fioretto-LDF/Hounie-RCL) is $\\ge\\tau$ \\emph{and} at least half its "
             "(backbone$\\times$cap) cells individually clear $\\tau$; \\emph{loss} is "
             "the mirror; else \\emph{tie}. Regimes bin the symmetric caps as "
             "tight~$=\\{$L10,L20,L30$\\}$, mid~$=\\{$L40,L50,L60$\\}$, "
             "loose~$=\\{$L70,L80,L90$\\}$. cc-F1 is reconstructed from frozen "
             "predictions on the paper backbones; rows marked macro-F1$^{\\ast}$ are "
             "the corpus MobileNetV3-only fallback (the corpus shares only that one "
             "backbone with the paper grid), pending the full server cc-F1 pull. "
             "The OctMNIST tight-cap win is stable across all $\\tau$; every other "
             "computable regime ties.}")
    L.append("\\label{tab:winbar_sensitivity}")
    L.append("\\begin{tabular}{l l l ccc}")
    L.append("\\toprule")
    L.append(" & & & \\multicolumn{3}{c}{Verdict at $\\tau$} \\\\")
    L.append("\\cmidrule(lr){4-6}")
    L.append("Regime & Metric (source) & Cells & $\\tau{=}.003$ & $.005$ & $.010$ \\\\")
    L.append("\\midrule")
    prev_ds = None
    for (regime_tex, rg, metric, source, ncells, used, reg_mean, verdicts) in tex:
        ds_tag = regime_tex.split("-")[0].split("$")[0]
        if rg == "bind":
            L.append("\\midrule")   # set the paper's headline regime apart
        elif prev_ds is not None and ds_tag != prev_ds:
            L.append("\\addlinespace[2pt]")
        prev_ds = ds_tag
        cov = ("%d (%s)" % (ncells, ",".join(used))) if ncells else "\\textsc{server-pending}"
        msrc = f"{metric} ({source})" if source != "server-pending" else "cc-F1 (\\emph{pending})"
        vv = []
        for v in verdicts:
            if v == "WIN":
                vv.append("\\textbf{win}")
            elif v == "loss":
                vv.append("loss")
            elif v == "tie":
                vv.append("tie")
            else:
                vv.append("--")
        L.append(f"{regime_tex} & {msrc} & {cov} & {vv[0]} & {vv[1]} & {vv[2]} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\\\[2pt]{\\footnotesize $^{\\ast}$macro-F1 fallback (corpus, MobileNetV3 "
             "only); the paper's RegNet/ViT and the full cc-F1 grid for these regimes "
             "are \\textsc{server-pending}. $^{\\dagger}$The paper's headline "
             "``tight-cap'' regime is the \\emph{binding} caps L30--L40 "
             "(Table~\\ref{tab:oct_backbone}); the literal L10--L30 \\emph{Oct-tight} "
             "bin above ties because the ultra-tight L10/L20 cells are cc-F1 no-ops "
             "(the win follows an inverted-U in cap tightness).}")
    L.append("\\end{table}")
    (tables / "tab_winbar_sensitivity.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- console ----
    print("[b5_v2] cc-F1 win-regime sensitivity (comparator = best trained dual)")
    print(f"{'regime':14s} {'metric':10s} {'source':16s} {'cells':16s} "
          f"{'mean':>8s}  t003 t005 t010")
    for r in rows:
        regime, metric, source, cov, rm, nc = r[0], r[1], r[2], r[3], r[4], r[5]
        print(f"{regime:14s} {metric:10s} {source:16s} {cov:16s} "
              f"{('' if rm=='' else f'{rm:+.4f}'):>8s}  "
              f"{r[6]:4s} {r[7]:4s} {r[8]:4s}")
    print(f"wrote {tables / 'tab_winbar_sensitivity.tex'}")
    print(f"wrote {outdir / 'b5_winbar_sensitivity.csv'}")


if __name__ == "__main__":
    main()

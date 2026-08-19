"""make_main_table.py -- build the expanded Table 1 (tab_ccf1.tex): full symmetric grid,
3 datasets x 3 backbones x {L30,L50,L70}, all 6 methods, constrained-class F1.

Professor's revision (2026-07-14): one consolidated results table instead of the old
Table 1 (MobileNetV3-only) + Table 2 (OctMNIST backbone deltas); Table 2 moves to the
supplement unchanged.

Conventions copied from the shipped tab_ccf1.tex (validated below by exact reproduction):
  - canonical cut: sweep=='paper_final', warmup 50, seeds 1-4, exactly one row per
    (dataset, model, cap, method, seed)
  - cell value = mean over the 4 seeds; {\tiny +-.0xx} = across-seed sample std (ddof=1)
  - bold = best rounded value among the constraint-TRAINED methods; underline = second
    distinct rounded value among trained; post-hoc clippers never marked (dagger'd)

Run:  python scripts/make_main_table.py [--validate-only]
Writes tables/tab_ccf1.tex and prints the caption/prose stats.
"""
# Re-homed to docs/paper/scripts/ on 2026-08-19. This file previously lived
# only in the gitignored archive/legacy/final_AAAI_PAPER/scripts/, so no clone
# of this repository could regenerate the float it emits. Paths below resolve
# against docs/paper/ -- ROOT is this file's parent's parent.

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # docs/paper/

DS3 = ["tissuemnist", "dermmnist", "octmnist"]
DSP = {"tissuemnist": "Tissue", "dermmnist": "Derm", "octmnist": "OctMNIST"}
BB3 = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
BBP = {"MobileNetV3": "MNetV3", "RegNetY400MF": "RegNet", "ViTB16": "ViT-B/16"}
CAPS = [30, 50, 70]
MORDER = ["heuristic", "danits_lp", "fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]
TRAINED = ["fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]


def load():
    df = pd.read_csv(os.path.join(ROOT, "data", "corpus", "corpus_final.csv"))
    df = df[df.sweep == "paper_final"].copy()
    df["pct"] = df.constraint_tag.str.extract(r"L(\d+)_").astype(int)
    key = ["dataset", "model", "pct", "method", "seed"]
    dup = df.duplicated(key).sum()
    assert dup == 0, f"{dup} duplicate rows for the atomic key -- refuse to average"
    n_per_cell = df.groupby(["dataset", "model", "pct", "method"]).seed.nunique()
    assert (n_per_cell == 4).all(), "every cell must have exactly seeds 1-4"
    return df


def cell_stats(df, ds, mo, p, me, col="cc_f1"):
    s = df[(df.dataset == ds) & (df.model == mo) & (df.pct == p) & (df.method == me)]
    s = s.sort_values("seed")[col].values
    assert len(s) == 4
    return float(s.mean()), float(s.std(ddof=1))


def fmt_val(mean, std):
    return f"{mean:.3f}" + r"{\tiny$\pm$" + f"{std:.3f}".lstrip("0") + "}"


def markup(vals_rounded):
    """bold = max rounded among trained; underline = second distinct rounded among trained."""
    tr = sorted({vals_rounded[m] for m in TRAINED}, reverse=True)
    best = tr[0]
    second = tr[1] if len(tr) > 1 else None
    out = {}
    for m in MORDER:
        if m in TRAINED and vals_rounded[m] == best:
            out[m] = "bold"
        elif m in TRAINED and second is not None and vals_rounded[m] == second:
            out[m] = "under"
        else:
            out[m] = ""
    return out


def render_cell(mean, std, mk):
    body = fmt_val(mean, std)
    if mk == "bold":
        num, rest = body.split("{", 1)
        return r"\textbf{" + num + r"}{" + rest
    if mk == "under":
        num, rest = body.split("{", 1)
        return r"\underline{" + num + r"}{" + rest
    return body


def validate_against_shipped(df):
    """Re-derive the 54 shipped MobileNetV3 numbers and diff against tables/tab_ccf1.tex."""
    shipped = open(os.path.join(ROOT, "tables", "tab_ccf1.tex"), encoding="utf-8").read()
    pat = re.compile(r"(\d\.\d{3})\}?\{\\tiny\$\\pm\$(\.\d{3})\}")
    got = pat.findall(shipped)
    assert len(got) == 54, f"expected 54 shipped numbers, parsed {len(got)}"
    it = iter(got)
    bad = []
    for ds in DS3:
        for p in CAPS:
            for me in MORDER:
                mean, std = cell_stats(df, ds, "MobileNetV3", p, me)
                sm, ss = next(it)
                if f"{mean:.3f}" != sm or f"{std:.3f}".lstrip("0") != ss:
                    bad.append((ds, p, me, f"{mean:.3f}±{std:.3f}", f"{sm}±{ss}"))
    if bad:
        for b in bad:
            print("MISMATCH", b)
        raise SystemExit("validation FAILED -- pipeline does not reproduce shipped table")
    print("validation OK: all 54 shipped MobileNetV3 values reproduce exactly (ddof=1)")


def build_table(df):
    L = []
    L.append(r"% tab_ccf1.tex -- headline constrained-class F1, FULL symmetric grid:")
    L.append(r"% 3 datasets x 3 backbones x {L30,L50,L70}, all 6 methods (professor revision")
    L.append(r"% 2026-07-14: single consolidated results table; the old per-backbone OctMNIST")
    L.append(r"% delta table tab_oct_backbone moved to the supplement, Supp. Table S6).")
    L.append(r"% Regenerable via scripts/make_main_table.py -- sweep=='paper_final', exactly one")
    L.append(r"% row per seed, mean +- across-seed sample std (ddof=1) over seeds 1-4.")
    L.append(r"% \input-ready: float-only. Requires: booktabs, multirow (loaded by main.tex).")
    L.append(r"\begin{table*}[t]\centering\small")
    L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L.append(r"\caption{\textbf{Constrained-class F1 on the full symmetric grid}: all three")
    L.append(r"datasets $\times$ all three backbones (mean over seeds $1$--$4$;")
    L.append(r"{\tiny$\pm$}\,across-seed std). \textbf{Bold} = best, \underline{underline} =")
    L.append(r"second-best among the constraint-\emph{trained} methods; $^{\dagger}$ = post-hoc")
    L.append(r"clippers, shown for reference. On Tissue and Derm the trained methods sit within")
    L.append(r"seed noise on every backbone (a paired-Wilcoxon tie; the marginal std shown here")
    L.append(r"is for transparency, the paired uncertainty is the seed-winrate of")
    L.append(r"Supp.~Table~S6). On OctMNIST at the tight $L30$ cap TraLO leads the trained")
    L.append(r"baselines on every backbone, largest on ViT-B/16.}")
    L.append(r"\label{tab:ccf1}")
    L.append(r"\vspace{4pt}")
    L.append(r"\begin{tabular}{lll cc cccc}")
    L.append(r"\toprule")
    L.append(r" & & & \multicolumn{2}{c}{Post-hoc$^{\dagger}$} & \multicolumn{4}{c}{Constraint-trained} \\")
    L.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}")
    L.append(r"Data & Backbone & Cap & Heur. & LP-LG & Fioretto & Hounie & TraLO-b & \textbf{TraLO} \\")
    L.append(r"\midrule")
    stats = {"best_or_tied": 0, "cells": 0}
    for di, ds in enumerate(DS3):
        for bi, mo in enumerate(BB3):
            for ci, p in enumerate(CAPS):
                ms = {m: cell_stats(df, ds, mo, p, m) for m in MORDER}
                rounded = {m: round(ms[m][0], 3) for m in MORDER}
                mk = markup(rounded)
                row = [render_cell(*ms[m], mk[m]) for m in MORDER]
                lead_ds = rf"\multirow{{9}}{{*}}{{{DSP[ds]}}}" if (bi == 0 and ci == 0) else ""
                lead_bb = rf"\multirow{{3}}{{*}}{{{BBP[mo]}}}" if ci == 0 else ""
                L.append(f"{lead_ds} & {lead_bb} & L{p} & " + " & ".join(row) + r" \\")
                # prose stat: TraLO best-or-tied (within 0.005 of best trained, full precision)
                best_tr = max(ms[m][0] for m in TRAINED)
                stats["cells"] += 1
                if ms["tralo"][0] >= best_tr - 0.005:
                    stats["best_or_tied"] += 1
            if bi < len(BB3) - 1:
                L.append(r"\cmidrule(lr){2-9}")
        if di < len(DS3) - 1:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n", stats


def markup_tieband(vals_full):
    """Bold = TIED with the best, not "is the best".

    The project adjudicates every comparison at a 0.005 band, so a table that
    bolds a single winner asserts a ranking the evidence does not support.
    Bold  = every trained entry within 0.005 of the best trained entry.
    Under = every trained entry within 0.005 of the best entry OUTSIDE that band.
    Both bands are computed on the FULL-PRECISION means -- rounding first puts
    entries on the wrong side of the boundary whenever the gap is near 0.005.
    """
    tr = {m: vals_full[m] for m in MORDER if m in TRAINED}
    best = max(tr.values())
    out = {m: "" for m in MORDER}
    bolded = [m for m, v in tr.items() if v >= best - 0.005]
    for m in bolded:
        out[m] = "bold"
    rest = {m: v for m, v in tr.items() if m not in bolded}
    if rest:
        second = max(rest.values())
        for m, v in rest.items():
            if v >= second - 0.005:
                out[m] = "under"
    return out


def render_plain(mean, mk):
    s = f"{mean:.3f}"
    if mk == "bold":
        return r"\textbf{" + s + "}"
    if mk == "under":
        return r"\underline{" + s + "}"
    return s


def build_table_two_metrics(df):
    """Combined Table 1: cc-F1 AND macro-F1 blocks, no per-cell std (moved to caption note;
    paired uncertainty lives in Supp. Table S6 and the Supp. Sec. D per-cell tables)."""
    all_stds = []
    L = []
    L.append(r"% tab_ccf1.tex -- consolidated headline table, FULL symmetric grid:")
    L.append(r"% 3 datasets x 3 backbones x {L30,L50,L70}, all 6 methods, cc-F1 AND macro-F1")
    L.append(r"% (professor revision 2026-07-14: one table, two quality metrics; the old")
    L.append(r"% per-backbone OctMNIST delta table tab_oct_backbone is Supp. Table S6).")
    L.append(r"% Regenerable via scripts/make_main_table.py --two-metrics")
    L.append(r"% sweep=='paper_final', one row per seed, cell = mean over seeds 1-4; per-cell")
    L.append(r"% across-seed stds omitted for width (range printed by the generator, noted in")
    L.append(r"% caption); paired uncertainty = seed-winrates, Supp. Table S6 / Supp. Sec. D.")
    L.append(r"% \input-ready: float-only. Requires: booktabs, multirow (loaded by main.tex).")
    L.append(r"\begin{table*}[t]\centering\small")
    L.append(r"\setlength{\tabcolsep}{2.2pt}")
    L.append(r"\caption{\textbf{Quality of the final satisfying predictions on the full")
    L.append(r"symmetric grid}: constrained-class F1 and macro-F1, all three datasets $\times$")
    L.append(r"all three backbones (mean over seeds $1$--$4$; across-seed stds omitted per")
    L.append(r"cell for width, range $.001$--$.055$). \textbf{Bold} = within $0.005$ of the")
    L.append(r"best constraint-\emph{trained} entry in the row --- the tie band every")
    L.append(r"comparison in this paper is adjudicated at --- so jointly bolded entries are")
    L.append(r"\emph{tied, not ranked}; \underline{underline} = best entry outside that band;")
    L.append(r"$^{\dagger}$ = post-hoc clippers, for reference; seed-winrates in")
    L.append(r"Table~\ref{tab:oct_backbone} and App.~\ref{app:regime}. Among trained methods the grid is a tie on")
    L.append(r"Tissue and Derm while TraLO leads every backbone at the OctMNIST tight caps;")
    L.append(r"on macro-F1 TraLO's paired gap over the best clipper clears $+0.005$ in $24$ of")
    L.append(r"$27$ cells, zero losses (grid mean $+0.031$). Per-dataset paired-gap")
    L.append(r"summaries over the symmetric caps for each backbone (including MobileNetV2)")
    L.append(r"are in Table~\ref{tab:bbgen}.}")
    L.append(r"\label{tab:ccf1}")
    L.append(r"\vspace{4pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{lll cccccc |c cccccc}")   # vrule divides cc-F1 | macro-F1
    L.append(r"\toprule")
    L.append(r" & & & \multicolumn{6}{c}{Constrained-class F1} & & \multicolumn{6}{c}{Macro-F1} \\")
    L.append(r"\cmidrule(lr){4-9}\cmidrule(lr){11-16}")
    hdr = r"Heur.$^{\dagger}$ & LP-LG$^{\dagger}$ & Fioretto & Hounie & TraLO-b & \textbf{TraLO}"
    L.append(r"Data & Backbone & Cap & " + hdr + " & & " + hdr + r" \\")
    L.append(r"\midrule")
    stats = {"best_or_tied": 0, "cells": 0}
    for di, ds in enumerate(DS3):
        for bi, mo in enumerate(BB3):
            for ci, p in enumerate(CAPS):
                blocks = []
                for col in ("cc_f1", "f1_macro"):
                    ms = {m: cell_stats(df, ds, mo, p, m, col) for m in MORDER}
                    all_stds += [ms[m][1] for m in MORDER]
                    mk = markup_tieband({m: ms[m][0] for m in MORDER})
                    blocks.append(" & ".join(render_plain(ms[m][0], mk[m]) for m in MORDER))
                    if col == "cc_f1":
                        best_tr = max(ms[m][0] for m in TRAINED)
                        stats["cells"] += 1
                        if ms["tralo"][0] >= best_tr - 0.005:
                            stats["best_or_tied"] += 1
                lead_ds = rf"\multirow{{9}}{{*}}{{{DSP[ds]}}}" if (bi == 0 and ci == 0) else ""
                lead_bb = rf"\multirow{{3}}{{*}}{{{BBP[mo]}}}" if ci == 0 else ""
                L.append(f"{lead_ds} & {lead_bb} & L{p} & {blocks[0]} & & {blocks[1]}" + r" \\")
            if bi < len(BB3) - 1:
                L.append(r"\cmidrule(lr){2-16}")
        if di < len(DS3) - 1:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"} % close \resizebox")
    L.append(r"\end{table*}")
    print(f"across-seed std range over all shown cells: "
          f"{min(all_stds):.3f} -- {max(all_stds):.3f}")
    return "\n".join(L) + "\n", stats


def prose_stats(df):
    """Numbers the surrounding prose cites."""
    print("\n--- prose stats ---")
    # oct L30/L40 paired delta vs per-seed best trained dual (cross-check vs old Table 2)
    for mo in BB3:
        for p in (30, 40):
            sub = df[(df.dataset == "octmnist") & (df.model == mo) & (df.pct == p)]
            piv = sub.pivot_table(index="seed", columns="method", values="cc_f1")
            d = piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
            print(f"oct {mo:12s} L{p}: paired dcc-F1 vs best dual = {d.mean():+.3f} "
                  f"(std {d.std(ddof=1):.3f}, {int((d > 0).sum())}/4 seeds)")


if __name__ == "__main__":
    df = load()
    if "--validate-only" in sys.argv:
        validate_against_shipped(df)
        sys.exit(0)
    if "--two-metrics" in sys.argv:
        tex, stats = build_table_two_metrics(df)
    else:
        tex, stats = build_table(df)
    out = os.path.join(ROOT, "tables", "tab_ccf1.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"wrote {out}")
    print(f"TraLO best-or-tied among trained (within 0.005): "
          f"{stats['best_or_tied']}/{stats['cells']} cells")
    prose_stats(df)

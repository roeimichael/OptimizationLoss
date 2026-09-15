r"""Build the TASK-CELL tables in the OFFICIAL metrics -- cc-F1 and macro-F1.

    python docs/paper/scripts/make_task_cell_table.py

Reads  docs/paper/data/task_cells_2026-09-10.json  (per seed, per arm)
Writes docs/paper/tables_task/tab_taskcell_ccf1.tex
       docs/paper/tables_task/tab_taskcell_macrof1.tex
       docs/paper/tables_task/tab_taskcell_contrast.tex

WHY THIS EXISTS, AND WHY IT IS NOT `make_main_table.py`
------------------------------------------------------
`make_main_table.py` builds the shipped `tab_ccf1.tex` from
`docs/paper/data/corpus/corpus_final.csv`, which is the dermmnist / octmnist /
tissuemnist generation -- a corpus that no longer exists (see
`docs/paper/WHICH_CORPUS.md`). Nothing here shares a row with it.

HOUSE CONVENTIONS, copied from `make_main_table.py` so the two read alike:
  * cell value = MEAN over the seeds, `{\tiny +-.0xx}` = across-seed sample sd
    (ddof=1, the same estimator `full_panel` reports).
  * bold      = best value among the CONSTRAINT-TRAINED arms, with a tie band.
  * underline = the next distinct value among the trained arms.
  * post-hoc clippers are never marked -- they carry a dagger instead.

WHAT IS DIFFERENT HERE, AND IT IS THE POINT
-------------------------------------------
1. The TIE BAND IS MEASURED, not fixed at 0.005. The shipped table hardcodes
   0.005; here it is the median across-seed sd of the cell's own trained arms,
   so a bold mark means "best, and nothing else is inside this cell's own
   noise". FRAMEWORK 2(z77): at 4 seeds the minimum detectable effect and the
   whole per-cell prize are the same size, so a bold mark on a fixed band
   promotes noise to a result.
2. A UNIT column. Ten task cells collapse to FOUR independent units
   (`paper_rows`' `MEASURED_UNITS`): two cap levels in one campaign share a
   warm-up. A sign test runs over units, never over rows.
3. `fioretto` and `hounie` carry a section mark on the PARTIAL campaigns
   (`dom1`, `dom1b`, `equaldose1`): 28.00 attempted constraint steps per run
   against 29.00 for every other arm, so those entries are not at equal
   compute and are excluded from the bold competition. `quarantine.REGISTRY`
   is the authority; DOSE_DEAD mirrors it.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA = os.path.join(PAPER, "data", "task_cells_2026-09-10.json")
OUT = os.path.join(PAPER, "tables_task")

# arm -> (printed name, kind).  kind: posthoc | lam0 | trained
ARMS = [
    ("clip",         r"Clip$^{\dagger}$",         "posthoc"),
    ("focal_clip",   r"Focal+Clip$^{\dagger}$",   "posthoc"),
    ("tralo_null",   r"$\lambda{=}0^{\ddagger}$", "lam0"),
    ("tralo_reseed", r"RNG floor$^{\ddagger}$",   "lam0"),
    ("tralo_reseed2", r"RNG floor 2$^{\ddagger}$", "lam0"),
    ("tralo",        r"\textsc{TraLO}",           "trained"),
    ("alm",          r"ALM",                      "trained"),
    ("fioretto",     r"LDF",                      "trained"),
    ("hounie",       r"RCL",                      "trained"),
]

# quarantine.REGISTRY: PARTIAL campaigns and the arms disqualified in each.
DOSE_DEAD = {
    "dom1":       {"fioretto", "hounie"},
    "dom1b":      {"fioretto", "hounie"},
    "equaldose1": {"fioretto", "hounie", "tralo_lam0"},
}

BACKBONE_TEX = {
    "MobileNetV2": "MNv2", "MobileNetV3": "MNv3",
    "RegNetY400MF": "RegNetY", "ViTB16": "ViT-B/16",
}

CONTRASTS = [
    ("clip",         r"vs.\ post-hoc clipping"),
    ("tralo_null",   r"vs.\ its own $\lambda{=}0$ twin"),
    ("tralo_reseed", r"vs.\ the RNG floor"),
    ("alm",          r"vs.\ ALM"),
]


def cell_stats(row, arm, metric):
    d = row["seeds"].get(arm)
    if not d:
        return None
    v = [s[metric] for s in d.values()]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def paired(row, a, b, metric):
    """Paired over the seeds BOTH arms ran -- never the union. FRAMEWORK 2(z50)."""
    da, db = row["seeds"].get(a), row["seeds"].get(b)
    if not da or not db:
        return None
    ks = sorted(set(da) & set(db))
    if len(ks) < 2:
        return None
    d = [da[k][metric] - db[k][metric] for k in ks]
    return st.mean(d), st.stdev(d), len(d), sum(1 for x in d if x > 0)


def fmt(mean, sd):
    return r"%s{\tiny $\pm$%s}" % (("%.3f" % mean).lstrip("0"),
                                   ("%.3f" % sd).lstrip("0"))


def live_trained(row):
    dead = DOSE_DEAD.get(row["campaign"], set())
    return [a for a, _, k in ARMS
            if k == "trained" and a not in dead and a in row["seeds"]]


def marks(row, metric):
    """bold / underline among the LIVE trained arms, on a MEASURED tie band.

    Returns (bold, under). A multi-way tie inside the band returns an EMPTY
    bold set -- the tied arms come back as `under` so the caller can see the
    tie existed rather than silently marking the arithmetic maximum.

    A cell holding FEWER THAN TWO live trained arms marks NOTHING. `coin1` and
    `coin2` stage `tralo` with no rival dual, and the first version of this
    function bolded it in both for leading a competition with one entrant --
    in `coin2` while sitting BELOW `clip`, `focal_clip` and its own
    $\\lambda{=}0$ twin. `tralo_wins` excludes rival-free cells from its
    denominator for exactly this reason; a table that marks them anyway
    manufactures two of its three wins.
    """
    vals = {a: cell_stats(row, a, metric) for a in live_trained(row)}
    if len(vals) < 2:
        return set(), set()
    band = st.median([v[1] for v in vals.values()])
    best = max(v[0] for v in vals.values())
    bold = {a for a, v in vals.items() if best - v[0] <= band}
    if len(bold) > 1:
        return set(), bold
    rest = [v[0] for a, v in vals.items() if a not in bold]
    if not rest:
        return bold, set()
    second = max(rest)
    return bold, {a for a, v in vals.items() if a not in bold and v[0] == second}


def _row_head(r):
    return "%s & %s & \\texttt{%s} & \\texttt{%s}" % (
        r["unit"], BACKBONE_TEX.get(r["backbone"], r["backbone"]),
        r["campaign"].replace("_", r"\_"), r["cap"].replace("_", r"\_"))


def metric_table(rows, metric, label, caption):
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3.0pt}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{cllc " + "c" * len(ARMS) + "}",
         r"\toprule",
         r"Unit & Backbone & Campaign & Cap & "
         + " & ".join(n for _, n, _ in ARMS) + r" \\",
         r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["unit"] != prev:
            L.append(r"\midrule")
        prev = r["unit"]
        bold, under = marks(r, metric)
        dead = DOSE_DEAD.get(r["campaign"], set())
        cells = []
        for a, _, _ in ARMS:
            s = cell_stats(r, a, metric)
            if s is None:
                cells.append("--")
                continue
            t = fmt(s[0], s[1])
            if a in bold:
                t = r"\textbf{%s}" % t
            elif a in under:
                t = r"\underline{%s}" % t
            if a in dead:
                t += r"$^{\S}$"
            cells.append(t)
        L.append("%s & %s \\\\" % (_row_head(r), " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{%s}" % caption, r"\label{%s}" % label,
          r"\end{table*}", ""]
    return "\n".join(L)


LAM0 = {"tralo_null", "tralo_reseed", "tralo_reseed2"}


def _res_where(rows):
    """Say WHAT the resolved contrasts are against -- computed, never asserted.

    The first draft of this caption hardcoded "both are against a lambda=0
    control". There were three, and a caption that states a count the generator
    does not compute is the same defect as a doc figure with no scorer behind
    it. Everything numeric in these captions is derived here.
    """
    hits = [b for r in rows for b, _ in CONTRASTS
            for p in [paired(r, "tralo", b, "ccf1")]
            if p and abs(p[0]) > 2 * p[1]]
    if not hits:
        return "so no difference in this table is resolved by the design that produced it"
    n0 = sum(1 for b in hits if b in LAM0)
    if n0 == len(hits):
        return (r"and \emph{all %d} are against a $\lambda{=}0$ control rather "
                r"than against a rival constrained method" % n0)
    return (r"of which %d are against a $\lambda{=}0$ control and %d against a "
            r"rival constrained method" % (n0, len(hits) - n0))


def contrast_table(rows):
    n_res = sum(1 for r in rows for b, _ in CONTRASTS
                for p in [paired(r, "tralo", b, "ccf1")]
                if p and abs(p[0]) > 2 * p[1])
    n_con = sum(1 for r in rows for b, _ in CONTRASTS
                if paired(r, "tralo", b, "ccf1"))
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4.0pt}",
         r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{cllc " + "cc" * len(CONTRASTS) + "}",
         r"\toprule",
         r"& & & & "
         + " & ".join(r"\multicolumn{2}{c}{%s}" % t for _, t in CONTRASTS) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (5 + 2 * i, 6 + 2 * i)
                  for i in range(len(CONTRASTS))),
         r"Unit & Backbone & Campaign & Cap & "
         + " & ".join(r"$\Delta$ & sign" for _ in CONTRASTS) + r" \\",
         r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["unit"] != prev:
            L.append(r"\midrule")
        prev = r["unit"]
        cells = []
        for b, _ in CONTRASTS:
            p = paired(r, "tralo", b, "ccf1")
            if p is None:
                cells += ["--", "--"]
                continue
            m, sd, n, w = p
            t = "%+.3f" % m
            if abs(m) > 2 * sd:
                t = r"\textbf{%s}" % t
            cells += [t, "%d/%d" % (w, n)]
        L.append("%s & %s \\\\" % (_row_head(r), " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{\textbf{Paired contrasts in cc-F1, over the seeds each pair "
          r"shares.} $\Delta$ is the mean paired difference and \emph{sign} counts "
          r"the seeds in which \textsc{TraLO} leads. \textbf{Bold} marks the "
          r"entries whose $|\Delta|$ exceeds twice its own paired standard "
          r"deviation -- the only differences this design resolves at 4 seeds. "
          r"\textbf{%d of %d contrasts resolve}, %s.}" % (n_res, n_con, _res_where(rows)),
          r"\label{tab:taskcell-contrast}", r"\end{table*}", ""]
    return "\n".join(L)


CAP_CC = (
    r"\textbf{Capped-class F1 (cc-F1) on the task cells.} Mean over 4 seeds, "
    r"$\pm$ across-seed sample standard deviation. Ten cells, but only "
    r"\textbf{four independent units}: two cap levels within one campaign share "
    r"a warm-up, so a sign test runs over the \emph{Unit} column and not over "
    r"rows. $^{\dagger}$post-hoc clippers (30 warm-up epochs, 0 constraint "
    r"epochs). $^{\ddagger}$$\lambda{=}0$ controls at identical compute: the "
    r"twin isolates the constraint, the RNG floor is that twin with only its "
    r"random stream perturbed. $^{\S}$28.00 attempted constraint steps per run "
    r"against 29.00 for every other arm -- not at equal compute, and excluded "
    r"from the comparison. \textbf{Bold} marks a constrained arm leading the "
    r"other constrained arms by more than the cell's own median across-seed "
    r"standard deviation; where no arm clears that band the tied leaders are "
    r"underlined instead. %s")

CAP_MF = (
    r"\textbf{Macro-F1 on the same cells}, over all classes. Conventions as in "
    r"Table~\ref{tab:taskcell-ccf1}. The capped classes are 2 of 8 (iWildCam) "
    r"and 2 of 7 (BCN), so this metric is carried by the classes the constraint "
    r"never touches; it is reported because cc-F1 alone cannot show collateral "
    r"damage. %s")


def _leader_sentence(rows, metric):
    """Who leads, counted -- so the caption cannot drift from the table."""
    winners, tied, norival = {}, 0, 0
    for r in rows:
        if len(live_trained(r)) < 2:
            norival += 1
            continue
        bold, _ = marks(r, metric)
        if bold:
            winners[list(bold)[0]] = winners.get(list(bold)[0], 0) + 1
        else:
            tied += 1
    testable = len(rows) - norival
    lead = (r"%d of these cells stage \textsc{TraLO} with no rival constrained "
            r"method and cannot test the comparison in either direction; they "
            r"are never marked. " % norival) if norival else ""
    if not winners:
        return lead + (r"Among the remaining %d, \textbf{no cell has an "
                       r"unambiguous leader} -- the constrained arms sit inside "
                       r"one another's seed noise throughout." % testable)
    name = {"tralo": r"\textsc{TraLO}", "alm": "ALM",
            "fioretto": "LDF", "hounie": "RCL"}
    got = ", ".join("%s in %d" % (name.get(a, a), n)
                    for a, n in sorted(winners.items(), key=lambda x: -x[1]))
    return lead + (r"Among the remaining %d an unambiguous leader exists in "
                   r"\textbf{%d} (%s); the other %d are ties inside the seed "
                   r"noise."
                   % (testable, sum(winners.values()), got, tied))


def main():
    rows = json.load(open(DATA))
    # Sort by UNIT first and only then by campaign/cap. A unit spans campaigns
    # (A2 is `coin2` + `equaldose1`) and a campaign spans units (`fmow1` is E1
    # and E2), so sorting on campaign before unit interleaves the unit column
    # and the \midrule grouping stops meaning anything. Any unit not named here
    # sorts last rather than colliding at one key.
    order = {"A1": 0, "A2": 1, "B1": 2, "C1": 3, "C2": 4, "D1": 5,
             "E1": 6, "E2": 7}
    rows.sort(key=lambda r: (order.get(r["unit"], 99), r["unit"],
                             r["campaign"], r["cap"]))
    os.makedirs(OUT, exist_ok=True)

    for name, text in (
            ("tab_taskcell_ccf1.tex",
             metric_table(rows, "ccf1", "tab:taskcell-ccf1",
                          CAP_CC % _leader_sentence(rows, "ccf1"))),
            ("tab_taskcell_macrof1.tex",
             metric_table(rows, "macrof1", "tab:taskcell-macrof1",
                          CAP_MF % _leader_sentence(rows, "macrof1"))),
            ("tab_taskcell_contrast.tex", contrast_table(rows))):
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(text)

    print("wrote 3 tables to %s" % OUT)
    print("cells %d, units %d" % (len(rows), len({r["unit"] for r in rows})))
    n_res = sum(1 for r in rows for b, _ in CONTRASTS
                for p in [paired(r, "tralo", b, "ccf1")]
                if p and abs(p[0]) > 2 * p[1])
    n_con = sum(1 for r in rows for b, _ in CONTRASTS
                if paired(r, "tralo", b, "ccf1"))
    print("contrasts that RESOLVE (|d| > 2 paired sd): %d of %d" % (n_res, n_con))
    nr = sum(1 for r in rows if len(live_trained(r)) < 2)
    print("cells staging tralo with NO rival (never marked): %d" % nr)
    for m in ("ccf1", "macrof1"):
        w = {}
        for r in rows:
            for a in marks(r, m)[0]:
                w[a] = w.get(a, 0) + 1
        print("%-8s unambiguous leader in %d of %d testable cells  %s"
              % (m, sum(w.values()), len(rows) - nr, w or "{}"))


def _cell(campaign="x", unit="U", cap="L80_G95", **arms):
    """A synthetic cell. arms: name -> list of per-seed cc-F1 values."""
    return {"unit": unit, "campaign": campaign, "backbone": "MobileNetV2",
            "dataset": "iwildcam", "cap": cap, "capped_classes": [2, 7],
            "seeds": {a: {"seed_%d" % (i + 1): {"ccf1": v, "macrof1": v}
                          for i, v in enumerate(vs)}
                      for a, vs in arms.items()}}


def self_test():
    ok, bad = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # 1. A cell with ONE trained arm marks NOTHING, however far ahead it is.
    #    This is the defect that manufactured two of three wins on first run.
    solo = _cell(tralo=[.90, .90, .90, .90], clip=[.10] * 4)
    chk("solo trained arm is never marked", marks(solo, "ccf1") == (set(), set()))

    # 1b. NEGATIVE CONTROL: add a rival and the same lead MUST now be marked.
    duo = _cell(tralo=[.90, .90, .90, .90], alm=[.10] * 4, clip=[.10] * 4)
    chk("with a rival present the same lead IS marked",
        marks(duo, "ccf1")[0] == {"tralo"})

    # 2. A lead inside the cell's own seed noise is NOT bold.
    noisy = _cell(tralo=[.80, .90, .70, .88], alm=[.79, .89, .69, .87])
    chk("a lead inside the seed band is a tie", marks(noisy, "ccf1")[0] == set())
    #    NEGATIVE CONTROL: same means, seeds tightened -> it becomes a win.
    tight = _cell(tralo=[.850, .851, .849, .850], alm=[.840, .841, .839, .840])
    chk("the same gap on tight seeds IS a win",
        marks(tight, "ccf1")[0] == {"tralo"})

    # 3. Dose-disqualified arms are out of the competition entirely.
    part = _cell(campaign="dom1", tralo=[.80] * 4, alm=[.79] * 4,
                 fioretto=[.99] * 4)
    chk("a 28-step arm cannot win a PARTIAL campaign",
        marks(part, "ccf1")[0] == {"tralo"})
    #    NEGATIVE CONTROL: the same numbers in a LIVE campaign, and it wins.
    live = _cell(campaign="bcn1mn3", tralo=[.80] * 4, alm=[.79] * 4,
                 fioretto=[.99] * 4)
    chk("the same arm in a live campaign DOES win",
        marks(live, "ccf1")[0] == {"fioretto"})

    # 4. `paired` uses the seeds BOTH arms ran -- never the union. FRAMEWORK 2(z50).
    rag = _cell(tralo=[.80, .80, .80], alm=[.70, .70, .70, .99])
    chk("paired uses the shared seeds only",
        abs(paired(rag, "tralo", "alm", "ccf1")[0] - 0.10) < 1e-9
        and paired(rag, "tralo", "alm", "ccf1")[2] == 3)

    # 5. The caption's leader count CANNOT drift from the table's bold marks --
    #    the first draft hardcoded both and both were wrong.
    rows = json.load(open(DATA))
    n_bold = sum(len(marks(r, "ccf1")[0]) for r in rows)
    chk("caption leader count matches the marked cells",
        (r"\textbf{%d}" % n_bold) in _leader_sentence(rows, "ccf1")
        or (n_bold == 0 and "no cell has an" in _leader_sentence(rows, "ccf1")))

    # 6. ...and the same for the contrast caption.
    n_res = sum(1 for r in rows for b, _ in CONTRASTS
                for p in [paired(r, "tralo", b, "ccf1")]
                if p and abs(p[0]) > 2 * p[1])
    tex = contrast_table(rows)
    chk("contrast caption states the resolved count it computed",
        (r"\textbf{%d of " % n_res) in tex)
    chk("every resolved contrast is bolded in the body",
        tex.count(r"\textbf{+") + tex.count(r"\textbf{-") == n_res)

    print("self-test: %d passed, %d failed" % (ok, len(bad)))
    for b in bad:
        print("  FAILED: %s" % b)
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test() if "--self-test" in sys.argv else main())

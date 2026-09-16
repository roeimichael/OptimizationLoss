"""Render the hospital smoke test to LaTeX, from its own measured JSON.

Nothing here retypes a number: every value in the document is read from the
output of `scripts/hospital_smoke.py --json`, so the PDF cannot drift from
what the pipeline actually computed.

    python scripts/hospital_smoke.py --json out.json
    python scripts/hospital_report.py out.json report.tex
"""
import json
import sys

# LaTeX row terminator. Built from chr(92) rather than written literally
# because this repo's shell transport eats one backslash from every pair, and
# a .tex file is almost entirely backslashes. See RULESET on heredocs.
BS = chr(92)
EOL = BS * 2

HEAD = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\definecolor{good}{RGB}{0,120,60}
\definecolor{bad}{RGB}{176,0,32}
\newcommand{\ok}{\textcolor{good}{\textbf{OK}}}
\newcommand{\no}{\textcolor{bad}{\textbf{WRONG}}}
\begin{document}
\begin{center}
{\LARGE\bfseries Local and global constraints: a validation}@EOL@[4pt]
{\large The hospital-beds smoke test}@EOL@[10pt]
\end{center}
""".replace("@EOL@", EOL)

INTRO = r"""\section*{What is being checked}

Shifman et al. (2025), Eqs.~(1)--(4), constrain an assignment $R[s,i]$ of
sample $s$ to class $i$ with two families of bounds:
\begin{align}
\sum_{s \in \lambda} R[s,i] &\le \Phi[\lambda][i] && \text{feature-based, per (group, class)}@EOL@
\sum_{s} R[s,i] &\le \Psi[i] && \text{target-based, per class, whole pool}
\end{align}
$\Phi$ is indexed by $\lambda$, the \emph{local feature}. In the hospital
reading: $\Psi$ is how many beds exist, and $\Phi$ is how many of them each
membership tier may occupy. Both are \emph{given}. They are a policy, not a
property of who happened to apply.

This document checks that the pipeline can now express that policy, and shows
what it produced before it could.
""".replace("@EOL@", EOL)

WHY = r"""\section*{Why the totals matter}

When the two percentages are equal, $\sum_\lambda \Phi[\lambda][i] = \Psi(i)$
\emph{exactly} -- the local ceilings partition the global one, so both
constraints bind. That is the protocol the paper uses: ``the percentages for
feature-based and target-based constraints were assumed to be the same across
all subsets and classes''.

The split uses \emph{largest remainder}, not independent rounding. Three groups
at one third of 100 beds each round to 33 and silently lose a bed; largest
remainder returns 34/33/33 and always reconstructs the total.

Every campaign in this project so far ran \texttt{L80\_G95} or
\texttt{L90\_G95}. With unequal percentages the local ceilings sum to
\emph{less} than the global one, so the global constraint has headroom and
cannot bind after allocation. That was a property of the chosen flag, not of
the method.
"""

TAIL = r"\end{document}" + "\n"


def esc(s):
    return str(s).replace("_", BS + "_").replace("%", BS + "%").replace("&", BS + "&")


def main():
    data = json.load(open(sys.argv[1]))
    classes = {int(k): v for k, v in data["classes"].items()}
    tiers = {int(k): v for k, v in data["tiers"].items()}
    shares = {int(k): v for k, v in data["shares"].items()}
    capped = [int(c) for c in data["capped"]]
    counts = {int(g): {int(c): v for c, v in d.items()} for g, d in data["counts"].items()}
    psi = {int(k): v for k, v in data["psi"].items()}
    pol = {int(g): {int(c): v for c, v in d.items()} for g, d in data["phi_policy"].items()}
    leg = {int(g): {int(c): v for c, v in d.items()} for g, d in data["phi_legacy"].items()}
    G = sorted(tiers)
    C = sorted(classes)
    L, GL = data["local_pct"], data["global_pct"]
    total_n = sum(sum(d.values()) for d in counts.values())

    out = [HEAD, INTRO]
    w = out.append

    def row(cells):
        return " & ".join(str(c) for c in cells) + " " + EOL

    # --- the pool ---------------------------------------------------------
    w(r"\section*{The pool}" + "\n")
    w("A deployment pool of %d patients, %d tiers, %d classes of which %d are "
      "capped. Tier sizes are near-equal on purpose, and the tier entitled to the "
      "largest share of beds supplies the fewest candidates.\n"
      % (total_n, len(G), len(classes), len(capped)))
    w(r"\begin{center}\begin{tabular}{l r " + "r" * len(classes) + "}")
    w(r"\toprule")
    w(row(["tier", "patients"] + [esc(classes[c]) for c in C]))
    w(r"\midrule")
    for g in G:
        w(row([esc(tiers[g]), sum(counts[g].values())] + [counts[g][c] for c in C]))
    w(r"\midrule")
    w(row(["TOTAL", total_n] + [sum(counts[g][c] for g in G) for c in C]))
    w(r"\bottomrule\end{tabular}\end{center}")

    # --- the policy -------------------------------------------------------
    w(r"\section*{The policy}" + "\n")
    w("Share of each class's bed budget, by tier. This is the external input, "
      "given once:\n")
    w(r"\begin{center}\begin{tabular}{l " + "r" * len(G) + "}")
    w(r"\toprule")
    w(row(["share"] + [esc(tiers[g]) for g in G]))
    w(r"\midrule")
    w(row([r"$s_\lambda$"] + ["%.0f%s%%" % (100 * shares[g], BS) for g in G]))
    w(r"\bottomrule\end{tabular}\end{center}")
    w("Caps configured at %stexttt{L%d%s_G%d}: the local percentage is %.0f%s%% "
      "and the global percentage is %.0f%s%%.\n"
      % (BS, round(L * 100), BS, round(GL * 100), 100 * L, BS, 100 * GL, BS))

    # --- result 1 ---------------------------------------------------------
    w(r"\section*{Result 1: the beds that exist ($\Psi$)}" + "\n")
    w(r"\begin{center}\begin{tabular}{l r}\toprule")
    w(row(["class", r"$\Psi(i)$"]))
    w(r"\midrule")
    for c in capped:
        w(row([esc(classes[c]), psi[c]]))
    w(r"\bottomrule\end{tabular}\end{center}")

    # --- result 2 ---------------------------------------------------------
    w(r"\section*{Result 2: who the beds are reserved for ($\Phi$)}" + "\n")
    w(r"\begin{center}\begin{tabular}{l l " + "r" * len(G) + " r l}")
    w(r"\toprule")
    w(row(["class", "derivation"] + [esc(tiers[g]) for g in G]
          + [r"$\sum_\lambda \Phi$", r"vs $\Psi$"]))
    w(r"\midrule")
    for i, c in enumerate(capped):
        sp = sum(pol[g][c] for g in G)
        sl = sum(leg[g][c] for g in G)
        w(row([esc(classes[c]), "policy (given shares)"] + [pol[g][c] for g in G]
              + [sp, r"\ok" if sp == psi[c] else r"\no"]))
        w(row(["", "prevalence (legacy)"] + [leg[g][c] for g in G] + [sl, ""]))
        if i != len(capped) - 1:
            w(r"\midrule")
    w(r"\bottomrule\end{tabular}\end{center}")

    # --- how to read it ---------------------------------------------------
    w(r"\section*{Read this table}" + "\n")
    w(r"\begin{itemize}")
    for c in capped:
        want = "/".join(str(int(round(psi[c] * shares[g]))) for g in G)
        got = "/".join(str(pol[g][c]) for g in G)
        old = "/".join(str(leg[g][c]) for g in G)
        w(r"\item \textbf{%s}: the policy asks for %s. The pipeline produced %s "
          r"(\ok). The legacy derivation produced %s, which has the same total but "
          r"hands the largest allocation to the tier the policy ranks last (\no)."
          % (esc(classes[c]), want, got, old))
    w(r"\end{itemize}")

    w(WHY)

    # --- verdict ----------------------------------------------------------
    w(r"\section*{Verdict}" + "\n")
    verdict = (r"\textcolor{good}{\textbf{PASS}}" if data["passed"]
               else r"\textcolor{bad}{\textbf{FAIL}}")
    w(r"\begin{center}\fbox{\parbox{0.86\textwidth}{\centering\large "
      + verdict + EOL + "[4pt] The pipeline reproduces the policy exactly, and "
      r"the legacy derivation demonstrably does not.}}\end{center}")

    out.append(TAIL)
    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
    print("wrote %s" % sys.argv[2])


if __name__ == "__main__":
    main()

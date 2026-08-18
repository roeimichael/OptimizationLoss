# warm-up-50 headline tables — ARCHIVED 2026-08-19, history not results

Moved out of `docs/paper/tables/A_headline/` because every file here fails at
least three of the project's own rules. None is `\input{}` by any `.tex`, so
nothing in the manuscript changed.

**What is wrong with all of them:**

* **They headline `flips`.** `stats_scoreboard.md` bolds `Flips | 5 | 0 | 0` per
  dataset; `win_matrix.csv` has 15 `Flips` rows all marked WIN; `scoreboard.csv`
  has 3 more. Post-hoc filling to the boundary is FREE, so "fewer flips"
  measures how much post-hoc surgery an arm needed, not how good its
  predictions were. Rejected roughly ten times. `full_panel.py` now refuses to
  render it as a verdict at all.
* **They pool 220 / 60 / 20 seed-pairs across cap levels and backbones.** The
  atomic cell is (dataset, backbone, cap, method) and averaging is over SEED
  only. `stats_scoreboard.md`'s `TraLO WIN vs Heuristic, F1 +0.0166, p=0.0009`
  at n=12 seed-pairs is exactly the pseudoreplication `full_panel.py` now
  rejects, and the underlying claim is separately retracted in FRAMEWORK
  section 2d.
* **They score `aider`**, which is out of scope, and whose loader was deleted.
* **They score `TraLO-bounded`**, which no longer exists: the optimizer reset
  and the undershoot hinge it ablates are both deleted, so under the current
  protocol it would be a bit-identical duplicate of `tralo`.
* **They use a paired bootstrap** where the protocol mandates Wilcoxon on cells.
* **They come from the warm-up-50 regime**, where CE has saturated and every
  method converges to the same thing (FRAMEWORK section 1).

**They cannot be regenerated.** No generator exists for any of them, and since
the aider loader was deleted they are unregenerable in principle, not just in
practice.

The paper body's own treatment of `flips` is more careful and is untouched: it
calls flips a deployment *property* rather than a headline metric, and carries
an explicit caveat that TraLO can record MORE flips at loose caps.

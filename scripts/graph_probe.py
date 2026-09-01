"""Does the TEST-SET GEOMETRY carry ordering information the allocator cannot see?

WHY THIS EXISTS. Six independent measurements now say training-time count
pressure cannot win, and they agree on the mechanism:

  * the penalty's gradient is NON-MONOTONE in the violation -- a scope 8x over
    budget receives 167x less pull than one 58% over (FRAMEWORK 2a2);
  * under `constraint_grad_mode: normalize` the DELIVERED step is exactly
    `lr * clip` every epoch, so its magnitude carries no violation information
    either (`src/training/constraint_step.py`, verified 2026-08-22:
    `clip_grad_norm_` caps, then anything below is scaled back UP to `clip`);
  * the count never approaches the budget anyway -- 2.5-3.5x over for the whole
    run (FRAMEWORK 2b-post);
  * post-hoc top-K allocation then fixes the count EXACTLY and is provably
    optimal for expected TP given the probabilities;
  * so the only thing training can win is the ORDERING -- and the constraint
    re-ranks about a third of the selected set while a pure RNG RESEED at zero
    dose re-ranks exactly as much (2026-08-22: the top-K set moved 12-20 of
    41-44 items for every treated arm, 15-20 of 41-44 for `tralo_reseed`).

So an ordering gain has to come from information the allocator does not have.
The allocator consumes a 2014x7 SCORE matrix. It has never seen the 2014x960
GEOMETRY. That asymmetry is the entire hypothesis this file tests.

THE SETTING MAKES THIS LEGITIMATE, and that is the point worth stating. The cap
is a TRANSDUCTIVE statement about the test set -- we are told how many positives
it contains -- so the test features are already inside the problem definition.
Diffusion here uses NO labels: only the test features and the model's own
probabilities. And the cap makes it unusually safe. A smoother that improves the
ordering but drifts in calibration is normally dangerous; here the allocator
re-imposes the exact budget afterwards, so ONLY the ordering survives to the
metric.

WHAT IS AND IS NOT NEW. Graph diffusion of scores is classical (Zhou 2004,
Zhu and Ghahramani 2002) and is not claimed here. What is new is the
composition: diffusing inside a transductive COUNT-CONSTRAINED problem, where an
optimal allocator re-imposes the budget afterwards and therefore strips
everything except the ordering. The classical method is the instrument, not the
result.

BUDGET. One documented default per knob, no search -- `k=10`, `alpha=0.5`.
A family that needs a search to clear a screen has failed the screen, and a
search here would be tuning on the very test set the endpoint is computed on.

TWO NEGATIVE CONTROLS, both reported every run, because a gain from
re-normalisation alone would look identical to a gain from geometry:

  C1  a degree-preserving SHUFFLED graph -- same k, same row count, neighbours
      drawn at random. Diffusion over noise must not help.
  C2  SHUFFLED FEATURES -- the real graph construction applied to a permuted
      feature matrix, which breaks the correspondence between an item and its
      own embedding while preserving the feature distribution exactly.

If either control moves the endpoint, the effect is not geometry and the run
says so.

    python -m scripts.graph_probe <run-dir> [<run-dir> ...]
    python -m scripts.graph_probe --campaign <root>
"""
import argparse
import glob
import io
import os

import numpy as np

from scripts.frozen_head_probe import (allocate, budgets, cc_f1, items_per_001,
                                       load_real, paired, seeds_needed)

K_DEFAULT = 10
ALPHA_DEFAULT = 0.5


def knn_affinity(feats, k, rng=None, shuffle_neighbours=False):
    """Symmetric cosine kNN affinity, self-loops removed.

    Cosine because the head that consumes these features is linear, so its
    decision geometry is angular; the raw norms vary several-fold across items
    and would otherwise weight the graph by nothing more than activation scale.
    """
    X = np.asarray(feats, np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)                 # never a neighbour of itself
    n = len(S)
    k = int(min(k, n - 1))
    if shuffle_neighbours:
        # C1: keep k neighbours per row, choose WHICH at random. Degree and
        # sparsity are preserved exactly; only the geometry is destroyed.
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.stack([rng.choice(n - 1, size=k, replace=False)
                        for _ in range(n)])
        idx = idx + (idx >= np.arange(n)[:, None])   # skip self without bias
    else:
        idx = np.argpartition(-S, k, axis=1)[:, :k]
    W = np.zeros((n, n), np.float32)
    rows = np.repeat(np.arange(n), k)
    W[rows, idx.ravel()] = 1.0
    W = np.maximum(W, W.T)                       # symmetrise
    return W


def diffuse(P, W, alpha):
    """Zhou-style propagation, solved exactly rather than iterated.

    F = (1-a) (I - a D^-1/2 W D^-1/2)^-1 P, then renormalised to a
    distribution. n is only a couple of thousand here, so the solve is exact
    and there is no iteration count to tune -- one fewer knob that could be
    searched.
    """
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    S = (W * dinv[:, None] * dinv[None, :]).astype(np.float64)
    F = (1.0 - alpha) * np.linalg.solve(
        np.eye(len(W), dtype=np.float64) - alpha * S,
        np.asarray(P, np.float64))
    F = np.clip(F, 0.0, None)
    return F / np.clip(F.sum(axis=1, keepdims=True), 1e-12, None)


def score(run_dir, k, alpha):
    """Baseline and each variant, through the run's OWN endpoint, in items."""
    d = load_real(run_dir)
    y, g, classes = d.y, d.groups, d.classes
    G, L = budgets(y, g, classes, d.local_pct, d.global_pct, d.n_classes)
    P = d.ref_probs

    def endpoint(Q):
        alloc = allocate(Q, g, G, L, classes)
        return cc_f1(y, alloc, classes), items_per_001(y, alloc, classes)

    base_f1, scale = endpoint(P)
    rng = np.random.default_rng(0)
    variants = {
        "diffused": lambda: diffuse(P, knn_affinity(d.features, k), alpha),
        "C1_shuffled_graph": lambda: diffuse(
            P, knn_affinity(d.features, k, rng, shuffle_neighbours=True), alpha),
        "C2_shuffled_features": lambda: diffuse(
            P, knn_affinity(d.features[rng.permutation(len(d.features))], k),
            alpha),
    }
    out = {}
    for name, build in variants.items():
        f1, _ = endpoint(build())
        out[name] = (f1 - base_f1) / 0.01 * scale       # items
    out["_base_ccF1"] = base_f1
    out["_items_per_001"] = scale
    return out



def _cell_of(run_dir):
    """(backbone, dataset, cap) for a run, from its path.

    <root>/<Backbone>/<dataset>/<cap>/<arm>/<seed>. Returns None when the path
    is too shallow to say, which is honest: an unknown cell must not silently
    join a known one.
    """
    parts = os.path.normpath(os.path.abspath(run_dir)).split(os.sep)
    return tuple(parts[-5:-2]) if len(parts) >= 5 else None


def _per_cell_report(names, rows, keys):
    """RULE 4: never pool across backbones, cap levels or datasets.

    The pooled block above keys on the REGIME NAME only, so a `--campaign`
    spanning three backbones and two cap levels produced ONE line per regime
    and ran a sign test over it. That is the aggregation this project has
    retracted a result over three times, and a direction-closing verdict was
    published off it. The pooled line stays so the published number remains
    reproducible; this block is what says whether it was legal.
    """
    cells = {}
    for i, nm in enumerate(names):
        cells.setdefault(_cell_of(nm) or ("?", "?", "?"), []).append(i)
    if len(cells) <= 1:
        print("")
        print("  ONE CELL (%s), so the pooled block above is a legal aggregate."
              % ("/".join(sorted(cells)[0]) if cells else "none"))
        return cells
    print("")
    print("  *** THE BLOCK ABOVE POOLS %d CELLS, AND RULE 4 FORBIDS THAT."
          % len(cells))
    print("      A backbone or a cap level is not a replicate: the")
    print("      unconstrained count, the ranking quality and K all move with")
    print("      both. Count CELLS, never runs.")
    try:
        from scripts.frozen_head_probe import seeds_needed
    except Exception:
        seeds_needed = None
    print("  %-30s %4s %s %8s %6s %7s"
          % ("cell", "n", "  ".join("%12s" % k[:12] for k in keys),
             "sd", "sign", "seeds"))
    for c in sorted(cells):
        idx = cells[c]
        v0 = [rows[keys[0]][i] for i in idx]
        m0 = sum(v0) / float(len(v0))
        sd0 = (sum((x - m0) ** 2 for x in v0) / max(1, len(v0) - 1)) ** 0.5
        pos = sum(1 for x in v0 if x > 0)
        need = ("%7s" % (seeds_needed(m0, sd0)
                         if seeds_needed and m0 > 0 and sd0 > 0 else "-"))
        vals = ["%+12.2f" % (sum(rows[k][i] for i in idx) / float(len(idx)))
                for k in keys]
        print("  %-30s %4d %s %8.2f %3d/%-2d %s"
              % ("/".join(c)[-30:], len(idx), "  ".join(vals), sd0, pos,
                 len(idx), need))
    n_pos = sum(1 for c in cells
                if sum(rows[keys[0]][i] for i in cells[c]) > 0)
    print("      CELL sign test on `%s`: %d of %d positive. That is the sample"
          % (keys[0], n_pos, len(cells)))
    print("      size, not %d run(s), and `seeds` is per cell at 80%% power."
          % len(names))
    return cells


KEYS = ("diffused", "C1_shuffled_graph", "C2_shuffled_features")


def write_dump(path, names, rows):
    """One row per RUN, with the arm split out of the path.

    \U0001f6d1 THIS FUNCTION EXISTS BECAUSE `--dump` WAS DECLARED AND NEVER
    READ -- it parsed, the probe ran to completion, and no file appeared. That
    is the inert-flag failure mode CLAUDE.md rule 3 counts, and this was the
    fifth occurrence. It prints what it wrote so the next reader does not have
    to trust it.
    """
    import csv
    cols = ["run", "backbone", "dataset", "cap", "arm", "seed"] + list(KEYS)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, nm in enumerate(names):
            parts = os.path.normpath(nm).split(os.sep)
            cell = _cell_of(nm) or ("?", "?", "?")
            arm = parts[-2] if len(parts) >= 2 else "?"
            seed = parts[-1] if parts else "?"
            w.writerow([nm] + list(cell) + [arm, seed] +
                       ["%.6f" % rows[k][i] for k in KEYS])
    print("  wrote %d row(s) x %d column(s) to %s"
          % (len(names), len(cols), path))
    print("  THE QUESTION IT IS FOR: group by `arm`. If every arm gains the "
          "same,\n  the geometry raises the BASELINE and no arm-vs-arm delta "
          "moves.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--campaign")
    ap.add_argument("-k", type=int, default=K_DEFAULT)
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--dump", help="write the per-run rows to this CSV, "
                                   "so the per-ARM question can be asked "
                                   "without recomputing the diffusion. "
                                   "The probe is paired against each run's "
                                   "OWN undiffused scores, so a gain here may "
                                   "be available to EVERY arm -- which would "
                                   "raise the baseline and change no "
                                   "arm-vs-arm delta. That question needs "
                                   "these rows")
    args = ap.parse_args()

    runs = list(args.runs)
    if args.campaign:
        runs += [os.path.dirname(p) for p in sorted(glob.glob(
            os.path.join(args.campaign, "**", "test_embeddings.npz"),
            recursive=True))]
    if not runs:
        raise SystemExit("no run directories given")

    print("GRAPH PROBE -- k=%d, alpha=%.2f, no search, %d run(s)"
          % (args.k, args.alpha, len(runs)))
    print("Diffusion reads the test FEATURES and the model's own probabilities.")
    print("It reads no labels. The allocator re-imposes the exact budget after,")
    print("so only a change in ORDERING can reach the number below.\n")

    rows, names = {}, []
    for r in runs:
        try:
            s = score(r, args.k, args.alpha)
        except SystemExit as exc:
            print("  skipped %s: %s" % (r, exc))
            continue
        names.append(os.path.relpath(r, args.campaign or os.path.dirname(r)))
        for key, v in s.items():
            rows.setdefault(key, []).append(v)

    if not names:
        raise SystemExit("no probeable runs")

    print("  %-46s %10s %10s %10s"
          % ("run", "diffused", "C1 graph", "C2 feats"))
    for i, nm in enumerate(names):
        print("  %-46s %+10.2f %+10.2f %+10.2f"
              % (nm[-46:], rows["diffused"][i], rows["C1_shuffled_graph"][i],
                 rows["C2_shuffled_features"][i]))

    print("\n  PAIRED OVER %d RUN(S), each against its OWN undiffused scores, "
          "in items" % len(names))
    print("  %-24s %9s %7s %7s %8s %9s"
          % ("variant", "d items", "sd", "sem", "sign", "sign p"))
    for key in ("diffused", "C1_shuffled_graph", "C2_shuffled_features"):
        st = paired(np.asarray(rows[key]))
        print("  %-24s %+9.2f %7.2f %7.2f %5d/%-2d %9.4f"
              % (key, st["mean"], st["sd"], st["sem"], st["pos"], st["n"],
                 st["sign_p"]))
    st = paired(np.asarray(rows["diffused"]))
    if st["mean"] > 0 and np.isfinite(st["sd"]) and st["sd"] > 0:
        print("\n  a GPU campaign would need ~%d seeds per cell to see this "
              "effect, vs the standard 4" % seeds_needed(st["mean"], st["sd"]))
    _per_cell_report(names, rows,
                     ("diffused", "C1_shuffled_graph",
                      "C2_shuffled_features"))
    print("\n  READ THE CONTROLS FIRST. If C1 or C2 moved, the effect is "
          "re-normalisation,\n  not geometry, and the `diffused` column means "
          "nothing.")
    if args.dump:
        write_dump(args.dump, names, rows)


if __name__ == "__main__":
    main()

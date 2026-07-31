"""Shared cc-F1 / macro-F1 data loaders for the Track-B B5/B6/B7 *_v2 analyses.

WHY THIS EXISTS
---------------
The handoff (paper/HANDOFF_TRACK_B.tex, sec. B5/B6/B7) is *constrained-class
F1* (cc-F1) centric, but the historical corpus ``docs/all_cells_raw.csv`` only
stores macro-F1 (``f1m``) + flips and NOT cc-F1.  cc-F1 is therefore
reconstructed here directly from the frozen per-sample predictions
(``final_predictions.csv``) using the SAME definition the paper uses
(``scripts/trackb_finalize.py::_cc_f1``):

    cc-F1(c) = 2*TP / (2*TP + FP + FN)   for the constrained class c
    macro-F1  = unweighted mean of per-class F1 over classes present.

Verified: the reconstructed per-seed-best cc-F1 gaps reproduce
``paper/tables/tab_oct_backbone.tex`` to the printed precision (e.g. ViT-B/16
L30 = +0.081, MobileNetV3 L30 = +0.016).

THREE SOURCES (kept explicitly separate -- never silently mixed):
  * LOCAL frozen predictions  -> true cc-F1 AND macro-F1, paper backbones
        (MobileNetV3 / RegNetY400MF / ViTB16).  Coverage is a SUBSET of the
        server ``paper_final`` tree: octmnist L10-L40, dermmnist L30+L50,
        tissuemnist L30.  This is the ONLY cc-F1 source available offline.
  * CORPUS all_cells_raw.csv  -> macro-F1 (``f1m``) + flips ONLY, and its
        backbones are MobileNetV3 / EfficientNetB0 / ResNet18 -- so it overlaps
        the paper grid on MobileNetV3 ONLY.  Used strictly as a *labeled*
        macro-F1 fallback.
  * GRAFT review_graft_2026-07.csv -> per-seed cc_f1 for the ablation/graft
        campaign (6 methods x 3 backbones x {L30,L40} x 4 seeds).  Feeds B7-c.

Constrained class per dataset (from the run configs): octmnist=2, dermmnist=4
(MEL), tissuemnist=4, aider=0.

Stdlib only (csv/glob/random/statistics) -- no numpy/pandas, matches the other
b5/b6/b7 scripts and runs in the base interpreter.
"""
import csv
import glob
import os
from statistics import mean

# ----------------------------------------------------------------------------
# Default paths.  EVIDENCE_ROOT points at the frozen-prediction tree that was
# pulled into this session's scratchpad; override with --evidence to point at
# the real server paper_final tree for a full-coverage rerun.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVIDENCE_ROOT_DEFAULT = ("C:/Users/roeym/AppData/Local/Temp/claude/"
                         "C--Users-roeym-Desktop-projects-OptimizationLoss/"
                         "2790751c-acab-4eaf-8454-d3d46c4b0668/scratchpad/evidence/"
                         "results/pending_runs/paper_final")
CORPUS_DEFAULT = os.path.join(ROOT, "docs", "all_cells_raw.csv")
GRAFT_DEFAULT = os.path.join(ROOT, "archive", "legacy", "final_AAAI_PAPER",
                             "data", "corpus", "review_graft_2026-07.csv")

CONSTRAINED = {"octmnist": 2, "dermmnist": 4, "tissuemnist": 4, "aider": 0}

# Canonical symmetric-grid constraint config per dataset in the corpus
# (constrained class + group column) -- used to pin the corpus fallback to the
# same slice the paper grid reports.
CORPUS_CANON = {"tissuemnist": ("4", "synth_group"),
                "dermmnist": ("4", "loc_group"),
                "aider": ("0", "synth_group")}

PRETTY_METHOD = {"fioretto_ldf": "Fioretto-LDF", "hounie_rcl": "Hounie-RCL",
                 "tralo_bounded": "TraLO-bounded", "danits_lp": "LP-LG",
                 "heuristic": "Heuristic", "tralo": "TraLO",
                 "fioretto_rh": "Fioretto+RH", "fioretto_restart": "Fioretto+restart",
                 "hounie_rh": "Hounie+RH"}

# comparator groups
TRAINED_DUAL = ["fioretto_ldf", "hounie_rcl"]      # "best trained dual"
CLIPPERS = ["heuristic", "danits_lp"]              # "best clipper"


# ----------------------------------------------------------------------------
# metric primitives
def f1_of_class(yt, yp, c):
    tp = sum(1 for a, b in zip(yt, yp) if b == c and a == c)
    fp = sum(1 for a, b in zip(yt, yp) if b == c and a != c)
    fn = sum(1 for a, b in zip(yt, yp) if b != c and a == c)
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else 0.0


def macro_f1(yt, yp):
    classes = sorted(set(yt) | set(yp))
    return mean(f1_of_class(yt, yp, c) for c in classes)


def _read_pred(path):
    yt, yp = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            yt.append(int(r["True_Label"]))
            yp.append(int(r["Predicted_Label"]))
    return yt, yp


# ----------------------------------------------------------------------------
# LOCAL frozen predictions  ->  cc-F1 + macro-F1
def load_local(evidence_root):
    """Return dict key (ds, bb, cap, method, seed) -> {'cc_f1', 'macro_f1'}.

    Parses .../lane*/{bb}/{ds}/{cap}/{method}/seed_{s}/final_predictions.csv.
    """
    out = {}
    pat = os.path.join(evidence_root, "**", "final_predictions.csv")
    for path in glob.glob(pat, recursive=True):
        p = path.replace("\\", "/").split("/")
        try:
            seed = int(p[-2].split("_")[1])
            method, cap, ds, bb = p[-3], p[-4], p[-5], p[-6]
        except (IndexError, ValueError):
            continue
        if ds not in CONSTRAINED:
            continue
        yt, yp = _read_pred(path)
        out[(ds, bb, cap, method, seed)] = {
            "cc_f1": f1_of_class(yt, yp, CONSTRAINED[ds]),
            "macro_f1": macro_f1(yt, yp),
        }
    return out


def local_cells(local):
    """Sorted unique (ds, bb, cap) present in the local map."""
    return sorted({(ds, bb, cap) for (ds, bb, cap, _m, _s) in local})


def local_methods(local, ds, bb, cap):
    return sorted({m for (d, b, c, m, _s) in local if (d, b, c) == (ds, bb, cap)})


def local_paired(local, ds, bb, cap, comparator_methods, metric, seeds=(1, 2, 3, 4)):
    """Matched-seed paired diffs TraLO - best(comparator) within one local cell.

    positive = TraLO better (higher cc/macro F1).  Returns [] if TraLO or the
    comparator set is not fully present.  ``best`` is per-seed max over the
    comparator methods (the paper's "best trained dual" / "best clipper").
    """
    diffs = []
    for s in seeds:
        t = local.get((ds, bb, cap, "tralo", s))
        if t is None:
            continue
        comps = [local[(ds, bb, cap, m, s)][metric]
                 for m in comparator_methods if (ds, bb, cap, m, s) in local]
        if not comps:
            continue
        diffs.append(t[metric] - max(comps))
    return diffs


def local_cell_mean(local, ds, bb, cap, method, metric, seeds=(1, 2, 3, 4)):
    vs = [local[(ds, bb, cap, method, s)][metric]
          for s in seeds if (ds, bb, cap, method, s) in local]
    return mean(vs) if vs else None


# ----------------------------------------------------------------------------
# CORPUS macro-F1 fallback (backbones differ from paper except MobileNetV3)
def _fnum(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def load_corpus(corpus_path):
    """key (ds, model, cls, grp, tight, seed, method) -> {'f1m', 'flips'}."""
    d = {}
    with open(corpus_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["ds"] == "eurosat":
                continue
            key = (r["ds"], r["model"], r["cls"], r["grp"], r["tight"],
                   r["seed"], r["method"])
            d[key] = {"f1m": _fnum(r["f1m"]), "flips": _fnum(r["flips"])}
    return d


def corpus_paired(corpus, ds, model, tight, comparator_methods,
                  seeds=("1", "2", "3", "4")):
    """Matched-seed paired macro-F1 diffs TraLO - best(comparator) in a corpus
    cell, pinned to the canonical symmetric constraint config for the dataset.
    positive = TraLO better."""
    cls, grp = CORPUS_CANON[ds]
    diffs = []
    for s in seeds:
        t = corpus.get((ds, model, cls, grp, tight, s, "tralo"))
        if t is None or t["f1m"] is None:
            continue
        comps = []
        for m in comparator_methods:
            r = corpus.get((ds, model, cls, grp, tight, s, m))
            if r and r["f1m"] is not None:
                comps.append(r["f1m"])
        if not comps:
            continue
        diffs.append(t["f1m"] - max(comps))
    return diffs


# ----------------------------------------------------------------------------
# GRAFT campaign per-seed cc_f1 (feeds B7-c)
def load_graft(graft_path):
    """key (model, tag, seed, method) -> {'cc_f1', 'macro_f1', 'flips'}."""
    d = {}
    with open(graft_path, newline="") as f:
        for r in csv.DictReader(f):
            d[(r["model"], r["tag"], r["seed"], r["method"])] = {
                "cc_f1": _fnum(r["cc_f1"]),
                "macro_f1": _fnum(r["macro_f1"]),
                "flips": _fnum(r["flips"]),
            }
    return d


# ----------------------------------------------------------------------------
# shared bootstrap machinery (identical definition to make_winning_results.py)
def boot_p(diffs, rng, B):
    """Two-sided paired percentile bootstrap p on the mean of diffs."""
    if len(diffs) < 2:
        return 1.0
    n = len(diffs)
    cnt = sum(1 for _ in range(B)
              if mean(rng.choice(diffs) for _ in range(n)) <= 0)
    return 2 * min(cnt, B - cnt) / B


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def boot_ci_mean(vals, rng, B, ci):
    """Percentile-bootstrap CI on the MEAN of vals -> (mean, lo, hi)."""
    m = mean(vals)
    n = len(vals)
    if n < 2:
        return m, float("nan"), float("nan")
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randint(0, n - 1)]
        means.append(s / n)
    means.sort()
    a = (1.0 - ci) / 2.0
    return m, _percentile(means, a), _percentile(means, 1.0 - a)

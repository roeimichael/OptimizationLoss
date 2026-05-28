"""Generic per-phase head-to-head analyzer.

Usage:
    python -m src.evaluation.phase_summary <sweep_dir> [<sweep_dir> ...]

For each sweep dir, loads all completed cells, applies the canonical-TraLO
filter, aggregates over seeds, and for every (model, cls, group, tight)
"cell" reports:
  - F1-macro winner (TraLO vs best of the 5 baselines)
  - TraLO mean flips vs best-baseline flips
  - in-training satisfaction rate
Then prints a verdict: how many cells TraLO wins / ties / loses on F1m,
and the flip-dominance count.
"""
import csv, glob, json, os, sys
from collections import defaultdict
from statistics import mean, stdev

METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
BASELINES = [m for m in METHODS if m != "tralo"]
TIGHT_ORDER = {f"L{l}_G{g}": (l, g) for l in (20, 30, 50, 70, 80)
               for g in (20, 30, 50, 70, 80)}


def load_metrics(path):
    out = {}
    try:
        with open(path) as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    out[r[0]] = r[1]
    except Exception:
        pass
    return out


def fnum(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def is_canonical_tralo(hp):
    return (hp.get("hybrid_mode") == "undershoot_hinge"
            and hp.get("reset_optimizer_at_sat") is True
            and hp.get("alpha_kl", 0.0) == 0.0
            and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
            and hp.get("penalty_mode") == "both"
            and hp.get("enable_ce_skip") is True)


def collect(sweep_dir):
    rows = []
    seen = set()
    for f in glob.glob(os.path.join(sweep_dir, "**", "config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        method = c.get("methodology")
        if method not in METHODS:
            continue
        hp = c.get("hyperparams", {})
        if method == "tralo" and not is_canonical_tralo(hp):
            continue
        seed = hp.get("seed")
        key = (c.get("model_name"),
               c.get("dataset_config", {}).get("constrained_class"),
               c.get("dataset_config", {}).get("group_column"),
               c.get("constraint_tag"), method, seed)
        if key in seen:
            continue
        seen.add(key)
        m = load_metrics(ev)
        rows.append({
            "model": key[0], "cls": key[1], "grp": key[2], "tight": key[3],
            "method": method, "seed": seed,
            "f1m": fnum(m.get("F1 (Macro)")),
            "flips": fnum(m.get("Flips Required")),
            "sat": fnum(m.get("Raw All Satisfied")),
            "acc": fnum(m.get("Accuracy")),
        })
    return rows


def agg_cells(rows):
    """(model,cls,grp,tight,method) -> {metric: mean}."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["cls"], r["grp"], r["tight"], r["method"])
        for met in ("f1m", "flips", "sat", "acc"):
            if r[met] is not None:
                groups[k][met].append(r[met])
    out = {}
    for k, d in groups.items():
        out[k] = {met: (mean(v) if v else None) for met, v in d.items()}
        out[k]["n"] = len(d.get("f1m", []))
    return out


def analyze(sweep_dir):
    rows = collect(sweep_dir)
    if not rows:
        print(f"\n### {sweep_dir}: NO completed cells found.")
        return
    cells = agg_cells(rows)
    # group cells by the comparison axis = (model, cls, grp, tight)
    axes = sorted({(k[0], k[1], k[2], k[3]) for k in cells})
    name = os.path.basename(sweep_dir.rstrip("/"))
    print(f"\n{'='*78}\n### {name}  ({len(rows)} cells, {len(axes)} (model,cls,grp,tight) comparison points)\n{'='*78}")

    f1_win = f1_tie = f1_loss = 0
    flip_win = flip_tie = flip_loss = 0
    detail = []
    for ax in axes:
        model, cls, grp, tight = ax
        tr = cells.get((*ax, "tralo"))
        if not tr or tr.get("f1m") is None:
            continue
        # best baseline F1m
        base_f1 = [(cells[(*ax, b)]["f1m"], b) for b in BASELINES
                   if (*ax, b) in cells and cells[(*ax, b)].get("f1m") is not None]
        if not base_f1:
            continue
        best_f1, best_f1_m = max(base_f1)
        gap = tr["f1m"] - best_f1
        if gap > 0.002:
            f1_win += 1; verdict = "WIN"
        elif gap < -0.002:
            f1_loss += 1; verdict = "LOSS"
        else:
            f1_tie += 1; verdict = "tie"
        # flips
        base_fl = [(cells[(*ax, b)].get("flips"), b) for b in BASELINES
                   if (*ax, b) in cells and cells[(*ax, b)].get("flips") is not None]
        tr_fl = tr.get("flips")
        flip_str = ""
        if base_fl and tr_fl is not None:
            best_fl, _ = min(base_fl)
            if tr_fl < best_fl - 0.5:
                flip_win += 1
            elif tr_fl > best_fl + 0.5:
                flip_loss += 1
            else:
                flip_tie += 1
            flip_str = f" flips tralo={tr_fl:.0f} vs best-base={best_fl:.0f}"
        detail.append(
            f"  {model:>14} cls{cls} {grp:>11} {tight:>8} | "
            f"F1m tralo={tr['f1m']:.3f} best-base={best_f1:.3f}({best_f1_m[:6]}) "
            f"gap={gap:+.3f} [{verdict}]{flip_str}")

    for d in detail:
        print(d)
    n = f1_win + f1_tie + f1_loss
    print(f"\n  --- VERDICT ({n} comparison points) ---")
    print(f"  F1-macro: TraLO WINS {f1_win}, TIES {f1_tie}, LOSES {f1_loss}  "
          f"(win/tie = {100*(f1_win+f1_tie)/n:.0f}% not-losing)")
    print(f"  Flips:    TraLO fewer {flip_win}, equal {flip_tie}, more {flip_loss}")


def main():
    if len(sys.argv) < 2:
        print("usage: python -m src.evaluation.phase_summary <sweep_dir> ...")
        sys.exit(1)
    for sweep_dir in sys.argv[1:]:
        analyze(sweep_dir)


if __name__ == "__main__":
    main()

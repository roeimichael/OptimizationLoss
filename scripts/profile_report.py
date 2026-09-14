"""The multi-metric profile. cc-F1 is ONE column, never the verdict on its own.

The acceptance bar was single-metric in the code: `deployed_h2h` computed the
paired effect on cc_f1 and nothing else, so every verdict this project issued
was cc-F1 by construction. This reports the whole profile and classifies each
cell as WIN / TRADE / LOSS:

  WIN    tralo leads or ties the best arm on cc-F1, AND is not below `clip` on
         accuracy, macro-F1 or collateral F1 by more than the seed noise
  TRADE  cc-F1 up, but at least one damage metric down beyond the noise. This
         is a real outcome, not a win, and it is named so it cannot be quoted
         as one
  LOSS   tralo trails on cc-F1 beyond the noise

"Beyond the noise" is the seed-paired sd of the DIFFERENCE, which is the noise
the contrast actually faces. Arm-vs-arm comparisons use only seeds both arms
ran, and the count is printed: comparing across different seed sets compares
populations.

The verdict is a summary of the table, never a replacement for it -- every
metric is printed for every arm whatever the verdict says.
"""
import csv, glob, json, os, sys, collections, math

QUALITY = ("accuracy", "macro_f1", "macro_p", "macro_r", "cc_f1", "collateral_f1")


def per_class(path, n_classes):
    tp = collections.Counter(); k = collections.Counter(); n = collections.Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            y = int(float(row["True_Label"])); p = int(float(row["Predicted_Label"]))
            n[y] += 1; k[p] += 1
            if y == p:
                tp[y] += 1
    return {c: dict(TP=tp[c], K=k[c], n=n[c]) for c in range(n_classes)}


def metrics(pc, capped):
    tot = sum(d["n"] for d in pc.values())
    cor = sum(d["TP"] for d in pc.values())
    f1 = {c: (2.0 * d["TP"] / (d["K"] + d["n"]) if (d["K"] + d["n"]) else 0.0) for c, d in pc.items()}
    pr = {c: (d["TP"] / d["K"] if d["K"] else 0.0) for c, d in pc.items()}
    rc = {c: (d["TP"] / d["n"] if d["n"] else 0.0) for c, d in pc.items()}
    other = [c for c in pc if c not in capped]
    mean = lambda v: (sum(v) / len(v)) if v else 0.0
    return dict(accuracy=cor / tot if tot else 0.0,
                macro_f1=mean(list(f1.values())), macro_p=mean(list(pr.values())),
                macro_r=mean(list(rc.values())),
                cc_f1=mean([f1[c] for c in capped if c in f1]),
                collateral_f1=mean([f1[c] for c in other]))


def paired(a, b, key):
    """mean and sd of (a - b) over the seeds BOTH arms ran."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None, None, 0
    d = [a[s][key] - b[s][key] for s in seeds]
    m = sum(d) / len(d)
    sd = (math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else None)
    return m, sd, len(seeds)



def classify(bestarm, cc_delta, cc_sd, deltas, have_clip):
    """The verdict line, as a pure function so it can be gated directly.

    Two overstatements this replaces, both seen on real fmow2 output:
    a NEGATIVE cc-F1 delta inside the noise printed as "WIN -- leads/ties", and
    "not dominated elsewhere" tested against `clip` ALONE, so an arm ahead of
    TraLO across the whole quality profile passed unnoticed unless it was clip.

    `deltas` is {arm: {metric: (mean_delta, sd)}}, TraLO minus that arm.
    """
    if cc_delta < -cc_sd:
        return "LOSS -- trails %s on cc-F1 by %.4f (sd %.4f)" % (bestarm, -cc_delta, cc_sd)
    dominators = []
    for a, per_metric in deltas.items():
        beats_tralo = any(dm < -sd for dm, sd in per_metric.values())
        tralo_ahead = any(dm > sd for dm, sd in per_metric.values())
        if beats_tralo and not tralo_ahead:
            dominators.append(a)
    if dominators:
        return ("DOMINATED -- cc-F1 holds vs %s (%+.4f), but dominated on the full "
                "profile by: %s" % (bestarm, cc_delta, ", ".join(sorted(dominators))))
    damage = ["%s %+0.4f" % (k, dm)
              for k, (dm, sd) in sorted(deltas.get("clip", {}).items())
              if k in ("accuracy", "macro_f1", "collateral_f1") and dm < -sd]
    if have_clip and damage:
        return "TRADE -- cc-F1 holds vs %s (%+.4f), but below clip on: %s" % (
            bestarm, cc_delta, ", ".join(damage))
    if cc_delta > cc_sd:
        return "WIN -- LEADS %s on cc-F1 by %+.4f beyond its sd %.4f, and is not dominated" % (
            bestarm, cc_delta, cc_sd)
    # The bar is "leading GROUP", so a deficit inside the noise is not a loss --
    # but it is not a win either, and calling it one is how a tie became a
    # headline four times in this project.
    return "LEADING GROUP -- within noise of %s on cc-F1 (%+.4f, sd %.4f), not dominated" % (
        bestarm, cc_delta, cc_sd)


def main(roots):
    rows = []
    for root in roots:
        for cj in glob.glob(os.path.join(root, "*/*/*/*/*/config.json")):
            cfg = json.load(open(cj))
            if cfg.get("status") != "completed":
                continue
            fp = os.path.join(os.path.dirname(cj), "final_predictions.csv")
            if not os.path.exists(fp):
                continue
            capped = tuple(cfg["dataset_config"]["constrained_class"])
            pc = per_class(fp, cfg["dataset_config"]["num_classes"])
            rows.append(dict(cell=(cfg["dataset_mode"], cfg["model_name"], cfg["constraint_tag"]),
                             arm=cfg["arm"], seed=cfg["hyperparams"]["seed"],
                             capped=capped, **metrics(pc, capped)))
    if not rows:
        print("no completed runs"); return 0
    seen = set()
    for r in rows:
        key = (r["cell"], r["arm"], r["seed"])
        if key in seen:
            raise SystemExit("duplicate observation %s" % (key,))
        seen.add(key)
    cells = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        cells[r["cell"]][r["arm"]][r["seed"]] = r
    print("Averaged over SEED only. Arm-vs-arm uses only seeds BOTH arms ran.")
    print("A cc-F1 gain paid for in damage is a TRADE, not a win.")
    for cell in sorted(cells):
        arms = cells[cell]
        ns = {a: len(v) for a, v in arms.items()}
        print(""); print("=" * 100); print(" / ".join(cell), "  seeds per arm:", ns); print("")
        print("  %-12s %4s %9s %9s %9s %9s %9s %9s" %
              ("arm", "n", "acc", "macroF1", "macroP", "macroR", "ccF1", "collatF1"))
        mean = {a: {k: sum(r[k] for r in v.values()) / len(v) for k in QUALITY} for a, v in arms.items()}
        for a in sorted(arms):
            print("  %-12s %4d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" %
                  (a, ns[a], *[mean[a][k] for k in QUALITY]))
        if "tralo" not in arms:
            print("\n  no tralo in this cell -- the claim cannot be posed here"); continue
        print("")
        print("  TraLO minus each arm, on common seeds (sd = seed-paired sd of the difference):")
        print("  %-12s %5s %22s %22s %22s" % ("arm", "seeds", "d ccF1", "d macroF1", "d accuracy"))
        for a in sorted(set(arms) - {"tralo"}):
            cols = []
            for k in ("cc_f1", "macro_f1", "accuracy"):
                m, sd, n = paired(arms["tralo"], arms[a], k)
                cols.append("%+8.4f +- %-8s" % (m, ("%.4f" % sd) if sd is not None else "n/a"))
            print("  %-12s %5d %22s %22s %22s" % (a, n, *cols))
        # --- the profile verdict ---
        print("")
        best = {k: max(mean[a][k] for a in arms) for k in QUALITY}
        verdicts = []
        lead_gap, lead_sd, _ = None, None, None
        others = [a for a in arms if a != "tralo"]
        bestarm = max(others, key=lambda a: mean[a]["cc_f1"])
        m, sd, n = paired(arms["tralo"], arms[bestarm], "cc_f1")
        noise = sd if sd else 0.0
        deltas = {}
        for a in others:
            deltas[a] = {}
            for k in QUALITY:
                dm, dsd, _ = paired(arms["tralo"], arms[a], k)
                if dm is not None:
                    deltas[a][k] = (dm, dsd or 0.0)
        v = classify(bestarm, m, noise, deltas, "clip" in arms)
        print("  VERDICT: %s" % v)
        if n < 4:
            print("  (only %d common seeds -- this is a direction, not a measurement)" % n)
    return 0


def cli(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", nargs="+", required=True,
                    help="campaign roots" if "--campaign" == "--campaign" else "run-dir globs")
    args = ap.parse_args(argv)
    return main(getattr(args, "--campaign".lstrip("-").replace("-", "_")))


if __name__ == "__main__":
    sys.exit(cli())

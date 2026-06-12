"""Aggregate the tissue low-warmup validation sweep (AAAI headroom test).

Question: does TraLO's F1 advantage GROW at low warmup on TissueMNIST across
backbones? The headroom theory says TraLO gains over POST-HOC baselines
(heuristic, danits) precisely when warmup CE is NOT saturated -- i.e. at low
warmup -- because the constraint phase still has gradient signal to reshape
predictions rather than just clip them.

Reads:  results/pending_runs/tissue_lowwarm_validation/
          {backbone}/tissuemnist/L50_G50/w{W}/{method}/seed_{S}/evaluation_metrics.csv
Writes: scripts/_audit/lowwarm_validation_agg.csv  (tidy per-run)
        + prints per-backbone warmup x method tables and the key trend.

Run on the server (results live on NFS):
    python -m scripts.agg_lowwarm_validation
or point it elsewhere:
    python -m scripts.agg_lowwarm_validation --root <path>
"""
import argparse
import csv
import os
from collections import defaultdict

ROOT_DEFAULT = "results/pending_runs/tissue_lowwarm_validation"
POSTHOC = ["heuristic", "danits_lp"]
TRAINED = ["tralo_bounded"]          # only trained comparator left = no-hinge ablation
ALL_METHODS = ["tralo", "tralo_bounded", "danits_lp", "heuristic"]
WARMUPS = [1, 2, 3, 4, 5]

# warmup=50 reference: TraLO - bestPostHoc F1(macro) gap, per backbone.
# Matched to THIS sweep's tightness (tissue cls4 L50_G50 ONLY, clean,
# from saturation_audit_v2.csv; n=3-4 tralo seeds each). The bar the
# low-warmup gaps must EXCEED for the headroom hypothesis to hold.
# (note MobileNetV2 is slightly NEGATIVE at w50/L50_G50.)
W50_REF = {
    "MobileNetV3": 0.024, "MobileNetV2": -0.015,
    "RegNetY400MF": 0.018, "ResNet18": 0.029,
}
# metric keys in evaluation_metrics.csv we care about
KEYS = {
    "F1 (Macro)": "f1_macro",
    "F1_Class4": "f1_cls4",
    "Accuracy": "acc",
    "Flips Required": "flips",
    "Raw All Satisfied": "sat",
    "Raw Total Excess": "excess",
}


def parse_eval(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            k, v = row[0], row[1]
            if k in KEYS:
                try:
                    out[KEYS[k]] = float(v)
                except ValueError:
                    out[KEYS[k]] = float("nan")
    return out


def collect(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        if "evaluation_metrics.csv" not in files:
            continue
        parts = dirpath.replace("\\", "/").split("/")
        # .../{backbone}/tissuemnist/L50_G50/w{W}/{method}/seed_{S}
        try:
            backbone = parts[-6]
            wtok = parts[-3]
            method = parts[-2]
            seedtok = parts[-1]
            warmup = int(wtok[1:]) if wtok.startswith("w") else None
            seed = int(seedtok.split("_")[1])
        except (IndexError, ValueError):
            continue
        if warmup is None:
            continue
        m = parse_eval(os.path.join(dirpath, "evaluation_metrics.csv"))
        if "f1_macro" not in m:
            continue
        m.update(backbone=backbone, warmup=warmup, method=method, seed=seed)
        rows.append(m)
    return rows


def mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()

    rows = collect(args.root)
    print(f"Collected {len(rows)} completed runs from {args.root}")
    if not rows:
        return

    # coverage
    backbones = sorted({r["backbone"] for r in rows})
    done = defaultdict(int)
    for r in rows:
        done[(r["backbone"], r["warmup"], r["method"])] += 1
    expected = len(backbones) * len(WARMUPS) * len(ALL_METHODS) * 4
    print(f"Backbones: {backbones}")
    print(f"Completed {len(rows)}/{expected} expected runs "
          f"({100*len(rows)/expected:.0f}%)\n")

    # persist tidy csv
    os.makedirs("scripts/_audit", exist_ok=True)
    outp = "scripts/_audit/lowwarm_validation_agg.csv"
    cols = ["backbone", "warmup", "method", "seed",
            "f1_macro", "f1_cls4", "acc", "flips", "sat", "excess"]
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote tidy per-run table -> {outp}\n")

    # mean over seeds: cell[(backbone,warmup,method)] = {metric: mean}
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["backbone"], r["warmup"], r["method"])
        for mk in ["f1_macro", "f1_cls4", "flips", "sat"]:
            cell[key][mk].append(r.get(mk, float("nan")))
    cm = {k: {mk: mean(v) for mk, v in d.items()} for k, d in cell.items()}

    def best_trained_f1(bb, w):
        vals = [cm[(bb, w, m)]["f1_macro"] for m in TRAINED
                if (bb, w, m) in cm]
        return max(vals) if vals else float("nan")

    def best_posthoc_f1(bb, w):
        vals = [cm[(bb, w, m)]["f1_macro"] for m in POSTHOC
                if (bb, w, m) in cm]
        return max(vals) if vals else float("nan")

    # ---- per-backbone warmup x method F1(macro) table ----
    for bb in backbones:
        print("=" * 78)
        print(f"{bb}   TissueMNIST L50_G50   F1(Macro) mean-over-seeds")
        print("=" * 78)
        hdr = "  w |" + "".join(f"{m[:8]:>9s}" for m in ALL_METHODS)
        print(hdr)
        for w in WARMUPS:
            line = f"  {w} |"
            for m in ALL_METHODS:
                v = cm.get((bb, w, m), {}).get("f1_macro", float("nan"))
                line += f"{v:9.3f}" if v == v else f"{'--':>9s}"
            print(line)

        # the key trend: TraLO advantage over post-hoc and over trained
        print(f"\n  {bb}: TraLO F1(macro) lead by warmup (headroom test)")
        print(f"  {'w':>3s} {'TraLO':>7s} {'bestPH':>7s} {'d_PH':>7s} "
              f"{'bestTr':>7s} {'d_Tr':>7s}  {'TraSat%':>8s} {'TraFlip':>8s}")
        for w in WARMUPS:
            t = cm.get((bb, w, "tralo"), {})
            tf1 = t.get("f1_macro", float("nan"))
            ph = best_posthoc_f1(bb, w)
            tr = best_trained_f1(bb, w)
            dph = tf1 - ph
            dtr = tf1 - tr
            sat = 100 * t.get("sat", float("nan"))
            flp = t.get("flips", float("nan"))
            print(f"  {w:>3d} {tf1:7.3f} {ph:7.3f} {dph:+7.3f} "
                  f"{tr:7.3f} {dtr:+7.3f}  {sat:8.1f} {flp:8.1f}")
        print()

    # ---- headline verdict: does d_PH at low warmup EXCEED the w=50 ref? ----
    print("=" * 78)
    print("VERDICT  -  hypothesis: TraLO's F1(macro) lead over POST-HOC is")
    print("LARGER at low warmup (1-5) than at warmup=50.")
    print("  d_PH = mean_seeds[ TraLO F1 ] - max( heuristic, danits ) F1")
    print("=" * 78)
    print(f"  {'backbone':14s}" + "".join(f"   w{w}  " for w in WARMUPS)
          + f"{'|lowW_mean':>10s}{'w50_ref':>9s}{'verdict':>10s}")
    for bb in backbones:
        line = f"  {bb:14s}"
        gaps = []
        for w in WARMUPS:
            tf1 = cm.get((bb, w, "tralo"), {}).get("f1_macro", float("nan"))
            dph = tf1 - best_posthoc_f1(bb, w)
            if dph == dph:
                gaps.append(dph)
            line += f"{dph:+7.3f}" if dph == dph else f"{'--':>7s}"
        lowmean = mean(gaps) if gaps else float("nan")
        ref = W50_REF.get(bb, float("nan"))
        if gaps and ref == ref:
            verdict = "CONFIRM" if lowmean > ref + 1e-9 else "no"
        else:
            verdict = "..."
        lm = f"{lowmean:+.3f}" if lowmean == lowmean else "--"
        rf = f"{ref:+.3f}" if ref == ref else "--"
        line += f"{lm:>10s}{rf:>9s}{verdict:>10s}"
        print(line)
    print("\n  CONFIRM = low-warmup mean gap exceeds the warmup=50 gap for that")
    print("  backbone (TraLO's post-hoc advantage genuinely grows at low warmup).")
    print("  Cells fill seed-by-seed; treat partial rows (n<4/seed) as provisional.")


if __name__ == "__main__":
    main()

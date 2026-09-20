"""Every Option C result on disk, both backbones, scored the same way.

Refuses to pool across backbone / host / precision / code stamp. De-duplicates
by prediction hash so a re-run cannot be counted as a seed. Averages over SEED
only. Primary endpoints cc_f1 then F1 (Macro); everything else is exploratory.
"""
import os, glob, csv, json, math, hashlib, collections

BASE = "/home/dsi/michaer8/optloss-rank"
MET = ["cc_f1", "F1 (Macro)"]
RIVALS = ["tralo_null", "clip", "focal_clip", "alm", "fioretto", "hounie"]
FAMILIES = {"polc2": "MobileNetV3", "polcv2": "ViTB16"}


def load(roots):
    runs, hashes = {}, collections.defaultdict(list)
    stamps, hosts, prec, models = set(), set(), set(), set()
    for root in roots:
        man = "%s/results/%s/campaign_manifest.json" % (BASE, root)
        if os.path.exists(man):
            rt = json.load(open(man)).get("runtime", {})
            hosts.add(rt.get("host")); prec.add(rt.get("precision"))
        for f in glob.glob("%s/results/%s/*/*/*/*/*/evaluation_metrics.csv" % (BASE, root)):
            d = os.path.dirname(f); p = f.split("/")
            model, cap, arm, seed = p[-6], p[-4], p[-3], p[-2]
            pred, cfg = d + "/final_predictions.csv", d + "/config.json"
            if not (os.path.exists(pred) and os.path.exists(cfg)):
                continue
            m = {}
            for r in csv.DictReader(open(f)):
                try:
                    m[r["Metric"]] = float(r["Value"])
                except Exception:
                    pass
            if not all(k in m for k in MET):
                continue
            models.add(model)
            stamps.add(json.load(open(cfg)).get("code_version"))
            h = hashlib.md5(open(pred, "rb").read()).hexdigest()
            runs[(cap, arm, seed)] = (m, h)
            hashes[h].append((cap, arm, seed))
    return runs, hashes, models, hosts, prec, stamps


def tstat(d):
    n = len(d)
    if n < 2:
        return (sum(d) / n if n else float("nan"), float("nan"), n)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return (mu, mu / (sd / math.sqrt(n)) if sd > 0 else float("nan"), n)


def paired(runs, cap, a, b, met):
    sa = {k[2] for k in runs if k[0] == cap and k[1] == a}
    sb = {k[2] for k in runs if k[0] == cap and k[1] == b}
    c = sorted(sa & sb)
    if not c:
        return None
    return tstat([runs[(cap, a, s)][0][met] - runs[(cap, b, s)][0][met] for s in c])


for fam, backbone in FAMILIES.items():
    roots = [fam + "_a", fam + "_b"]
    runs, hashes, models, hosts, prec, stamps = load(roots)
    if not runs:
        print("%s: no runs" % backbone)
        continue
    bad = []
    if len(models) > 1: bad.append("backbones differ: %s" % sorted(models))
    if len(hosts) > 1: bad.append("hosts differ: %s" % sorted(hosts))
    if len(prec) > 1: bad.append("precision differs: %s" % sorted(prec))
    if len(stamps) > 1: bad.append("stamps differ: %s" % sorted(stamps))
    print("")
    print("#" * 104)
    print("# %s   (%s)" % (backbone, ", ".join(roots)))
    print("#" * 104)
    if bad:
        print("REFUSED TO POOL: " + "; ".join(bad))
        continue
    dup = sum(1 for h, k in hashes.items() if len(k) > 1)
    cells = collections.Counter((k[0], k[1]) for k in runs)
    lo, hi = min(cells.values()), max(cells.values())
    print("%d runs | %d distinct prediction hashes | %d duplicated | host %s | precision %s"
          % (len(runs), len(hashes), dup, sorted(hosts), sorted(prec)))
    print("%d cells, seeds/cell min=%d max=%d -> %s"
          % (len(cells), lo, hi, "BALANCED" if lo == hi else "UNBALANCED, DO NOT POOL"))
    caps = sorted({k[0] for k in runs}, key=lambda c: int(c.split("_")[0][1:]))
    for met in MET:
        print("")
        print("  %s" % met)
        print("  %-10s | %s" % ("cap", " | ".join("%-18s" % ("tralo - " + r) for r in RIVALS)))
        print("  " + "-" * 100)
        for cap in caps:
            cells_out = []
            for r in RIVALS:
                v = paired(runs, cap, "tralo", r, met)
                if v is None:
                    cells_out.append("%-18s" % "--")
                else:
                    mu, t, n = v
                    ts = "t%+4.1f" % t if t == t else "t  --"
                    cells_out.append("%-18s" % ("%+.4f %s n%d" % (mu, ts, n)))
            print("  %-10s | %s" % (cap, " | ".join(cells_out)))
    print("")
    wins = tot = 0
    for met in MET:
        for cap in caps:
            for r in RIVALS:
                v = paired(runs, cap, "tralo", r, met)
                if v:
                    tot += 1
                    wins += 1 if v[0] > 0 else 0
    print("  TraLO ahead in %d of %d (cap x rival x endpoint) contrasts." % (wins, tot))

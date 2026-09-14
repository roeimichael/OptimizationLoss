"""Measure grouping, split and distribution diagnostics from dataset metadata."""

import argparse
import os
import sys
import numpy as np
import pandas as pd

SEED_NOISE_ITEMS = 2.7
GROUP_CANDIDATES = (
    "loc_group",
    "synth_group",
    "group",
    "domain",
    "site",
    "location",
    "hospital",
    "region",
    "group_id",
)


def _group_column(df):
    for c in GROUP_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _dev(counts_obs, counts_exp):
    return float(np.abs(counts_obs - counts_exp).sum())


GENERIC_SLICE_DIRS = ("oodslice", "slice_1", "shift_1", "data", ".")


def slice_label(path):
    parts = [p for p in os.path.normpath(path).replace("\\", "/").split("/") if p]
    keep = []
    for p in reversed(parts):
        keep.insert(0, p)
        if p not in GENERIC_SLICE_DIRS:
            break
    return "/".join(keep) if keep else path


def novelty_items(train, test, gcol, label="label", n_null=200, seed=0):
    rng = np.random.default_rng(seed)
    classes = sorted(set(train[label]) | set(test[label]))
    idx = {c: i for (i, c) in enumerate(classes)}
    (n_tr, n_te) = (len(train), len(test))

    def cell_counts(frame):
        out = np.zeros(len(classes))
        for c, k in frame[label].value_counts().items():
            out[idx[c]] = k
        return out

    p_glob = cell_counts(train) / n_tr
    glob_obs = _dev(cell_counts(test), p_glob * n_te)
    glob_null = np.array(
        [_dev(rng.multinomial(n_te, p_glob), p_glob * n_te) for _ in range(n_null)]
    )
    (units, unseen_groups, unseen_items) = ([], [], 0)
    if gcol is not None:
        for g in sorted(test[gcol].unique()):
            te_g = test[test[gcol] == g]
            tr_g = train[train[gcol] == g]
            if len(tr_g) == 0:
                unseen_groups.append(g)
                unseen_items += len(te_g)
                units.append((cell_counts(te_g), p_glob, len(te_g)))
                continue
            units.append((cell_counts(te_g), cell_counts(tr_g) / len(tr_g), len(te_g)))
    else:
        units = []
    shift = np.divide(
        cell_counts(test) / n_te, p_glob, out=np.ones_like(p_glob), where=p_glob > 0
    )

    def net_expect(p, n):
        q = p * shift
        tot = q.sum()
        return q / tot * n if tot > 0 else p * n

    loc_obs = sum((_dev(obs, p * n) for (obs, p, n) in units))
    net_obs = sum((_dev(obs, net_expect(p, n)) for (obs, p, n) in units))
    net_null = (
        np.array(
            [
                sum(
                    (
                        _dev(rng.multinomial(n, net_expect(p, n) / n), net_expect(p, n))
                        for (_, p, n) in units
                    )
                )
                for _ in range(n_null)
            ]
        )
        if units
        else np.zeros(n_null)
    )
    loc_null = (
        np.array(
            [
                sum((_dev(rng.multinomial(n, p), p * n) for (_, p, n) in units))
                for _ in range(n_null)
            ]
        )
        if units
        else np.zeros(n_null)
    )

    def summarise(obs, null):
        sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
        excess = obs - float(null.mean())
        return (excess, excess / sd if sd > 0 else float("nan"))

    (g_ex, g_z) = summarise(glob_obs, glob_null)
    (l_ex, l_z) = summarise(loc_obs, loc_null)
    (n_ex, n_z) = summarise(net_obs, net_null)
    return {
        "net_items": n_ex,
        "net_z": n_z,
        "net_raw": net_obs,
        "net_null": float(net_null.mean()),
        "global_items": g_ex,
        "global_z": g_z,
        "global_raw": glob_obs,
        "global_null": float(glob_null.mean()),
        "local_items": l_ex,
        "local_z": l_z,
        "local_raw": loc_obs,
        "local_null": float(loc_null.mean()),
        "unseen_groups": unseen_groups,
        "unseen_items": unseen_items,
    }


def heterogeneity_items(test, gcol, label="label"):
    if gcol is None:
        return 0.0
    n = len(test)
    out = 0.0
    for c in sorted(test[label].unique()):
        n_c = int((test[label] == c).sum())
        for g in sorted(test[gcol].unique()):
            te_g = test[test[gcol] == g]
            expected = n_c * len(te_g) / n
            out += abs(int((te_g[label] == c).sum()) - expected)
    return out


def screen(path):
    tr = pd.read_csv(os.path.join(path, "train_meta.csv"))
    te = pd.read_csv(os.path.join(path, "test_meta.csv"))
    gcol = _group_column(te)
    counts = te["label"].value_counts().sort_index()
    ratio = counts.max() / max(counts.min(), 1)
    nov = novelty_items(tr, te, gcol)
    return {
        "path": path,
        "n_train": len(tr),
        "n_test": len(te),
        "n_classes": int(te["label"].nunique()),
        "gcol": gcol,
        "n_groups": int(te[gcol].nunique()) if gcol else 0,
        "counts": counts.to_dict(),
        "imbalance": float(ratio),
        "rarest": int(counts.min()),
        "heterogeneity": heterogeneity_items(te, gcol),
        **nov,
    }


def verdict_lines(r, name, noise=None):
    noise = SEED_NOISE_ITEMS if noise is None else float(noise)
    out = []
    if r["gcol"] is None:
        out.append(
            "  %-22s NO GROUP COLUMN -- the local scope does not exist here." % name
        )
    elif not np.isfinite(r["net_z"]):
        out.append(
            "  %-22s UNDECIDABLE: the sampling-noise null has zero spread, so z is"
            % name
        )
        out.append(
            "  %-22s   undefined and NOTHING was tested. This is not a pass. Usually it"
            % ""
        )
        out.append(
            "  %-22s   means one group, or identical groups -- check the group column."
            % ""
        )
    elif r["net_z"] < 2.0:
        out.append(
            "  %-22s DEAD: NET per-group novelty %+.0f items is within sampling noise (z=%.1f)."
            % (name, r["net_items"], r["net_z"])
        )
    elif r["net_items"] < noise:
        out.append(
            "  %-22s DEAD: NET novelty %.0f items is BELOW the %.1f-item seed noise (local reads %.0f, but that includes the global shift)."
            % (name, r["net_items"], noise, r["local_items"])
        )
    elif r["net_items"] < 3 * noise:
        out.append(
            "  %-22s MARGINAL: NET novelty %.0f items against %.1f-item noise (local %.0f)."
            % (name, r["net_items"], noise, r["local_items"])
        )
    else:
        out.append(
            "  %-22s STAGE 1 PASS (necessary, not sufficient): NET novelty %.0f items, %.0fx seed noise (local %.0f)."
            % (name, r["net_items"], r["net_items"] / noise, r["local_items"])
        )
    return out


def _synthetic(tmp, kind, n_class=4, n_group=6, per=500, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(n_class))
    (tr, te) = ([], [])
    for g in range(n_group):
        p = base if kind == "dead" else rng.dirichlet(np.ones(n_class) * 0.35)
        lab = rng.choice(n_class, size=per, p=p)
        tr += [{"location": "g%d" % g, "label": int(c)} for c in lab]
    for g in range(n_group, n_group + 4):
        p = base if kind == "dead" else rng.dirichlet(np.ones(n_class) * 0.35)
        lab = rng.choice(n_class, size=per, p=p)
        te += [{"location": "g%d" % g, "label": int(c)} for c in lab]
    if kind == "dead":
        for i, r in enumerate(te):
            r["location"] = "g%d" % (n_group + i % 4)
    os.makedirs(tmp, exist_ok=True)
    pd.DataFrame(tr).to_csv(os.path.join(tmp, "train_meta.csv"), index=False)
    pd.DataFrame(te).to_csv(os.path.join(tmp, "test_meta.csv"), index=False)
    return tmp


def self_test(out=sys.stdout):
    import tempfile

    root = tempfile.mkdtemp()
    ok = True
    seen = {}
    for kind in ("dead", "live"):
        r = screen(_synthetic(os.path.join(root, kind), kind))
        seen[kind] = r
        for line in verdict_lines(r, kind):
            out.write(line + chr(10))
    if seen["dead"]["net_z"] >= 2.0:
        out.write(
            "SELF-TEST FAIL: groups built as an INDEX are i.i.d. draws from one distribution and MUST read DEAD, got NET %+.0f at z=%.1f%s"
            % (seen["dead"]["net_items"], seen["dead"]["net_z"], chr(10))
        )
        ok = False
    if not (
        seen["live"]["net_z"] >= 2.0 and seen["live"]["net_items"] > SEED_NOISE_ITEMS
    ):
        out.write(
            "SELF-TEST FAIL: per-group prevalences ARE the signal this screen exists to find; it must clear stage 1, got NET %+.0f at z=%.1f%s"
            % (seen["live"]["net_items"], seen["live"]["net_z"], chr(10))
        )
        ok = False
    lines = chr(10).join(verdict_lines(dict(seen["live"], net_z=float("nan")), "nan-z"))
    if "UNDECIDABLE" not in lines or "PASS" in lines:
        out.write(
            "SELF-TEST FAIL: an undefined z must not upgrade the verdict:"
            + chr(10)
            + lines
            + chr(10)
        )
        ok = False
    out.write("SELF-TEST %s%s" % ("PASSED" if ok else "FAILED", chr(10)))
    return 0 if ok else 1


def main():
    global SEED_NOISE_ITEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="slice dirs with train/test_meta.csv")
    ap.add_argument(
        "--noise",
        type=float,
        default=SEED_NOISE_ITEMS,
        help="paired seed sd in items, the divisor every verdict below is scaled by. The default %.1f is dermmnist x MobileNetV3, and dermmnist is REMOVED and was leaked; iwildcam measures 4.75 to 27.83. Pass the number for the dataset and backbone you actually intend to run."
        % SEED_NOISE_ITEMS,
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.paths:
        ap.error("give at least one slice dir, or --self-test")
    SEED_NOISE_ITEMS = float(args.noise)
    print("DATASET SCREEN -- can a count constraint carry information here?")
    print(
        "Everything is in ITEMS. Every verdict below is scaled by a paired seed sd of"
    )
    print(
        "%.2f items%s. On iwildcam the measured range is 4.75 to 27.83, so a PASS at 2.7"
        % (
            SEED_NOISE_ITEMS,
            " (the default: dermmnist x MobileNetV3, a REMOVED dataset)"
            if abs(SEED_NOISE_ITEMS - 2.7) < 1e-09
            else " (--noise)",
        )
    )
    print(
        "can be a DEAD at 27.8. Pass --noise to price it for the dataset you will run."
    )
    print("")
    rows = [screen(p) for p in args.paths]
    print(
        "  %-34s %7s %6s %7s %7s %9s"
        % ("dataset", "n_test", "cls", "groups", "imbal", "rarest")
    )
    for r in rows:
        print(
            "  %-34s %7d %6d %7d %7.1fx %9d"
            % (
                slice_label(r["path"])[-34:],
                r["n_test"],
                r["n_classes"],
                r["n_groups"],
                r["imbalance"],
                r["rarest"],
            )
        )
    print("")
    print("  NOVELTY = observed deviation MINUS the sampling-noise null, in items.")
    print(
        "  %-30s %9s %6s %9s %6s %9s %6s %7s"
        % ("dataset", "NET ex", "z", "LOCAL ex", "z", "GLOBAL ex", "z", "unseen")
    )
    for r in rows:
        print(
            "  %-30s %+9.0f %6.1f %+9.0f %6.1f %+9.0f %6.1f %7d"
            % (
                slice_label(r["path"])[-30:],
                r["net_items"],
                r["net_z"],
                r["local_items"],
                r["local_z"],
                r["global_items"],
                r["global_z"],
                len(r["unseen_groups"]),
            )
        )
    print("")
    print("  !! STAGE 1 ONLY. Passing here is NECESSARY, NOT SUFFICIENT.")
    print("  Distribution differences do not establish learnable cap headroom.")
    print("  Inspect development predictions with scripts.headroom before launch.")
    print("")
    for r in rows:
        name = slice_label(r["path"])
        for line in verdict_lines(r, name):
            print(line)
        if r["unseen_groups"]:
            print(
                "  %-22s   and %d test group(s) are ABSENT from train (%d items) -- training carries no prior for them at all."
                % ("", len(r["unseen_groups"]), r["unseen_items"])
            )


if __name__ == "__main__":
    main()

"""Which classes does THIS campaign cap? Ask the campaign, never assume.

🛑 THE DEFECT THIS EXISTS FOR (found 2026-09-09). Seven tools defaulted
`--classes` to `[2, 7]` and `paired_seeds` defaulted `--capped` to `[2, 4]`.
`[2, 7]` is iwildcam's pair; `[2, 4]` is dermmnist's, and dermmnist is REMOVED.
**bcn and fmow both cap classes 3 and 5.** Every one of those tools genuinely
consumes its default, so running one on a bcn or fmow campaign without the flag
scores two classes the experiment never constrained -- and prints a plausible
number rather than raising. Two of the three campaigns live when this was found
were on bcn and fmow.

It is the same shape as the inert-flag catalogue in the opposite direction: not
a knob that does nothing, but a default that does the WRONG thing while looking
right. The signature is identical -- a number arrives, nothing is red.

THE RULE HERE. The capped classes are a property of the CONFIG, which every run
carries, so no tool that can see a `config.json` has any business guessing:

  * no override               -> read it from the campaign
  * override AGREES           -> proceed, silently
  * override DISAGREES        -> REFUSE. That is the exact silent-wrong-answer
                                case, and a warning would be read after the
                                number instead of before it.
  * campaign disagrees with   -> REFUSE. Two datasets under one root cannot be
    ITSELF                       scored on one class pair.
  * no config reachable       -> fall back to the override and SAY SO, because
                                a pre-GPU screen (`ceiling_screen`) legitimately
                                has no run to read.
"""
import glob as _glob
import json
import os
import sys

CONFIG_SCAN_LIMIT = 400


def find_configs(target, limit=CONFIG_SCAN_LIMIT):
    """Every `config.json` under a root, a run dir, or a glob pattern.

    Accepts what the callers actually pass: `--campaign results/x` (a root),
    `--glob 'results/x/*/*/*/tralo/seed_*'` (run dirs), a single run dir, or a
    list of any of those.
    """
    if isinstance(target, (list, tuple, set)):
        seen, out = set(), []
        for t in target:
            for c in find_configs(t, limit):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out
    t = str(target)
    hits = []
    if any(ch in t for ch in "*?["):
        for d in sorted(_glob.glob(t)):
            p = os.path.join(d, "config.json")
            if os.path.isfile(p):
                hits.append(p)
        # a glob may already name the config files themselves
        hits += [p for p in sorted(_glob.glob(t)) if p.endswith("config.json")]
    elif os.path.isfile(t):
        if t.endswith("config.json"):
            hits.append(t)
    elif os.path.isdir(t):
        direct = os.path.join(t, "config.json")
        if os.path.isfile(direct):
            hits.append(direct)
        else:
            for root, _dirs, files in os.walk(t):
                if "config.json" in files:
                    hits.append(os.path.join(root, "config.json"))
                    if len(hits) >= limit:
                        break
    return hits[:limit]


def _from_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    raw = (cfg.get("dataset_config") or {}).get("constrained_class")
    if raw is None:
        return None
    if isinstance(raw, int):
        return (int(raw),)
    return tuple(sorted(int(x) for x in raw))


def resolve(target, override=None, out=sys.stdout, what="--classes"):
    """The capped classes for `target`. Raises SystemExit rather than guess."""
    found = {}
    for p in find_configs(target):
        cls = _from_config(p)
        if cls:
            found.setdefault(cls, p)

    if len(found) > 1:
        raise SystemExit(
            "REFUSED: %s carries MORE THAN ONE capped-class set -- %s. A single "
            "class pair cannot score two datasets, and scoring them together "
            "would silently report one dataset's classes for both. Split the "
            "roots." % (target, "; ".join(
                "%s (e.g. %s)" % (list(k), v) for k, v in sorted(found.items()))))

    if not found:
        if override is None:
            raise SystemExit(
                "REFUSED: no config.json under %s carries "
                "`dataset_config.constrained_class`, and no %s was given. This "
                "tool used to default to iwildcam's (2, 7), which is WRONG on "
                "bcn and fmow (both cap 3 and 5) and prints a plausible number "
                "rather than raising. Pass %s explicitly."
                % (target, what, what))
        out.write("NOTE: no config.json reachable under %s, so %s %s is taken "
                  "on trust and was not verified against any run.\n"
                  % (target, what, list(override)))
        return sorted(int(x) for x in override)

    (actual, example), = found.items()
    if override is not None:
        want = tuple(sorted(int(x) for x in override))
        if want != actual:
            raise SystemExit(
                "REFUSED: %s %s contradicts the campaign, which caps %s "
                "(read from %s). One of the two is wrong and the numbers would "
                "not say which -- they would just be about the other classes."
                % (what, list(want), list(actual), example))
    return list(actual)


def self_test(out=sys.stdout):
    import shutil
    import tempfile
    checks = []
    tmp = tempfile.mkdtemp(prefix="cappedcls_")

    def mk(sub, classes, dataset="bcn"):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"dataset_config": {"constrained_class": list(classes),
                                          "name": dataset}}, fh)
        return d

    try:
        bcn = mk("bcn/MNv3/bcn/L70_G95/tralo/seed_1", [3, 5])
        checks.append(("reads the capped classes off the campaign",
                       resolve(os.path.join(tmp, "bcn")) == [3, 5]))
        checks.append(("a run DIRECTORY resolves too",
                       resolve(bcn) == [3, 5]))
        checks.append(("a GLOB of run dirs resolves too",
                       resolve(os.path.join(tmp, "bcn/*/*/*/tralo/seed_*"))
                       == [3, 5]))
        checks.append(("an override that AGREES is accepted",
                       resolve(os.path.join(tmp, "bcn"), [5, 3]) == [3, 5]))

        # ---- the defect itself, as a negative control -------------------
        try:
            resolve(os.path.join(tmp, "bcn"), [2, 7])
            ok = False
        except SystemExit as e:
            ok = "contradicts the campaign" in str(e)
        checks.append(("NEGATIVE CONTROL: the old [2, 7] default on a bcn "
                       "campaign is REFUSED, not silently scored", ok))

        mk("mixed/A/iwildcam/L70_G95/tralo/seed_1", [2, 7], "iwildcam")
        mk("mixed/B/bcn/L70_G95/tralo/seed_1", [3, 5], "bcn")
        try:
            resolve(os.path.join(tmp, "mixed"))
            ok = False
        except SystemExit as e:
            ok = "MORE THAN ONE capped-class set" in str(e)
        checks.append(("NEGATIVE CONTROL: two datasets under one root are "
                       "REFUSED", ok))

        empty = os.path.join(tmp, "empty")
        os.makedirs(empty, exist_ok=True)
        try:
            resolve(empty)
            ok = False
        except SystemExit as e:
            ok = "no config.json" in str(e)
        checks.append(("NEGATIVE CONTROL: no config AND no override is "
                       "REFUSED, never defaulted", ok))

        buf = _Buf()
        got = resolve(empty, [2, 7], out=buf)
        checks.append(("no config WITH an override proceeds and SAYS it was "
                       "not verified",
                       got == [2, 7] and "not verified" in buf.text))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = [c for c, ok in checks if not ok]
    for c, ok in checks:
        out.write("  %s %s\n" % ("PASS" if ok else "FAIL", c))
    out.write("%s: %d/%d\n" % ("OK" if not bad else "FAILED",
                               len(checks) - len(bad), len(checks)))
    return 1 if bad else 0


class _Buf(object):
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.capped_classes <root|glob> "
                 "[--self-test]")
    print(" ".join(str(c) for c in resolve(sys.argv[1])))

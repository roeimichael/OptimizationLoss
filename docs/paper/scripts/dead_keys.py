"""Which hyperparameters in our configs is the code actually reading?

`rho_step` turned out to be a dead key -- every generator sets it, nothing reads
it, and the resulting misreading of the rho schedule made a bounded penalty look
like an unbounded explosion. That is not a one-off risk: an inert flag is this
project's most frequent failure mode (it has now happened four times: the CE-skip
asymmetry, focal_clip in arm_joint, by_k on octmnist, and rho_step).

So instead of finding them one at a time, enumerate every key that appears in a
config's `hyperparams` and check whether ANY source file reads it, by either
`hp.get("key")` / `hp["key"]` / `hyperparams[...]` or a bare quoted occurrence.

A key reported here is not automatically a bug -- some are consumed by the
config generator, or by a methodology this campaign never dispatches. It is a
list of things to verify, and each one is a place where a config can quietly
promise something the run does not do.
"""
import argparse
import glob
import json
import os
import ast
import re
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="campaign roots to scan for config.json")
    ap.add_argument("--src", default="src", help="source tree to grep")
    a = ap.parse_args()

    keys = Counter()
    for root in a.configs:
        for p in glob.glob(root + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(p))
            except Exception:
                continue
            for k in (cfg.get("hyperparams") or {}):
                keys[k] += 1

    # Parsed, not grepped. "Mentioned somewhere" is far too weak: after
    # `rho_step` was found dead a warning was added naming it, and BOTH a
    # whole-file grep and a line-by-line one then report it as read -- the
    # `hp["rho_step"]` sits on a continuation line of the `log.warning(` call,
    # so the line itself looks innocent. Only the AST sees that the extraction
    # is an argument to a logging call and therefore not a real read.
    LOG_FNS = {"debug", "info", "warning", "warn", "error", "critical",
               "exception"}
    MAPS = {"hp", "hyperparams", "hyper", "cfg", "config", "params"}
    reads = {}          # key -> set of "real" / "log"

    def record(key, kind):
        reads.setdefault(key, set()).add(kind)

    for dirpath, _, files in os.walk(a.src):
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dirpath, f),
                                      encoding="utf-8").read())
            except Exception:
                continue
            log_spans = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    is_log = (
                        (isinstance(fn, ast.Attribute) and fn.attr in LOG_FNS)
                        or (isinstance(fn, ast.Name) and fn.id == "print"))
                    if is_log and getattr(node, "end_lineno", None):
                        log_spans.append((node.lineno, node.end_lineno))

            def in_log(ln):
                return any(a_ <= ln <= b_ for a_, b_ in log_spans)

            for node in ast.walk(tree):
                key = base = None
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    base, key = node.func.value, node.args[0].value
                elif (isinstance(node, ast.Subscript)
                      and isinstance(node.slice, ast.Constant)
                      and isinstance(node.slice.value, str)):
                    base, key = node.value, node.slice.value
                if key is None:
                    continue
                name = base.id if isinstance(base, ast.Name) else None
                if name not in MAPS:
                    continue
                record(key, "log" if in_log(node.lineno) else "real")

    dead, logged_only, alive = [], [], []
    for k, n in sorted(keys.items(), key=lambda kv: -kv[1]):
        kinds = reads.get(k, set())
        if "real" in kinds:
            alive.append((k, n))
        elif "log" in kinds:
            logged_only.append((k, n))
        else:
            dead.append((k, n))

    print("hyperparameter keys seen in configs: %d   (%d runs scanned)"
          % (len(keys), max(keys.values()) if keys else 0))
    print("\nREAD BY THE CODE (%d):" % len(alive))
    print("  " + ", ".join(k for k, _ in alive))
    if logged_only:
        print("\n*** MENTIONED ONLY IN A LOG/COMMENT -- the value is never used"
              " (%d): ***" % len(logged_only))
        for k, n in logged_only:
            print("  %-34s present in %d configs" % (k, n))
    print("\n*** NEVER READ AT ALL -- verify each (%d): ***" % len(dead))
    for k, n in dead:
        print("  %-34s present in %d configs" % (k, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())

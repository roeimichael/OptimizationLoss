"""Audit emitted hyperparameters against live runtime reads and cache identity."""

import ast
import glob
import json
import os
import subprocess
import sys
import tempfile

SEED_NAMES = {
    "hyperparams": {"hp", "hyperparams", "hparams"},
    "config": {"config", "cfg", "conf"},
    "dataset_config": {"dataset_config"},
}
SECTION_KEYS = {"hyperparams": "hyperparams", "dataset_config": "dataset_config"}


def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class Reads(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.hits = []
        self.opaque = []
        self.alias = {}

    def _kind_of(self, base):
        for kind, names in SEED_NAMES.items():
            if base in names:
                return kind
        return self.alias.get(base)

    ACCESSOR_DEFS = ("_required",)

    def visit_FunctionDef(self, node):
        (outer, self._fn) = (getattr(self, "_fn", None), node.name)
        self.generic_visit(node)
        self._fn = outer

    def _opaque(self, base, how, lineno):
        if getattr(self, "_fn", None) in self.ACCESSOR_DEFS:
            return
        "A read the walker cannot resolve to a literal key. Recorded so the\n        audit fails loudly instead of silently under-reporting its read set."
        self.opaque.append((base, how, self.path, lineno))

    def _record(self, base, key, lineno):
        kind = self._kind_of(base)
        if kind is None:
            return
        self.hits.append((kind, key, self.path, lineno))

    def visit_Assign(self, node):
        tgt = node.targets[0] if len(node.targets) == 1 else None
        name = tgt.id if isinstance(tgt, ast.Name) else None
        if name:
            v = node.value
            key = None
            if (
                isinstance(v, ast.Subscript)
                and isinstance(v.slice, ast.Constant)
                and isinstance(v.slice.value, str)
            ):
                key = v.slice.value
            elif (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute)
                and (v.func.attr == "get")
                and v.args
                and isinstance(v.args[0], ast.Constant)
            ):
                key = v.args[0].value
            elif isinstance(v, ast.Attribute):
                key = v.attr
            if key in SECTION_KEYS:
                self.alias[name] = SECTION_KEYS[key]
        self.generic_visit(node)

    def visit_Subscript(self, node):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            self._record(_base_name(node.value), sl.value, node.lineno)
        elif self._kind_of(_base_name(node.value)) is not None:
            self._opaque(_base_name(node.value), "subscript", node.lineno)
        self.generic_visit(node)

    OPAQUE_METHODS = ("get", "pop", "setdefault")

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in self.OPAQUE_METHODS and node.args:
            base = _base_name(f.value)
            if isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str
            ):
                if f.attr == "get":
                    self._record(base, node.args[0].value, node.lineno)
                else:
                    self._record(base, node.args[0].value, node.lineno)
            elif self._kind_of(base) is not None:
                self._opaque(base, ".%s()" % f.attr, node.lineno)
        else:
            self.visit_Call_helper(node)
        self.generic_visit(node)

    def visit_Call_helper(self, node):
        if not isinstance(node.func, ast.Name) or len(node.args) < 2:
            return
        base = _base_name(node.args[0])
        if self._kind_of(base) is None:
            return
        key = node.args[1]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            self._record(base, key.value, node.lineno)

    def visit_Dict(self, node):
        for k in node.keys:
            if k is None:
                self._opaque(None, "** splat", node.lineno)
        self.generic_visit(node)


SHARED_DIRS = [
    "src/pipeline",
    "src/training",
    "src/utils",
    "src/losses",
    "src/experiments",
    "src/models",
]
METH_DIR = "src/methodologies"


def _walk(d):
    return [
        os.path.join(dp, f).replace("\\", "/")
        for (dp, dn, fn) in os.walk(d)
        if "__pycache__" not in dp
        for f in fn
        if f.endswith(".py")
    ]


def _keys_in(paths):
    out = set()
    for p in paths:
        if p.replace("\\", "/") == "src/pipeline/config.py":
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except (SyntaxError, OSError):
            continue
        v = Reads(p)
        v.visit(tree)
        v.hits = []
        v.visit(tree)
        out |= {k for (kind, k, _f, _l) in v.hits if kind == "hyperparams"}
    return out


def per_methodology_reads():
    shared = set()
    for d in SHARED_DIRS:
        shared |= _keys_in(_walk(d))
    DRIVERS = {
        "dual_common.py": ("tralo", "fioretto_ldf", "fioretto_alm", "hounie_rcl"),
    }
    driver_keys = {f: _keys_in([os.path.join(METH_DIR, f)]) for f in DRIVERS}
    out = {}
    for m in sorted(os.listdir(METH_DIR)):
        d = os.path.join(METH_DIR, m)
        if not os.path.isdir(d) or m == "__pycache__":
            continue
        own = _keys_in(_walk(d))
        for f, users in DRIVERS.items():
            if m in users:
                own |= driver_keys[f]
        out[m] = own | shared
    return out


def audit_per_arm(root):
    from src.pipeline.config import validate_hyperparams

    reads = per_methodology_reads()
    bad = 0
    for p in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
        c = json.load(open(p, encoding="utf-8"))
        try:
            validate_hyperparams(c["methodology"], c["hyperparams"])
            dead = (
                set(c["hyperparams"])
                - reads[c["methodology"]]
                - {"seed", "constraint_epochs"}
            )
            if dead:
                raise ValueError("no live runtime reader: " + ", ".join(sorted(dead)))
        except (ValueError, KeyError) as exc:
            print("FAIL %s: %s" % (p, exc))
            bad += 1
    return bad


WARMUP_PATH = [
    "src/pipeline/warmup.py",
    "src/losses/imbalanced_losses.py",
    "src/training/model_cache.py",
    "src/pipeline/setup.py",
    "src/models",
]
WARMUP_EXTRA = {"seed"}


def audit_identity(root):
    from configs.gen_campaign import load_protocol, compute_base_model_id

    P = load_protocol()
    paths = [p for f in WARMUP_PATH for p in (_walk(f) if os.path.isdir(f) else [f])]
    missing = (_keys_in(paths) | WARMUP_EXTRA) - set(P["warmup_identity_keys"])
    bad = len(missing)
    if missing:
        print("FAIL warm-up identity lacks live keys: %s" % sorted(missing))
    for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = json.load(open(p, encoding="utf-8"))
        expected = compute_base_model_id(
            P, c["model_name"], c["hyperparams"], c["dataset_mode"], c["dataset_config"]
        )
        if c["base_model_id"] != expected:
            print("FAIL base_model_id: %s" % p)
            bad += 1
    return bad


def audit(root):
    paths = glob.glob(os.path.join(root, "**", "config.json"), recursive=True)
    if not paths:
        print("FAIL no config.json under %s" % root)
        return 1
    bad = audit_per_arm(root) + audit_identity(root)
    print(
        "audited %d configs from %s: %s" % (len(paths), root, "FAIL" if bad else "OK")
    )
    return int(bool(bad))


def main():
    import argparse
    from configs.gen_campaign import load_protocol

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?")
    args = ap.parse_args()
    try:
        if args.root:
            return audit(args.root)
        with tempfile.TemporaryDirectory(prefix="cfgaudit_") as root:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "configs.gen_campaign",
                    "--root",
                    root,
                    "--datasets",
                    *load_protocol()["datasets"],
                    "--models",
                    *load_protocol()["models"],
                    "--arms",
                    "all",
                    "--seeds",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                print(result.stdout + result.stderr)
                return 1
            return audit(root)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print("FAIL audit_config: %s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

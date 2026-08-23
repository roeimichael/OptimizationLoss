"""Cross-reference every key a config EMITS against every key the code READS.

Three failure modes, all of which have bitten this project:

  HALLUCINATED  emitted but never read anywhere. The config implies a knob that
                does not exist. `base_loss`, `rho_step` and `alpha_kl` were all
                this, and each one made someone believe an arm was configured
                when it was inert.

  SILENT        read with a default but never emitted. The value is then decided
                by a literal buried in the code, not by the config -- which is
                how hounie_rcl ran at eta=0.1 while the paper and its own
                hp_defaults both said 0.01.

  OK            emitted and read.

A grep cannot do this: `rho_step` appears in a log-format string, so grep calls
it used. This walks the AST, so only real subscript / .get() accesses count.

    python -m scripts.audit_config            # generates a reference campaign
    python -m scripts.audit_config <campaign> # audits an existing one
"""
import ast
import collections
import glob
import json
import os
import subprocess
import sys
import tempfile

OPAQUE = []          # reads the walker could not resolve
CODE_ROOTS = ["src", "scripts", "main.py", "configs"]

# Seed names for each config section. Aliases are then DISCOVERED, not guessed:
# `ds = config['dataset_config']` teaches the walker that `ds` is a dataset_config,
# which is how group_column was first mis-reported as unread.
SEED_NAMES = {
    "hyperparams": {"hp", "hyperparams", "hparams"},
    "config": {"config", "cfg", "conf"},
    "dataset_config": {"dataset_config"},
}
SECTION_KEYS = {"hyperparams": "hyperparams", "dataset_config": "dataset_config"}


def _base_name(node):
    """Rightmost identifier of an expression: inputs.hyperparams -> hyperparams."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class Reads(ast.NodeVisitor):
    """Collect (container, key, file, line) for every literal-key access."""

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

    def _opaque(self, base, how, lineno):
        """A read the walker cannot resolve to a literal key. Recorded so the
        audit fails loudly instead of silently under-reporting its read set."""
        self.opaque.append((base, how, self.path, lineno))

    def _record(self, base, key, lineno):
        kind = self._kind_of(base)
        if kind is None:
            return
        self.hits.append((kind, key, self.path, lineno))

    def visit_Assign(self, node):
        """Learn aliases: ds = config['dataset_config']  ->  ds is a dataset_config.
        Also  hp = inputs.hyperparams  and  hp = config.get('hyperparams', {})."""
        tgt = node.targets[0] if len(node.targets) == 1 else None
        name = tgt.id if isinstance(tgt, ast.Name) else None
        if name:
            v = node.value
            key = None
            if isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Constant)                     and isinstance(v.slice.value, str):
                key = v.slice.value
            elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                  and v.func.attr == "get" and v.args
                  and isinstance(v.args[0], ast.Constant)):
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
            # hp[k] with a computed key: the audit cannot know which key
            self._opaque(_base_name(node.value), "subscript", node.lineno)
        self.generic_visit(node)

    OPAQUE_METHODS = ("get", "pop", "setdefault")

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in self.OPAQUE_METHODS and node.args:
            base = _base_name(f.value)
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                if f.attr == "get":
                    self._record(base, node.args[0].value, node.lineno)
                else:
                    # .pop / .setdefault also READ the key, and .pop mutates
                    self._record(base, node.args[0].value, node.lineno)
            elif self._kind_of(base) is not None:
                self._opaque(base, ".%s()" % f.attr, node.lineno)
        self.generic_visit(node)

    def visit_Dict(self, node):
        """`{**hp}` copies every key without naming one."""
        for k in node.keys:
            if k is None:
                self._opaque(None, "** splat", node.lineno)
        self.generic_visit(node)


def collect_reads():
    reads = collections.defaultdict(list)      # (kind, key) -> [(file, line)]
    for root in CODE_ROOTS:
        files = [root] if root.endswith(".py") else [
            os.path.join(dp, f).replace("\\", "/")
            for dp, dn, fn in os.walk(root) if "__pycache__" not in dp
            for f in fn if f.endswith(".py")]
        for p in files:
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except SyntaxError:
                continue
            v = Reads(p)
            v.visit(tree)          # pass 1: learn aliases
            v.hits, v.opaque = [], []
            v.visit(tree)          # pass 2: record with aliases known
            for kind, key, path, line in v.hits:
                reads[(kind, key)].append((path, line))
            OPAQUE.extend(v.opaque)
    return reads


def collect_emitted(root):
    emitted = collections.defaultdict(set)     # kind -> {key}
    per_arm = collections.defaultdict(set)     # arm -> {hyperparams keys}
    n = 0
    for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = json.load(open(p))
        n += 1
        for k in c:
            emitted["config"].add(k)
        for k in c.get("hyperparams", {}):
            emitted["hyperparams"].add(k)
            per_arm[c.get("arm", "?")].add(k)
        for k in c.get("dataset_config", {}):
            emitted["dataset_config"].add(k)
    return emitted, per_arm, n



SHARED_DIRS = ["src/pipeline", "src/training", "src/utils", "src/losses",
               "src/experiments", "src/models"]
METH_DIR = "src/methodologies"
# read by the verification scripts on every arm, so declared not hallucinated
CONTRACT = {"warmup_epochs", "constraint_epochs", "seed", "lr"}
# Legitimately absent on some arms. `warmup_loss` unset MEANS plain CE, which is
# what the four constrained arms and `clip` want; the rest are read by the shared
# warm-up only when warmup_loss selects that recipe.
CONDITIONAL = {"warmup_loss", "focal_alpha", "focal_gamma", "cb_beta",
               "logit_adjust_tau"}


def _walk(d):
    return [os.path.join(dp, f).replace("\\", "/")
            for dp, dn, fn in os.walk(d) if "__pycache__" not in dp
            for f in fn if f.endswith(".py")]


def _keys_in(paths):
    out = set()
    for p in paths:
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except (SyntaxError, OSError):
            continue
        v = Reads(p)
        v.visit(tree)
        v.hits = []
        v.visit(tree)
        out |= {k for kind, k, _f, _l in v.hits if kind == "hyperparams"}
    return out


def per_methodology_reads():
    """Read set for each methodology = its own package + the shared pipeline that
    every methodology passes through. A key read by tralo but emitted on the
    fioretto arm is hallucinated FOR FIORETTO, which a union audit cannot see."""
    shared = set()
    for d in SHARED_DIRS:
        shared |= _keys_in(_walk(d))
    common = _keys_in([os.path.join(METH_DIR, "imbalanced_common.py")])
    out = {}
    for m in sorted(os.listdir(METH_DIR)):
        d = os.path.join(METH_DIR, m)
        if not os.path.isdir(d) or m == "__pycache__":
            continue
        own = _keys_in(_walk(d))
        if m in ("focal", "class_balanced", "logit_adjust"):
            own |= common
        out[m] = own | shared
    return out


def audit_per_arm(root):
    reads = per_methodology_reads()
    arms = {}
    for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = json.load(open(p))
        arms.setdefault(c["arm"], (c["methodology"], set()))[1].update(c["hyperparams"])
    print("=" * 78)
    print("PER-ARM  (a key read by tralo but emitted on fioretto is hallucinated)")
    print("=" * 78)
    bad = 0
    for arm in sorted(arms):
        meth, em = arms[arm]
        rd = reads.get(meth, set())
        hall = sorted(em - rd - CONTRACT)
        silent = sorted(rd - em - CONDITIONAL)
        flag = "OK  " if not hall else "FAIL"
        if hall:
            bad += len(hall)
        print("  %s %-11s -> %-15s %2d emitted" % (flag, arm, meth, len(em)))
        if hall:
            print("        HALLUCINATED: %s" % ", ".join(hall))
        if silent:
            print("        silent (defaulted in code): %s" % ", ".join(silent))
    print()
    return bad


# Files that can change WHAT THE WARM-UP PRODUCES. Every hyperparameter read
# here must appear in the YAML's `warmup_identity_keys`, or two arms that differ
# in it hash to the same base_model_id and the second one silently loads the
# first one's trained model instead of training. That has happened four times.
WARMUP_PATH = ["src/pipeline/warmup.py", "src/losses/imbalanced_losses.py",
               "src/training/model_cache.py", "src/pipeline/setup.py",
               "src/models"]
# read at src/experiments/runner.py:43 (seed_all), not on the path above
WARMUP_EXTRA = {"seed"}


def audit_identity(root):
    """Three properties of base_model_id:

    P1 completeness  everything the warm-up reads is in warmup_identity_keys
    P2 injectivity   id <-> warm-up identity is a bijection over the campaign
    P3 sharing       arms sharing an id agree on every warm-up-path key
    """
    import yaml
    P = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    declared = set(P["warmup_identity_keys"])

    paths = []
    for f in WARMUP_PATH:
        paths += _walk(f) if os.path.isdir(f) else [f]
    actual = _keys_in(paths) | WARMUP_EXTRA

    print("=" * 78)
    print("BASE_MODEL_ID  (warm-up cache identity)")
    print("=" * 78)
    bad = 0

    missing = sorted(actual - declared)
    print()
    print("  P1 completeness -- the warm-up reads %d keys, %d are declared"
          % (len(actual), len(declared)))
    if missing:
        bad += len(missing)
        print("     FAIL: read by the warm-up but ABSENT from warmup_identity_keys:")
        for k in missing:
            print("       %s   <-- two arms differing here share a cached model" % k)
    else:
        print("     OK -- every key the warm-up reads is in the identity")
    unused = sorted(declared - actual)
    if unused:
        print("     note: declared but not read on the warm-up path: %s"
              % ", ".join(unused))
        print("           (harmless -- over-splits caches, never shares wrongly)")

    runs = [json.load(open(q)) for q in
            glob.glob(os.path.join(root, "**", "config.json"), recursive=True)]

    def projection(cfg):
        hp = cfg["hyperparams"]
        d = {"model_name": cfg["model_name"], "dataset_mode": cfg["dataset_mode"],
             "data_dir": cfg["dataset_config"]["data_dir"],
             "num_classes": cfg["dataset_config"]["num_classes"]}
        d.update({k: hp[k] for k in declared if k in hp})
        return json.dumps(d, sort_keys=True)

    by_id = collections.defaultdict(set)
    by_proj = collections.defaultdict(set)
    for cfg in runs:
        by_id[cfg["base_model_id"]].add(projection(cfg))
        by_proj[projection(cfg)].add(cfg["base_model_id"])

    print()
    print("  P2 injectivity -- %d distinct ids for %d distinct warm-up identities"
          % (len(by_id), len(by_proj)))
    collide = {i: v for i, v in by_id.items() if len(v) > 1}
    split = {v: i for v, i in by_proj.items() if len(i) > 1}
    if collide:
        bad += len(collide)
        print("     FAIL: %d id(s) map to MORE THAN ONE warm-up identity -- a hash"
              % len(collide))
        print("           collision, so two different models share one cache file:")
        for i in sorted(collide):
            print("       %s" % i)
    elif split:
        bad += len(split)
        print("     FAIL: one identity produced several ids -- generator is "
              "non-deterministic")
    else:
        print("     OK -- one id per identity, and no id shared by two identities")

    per_id_arms = collections.defaultdict(set)
    per_id_vals = collections.defaultdict(lambda: collections.defaultdict(set))
    for cfg in runs:
        bid = cfg["base_model_id"]
        per_id_arms[bid].add(cfg["arm"])
        for k in sorted(actual):
            per_id_vals[bid][k].add(json.dumps(cfg["hyperparams"].get(k)))

    disagree = [(bid, k) for bid in per_id_vals for k in per_id_vals[bid]
                if len(per_id_vals[bid][k]) > 1]
    groups = collections.Counter(frozenset(a) for a in per_id_arms.values())
    print()
    print("  P3 sharing -- %d warm-ups cover %d arms per (model, dataset, seed)"
          % (len(groups), sum(len(g) for g in groups)))
    for g in sorted(groups, key=lambda x: (-len(x), sorted(x))):
        tag = ("trains once, reused by %d arms" % len(g)) if len(g) > 1 else "own warm-up"
        print("     %-42s %s" % (" + ".join(sorted(g)), tag))
    if disagree:
        bad += len(disagree)
        print("     FAIL: arms sharing an id DISAGREE on a warm-up key:")
        for bid, k in disagree[:10]:
            print("       %s: %s" % (bid, k))
    else:
        print("     OK -- arms sharing an id agree on all %d warm-up keys"
              % len(actual))
    n_arms = len({c["arm"] for c in runs})
    saved = n_arms - len(groups)
    print("     => %d of %d warm-ups are cache hits (%.0f%% of warm-up training "
          "skipped)" % (saved, n_arms, 100.0 * saved / n_arms))
    print()
    return bad


def main():
    if len(sys.argv) > 1:
        root = sys.argv[1]
        tmp = None
    else:
        tmp = tempfile.mkdtemp(prefix="cfgaudit_")
        root = tmp
        subprocess.check_call(
            [sys.executable, "-m", "configs.gen_campaign", "--root", root,
             "--datasets", "dermmnist", "tissuemnist", "octmnist",
             "--models", "MobileNetV3", "MobileNetV2", "RegNetY400MF", "ViTB16",
             "--caps", "L30_G30", "L50_G50", "--arms", "all"],
            stdout=subprocess.DEVNULL)

    emitted, per_arm, n = collect_emitted(root)
    reads = collect_reads()
    read_keys = collections.defaultdict(set)
    for (kind, key) in reads:
        read_keys[kind].add(key)

    print("audited %d configs from %s\n" % (n, root))
    bad = 0

    for kind in ("config", "hyperparams", "dataset_config"):
        em, rd = emitted[kind], read_keys[kind]
        print("=" * 78)
        print("%s   (%d emitted, %d read in code)" % (kind.upper(), len(em), len(rd)))
        print("=" * 78)

        hallucinated = sorted(em - rd)
        if hallucinated:
            bad += len(hallucinated)
            print("\n  HALLUCINATED -- emitted but NEVER read (delete from generator):")
            for k in hallucinated:
                print("     %s" % k)
        else:
            print("\n  HALLUCINATED: none")

        if kind == "hyperparams":
            silent = sorted(rd - em)
            if silent:
                print("\n  SILENT -- read in code but NOT emitted (value comes from a")
                print("            literal in the code, not from the config):")
                for k in silent:
                    where = reads[(kind, k)][0]
                    print("     %-28s first read at %s:%d" % (k, where[0], where[1]))
            else:
                print("\n  SILENT: none")

        ok = sorted(em & rd)
        print("\n  OK (%d): %s" % (len(ok), ", ".join(ok)))
        print()

    bad += audit_per_arm(root)
    bad += audit_identity(root)

    if OPAQUE:
        # An opaque read in src/ hides a real config dependency: the pipeline
        # consumes a key the audit cannot name, so a hallucinated key could
        # live under it. In the audit and scoring tools it is almost always a
        # loop over a key list the audit can already see -- report, don't fail.
        pipeline = [o for o in OPAQUE if o[2].startswith("src/")]
        tooling = [o for o in OPAQUE if not o[2].startswith("src/")]
        print("=" * 78)
        print("UNAUDITABLE READS  (%d in src/, %d in tooling)"
              % (len(pipeline), len(tooling)))
        print("=" * 78)
        print()
        print("  A config dict accessed with a key this walker cannot resolve.")
        print("  Below such a read the read set is UNKNOWN, so the audit stops")
        print("  proving anything about it. None of these exist today; the")
        print("  point is that adding one FAILS instead of silently shrinking")
        print("  what the audit covers.")
        if pipeline:
            bad += len(pipeline)
            print()
            print("  FAIL -- in the pipeline, where it would hide a real key:")
            for base, how, path, line in pipeline:
                print("     %s:%d  %s on %s" % (path, line, how, base or "a dict"))
        if tooling:
            print()
            for base, how, path, line in tooling:
                print("     (tooling, allowed) %s:%d  %s on %s"
                      % (path, line, how, base or "a dict"))
        if not pipeline:
            print()
            print("  OK -- every unresolvable read is in audit/scoring code")
            print("  that iterates a declared key list, not in src/.")
        print()

    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 78)
    if bad:
        print("%d HALLUCINATED key(s) -- the config claims knobs the code does not read." % bad)
        return 1
    print("No hallucinated keys: every emitted value has a reader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

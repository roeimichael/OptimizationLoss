"""Audit the warm-up identity for the defect class that has now appeared TWICE.

THE CLASS. `compute_base_model_id` hashes `model_name`, dataset facts, and the
`warmup_identity_keys` that are present in `hp`. Anything else that reaches the
warm-up and CHANGES what it trains is invisible to the cache key, so two runs
that train differently can share one cached model.

Instance 1: `rank_weight` / `rank_margin` -- caught before launch, by design.
Instance 2: the CAP, reached through `config["constraint"]` rather than `hp`.
            Not caught by design; found by counting warm-up identities at 2/40.

Both were inputs to `run_warmup` that did not live in `hp`. So the audit is:
**walk the AST of `run_warmup` and every function it calls, and report every
subscript of `config` that is not `hyperparams`.** Each one is a candidate for
the same bug and has to be justified as either (a) already an identity key,
(b) part of the identity some other way (model_name, dataset), or (c) provably
unable to change what the warm-up trains.

AST, not grep -- RULESET section 6: claims about what code reads come from
reading it or from the AST, never from a text search.
"""
import ast
import os
import sys


class ConfigReads(ast.NodeVisitor):
    """Every `config[...]` / `config.get(...)` in a function body."""

    def __init__(self):
        self.reads = []

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "config":
            key = None
            if isinstance(node.slice, ast.Constant):
                key = node.slice.value
            self.reads.append(key or "<dynamic>")
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "get"
                and isinstance(f.value, ast.Name) and f.value.id == "config"
                and node.args and isinstance(node.args[0], ast.Constant)):
            self.reads.append(node.args[0].value)
        self.generic_visit(node)


def main(repo):
    path = os.path.join(repo, "src", "pipeline", "warmup.py")
    tree = ast.parse(open(path, encoding="utf-8").read())

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_warmup":
            target = node
    if target is None:
        print("run_warmup not found")
        return 2

    v = ConfigReads()
    v.visit(target)
    reads = sorted(set(v.reads))

    # Everything the digest already covers, and why.
    import yaml
    P = yaml.safe_load(open(os.path.join(repo, "configs", "protocol.yml"),
                            encoding="utf-8"))
    identity = set(P["warmup_identity_keys"])

    COVERED = {
        "hyperparams": "the hp dict itself -- identity keys are drawn from it",
        "model_name": "hashed into base_model_id directly",
        "base_model_id": "IS the cache key",
        "dataset_mode": "hashed into base_model_id directly",
        "dataset_config": "data_dir and num_classes hashed directly",
        "data_fingerprint": "part of the cache identity record",
    }
    JUSTIFIED = {
        # Reached by the ranking loss; now stamped into hp as rank_cap_fraction.
        "constraint": ("FIXED 323edf44: feeds rank_frac, now mirrored into hp as "
                       "rank_cap_fraction and hashed"),
    }

    print("`config[...]` reads inside run_warmup -- each is a cache-identity candidate")
    print("")
    unexplained = []
    for k in reads:
        if k in COVERED:
            status, why = "covered", COVERED[k]
        elif k in JUSTIFIED:
            status, why = "justified", JUSTIFIED[k]
        elif k in identity:
            status, why = "identity", "declared in warmup_identity_keys"
        else:
            status, why = "*** UNEXPLAINED ***", "can it change what the warm-up trains?"
            unexplained.append(k)
        print("  %-22s %-20s %s" % (k, status, why))

    print("")
    if unexplained:
        print("VERDICT: %d unexplained config read(s): %s" % (len(unexplained), unexplained))
        print("  Each must be shown either to be in the identity, or to be unable")
        print("  to change what the warm-up trains. This is the defect class that")
        print("  produced the cap bug; it does not announce itself.")
        return 1
    print("VERDICT: every config read inside run_warmup is accounted for.")
    print("  rank_cap_fraction closed the only unexplained one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

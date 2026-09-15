"""Every hyperparameter the warm-up READS must be a declared identity key.

WHY THIS IS DERIVED AND NOT A HARDCODED LIST.

The same defect has now appeared twice, and both times the existing tests passed:

  1. `rank_weight` / `rank_margin` -- caught by design, before launch.
  2. `rank_cap_fraction` -- the CAP, reaching the loss through
     `config["constraint"]` instead of `hp`. Invisible to the digest, so L80 and
     L90 shared one cached warm-up and L90 became a model trained for the wrong
     cut. Found at 2/40 runs by COUNTING warm-up identities, not by any test.
  3. `rank_min_group` -- latent, found by auditing the class afterwards. Nothing
     sweeps it today, so nothing collides today; the dose analysis points
     straight at lowering it, which is when it would have bitten.

A test that lists the keys by hand would have to be edited by the same person
who forgets to declare the key, so it protects nothing. This one reads
`read_rank_config`'s AST and requires every `hp` key it consumes to be declared
in `warmup_identity_keys`. A new knob is then covered the moment it is written.

The general rule, from LEDGER PART 1: **an identity key must cover every input
that changes the warm-up.** The cache cannot see intent, only the digest.
"""
import ast
import inspect

import yaml

from configs.gen_campaign import load_protocol


def _hp_keys_read_by(func):
    """Every `hp[...]` / `hp.get(...)` constant key inside a function."""
    tree = ast.parse(inspect.getsource(func))
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "hp"
                and isinstance(node.slice, ast.Constant)):
            keys.add(node.slice.value)
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "get"
                    and isinstance(f.value, ast.Name) and f.value.id == "hp"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                keys.add(node.args[0].value)
    return keys


def test_every_hp_key_the_ranking_loss_reads_is_a_declared_identity_key():
    from src.training.rank_loss import read_rank_config

    read = _hp_keys_read_by(read_rank_config)
    assert read, "the AST walk found no hp keys; the probe is broken, not the code"

    declared = set(load_protocol()["warmup_identity_keys"])
    missing = sorted(read - declared)
    assert not missing, (
        "read_rank_config consumes %s, which change what the warm-up trains but "
        "are NOT in warmup_identity_keys. Two arms differing only in one of them "
        "would share a cached warm-up -- the defect recorded twice in LEDGER "
        "PART 1." % missing)


def test_the_cap_fraction_reaches_the_loss_and_is_declared():
    """The second instance, pinned at its actual mechanism.

    `rank_cap_fraction` is not read by `read_rank_config` -- it is stamped by
    `gen_campaign` to mirror `config["constraint"][0]`, which `run_warmup`
    derives `rank_frac` from. The AST probe above cannot see it, so it is
    asserted directly, together with the guard that keeps the two in step.
    """
    declared = set(load_protocol()["warmup_identity_keys"])
    assert "rank_cap_fraction" in declared

    import src.pipeline.warmup as warmup
    source = inspect.getsource(warmup.run_warmup)
    assert "rank_cap_fraction" in source, (
        "run_warmup no longer checks the stamp against the cap it trains; the "
        "digest and the trained cut can now drift apart")


def test_controls_carry_none_of_the_rank_identity_keys():
    """The safety property that makes these keys addable at all.

    `compute_base_model_id` hashes only keys PRESENT in hp, so declaring a new
    rank key must leave every non-rank digest untouched. If a control ever
    started carrying one, adding a key would silently invalidate the entire
    cached corpus.
    """
    from configs.gen_campaign import build_hyperparams

    P = load_protocol()
    rank_keys = {"rank_weight", "rank_margin", "rank_min_group", "rank_cap_fraction"}
    for arm in ("clip", "focal_clip", "aug_clip"):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        leaked = sorted(rank_keys & set(hp))
        assert not leaked, (
            "control arm %s carries %s; declaring a rank key would move its "
            "digest and orphan its cached warm-up" % (arm, leaked))

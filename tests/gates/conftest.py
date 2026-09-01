"""Shared scaffolding for the STAGED PRE-FLIGHT GATES.

WHY THESE EXIST, AND WHY THEY ARE SEPARATE FROM `tests/test_pipeline.py`.
The main suite gates the CODE. These gate the EXPERIMENT: they encode failure
modes this project actually paid for, at the stage of the pipeline where each
one is still cheap to catch. Every gate below traces to a measured incident in
`docs/FRAMEWORK.md` section 2 or `docs/archive/REJECTED_full_2026-08-18.md`,
and each says which.

    stage 1  data      the slice, before a single image is loaded
    stage 2  budget    the cap arithmetic, before a config is written
    stage 3  model     the backbone and the warm-up cache, before training
    stage 4  grid      apples-to-apples across the campaign, before launch
    stage 5  trainlog  what the optimiser actually did, during and after
    stage 6  results   what may be read off the output, before any claim

Run one stage with `python -m scripts.preflight --stage budget`, or all of
them with `--stage all`. `pytest tests/gates -m stage4_grid` is the same thing.

THREE RULES FOR ANYTHING ADDED HERE, because a gate suite that fails for
uninteresting reasons gets switched off:

1. **A gate encodes a MEASURED failure, not a possibility.** "the config is not
   None" is noise. "`class_balanced` is byte-identical to `clip` in 24/24
   because iwildcam's TRAIN set is exactly 2500/class" is a gate.
2. **A gate is not done until a NEGATIVE CONTROL shows it FAIL.** Build the
   broken input in the same test and assert the check rejects it. A gate that
   has never failed has never been shown to work.
3. **TABLE-DRIVEN, not parametrised into hundreds of node ids.** One test
   function walks a table of cases, collects EVERY failure, and reports them
   together. The point is a short, readable report, not a large test count.
"""
import io
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The only runnable dataset, and the four backbones the paper claims. Both
# lists are closed: an arm or a backbone outside them is a defect, not a
# variant. FRAMEWORK 2(n) removed the other three datasets; ShuffleNetV2 and
# the small CNNs appear in no .tex file.
RUNNABLE_DATASETS = ("iwildcam",)
CLAIMED_BACKBONES = ("ViTB16", "MobileNetV3", "MobileNetV2", "RegNetY400MF")
HEADLINE_BACKBONE = "ViTB16"          # fixed a priori 2026-08-20, FRAMEWORK 1-pre
CAPPED_CLASSES = (2, 7)               # impala, cattle


def rel(*parts):
    return os.path.join(ROOT, *parts)


def read(*parts):
    return io.open(rel(*parts), encoding="utf-8").read()


def load_yaml(*parts):
    return yaml.safe_load(read(*parts))


@pytest.fixture(scope="session")
def protocol():
    return load_yaml("configs", "protocol.yml")


@pytest.fixture(scope="session")
def windows():
    from configs.task_cells import load_windows
    return load_windows()


@pytest.fixture(scope="session")
def slice_dir():
    """The iwildcam oodslice, or a skip.

    NOT a silent fallback to synthetic data. A gate that quietly measures a
    toy when the real slice is absent reports a pass about nothing, which is
    the exact defect class FRAMEWORK 2(z25) is about.
    """
    d = rel("data", "iwildcam", "oodslice")
    if not os.path.exists(os.path.join(d, "test_meta.csv")):
        pytest.skip("iwildcam/oodslice not on this machine -- gate NOT run")
    return d


def report(failures, what):
    """Collect-then-assert, so one run names every problem rather than the
    first. `failures` is a list of strings."""
    assert not failures, (
        "%d %s:\n  - %s" % (len(failures), what, "\n  - ".join(failures)))


def items_from_f1(d_f1, K, n):
    """cc-F1 delta -> ITEMS. With exactly K predictions emitted,
    `F1 = 2*TP/(K+n)`, so `items = dF1 * (K+n)/2`. Quoting a raw F1 delta
    hides that the whole gap from `clip` to a PERFECT allocator is 1.9-9.9
    items -- 0.02 is not a small effect there, it can be the entire headroom.
    """
    return d_f1 * (K + n) / 2.0

"""Maintained behavioral regression fixtures."""

import pytest

pytestmark = pytest.mark.stage6_results


def test_headroom_keys_the_cell_by_BACKBONE_and_refuses_when_it_cannot():
    import pytest
    from scripts.headroom import run_axes

    parts = ("results", "dom1", "MobileNetV2", "iwildcam", "L80_G95", "tralo", "seed_1")
    assert run_axes(parts) == ("MobileNetV2", "iwildcam")
    other = ("results", "dom1", "MobileNetV3", "iwildcam", "L80_G95", "tralo", "seed_1")
    assert run_axes(other) != run_axes(parts)
    assert run_axes(parts[:-1] + ("seed_4",)) == run_axes(parts)
    assert run_axes(parts[:-2] + ("alm", "seed_1")) == run_axes(parts)
    with pytest.raises(SystemExit):
        run_axes(("seed_1",))
    with pytest.raises(SystemExit):
        run_axes(("iwildcam", "L80_G95", "tralo", "seed_1"))

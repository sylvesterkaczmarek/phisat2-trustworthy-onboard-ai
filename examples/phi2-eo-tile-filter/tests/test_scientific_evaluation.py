from __future__ import annotations

import math

import pytest

from phi2_tile_filter.calibrate_threshold import clopper_pearson_lower_bound


def test_clopper_pearson_one_sided_lower_bound_for_all_successes() -> None:
    bound = clopper_pearson_lower_bound(10, 10, confidence_level=0.95)
    assert bound == pytest.approx(math.pow(0.05, 1.0 / 10.0))
    assert bound < 1.0


def test_clopper_pearson_lower_bound_handles_zero_successes() -> None:
    assert clopper_pearson_lower_bound(0, 10, confidence_level=0.95) == 0.0


def test_clopper_pearson_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(1, 10, confidence_level=1.0)

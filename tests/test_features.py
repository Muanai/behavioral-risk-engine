import numpy as np
import pytest
from src.features_numba import get_max_consecutive_late, get_velocity_slope


def test_late_streak_logic():
    pay_matrix = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0],
    ], dtype=np.int32)

    expected = np.array([2, 0, 6, 1], dtype=np.int32)

    result = get_max_consecutive_late(pay_matrix)

    np.testing.assert_array_equal(result, expected, err_msg="Streak logic is incorrect!")


def test_velocity_slope_logic():
    ratio_matrix = np.array([
        [0.1, 0.5, 0.9],
        [0.9, 0.5, 0.1],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)

    result = get_velocity_slope(ratio_matrix)

    assert result[0] > 0, "Rising slope should be positive"
    assert result[0] == pytest.approx(0.4, abs=1e-5)

    assert result[1] < 0, "Falling slope should be negative"
    assert result[1] == pytest.approx(-0.4, abs=1e-5)

    assert result[2] == pytest.approx(0.0, abs=1e-5)
    assert result[3] == pytest.approx(0.0, abs=1e-5)


def test_velocity_robustness():
    ratio_matrix = np.array([
        [np.nan, 0.5, 0.5],
        [np.inf, 0.5, 0.5],
    ], dtype=np.float64)

    result = get_velocity_slope(ratio_matrix)

    assert not np.isnan(result).any(), "Output contains NaN! Handler failed."
    assert not np.isinf(result).any(), "Output contains Inf! Handler failed."
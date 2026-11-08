import math

import numpy as np

from data_process.tool.repair_nero_dynamics_h5 import (
    causal_derivative,
    causal_one_pole_lowpass,
)


def test_causal_derivative_uses_each_measured_timestamp_interval():
    values = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64)
    timestamps_s = np.asarray([0.0, 0.5, 1.5], dtype=np.float64)

    actual = causal_derivative(values, timestamps_s)

    np.testing.assert_allclose(actual[:, 0], [0.0, 2.0, 2.0])


def test_causal_lowpass_matches_nero_one_pole_discretization():
    values = np.asarray([[0.0], [1.0], [1.0]], dtype=np.float64)
    timestamps_s = np.asarray([0.0, 0.01, 0.03], dtype=np.float64)

    actual = causal_one_pole_lowpass(values, timestamps_s, cutoff_hz=3.0)

    alpha_1 = 1.0 - math.exp(-2.0 * math.pi * 3.0 * 0.01)
    alpha_2 = 1.0 - math.exp(-2.0 * math.pi * 3.0 * 0.02)
    expected_1 = alpha_1
    expected_2 = expected_1 + alpha_2 * (1.0 - expected_1)
    np.testing.assert_allclose(actual[:, 0], [0.0, expected_1, expected_2])

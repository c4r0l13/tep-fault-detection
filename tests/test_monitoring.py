import numpy as np
import pytest

from tep_fdd.metrics import detection_delay, false_alarm_rate, fault_detection_rate
from tep_fdd.monitoring import PCAMonitor, lag_matrix


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 3))
    W = rng.normal(size=(3, 10))
    return latent @ W + 0.1 * rng.normal(size=(2000, 10))


def test_lag_matrix_shape_and_values():
    X = np.arange(10, dtype=float).reshape(5, 2)
    L = lag_matrix(X, lags=1)
    assert L.shape == (5, 4)
    np.testing.assert_array_equal(L[2], [4, 5, 2, 3])
    np.testing.assert_array_equal(L[0, 2:], X[0])  # padded with first observation


def test_lag_matrix_does_not_cross_groups():
    X = np.arange(8, dtype=float).reshape(4, 2)
    L = lag_matrix(X, lags=1, groups=np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(L[2, 2:], X[2])  # first row of group 1 uses itself


def test_false_alarm_rate_close_to_alpha(normal_data):
    mon = PCAMonitor(var_explained=0.9, alpha=0.01).fit(normal_data[:1500])
    alarms = mon.predict(normal_data[1500:])
    assert alarms.mean() < 0.06


def test_detects_mean_shift(normal_data):
    mon = PCAMonitor().fit(normal_data[:1500])
    faulty = normal_data[1500:].copy()
    faulty[:, 4] += 4 * normal_data[:, 4].std()
    assert mon.predict(faulty).mean() > 0.9


def test_calibrate_sets_empirical_quantiles(normal_data):
    mon = PCAMonitor().fit(normal_data[:1000]).calibrate(normal_data[1000:], quantile=0.99)
    t2, spe = mon.statistics(normal_data[1000:])
    assert np.isclose((t2 > mon.t2_limit_).mean(), 0.01, atol=0.005)
    assert np.isclose((spe > mon.spe_limit_).mean(), 0.01, atol=0.005)


def test_spe_contributions_point_to_faulty_variable(normal_data):
    mon = PCAMonitor().fit(normal_data[:1500])
    faulty = normal_data[1500:].copy()
    faulty[:, 7] += 5.0
    contrib = mon.spe_contributions(faulty).mean(axis=0)
    assert contrib.argmax() == 7


def test_metrics():
    alarms = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    assert false_alarm_rate(alarms, onset=4) == 0.25
    assert fault_detection_rate(alarms, onset=4) == 0.75
    assert detection_delay(alarms, onset=4, consecutive=3) == 0
    assert detection_delay(alarms, onset=4, consecutive=4) is None

import numpy as np

from tep_fdd.data import FAULT_DESCRIPTIONS, VARIABLE_CODES, VARIABLE_NAMES
from tep_fdd.diagnosis import window_features


def test_window_features_shape_and_no_leakage():
    X = np.vstack([np.zeros((5, 2)), np.full((5, 2), 10.0)])
    groups = np.array([0] * 5 + [1] * 5)
    F = window_features(X, groups, window=3)
    assert F.shape == (10, 6)
    # first row of group 1: rolling mean must only see group 1 values
    assert F[5, 2] == 10.0
    assert F[5, 4] == 0.0  # std of a single value -> 0


def test_metadata_consistency():
    assert len(VARIABLE_CODES) == len(VARIABLE_NAMES) == 52
    assert set(FAULT_DESCRIPTIONS) == set(range(22))

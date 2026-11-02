"""Loading utilities for the Tennessee Eastman Process (TEP) benchmark.

Dataset: Braatz et al. TEP simulation data (52 variables, 3-minute sampling).

* ``d00.dat``      - normal operation, training (stored transposed: 52 x 500).
* ``dXX.dat``      - fault XX, training (480 samples, fault active from the start).
* ``d00_te.dat``   - normal operation, test (960 samples).
* ``dXX_te.dat``   - fault XX, test (960 samples, fault introduced after sample 160).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_FAULTS = 21
FAULT_ONSET_TEST = 160  # samples (8 h at 3-min sampling)
SAMPLE_MINUTES = 3

XMEAS_NAMES = [
    "A feed (stream 1)",
    "D feed (stream 2)",
    "E feed (stream 3)",
    "A and C feed (stream 4)",
    "Recycle flow (stream 8)",
    "Reactor feed rate (stream 6)",
    "Reactor pressure",
    "Reactor level",
    "Reactor temperature",
    "Purge rate (stream 9)",
    "Product separator temperature",
    "Product separator level",
    "Product separator pressure",
    "Product separator underflow",
    "Stripper level",
    "Stripper pressure",
    "Stripper underflow",
    "Stripper temperature",
    "Stripper steam flow",
    "Compressor work",
    "Reactor cooling water outlet temp",
    "Separator cooling water outlet temp",
    *[f"Reactor feed analysis {c}" for c in "ABCDEF"],
    *[f"Purge gas analysis {c}" for c in "ABCDEFGH"],
    *[f"Product analysis {c}" for c in "DEFGH"],
]
XMV_NAMES = [
    "D feed flow",
    "E feed flow",
    "A feed flow",
    "A and C feed flow",
    "Compressor recycle valve",
    "Purge valve",
    "Separator pot liquid flow",
    "Stripper liquid product flow",
    "Stripper steam valve",
    "Reactor cooling water flow",
    "Condenser cooling water flow",
]
VARIABLE_CODES = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]
VARIABLE_NAMES = XMEAS_NAMES + XMV_NAMES

FAULT_DESCRIPTIONS = {
    0: "Normal operation",
    1: "A/C feed ratio, B composition constant (step)",
    2: "B composition, A/C ratio constant (step)",
    3: "D feed temperature (step)",
    4: "Reactor cooling water inlet temperature (step)",
    5: "Condenser cooling water inlet temperature (step)",
    6: "A feed loss (step)",
    7: "C header pressure loss (step)",
    8: "A, B, C feed composition (random variation)",
    9: "D feed temperature (random variation)",
    10: "C feed temperature (random variation)",
    11: "Reactor cooling water inlet temperature (random variation)",
    12: "Condenser cooling water inlet temperature (random variation)",
    13: "Reaction kinetics (slow drift)",
    14: "Reactor cooling water valve (sticking)",
    15: "Condenser cooling water valve (sticking)",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Stream 4 valve fixed at steady-state position",
}

assert len(VARIABLE_CODES) == len(VARIABLE_NAMES) == 52


@dataclass(frozen=True)
class TEPData:
    """Container with train/test matrices and labels."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Index of each test sample inside its 960-sample run (0..959)
    t_test: np.ndarray


def _read_dat(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape[0] == 52 and arr.shape[1] != 52:  # d00.dat is stored transposed
        arr = arr.T
    if arr.shape[1] != 52:
        raise ValueError(f"Unexpected shape {arr.shape} in {path.name}")
    return arr


def load_run(data_dir: str | Path, fault: int, split: str) -> np.ndarray:
    """Load one simulation run as a (n_samples, 52) array."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    suffix = "_te" if split == "test" else ""
    return _read_dat(Path(data_dir) / f"d{fault:02d}{suffix}.dat")


def load_tep(data_dir: str | Path, faults: range | list[int] | None = None) -> TEPData:
    """Load all runs and build labelled train/test sets.

    Test labels follow the benchmark convention: samples before the fault onset
    (first 160) are labelled as normal (0).
    """
    faults = list(range(N_FAULTS + 1)) if faults is None else list(faults)
    Xtr, ytr, Xte, yte, tte = [], [], [], [], []
    for f in faults:
        tr = load_run(data_dir, f, "train")
        te = load_run(data_dir, f, "test")
        Xtr.append(tr)
        ytr.append(np.full(len(tr), f))
        labels = np.full(len(te), f)
        if f != 0:
            labels[:FAULT_ONSET_TEST] = 0
        Xte.append(te)
        yte.append(labels)
        tte.append(np.arange(len(te)))
    return TEPData(
        np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xte), np.concatenate(yte), np.concatenate(tte)
    )


def to_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=VARIABLE_CODES)

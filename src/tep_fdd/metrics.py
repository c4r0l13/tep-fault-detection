"""Evaluation metrics used in the process-monitoring literature."""

from __future__ import annotations

import numpy as np


def fault_detection_rate(alarms: np.ndarray, onset: int) -> float:
    """Share of post-onset samples flagged as faulty (higher is better)."""
    post = np.asarray(alarms)[onset:]
    return float(post.mean()) if len(post) else float("nan")


def false_alarm_rate(alarms: np.ndarray, onset: int) -> float:
    """Share of pre-onset (normal) samples flagged as faulty (lower is better)."""
    pre = np.asarray(alarms)[:onset]
    return float(pre.mean()) if len(pre) else float("nan")


def detection_delay(alarms: np.ndarray, onset: int, consecutive: int = 6) -> int | None:
    """Samples from fault onset until ``consecutive`` alarms in a row.

    Requiring several consecutive alarms is how alarm systems avoid chattering
    in practice. Returns ``None`` if the fault is never confirmed.
    """
    post = np.asarray(alarms)[onset:].astype(int)
    if consecutive <= 0:
        raise ValueError("consecutive must be positive")
    run = 0
    for i, a in enumerate(post):
        run = run + 1 if a else 0
        if run >= consecutive:
            return i - consecutive + 1
    return None

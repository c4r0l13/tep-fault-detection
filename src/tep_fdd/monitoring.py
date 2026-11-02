"""Multivariate statistical process monitoring (MSPM).

Implements the industry-standard PCA monitor with Hotelling's T^2 and SPE (Q)
statistics, plus its dynamic extension (DPCA) that augments each observation
with time-lagged copies to capture process autocorrelation.

References
----------
* Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals
  associated with principal component analysis. Technometrics.
* Ku, W., Storer, R. H., & Georgakis, C. (1995). Disturbance detection and
  isolation by dynamic principal component analysis. Chemom. Intell. Lab. Syst.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def lag_matrix(X: np.ndarray, lags: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Stack ``[x_t, x_{t-1}, ..., x_{t-lags}]`` row-wise.

    Lags never cross run boundaries: when ``groups`` is given, the first
    ``lags`` rows of every group are padded with the earliest available
    observation of that group, so the output keeps the input length and
    stays aligned with labels.
    """
    if lags == 0:
        return X
    groups = np.zeros(len(X), dtype=int) if groups is None else np.asarray(groups)
    out = np.empty((X.shape[0], X.shape[1] * (lags + 1)), dtype=float)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        Xg = X[idx]
        blocks = []
        for k in range(lags + 1):
            shifted = np.vstack([np.repeat(Xg[:1], k, axis=0), Xg[: len(Xg) - k]]) if k else Xg
            blocks.append(shifted)
        out[idx] = np.hstack(blocks)
    return out


@dataclass
class PCAMonitor:
    """PCA-based fault detector using T^2 and SPE statistics.

    Parameters
    ----------
    var_explained:
        Cumulative variance used to choose the number of retained components.
    alpha:
        Significance level of the control limits (0.01 -> 99% limits).
    lags:
        Number of time lags; ``lags > 0`` turns the model into Dynamic PCA.
    """

    var_explained: float = 0.90
    alpha: float = 0.01
    lags: int = 0

    mean_: np.ndarray = field(init=False, repr=False)
    std_: np.ndarray = field(init=False, repr=False)
    P_: np.ndarray = field(init=False, repr=False)
    eigvals_: np.ndarray = field(init=False, repr=False)
    n_components_: int = field(init=False)
    t2_limit_: float = field(init=False)
    spe_limit_: float = field(init=False)

    def fit(self, X: np.ndarray, groups: np.ndarray | None = None) -> PCAMonitor:
        Xl = lag_matrix(X, self.lags, groups)
        self.mean_ = Xl.mean(axis=0)
        self.std_ = Xl.std(axis=0, ddof=1)
        self.std_[self.std_ < 1e-12] = 1.0
        Z = (Xl - self.mean_) / self.std_
        n = Z.shape[0]
        cov = (Z.T @ Z) / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = np.clip(eigvals[order], 0, None), eigvecs[:, order]
        cum = np.cumsum(eigvals) / eigvals.sum()
        a = int(np.searchsorted(cum, self.var_explained) + 1)
        self.n_components_ = a
        self.P_ = eigvecs[:, :a]
        self.eigvals_ = eigvals

        # Hotelling T^2 limit (F distribution)
        f_crit = stats.f.ppf(1 - self.alpha, a, n - a)
        self.t2_limit_ = a * (n - 1) * (n + 1) / (n * (n - a)) * f_crit

        # SPE / Q limit (Jackson & Mudholkar, 1979)
        resid = eigvals[a:]
        th1, th2, th3 = (np.sum(resid**k) for k in (1, 2, 3))
        h0 = 1 - 2 * th1 * th3 / (3 * th2**2)
        z = stats.norm.ppf(1 - self.alpha)
        self.spe_limit_ = th1 * (z * np.sqrt(2 * th2 * h0**2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1**2) ** (
            1 / h0
        )
        return self

    def calibrate(
        self, X_cal: np.ndarray, quantile: float | None = None, groups: np.ndarray | None = None
    ) -> PCAMonitor:
        """Replace the theoretical limits with empirical quantiles on held-out normal data.

        Theoretical limits assume i.i.d. Gaussian data and a training set that
        represents all normal variability. Neither holds for TEP (autocorrelated
        signals, only 500 training samples), which produces many false alarms.
        Calibrating on an independent normal run fixes the false-alarm rate
        without touching any faulty data.
        """
        q = 1 - self.alpha if quantile is None else quantile
        t2, spe = self.statistics(X_cal, groups)
        self.t2_limit_ = float(np.quantile(t2, q))
        self.spe_limit_ = float(np.quantile(spe, q))
        return self

    def _scale(self, X: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
        return (lag_matrix(X, self.lags, groups) - self.mean_) / self.std_

    def statistics(self, X: np.ndarray, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (T^2, SPE) for every row of ``X``."""
        Z = self._scale(X, groups)
        T = Z @ self.P_
        t2 = np.sum(T**2 / self.eigvals_[: self.n_components_], axis=1)
        resid = Z - T @ self.P_.T
        spe = np.sum(resid**2, axis=1)
        return t2, spe

    def predict(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Alarm (1) if either statistic exceeds its control limit."""
        t2, spe = self.statistics(X, groups)
        return ((t2 > self.t2_limit_) | (spe > self.spe_limit_)).astype(int)

    def spe_contributions(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Per-variable SPE contributions (squared residuals), summed over lags.

        Returns an array of shape (n_samples, n_original_variables).
        """
        Z = self._scale(X, groups)
        resid = Z - (Z @ self.P_) @ self.P_.T
        contrib = resid**2
        n_vars = X.shape[1]
        return contrib.reshape(len(X), self.lags + 1, n_vars).sum(axis=1)

"""Deterministic linear cross-asset scenario and stress analytics.

All inputs are explicitly supplied; no external market data is fetched.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty finite 1-D vector")
    return arr


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or not all(arr.shape) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty finite 2-D matrix")
    return arr


def scenario_returns(exposures: ArrayLike, factor_shocks: ArrayLike) -> NDArray[np.float64]:
    """Return asset impacts from linear factor exposures.

    exposures: (n_assets, n_factors); factor_shocks: (n_factors,).
    Impacts and shocks are decimal returns, not currency amounts.
    """
    b = _matrix(exposures, "exposures")
    s = _vector(factor_shocks, "factor_shocks")
    if b.shape[1] != s.size:
        raise ValueError("factor dimension mismatch")
    return b @ s


def portfolio_impact(weights: ArrayLike, asset_impacts: ArrayLike) -> float:
    """Weighted portfolio return impact; weights need not sum to one."""
    w = _vector(weights, "weights")
    r = _vector(asset_impacts, "asset_impacts")
    if w.size != r.size:
        raise ValueError("asset dimension mismatch")
    return float(w @ r)


def scenario_grid(exposures: ArrayLike, shocks: ArrayLike) -> NDArray[np.float64]:
    """Asset return impacts for rows of scenarios: (n_scenarios, n_assets)."""
    b = _matrix(exposures, "exposures")
    s = _matrix(shocks, "shocks")
    if s.shape[1] != b.shape[1]:
        raise ValueError("factor dimension mismatch")
    return s @ b.T


def factor_sensitivity(weights: ArrayLike, exposures: ArrayLike) -> NDArray[np.float64]:
    """Portfolio return sensitivity to one unit of each factor shock."""
    b = _matrix(exposures, "exposures")
    w = _vector(weights, "weights")
    if w.size != b.shape[0]:
        raise ValueError("asset dimension mismatch")
    return w @ b


def gaussian_factor_scenarios(
    mean: ArrayLike,
    covariance: ArrayLike,
    n_scenarios: int,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Reproducible multivariate normal factor shocks.

    Covariance must be symmetric positive semidefinite within tolerance.
    """
    mu = _vector(mean, "mean")
    cov = _matrix(covariance, "covariance")
    if cov.shape != (mu.size, mu.size):
        raise ValueError("covariance dimension mismatch")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(cov)) < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    if (
        isinstance(n_scenarios, (bool, np.bool_))
        or not isinstance(n_scenarios, (int, np.integer))
        or n_scenarios < 1
    ):
        raise ValueError("n_scenarios must be a positive integer")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return (
        np.random.default_rng(seed).standard_normal((n_scenarios, mu.size)) @ root.T + mu
    )


def stress_loss(
    weights: ArrayLike, exposures: ArrayLike, factor_shocks: ArrayLike, *, notional: float
) -> float:
    """Signed currency loss: positive means loss, negative means gain."""
    if not np.isfinite(notional) or notional < 0:
        raise ValueError("notional must be finite and nonnegative")
    return -portfolio_impact(weights, scenario_returns(exposures, factor_shocks)) * float(notional)

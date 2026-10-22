"""Finite-state Markov regime dynamics for scenario research."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _transition_matrix(
    transition: Sequence[Sequence[float]],
) -> np.ndarray:
    """Validate a finite row-stochastic transition matrix."""
    matrix = np.asarray(transition, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("transition must be a nonempty square matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition probabilities must be finite")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("transition probabilities must lie in [0, 1]")
    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("each transition row must sum to one")

    return matrix


def regime_distribution(
    transition: Sequence[Sequence[float]],
    initial: Sequence[float],
    steps: int,
) -> np.ndarray:
    """Propagate a regime probability distribution by a Markov chain."""
    matrix = _transition_matrix(transition)
    probabilities = np.asarray(initial, dtype=float)

    if probabilities.shape != (matrix.shape[0],):
        raise ValueError("initial distribution dimensions mismatch")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("initial probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("initial probabilities must lie in [0, 1]")
    if not math.isclose(
        float(probabilities.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("initial probabilities must sum to one")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    return probabilities @ np.linalg.matrix_power(matrix, steps)


def simulate_regimes(
    transition: Sequence[Sequence[float]],
    initial_state: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Simulate a reproducible finite-state Markov regime path.

    The output includes the initial state and `steps` transitions.
    """
    matrix = _transition_matrix(transition)
    n_states = matrix.shape[0]

    if (
        isinstance(initial_state, bool)
        or not isinstance(initial_state, int)
        or not 0 <= initial_state < n_states
    ):
        raise ValueError("initial_state is invalid")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    path = np.empty(steps + 1, dtype=int)
    path[0] = initial_state

    for t in range(steps):
        path[t + 1] = rng.choice(n_states, p=matrix[path[t]])

    return path

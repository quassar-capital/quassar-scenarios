"""Mathematical and adversarial tests for finite-state Markov regimes."""

import numpy as np
import pytest

from quassar_scenarios.regime_switching import (
    regime_distribution,
    simulate_regimes,
)

TRANSITION = [[0.9, 0.1], [0.2, 0.8]]


def test_zero_steps_preserves_initial_distribution():
    result = regime_distribution(TRANSITION, [1.0, 0.0], 0)
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_one_step_matches_transition_row():
    result = regime_distribution(TRANSITION, [1.0, 0.0], 1)
    np.testing.assert_allclose(result, [0.9, 0.1])


def test_two_steps_match_analytical_result():
    result = regime_distribution(TRANSITION, [1.0, 0.0], 2)
    np.testing.assert_allclose(result, [0.83, 0.17])


@pytest.mark.parametrize("steps", [0, 1, 2, 10, 100])
def test_probability_conservation(steps):
    result = regime_distribution(TRANSITION, [0.3, 0.7], steps)
    assert np.all(result >= -1e-12)
    assert np.all(result <= 1.0 + 1e-12)
    assert np.isclose(result.sum(), 1.0, atol=1e-12)


def test_stationary_distribution():
    stationary = [2.0 / 3.0, 1.0 / 3.0]
    result = regime_distribution(TRANSITION, stationary, 50)
    np.testing.assert_allclose(result, stationary, atol=1e-12)


def test_long_run_convergence():
    result = regime_distribution(TRANSITION, [1.0, 0.0], 200)
    np.testing.assert_allclose(result, [2.0 / 3.0, 1.0 / 3.0], atol=1e-10)


def test_absorbing_state():
    transition = [[1.0, 0.0], [0.4, 0.6]]
    result = regime_distribution(transition, [1.0, 0.0], 100)
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_identity_transition():
    transition = np.eye(3)
    initial = [0.2, 0.3, 0.5]
    result = regime_distribution(transition, initial, 100)
    np.testing.assert_allclose(result, initial)


def test_simulation_reproducibility():
    first = simulate_regimes(TRANSITION, 0, 1000, seed=12345)
    second = simulate_regimes(TRANSITION, 0, 1000, seed=12345)
    np.testing.assert_array_equal(first, second)


def test_simulation_path_length_and_states():
    path = simulate_regimes(TRANSITION, 1, 100, seed=42)
    assert len(path) == 101
    assert path[0] == 1
    assert set(path).issubset({0, 1})


def test_simulation_absorbing_state():
    path = simulate_regimes(
        [[1.0, 0.0], [0.0, 1.0]],
        initial_state=1,
        steps=100,
        seed=42,
    )
    assert np.all(path == 1)


def test_simulation_zero_steps():
    path = simulate_regimes(TRANSITION, 1, 0, seed=42)
    np.testing.assert_array_equal(path, [1])


@pytest.mark.parametrize(
    "transition",
    [
        [[0.5, 0.5]],
        [[0.8, 0.3], [0.2, 0.8]],
        [[-0.1, 1.1], [0.2, 0.8]],
        [[float("nan"), 0.0], [0.2, 0.8]],
        [[float("inf"), 0.0], [0.2, 0.8]],
        [],
    ],
)
def test_invalid_transition_matrices(transition):
    with pytest.raises(ValueError):
        regime_distribution(transition, [1.0, 0.0], 1)


@pytest.mark.parametrize(
    "initial",
    [
        [0.5, 0.6],
        [-0.1, 1.1],
        [float("nan"), 0.0],
        [1.0],
    ],
)
def test_invalid_initial_distributions(initial):
    with pytest.raises(ValueError):
        regime_distribution(TRANSITION, initial, 1)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        regime_distribution(TRANSITION, [1.0, 0.0], steps)


@pytest.mark.parametrize("state", [-1, 2, True])
def test_invalid_initial_state(state):
    with pytest.raises(ValueError):
        simulate_regimes(TRANSITION, state, 10, seed=42)


@pytest.mark.parametrize("seed", [1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        simulate_regimes(TRANSITION, 0, 10, seed=seed)

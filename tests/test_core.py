"""Analytical reference and input-contract tests."""

import numpy as np
import pytest

from quassar_scenarios import (
    factor_sensitivity,
    gaussian_factor_scenarios,
    portfolio_impact,
    scenario_grid,
    scenario_returns,
    stress_loss,
)


@pytest.fixture
def exposures():
    return [[1.0, 0.5], [-0.2, 2.0]]


def test_scenario_returns(exposures):
    np.testing.assert_allclose(scenario_returns(exposures, [0.1, -0.02]), [0.09, -0.06])


def test_portfolio_impact():
    assert portfolio_impact([0.6, 0.4], [0.09, -0.06]) == pytest.approx(0.03)


def test_grid_matches_individual_scenarios(exposures):
    shocks = [[0.1, -0.02], [-0.03, 0.04]]
    actual = scenario_grid(exposures, shocks)
    expected = np.stack([scenario_returns(exposures, row) for row in shocks])
    np.testing.assert_allclose(actual, expected)


def test_sensitivity_finite_difference(exposures):
    w = [0.6, 0.4]
    s = np.array([0.02, -0.03])
    eps = 1e-6
    actual = factor_sensitivity(w, exposures)
    for i in range(2):
        direction = np.eye(2)[i] * eps
        fd = (
            portfolio_impact(w, scenario_returns(exposures, s + direction))
            - portfolio_impact(w, scenario_returns(exposures, s - direction))
        ) / (2 * eps)
        assert actual[i] == pytest.approx(fd)


def test_stress_loss_sign_and_notional(exposures):
    assert stress_loss(
        [0.6, 0.4], exposures, [0.1, -0.02], notional=1_000_000
    ) == pytest.approx(-30_000)


def test_gaussian_seed_reproducibility():
    a = gaussian_factor_scenarios([0, 0], [[1, 0.2], [0.2, 1]], 50, seed=7)
    b = gaussian_factor_scenarios([0, 0], [[1, 0.2], [0.2, 1]], 50, seed=7)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 2)


def test_gaussian_singular_covariance():
    a = gaussian_factor_scenarios([1, 2], [[0, 0], [0, 0]], 5, seed=0)
    np.testing.assert_allclose(a, [[1, 2]] * 5)


@pytest.mark.parametrize("bad", [[[1, 2, 3]], [[1, float("nan")]], [[1, float("inf")]]])
def test_invalid_exposure_dimensions_or_values(bad):
    with pytest.raises(ValueError):
        scenario_returns(bad, [1, 2])


@pytest.mark.parametrize("bad", [[1, 2, 3], [float("nan"), 0]])
def test_invalid_factor_shocks(bad):
    with pytest.raises(ValueError):
        scenario_returns([[1, 2]], bad)


@pytest.mark.parametrize("bad", [[[1, 2], [0, 1]], [[1, 0], [0, -1]], [[1, 2, 3], [2, 1, 0]]])
def test_invalid_covariance(bad):
    with pytest.raises(ValueError):
        gaussian_factor_scenarios([0, 0], bad, 10, seed=0)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_scenario_count(n):
    with pytest.raises(ValueError):
        gaussian_factor_scenarios([0], [[1]], n, seed=0)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        gaussian_factor_scenarios([0], [[1]], 2, seed=seed)


@pytest.mark.parametrize("notional", [-1, float("nan"), float("inf")])
def test_invalid_notional(notional):
    with pytest.raises(ValueError):
        stress_loss([1], [[1]], [1], notional=notional)


def test_portfolio_weight_mismatch():
    with pytest.raises(ValueError):
        portfolio_impact([1], [1, 2])


def test_grid_factor_mismatch():
    with pytest.raises(ValueError):
        scenario_grid([[1, 2]], [[1, 2, 3]])


def test_sensitivity_asset_mismatch():
    with pytest.raises(ValueError):
        factor_sensitivity([1, 2], [[1, 2]])

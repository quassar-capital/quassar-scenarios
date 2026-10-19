"""Quassar Scenarios: research-grade linear scenario analytics prototype."""

from .core import (
    factor_sensitivity,
    gaussian_factor_scenarios,
    portfolio_impact,
    scenario_grid,
    scenario_returns,
    stress_loss,
)

__all__ = [
    "factor_sensitivity",
    "gaussian_factor_scenarios",
    "portfolio_impact",
    "scenario_grid",
    "scenario_returns",
    "stress_loss",
]

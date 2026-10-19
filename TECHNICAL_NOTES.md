# Quassar Scenarios — Technical Notes (v0.1)

This is a **research prototype**, not a production risk-management system or an investment decision engine.

## Model and units

Let B be an (assets × factors) matrix of linear exposures, s a vector of factor shocks, and w a vector of portfolio weights. Asset return impacts are `r = B @ s`; the portfolio return impact is `w @ r`, and portfolio factor sensitivities are `w @ B`. For a notional N, the signed currency stress loss is `-N * w @ B @ s` (positive indicates loss). Inputs are decimal returns, not percentage points. The library does not infer market-data units or normalize weights.

The scenario grid accepts one scenario per row and returns one asset impact per column. Gaussian factor scenarios draw independent standard normals with a local seeded NumPy generator and transform them using the eigen-decomposition of a symmetric positive-semidefinite covariance matrix; a zero eigenvalue is permitted. The seed enables deterministic replay for the same environment and dependencies; exact values across library versions are not guaranteed.

## Limitations and next validation stages

Linear, static exposures omit nonlinear derivatives, liquidity, path dependency, contagion, endogenous responses, regime changes, transaction costs, and market-data provenance. Gaussian assumptions may understate tail risk. Covariance is a model input, not estimated or validated by this package. No data is downloaded, no trades are executed, and no investment advice is generated. Future phases require independent financial-model validation, sensitivity and stress calibration, property-based tests, and performance benchmarks.

## Developer checks

`python -m pip install -e '.[dev]'`

`ruff format --check . && ruff check . && pytest -q`

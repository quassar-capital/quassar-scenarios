"""Independent Monte Carlo validation of finite-state Markov regimes."""

import os

import numpy as np

from quassar_scenarios.regime_switching import (
    regime_distribution,
    simulate_regimes,
)


def main():
    transition = [[0.9, 0.1], [0.2, 0.8]]
    initial_state = 0
    steps = 50
    ci_mode = os.environ.get("QUASSAR_CI") == "1"
    replications = 500 if ci_mode else 5000
    experiments = 8 if ci_mode else 30

    theoretical = regime_distribution(transition, [1.0, 0.0], steps)

    errors = []

    for experiment in range(experiments):
        terminal_states = np.empty(replications, dtype=int)

        for replication in range(replications):
            seed = experiment * replications + replication
            path = simulate_regimes(transition, initial_state, steps, seed)
            terminal_states[replication] = path[-1]

        empirical = np.bincount(terminal_states, minlength=2) / replications

        errors.append(empirical - theoretical)

    errors = np.asarray(errors)
    mean_error = errors.mean(axis=0)
    standard_error = errors.std(axis=0, ddof=1) / np.sqrt(experiments)

    print("QUASSAR SCENARIOS — MARKOV MONTE CARLO VALIDATION")
    print(f"Independent experiments: {experiments}")
    print(f"Paths per experiment: {replications:,}")
    print(f"Transitions per path: {steps}")
    print(f"Theoretical distribution: {theoretical}")
    print(f"Mean probability error: {mean_error}")
    print(f"Standard error of mean: {standard_error}")

    if not np.all(np.abs(mean_error) <= 4.0 * standard_error + 1e-12):
        raise AssertionError("Monte Carlo means exceed the validation tolerance")

    print("MONTE CARLO VALIDATION PASSED")


if __name__ == "__main__":
    main()

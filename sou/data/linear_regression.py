from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LinearExperimentMeta:
    """Metadata for one linear-regression simulation scenario."""

    experiment: int
    spec: Optional[int]
    rep: int
    n_train: int
    n_total: int
    p: int
    noise_sd: float


def _split_xy(
    X: np.ndarray, y: np.ndarray, n_train: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _noise(n: int, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n) * float(noise_sd)


def experiment_1(
    spec: int = 0,
    rep: int = 1,
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
):
    """Paper Experiment 1: signals based on coefficients 1, ..., 10.

    The five specifications use (n, p) = (25, 20), (55, 50),
    (105, 100), (205, 200), and (405, 400). ``rep`` controls the
    number of non-zero coefficients through s = 10 * rep.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = {
        0: (25, 20),
        1: (55, 50),
        2: (105, 100),
        3: (205, 200),
        4: (405, 400),
    }
    if spec not in grid:
        raise ValueError("spec must be in {0,1,2,3,4}")
    n_train, p = grid[spec]

    n_total = n_train + 100
    X = rng.standard_normal((n_total, p))

    beta_base = np.arange(1, 11, dtype=float)
    beta_signal = np.repeat(beta_base, rep)
    if beta_signal.size > p:
        raise ValueError(f"rep too large: {beta_signal.size} non-zero coefficients for p={p}")
    beta = np.concatenate([beta_signal, np.zeros(p - beta_signal.size)])

    y = (X @ beta + _noise(n_total, noise_sd, rng)).reshape(-1, 1)
    X_train, y_train, X_test, y_test = _split_xy(X, y, n_train)
    meta = LinearExperimentMeta(1, spec, rep, n_train, n_total, p, noise_sd)
    return X_train, y_train, X_test, y_test, beta, meta


def experiment_2(
    spec: int = 0,
    rep: int = 1,
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
):
    """Paper Experiment 2: alternative fixed signal design.

    The non-zero base coefficients are (2,2,2,2,2,2,2,2,5,20).
    The five specifications use (n, p) = (24, 20), (60, 50),
    (120, 100), (240, 200), and (480, 400).
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = {
        0: (24, 20),
        1: (60, 50),
        2: (120, 100),
        3: (240, 200),
        4: (480, 400),
    }
    if spec not in grid:
        raise ValueError("spec must be in {0,1,2,3,4}")
    n_train, p = grid[spec]

    n_total = n_train + 100
    X = rng.standard_normal((n_total, p))

    beta_base = np.array([2, 2, 2, 2, 2, 2, 2, 2, 5, 20], dtype=float)
    beta_signal = np.repeat(beta_base, rep)
    if beta_signal.size > p:
        raise ValueError(f"rep too large: {beta_signal.size} non-zero coefficients for p={p}")
    beta = np.concatenate([beta_signal, np.zeros(p - beta_signal.size)])

    y = (X @ beta + _noise(n_total, noise_sd, rng)).reshape(-1, 1)
    X_train, y_train, X_test, y_test = _split_xy(X, y, n_train)
    meta = LinearExperimentMeta(2, spec, rep, n_train, n_total, p, noise_sd)
    return X_train, y_train, X_test, y_test, beta, meta


def experiment_3(
    prop_nonzero: float = 0.2,
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
):
    """Paper Experiment 3: heavy-tailed signals with random sparsity."""
    if rng is None:
        rng = np.random.default_rng()

    n_train, p = 55, 50
    n_total = n_train + 100
    X = rng.standard_normal((n_total, p))

    beta_base = rng.standard_t(df=2, size=p)
    mask = rng.binomial(n=1, p=float(prop_nonzero), size=p)
    beta = beta_base * mask

    y = (X @ beta + _noise(n_total, noise_sd, rng)).reshape(-1, 1)
    X_train, y_train, X_test, y_test = _split_xy(X, y, n_train)
    meta = LinearExperimentMeta(3, None, 1, n_train, n_total, p, noise_sd)
    return X_train, y_train, X_test, y_test, beta, meta


def generate_experiment(
    experiment: int,
    *,
    spec: Optional[int],
    rep: int,
    prop_nonzero: Optional[float],
    noise_sd: float,
    rng: np.random.Generator,
):
    """Dispatch using the experiment numbering used in the paper."""
    if experiment == 1:
        if spec is None:
            raise ValueError("Experiment 1 requires spec")
        return experiment_1(spec=spec, rep=rep, noise_sd=noise_sd, rng=rng)
    if experiment == 2:
        if spec is None:
            raise ValueError("Experiment 2 requires spec")
        return experiment_2(spec=spec, rep=rep, noise_sd=noise_sd, rng=rng)
    if experiment == 3:
        if prop_nonzero is None:
            raise ValueError("Experiment 3 requires prop_nonzero")
        return experiment_3(prop_nonzero=prop_nonzero, noise_sd=noise_sd, rng=rng)
    raise ValueError("experiment must be one of {1,2,3}")

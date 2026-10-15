from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.utils.validation import check_X_y

from sou.models.sou_estimators import SoUGlobalLocalRegressor


PAPER_N_SIM = 100
PAPER_SEED_BASE = 1
PAPER_NOISE_SD = 1.0
PAPER_REPS_BY_SPEC = {
    0: [1],
    1: [1, 2],
    2: [1, 2, 5],
    3: [1, 5, 10],
    4: [1, 5, 10, 20],
}

# Fixed SoU-SGL initialization used in all linear-regression fits.
INITIAL_LOCAL_PRECISION = 1e-2
INITIAL_GLOBAL_PRECISION = 1.0


@dataclass(frozen=True)
class Scenario:
    experiment: int
    spec: Optional[int]
    rep: int
    prop_nonzero: Optional[float]


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def prop_small_on_true_zeros(
    coef: np.ndarray, beta_zero_mask: np.ndarray, threshold: float
) -> float:
    coef = np.asarray(coef).reshape(-1)
    beta_zero_mask = np.asarray(beta_zero_mask).reshape(-1).astype(bool)
    if beta_zero_mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(coef[beta_zero_mask]) <= threshold))


def normalize_scenario(
    experiment: int,
    spec: int,
    rep: int,
    prop_nonzero: Optional[float],
) -> Scenario:
    if experiment in (1, 2):
        if spec not in PAPER_REPS_BY_SPEC:
            raise ValueError("spec must be in {0,1,2,3,4}")
        if rep not in PAPER_REPS_BY_SPEC[spec]:
            raise ValueError(
                f"rep={rep} is not part of the paper grid for spec={spec}; "
                f"allowed values are {PAPER_REPS_BY_SPEC[spec]}"
            )
        return Scenario(experiment, int(spec), int(rep), None)
    if experiment == 3:
        if prop_nonzero is None:
            raise ValueError("Experiment 3 requires --prop-nonzero")
        return Scenario(3, None, 1, float(prop_nonzero))
    raise ValueError("experiment must be one of {1,2,3}")


def dataset_dir(data_root: Path, scenario: Scenario) -> Path:
    spec_str = "none" if scenario.spec is None else str(scenario.spec)
    out = (
        data_root
        / f"experiment={scenario.experiment}"
        / f"spec={spec_str}"
        / f"rep={scenario.rep}"
        / f"noise={PAPER_NOISE_SD}"
        / f"seedbase={PAPER_SEED_BASE}"
    )
    if scenario.experiment == 3:
        out = out / f"prop={scenario.prop_nonzero}"
    return out


def result_dir(results_root: Path, model: str, scenario: Scenario) -> Path:
    spec_str = -1 if scenario.spec is None else scenario.spec
    out = (
        results_root
        / f"model={model}"
        / f"experiment={scenario.experiment}"
        / f"spec={spec_str}"
        / f"rep={scenario.rep}"
    )
    if scenario.experiment == 3:
        out = out / f"prop={scenario.prop_nonzero}"
    return out


def load_sim_npz_regression(
    sim_path: Path,
    y_as_2d: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(sim_path, allow_pickle=False)
    X_train = np.asarray(data["X_train"])
    X_test = np.asarray(data["X_test"])
    y_train = np.asarray(data["y_train"])
    y_test = np.asarray(data["y_test"])

    if y_as_2d:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    else:
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    beta_true = np.asarray(data["beta_true"]).reshape(-1)
    beta_nonzero_mask = np.asarray(data["beta_nonzero_mask"]).reshape(-1).astype(bool)
    beta_zero_mask = ~beta_nonzero_mask
    return X_train, X_test, y_train, y_test, beta_true, beta_zero_mask


def normalize_range(start: int, end: Optional[int]) -> int:
    end2 = PAPER_N_SIM if end is None else int(end)
    if not (0 <= start < end2 <= PAPER_N_SIM):
        raise ValueError(
            f"Invalid range: start={start}, end={end2}; paper uses {PAPER_N_SIM} repetitions"
        )
    return end2


def fit_paper_sou_regression(
    model: SoUGlobalLocalRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> SoUGlobalLocalRegressor:
    """Fit SoU-SGL with the fixed initialization used in the paper."""
    X, y = check_X_y(X, y)
    model.n_samples_, model.n_features_ = X.shape
    model.sigma_alpha = 10.0

    XtX = X.T @ X
    Xty = X.T @ y
    Xt1 = X.T @ np.ones(model.n_samples_) if model.fit_intercept else None

    model.a_sigma_ = model.a_pi_sigma
    model.b_sigma_ = model.b_pi_sigma
    model.a_tau_ = np.full(model.n_features_, 3.1)
    model.b_tau_ = model.a_tau_ / INITIAL_LOCAL_PRECISION
    model.a_nu_ = 3.1
    model.b_nu_ = model.a_nu_ / INITIAL_GLOBAL_PRECISION
    model.coef_ = np.zeros(model.n_features_)
    model.m_alpha_ = np.mean(y) if model.fit_intercept else 0.0
    model.s_alpha_ = 1.0 if model.fit_intercept else 0.0
    model.n_iter_ = 0
    model.converged_ = False

    for iteration in range(model.max_iter):
        D = np.diag(model.a_tau_ / model.b_tau_)
        model.m_alpha_, model.s_alpha_ = model._update_intercept(
            X, y, model.coef_, model.a_sigma_, model.b_sigma_
        )
        model.coef_, model.S_, S_diag = model._update_coefficients(
            XtX,
            Xty,
            Xt1,
            D,
            model.a_sigma_,
            model.b_sigma_,
            model.a_nu_,
            model.b_nu_,
            model.m_alpha_,
        )
        a_tau_new, b_tau_new = model._update_tau_parameters(
            S_diag,
            model.coef_,
            model.a_sigma_,
            model.b_sigma_,
            model.a_nu_,
            model.b_nu_,
        )
        a_nu_new, b_nu_new = model._update_nu_parameters(
            S_diag,
            model.coef_,
            model.a_sigma_,
            model.b_sigma_,
            a_tau_new,
            b_tau_new,
        )
        a_sigma_new, b_sigma_new = model._update_sigma_parameters(
            X,
            y,
            model.S_,
            model.coef_,
            model.a_sigma_,
            model.b_sigma_,
            a_tau_new,
            b_tau_new,
            a_nu_new,
            b_nu_new,
            model.m_alpha_,
            model.s_alpha_,
        )

        tau_converged = np.allclose(
            b_tau_new / (a_tau_new - 1),
            model.b_tau_ / (model.a_tau_ - 1),
            rtol=model.tol,
        )
        nu_converged = np.isclose(
            b_nu_new / (a_nu_new - 1),
            model.b_nu_ / (model.a_nu_ - 1),
            rtol=model.tol,
        )
        sigma_converged = np.isclose(
            b_sigma_new / (a_sigma_new - 1),
            model.b_sigma_ / (model.a_sigma_ - 1),
            rtol=model.tol,
        )

        model.a_tau_, model.b_tau_ = a_tau_new, b_tau_new
        model.a_nu_, model.b_nu_ = a_nu_new, b_nu_new
        model.a_sigma_, model.b_sigma_ = a_sigma_new, b_sigma_new
        model.n_iter_ = iteration + 1

        if tau_converged and nu_converged and sigma_converged:
            model.converged_ = True
            break

    model.is_fitted_ = True
    return model

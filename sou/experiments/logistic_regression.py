from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_X_y

from sou.models.sou_estimators import SoUGlobalLocalClassifier


INITIAL_LOCAL_PRECISION = 1e-2
INITIAL_GLOBAL_PRECISION = 1.0


def fit_paper_classifier(
    model: SoUGlobalLocalClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> SoUGlobalLocalClassifier:
    """Fit GL/SoU-SGL with the fixed initialization used in the paper."""
    X, y = check_X_y(X, y)
    n_samples, n_features = X.shape

    model.m_alpha_ = 0.0
    model.s_alpha_sq_ = 1.0
    model.m_ = np.zeros(n_features)
    model.S_ = np.eye(n_features)
    model.a_tau_ = np.full(n_features, 3.1)
    model.b_tau_ = model.a_tau_ / INITIAL_LOCAL_PRECISION
    model.a_nu_ = 3.1
    model.b_nu_ = model.a_nu_ / INITIAL_GLOBAL_PRECISION
    model.xi_ = np.ones(n_samples)
    model.n_iter_ = 0
    model.converged_ = False

    for iteration in range(model.max_iter):
        eta_mean = model.m_alpha_ + X @ model.m_
        eta_var = model.s_alpha_sq_ + np.sum(X @ model.S_ * X, axis=1)
        model.xi_ = np.sqrt(eta_mean**2 + eta_var)
        Lambda_diag = model._compute_lambda(model.xi_)

        D_diag = model.a_tau_ / model.b_tau_ * model.a_nu_ / model.b_nu_
        model.m_alpha_, model.s_alpha_sq_ = model._update_intercept(
            X, y, model.m_, model.S_, Lambda_diag
        )
        model.m_, model.S_ = model._update_coefficients(
            X, y, model.m_alpha_, Lambda_diag, D_diag
        )

        S_diag = np.diag(model.S_)
        a_tau_new, b_tau_new = model._update_tau_params(
            S_diag, model.m_, model.a_nu_, model.b_nu_
        )
        a_nu_new, b_nu_new = model._update_nu_params(
            S_diag, model.m_, a_tau_new, b_tau_new
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

        model.a_tau_, model.b_tau_ = a_tau_new, b_tau_new
        model.a_nu_, model.b_nu_ = a_nu_new, b_nu_new
        model.n_iter_ = iteration + 1

        if tau_converged and nu_converged:
            model.converged_ = True
            break

    model.is_fitted_ = True
    return model

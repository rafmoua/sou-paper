"""Horseshoe estimators backed by Stan via rpy2/rstan.

Provides normal-means and linear-regression estimators with scikit-learn style APIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# Enable automatic NumPy <-> R conversion
numpy2ri.activate()

# Import required R packages
rstan = importr("rstan")


def _resolve_stan_file(stan_file: str) -> str:
    """Resolve a Stan file path from absolute/local package locations."""
    path = Path(stan_file)
    if path.is_file():
        return str(path)
    local_path = Path(__file__).resolve().parent / "stan" / stan_file
    if local_path.is_file():
        return str(local_path)
    return str(path)


class HorseshoeNormalMeans(BaseEstimator, RegressorMixin):
    """Horseshoe estimator for replicated normal-means data."""

    def __init__(
        self,
        tau_scale: float = 1.0,
        lambda_scale: float = 1.0,
        sigma_alpha: float = 3.0,
        sigma_beta: float = 2.0,
        n_iter: int = 2000,
        n_chains: int = 4,
        stan_file: str = "horseshoe_normal_means_noncentered.stan",
        seed: int = 123,
    ):
        """Initialize normal-means horseshoe hyperparameters and Stan settings."""
        self.tau_scale = tau_scale
        self.lambda_scale = lambda_scale
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.n_iter = n_iter
        self.n_chains = n_chains
        self.stan_file = _resolve_stan_file(stan_file)
        self.seed = seed

        # Fitted attributes
        self.coef_ = None
        self.intercept_ = 0.0  # kept for sklearn compatibility
        self.lambda_ = None
        self.tau_ = None
        self.sigma_ = None
        self._fit_result = None

    def fit(self, y, X=None):  # X unused, but included for sklearn-style compatibility if needed
        """Fit the estimator from a 2D array of replicated observations."""
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError("y must be a 2D array of shape (n_groups, n_replicates).")

        n_groups, n_replicates = y.shape

        stan_data = robjects.ListVector(
            {
                "n": int(n_groups),
                "p": int(n_replicates),
                "Y": y,
                "tau_scale": float(self.tau_scale),
                "lambda_scale": float(self.lambda_scale),
                "sigma_alpha": float(self.sigma_alpha),
                "sigma_beta": float(self.sigma_beta),
            }
        )

        self._fit_result = rstan.stan(
            file=self.stan_file,
            data=stan_data,
            iter=int(self.n_iter),
            chains=int(self.n_chains),
            seed=int(self.seed),
        )

        extract = robjects.r["extract"](self._fit_result)

        beta_samples = np.array(extract.rx2("beta"))
        self.coef_ = beta_samples.mean(axis=0)

        self.lambda_ = np.array(extract.rx2("lambda"))
        self.tau_ = np.array(extract.rx2("tau"))
        if "sigma" in list(extract.names):
            self.sigma_ = np.array(extract.rx2("sigma"))
        elif "sigma_sq" in list(extract.names):
            self.sigma_ = np.sqrt(np.array(extract.rx2("sigma_sq")))
        else:
            self.sigma_ = None

        return self

    def get_params(self, deep: bool = True):
        """Return estimator parameters for scikit-learn compatibility."""
        return {
            "tau_scale": self.tau_scale,
            "lambda_scale": self.lambda_scale,
            "sigma_alpha": self.sigma_alpha,
            "sigma_beta": self.sigma_beta,
            "n_iter": self.n_iter,
            "n_chains": self.n_chains,
            "stan_file": self.stan_file,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        """Set estimator parameters for scikit-learn compatibility."""
        for key, value in parameters.items():
            setattr(self, key, value)
        return self


class HorseshoeRegressor(BaseEstimator, RegressorMixin):
    """Horseshoe linear regressor fitted through Stan."""

    def __init__(
        self,
        tau_scale: float = 1.0,
        lambda_scale: float = 1.0,
        sigma_alpha: float = 3.0,
        sigma_beta: float = 2.0,
        iter: int = 2000,
        chains: int = 4,
        stan_file: str = "horseshoe_linear_regression.stan",
        seed: int = 1,
    ):
        """Initialize regression horseshoe hyperparameters and Stan settings."""
        self.tau_scale = tau_scale
        self.lambda_scale = lambda_scale
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.iter = iter
        self.chains = chains
        self.stan_file = _resolve_stan_file(stan_file)
        self.seed = seed

        self._fit_result = None
        self.coef_ = None
        self.intercept_ = 0.0
        self.lambda_ = None
        self.tau_ = None
        self.sigma_ = None

    def fit(self, X, y):
        """Fit the regressor from feature matrix X and target vector y."""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n, p = X.shape

        # Convert X to an R matrix (column-major by default)
        X_r = robjects.r.matrix(X, nrow=int(n), ncol=int(p), byrow=False)

        stan_data = robjects.ListVector(
            {
                "n": int(n),
                "p": int(p),
                "X": X_r,
                "y": robjects.FloatVector(y.astype(float)),
                "tau_scale": float(self.tau_scale),
                "lambda_scale": float(self.lambda_scale),
                "sigma_alpha": float(self.sigma_alpha),
                "sigma_beta": float(self.sigma_beta),
            }
        )

        self._fit_result = rstan.stan(
            file=self.stan_file,
            data=stan_data,
            iter=int(self.iter),
            chains=int(self.chains),
            seed=int(self.seed),
        )

        extract = robjects.r["extract"](self._fit_result)

        beta_samples = np.array(extract.rx2("beta"))
        self.coef_ = beta_samples.mean(axis=0)

        lambda_samples = np.array(extract.rx2("lambda"))
        tau_samples = np.array(extract.rx2("tau"))
        sigma_samples = np.array(extract.rx2("sigma"))

        # Keep the same logic as your original code (posterior means)
        self.lambda_ = lambda_samples.mean(axis=0)
        self.tau_ = tau_samples.mean(axis=0)
        self.sigma_ = sigma_samples.mean(axis=0)

        # Intercept is optional (depends on Stan file)
        if "intercept" in list(extract.names):
            intercept_samples = np.array(extract.rx2("intercept"))
            self.intercept_ = float(intercept_samples.mean())

        return self

    def predict(self, X):
        """Predict using posterior mean coefficients and intercept."""
        if self.coef_ is None:
            raise RuntimeError("You must call .fit() before .predict().")

        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

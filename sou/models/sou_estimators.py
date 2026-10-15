from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from scipy.special import polygamma  # psi is digamma, polygamma(1, x) is trigamma
from scipy.special import expit as sigmoid
from scipy.optimize import root
from tqdm import tqdm


def _compute_term(X, y, S, m, m_alpha=None, s_alpha=None, intercept=False):
    """Helper function to compute the SSR term"""
    if intercept:
        n = len(y)
        term = (np.sum(y ** 2) - 2 * m_alpha * np.sum(y) - 2 * y.T @ X @ m
                + n * m_alpha ** 2 + 2 * m_alpha * np.sum(X @ m)
                + m.T @ X.T @ X @ m + np.trace(X.T @ X @ S) + n * s_alpha ** 2)
    else:
        term = (y.T @ y - 2 * y.T @ X @ m + m.T @ X.T @ X @ m + np.trace(X.T @ X @ S))
    return term

class SoUNormalMeans(BaseEstimator):
    """Normal-means SoU estimator with global-local shrinkage."""
    def __init__(self, gamma_beta=1.0, gamma_tau=1.0, gamma_nu=1.0, gamma_sigma=1.0,
                 a_pi_sigma=1.0, b_pi_sigma=1.0, lambda_tau=1.0, lambda_nu=1.0,
                 tol=1e-6, max_iter=1000, verbose=False):
        """Initialize normal-means hyperparameters and optimization settings."""
        self.gamma_beta = gamma_beta
        self.gamma_tau = gamma_tau
        self.gamma_nu = gamma_nu
        self.gamma_sigma = gamma_sigma
        self.a_pi_sigma = a_pi_sigma
        self.b_pi_sigma = b_pi_sigma
        self.lambda_tau = lambda_tau
        self.lambda_nu = lambda_nu
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _compute_m_s(self, y, k, y_bar, a_sigma, b_sigma, a_tau, b_tau, a_nu, b_nu):
        """Compute optimal means m and variances s for all groups (vectorized)"""
        term1 = (a_sigma * k) / (b_sigma * self.gamma_beta)
        term2 = (a_nu * a_tau) / (b_nu * b_tau)
        s = 1 / (term1 + term2)
        m = s * term1 * y_bar


        return m, s

    def _compute_b_sigma(self, y, m, s, a_sigma):
        """Compute optimal b_sigma (vectorized)"""
        total_k = sum(len(y_i) for y_i in y)

        # Vectorized sum-term computation
        sum_term = 0
        for i, y_i in enumerate(y):
            sum_term += np.sum((y_i - m[i]) ** 2) + len(y_i) * s[i]

        numerator = a_sigma * sum_term + 2 * self.gamma_sigma * a_sigma * self.b_pi_sigma
        denominator = total_k + 2 * self.gamma_sigma * self.a_pi_sigma

        return np.maximum(1e-16, numerator / denominator)

    def _compute_b_tau(self, a_tau, m, s, a_sigma, b_sigma, a_nu, b_nu):
        """Compute optimal b_tau for all groups (vectorized)"""
        numerator = (self.gamma_tau - self.gamma_beta / 2) * (a_tau - 1)

        sqrt_term = np.sqrt(
            (self.gamma_tau - self.gamma_beta / 2) ** 2 +
            (2 * self.lambda_tau * self.gamma_tau * self.gamma_beta * a_tau * a_nu *
             (m ** 2 + s)) / ((a_tau - 1) * b_nu)
        )

        denominator = 2 * self.lambda_tau * self.gamma_tau

        return np.maximum(1e-16, (numerator + (a_tau - 1) * sqrt_term) / denominator)

    def _compute_b_nu(self, a_nu, m, s, a_tau, b_tau, a_sigma, b_sigma):
        """Compute optimal b_nu (global shrinkage)"""
        n = len(m)
        numerator = (self.gamma_nu - (n * self.gamma_beta) / 2) * (a_nu - 1)

        # Vectorized sum-term computation
        sum_term = np.sum((a_tau / b_tau) * (m ** 2 + s))

        sqrt_term = np.sqrt(
            (self.gamma_nu - (n * self.gamma_beta) / 2) ** 2 +
            (2 * self.lambda_nu * self.gamma_nu * self.gamma_beta * a_nu * sum_term) / (a_nu - 1)
        )

        denominator = 2 * self.lambda_nu * self.gamma_nu

        return np.maximum(1e-16, (numerator + (a_nu - 1) * sqrt_term) / denominator)

    def _derivative_a_tau(self, a_tau, b_tau, m, s, a_sigma, b_sigma, a_nu, b_nu):
        """Derivative of ELBO w.r.t. a_tau (vectorized)"""
        term1 = -(self.gamma_beta / 2) * (
                (a_nu) / (b_nu * b_tau) * (m ** 2 + s) - polygamma(1, a_tau)
        )
        term2 = -self.gamma_tau * (
                polygamma(1, a_tau) * (a_tau + 1) - 1 - (self.lambda_tau * b_tau) / (a_tau - 1) ** 2
        )
        return term1 + term2

    def _derivative_a_nu(self, a_nu, b_nu, m, s, a_tau, b_tau, a_sigma, b_sigma):
        """Derivative of ELBO w.r.t. a_nu (global shrinkage)"""
        n = len(m)
        sum_term = np.sum((a_tau / b_tau) * (m ** 2 + s))

        term1 = -(self.gamma_beta / 2) * (
                (1 / b_nu) * sum_term - n * polygamma(1, a_nu)
        )
        term2 = -self.gamma_nu * (
                polygamma(1, a_nu) * (a_nu + 1) - 1 - (self.lambda_nu * b_nu) / (a_nu - 1) ** 2
        )
        return term1 + term2

    def _derivative_a_sigma(self, a_sigma, b_sigma, y, m, s):
        """Derivative of ELBO w.r.t. a_sigma"""
        total_k = sum(len(y_i) for y_i in y)

        # Sum-term computation
        sum_term = 0
        for i, y_i in enumerate(y):
            sum_term += np.sum((y_i - m[i]) ** 2) + len(y_i) * s[i]

        term1 = (total_k / 2) * polygamma(1, a_sigma)
        term2 = -(1 / (2 * b_sigma)) * sum_term
        term3 = -self.gamma_sigma * (
                -polygamma(1, a_sigma) * (self.a_pi_sigma - a_sigma) +
                (self.b_pi_sigma / b_sigma) - 1
        )
        return term1 + term2 + term3

    def fit(self, y):
        """Fit the model with global-local shrinkage priors (optimized version)"""
        n = len(y)

        # Parameter initialization
        self.a_sigma_ = self.a_pi_sigma
        self.b_sigma_ = self.b_pi_sigma
        self.a_tau_ = np.full(n, 3.0)  # Local shrinkage for each group
        self.b_tau_ = np.full(n, 2.0)
        self.a_nu_ = 3.0  # Global shrinkage
        self.b_nu_ = 2.0

        # Precompute group sizes
        k = np.array([len(y_i) for y_i in y])
        y_bar = np.array([np.mean(y_i) for y_i in y])

        for _ in tqdm(range(self.max_iter), disable=not self.verbose):
            # 1. Update m e s (vectorized)
            m, s = self._compute_m_s(y, k, y_bar, self.a_sigma_, self.b_sigma_,
                                     self.a_tau_, self.b_tau_, self.a_nu_, self.b_nu_)

            # 2. Update b_sigma
            b_sigma_new = self._compute_b_sigma(y, m, s, self.a_sigma_)

            # 3. Update b_tau (vectorized)
            b_tau_new = self._compute_b_tau(self.a_tau_, m, s, self.a_sigma_,
                                            self.b_sigma_, self.a_nu_, self.b_nu_)

            # 4. Update b_nu
            b_nu_new = self._compute_b_nu(self.a_nu_, m, s, self.a_tau_, self.b_tau_,
                                          self.a_sigma_, self.b_sigma_)

            # 5. Update a_tau (vectorized using root finding)
            a_tau_new = np.empty(n)
            for i in range(n):
                def eq_log_a_tau(log_a):
                    a = np.exp(log_a)
                    return self._derivative_a_tau(
                        a, b_tau_new[i], m[i], s[i],
                        self.a_sigma_, self.b_sigma_, self.a_nu_, self.b_nu_
                    )

                sol = root(eq_log_a_tau, np.log(self.a_tau_[i]), method='lm')
                a_tau_new[i] = np.exp(sol.x[0])

            a_tau_new = np.maximum(1.1, a_tau_new)

            # 6. Update a_nu
            def eq_log_a_nu(log_a):
                a = np.exp(log_a)
                return self._derivative_a_nu(
                    a, b_nu_new, m, s, a_tau_new, b_tau_new,
                    self.a_sigma_, self.b_sigma_
                )

            sol_log_a_nu = root(eq_log_a_nu, np.log(self.a_nu_), method='lm')
            a_nu_new = np.maximum(1.1, np.exp(sol_log_a_nu.x[0]))

            # 7. Update a_sigma
            def eq_log_a_sigma(log_a):
                a = np.exp(log_a)
                return self._derivative_a_sigma(a, b_sigma_new, y, m, s)

            sol_log_a_sigma = root(eq_log_a_sigma, np.log(self.a_sigma_), method='lm')
            a_sigma_new = np.maximum(1.1, np.exp(sol_log_a_sigma.x[0]))

            # Convergence check
            tau_converged = np.allclose(
                b_tau_new / (a_tau_new - 1), self.b_tau_ / (self.a_tau_ - 1), rtol=self.tol)
            nu_converged = np.isclose(
                b_nu_new / (a_nu_new - 1), self.b_nu_ / (self.a_nu_ - 1), rtol=self.tol)
            sigma_converged = np.isclose(
                b_sigma_new / (a_sigma_new - 1), self.b_sigma_ / (self.a_sigma_ - 1), rtol=self.tol)

            if tau_converged and nu_converged and sigma_converged:
                if self.verbose:
                    print("Converged!")
                break

            # Update parameters
            self.a_sigma_, self.b_sigma_ = a_sigma_new, b_sigma_new
            self.a_tau_, self.b_tau_ = a_tau_new, b_tau_new
            self.a_nu_, self.b_nu_ = a_nu_new, b_nu_new

        # Store final parameters
        self.m_ = m
        self.s_ = s
        self.is_fitted_ = True

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'gamma_beta': self.gamma_beta,
            'gamma_tau': self.gamma_tau,
            'gamma_nu': self.gamma_nu,
            'gamma_sigma': self.gamma_sigma,
            'a_pi_sigma': self.a_pi_sigma,
            'b_pi_sigma': self.b_pi_sigma,
            'lambda_tau': self.lambda_tau,
            'lambda_nu': self.lambda_nu,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'verbose': self.verbose
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class SoUGlobalLocalRegressor(BaseEstimator, RegressorMixin):
    """Linear-regression SoU estimator with global-local shrinkage."""
    def __init__(self, fit_intercept=True, gamma_beta=1.0, gamma_tau=1.0,
                 gamma_nu=1.0, gamma_sigma=1.0, gamma_alpha=1.0,
                 a_pi_sigma=1.0, b_pi_sigma=1.0, lambda_tau=1.0, lambda_nu=1.0,
                 tol=1e-6, max_iter=1000, verbose=False):
        """Initialize regression hyperparameters and optimization settings."""
        self.fit_intercept = fit_intercept
        self.gamma_beta = gamma_beta
        self.gamma_tau = gamma_tau
        self.gamma_nu = gamma_nu
        self.gamma_sigma = gamma_sigma
        self.gamma_alpha = gamma_alpha
        self.a_pi_sigma = a_pi_sigma
        self.b_pi_sigma = b_pi_sigma
        self.lambda_tau = lambda_tau
        self.lambda_nu = lambda_nu
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _update_intercept(self, X, y, m, a_sigma, b_sigma):
        """Update intercept parameters m_alpha and s_alpha"""
        if not self.fit_intercept:
            return 0.0, 0.0

        n = len(y)
        Xm = X @ m
        numerator = (a_sigma / b_sigma) * np.sum(y - Xm)
        denominator = n * a_sigma / b_sigma + self.gamma_alpha / self.sigma_alpha ** 2
        m_alpha = numerator / denominator
        s_alpha = np.sqrt(1 / (n * a_sigma / (self.gamma_alpha * b_sigma) + 1 / self.sigma_alpha ** 2))

        return m_alpha, s_alpha

    def _update_coefficients(self, XtX, Xty, Xt1, D, a_sigma, b_sigma, a_nu, b_nu, m_alpha):
        """Update coefficients m and S"""
        # Precomputations
        # Xt1 = np.sum(XtX, axis=1) if self.fit_intercept else 0

        # Update m
        term = Xty - m_alpha * Xt1 if self.fit_intercept else Xty
        S_inv = XtX + self.gamma_beta * (a_nu / b_nu) * D
        # m = np.linalg.solve(S_inv, term)

        S_not_scaled = np.linalg.inv(S_inv)
        m=S_not_scaled@term

        # Update S
        # S_inv_scaled = (a_sigma / (self.gamma_beta * b_sigma)) * S_inv
        # S = np.linalg.inv(S_inv_scaled)
        S=((self.gamma_beta * b_sigma)/a_sigma)*S_not_scaled

        return m, S, np.diag(S)

    def _update_tau_parameters(self, S_diag, m, a_sigma, b_sigma, a_nu, b_nu):
        """Update tau parameters (local shrinkage)"""
        p = len(m)
        m_sq = m ** 2
        ratio = (a_nu / b_nu) * (a_sigma / b_sigma)

        # Vectorized computation of b_tau
        terms = (self.gamma_tau - self.gamma_beta / 2) * (self.a_tau_ - 1)
        sqrt_terms = np.sqrt(
            (self.gamma_tau - self.gamma_beta / 2) ** 2 +
            (2 * self.lambda_tau * self.gamma_tau * self.gamma_beta *
             ratio * self.a_tau_ * (S_diag + m_sq)) / (self.a_tau_ - 1)
        )
        b_tau = (terms + (self.a_tau_ - 1) * sqrt_terms) / (2 * self.lambda_tau * self.gamma_tau)
        b_tau = np.maximum(1e-16, b_tau)

        # Numerical update of a_tau
        a_tau = np.empty(p)
        for j in range(p):
            def eq_a_tau(a):
                return self._derivative_a_tau(a, b_tau[j], S_diag[j], m[j], a_sigma, b_sigma, a_nu, b_nu)

            sol = root(eq_a_tau, self.a_tau_[j], method='lm')
            a_tau[j] = sol.x[0]

        return np.maximum(1.1, a_tau), b_tau

    def _derivative_a_tau(self, a, b, S_jj, m_j, a_sigma, b_sigma, a_nu, b_nu):
        """Derivative of ELBO w.r.t. a_tau_j"""
        term1 = -self.gamma_beta / 2 * (
                (a_sigma * a_nu) / (b_sigma * b * b_nu) * (S_jj + m_j ** 2) - polygamma(1, a))
        term2 = -self.gamma_tau * (
                polygamma(1, a) * (a + 1) - 1 - (self.lambda_tau * b) / (a - 1) ** 2)
        return term1 + term2

    def _update_nu_parameters(self, S_diag, m, a_sigma, b_sigma, a_tau, b_tau):
        """Update nu parameters (global shrinkage)"""
        sum_term = np.sum((S_diag + m ** 2) * (a_tau / b_tau))
        p = len(m)
        ratio = a_sigma / b_sigma

        # Update b_nu
        numerator = (self.gamma_nu - p * self.gamma_beta / 2) * (self.a_nu_ - 1)
        sqrt_term = np.sqrt(
            (self.gamma_nu - p * self.gamma_beta / 2) ** 2 +
            (2 * self.lambda_nu * self.gamma_nu * self.gamma_beta * self.a_nu_ * ratio * sum_term) / (self.a_nu_ - 1))
        b_nu = (numerator + (self.a_nu_ - 1) * sqrt_term) / (2 * self.lambda_nu * self.gamma_nu)
        b_nu = np.maximum(1e-16, b_nu)

        # Update a_nu numerically
        def eq_a_nu(a):
            return self._derivative_a_nu(a, b_nu, S_diag, m, a_sigma, b_sigma, a_tau, b_tau)

        sol = root(eq_a_nu, self.a_nu_, method='lm')
        a_nu = sol.x[0]

        return np.maximum(1.1, a_nu), b_nu

    def _derivative_a_nu(self, a, b, S_diag, m, a_sigma, b_sigma, a_tau, b_tau):
        """Derivative of ELBO w.r.t. a_nu"""
        p = len(m)
        sum_term = np.sum((S_diag + m ** 2) * (a_tau / b_tau))

        term1 = -self.gamma_beta / 2 * (
                (a_sigma / (b_sigma * b)) * sum_term - p * polygamma(1, a))
        term2 = -self.gamma_nu * (
                polygamma(1, a) * (a + 1) - 1 - (self.lambda_nu * b) / (a - 1) ** 2)
        return term1 + term2
    def _update_sigma_parameters(self, X, y, S, m, a_sigma, b_sigma, a_tau, b_tau, a_nu, b_nu, m_alpha, s_alpha):
        """Update sigma parameters"""
        n = len(y)
        p = X.shape[1]

        term = _compute_term(X, y, S, m, m_alpha, s_alpha, self.fit_intercept)
        sum_term = np.sum((np.diag(S) + m ** 2) * (a_tau / b_tau))

        # Update b_sigma
        numerator = a_sigma * term + self.gamma_beta * a_sigma * (a_nu / b_nu) * sum_term
        numerator += 2 * self.gamma_sigma * a_sigma * self.b_pi_sigma
        denominator = n + self.gamma_beta * p + 2 * self.gamma_sigma * self.a_pi_sigma
        b_sigma = numerator / denominator
        b_sigma = np.maximum(1e-16, b_sigma)

        # Update a_sigma numerically
        def eq_a_sigma(a):
            return self._derivative_a_sigma(a, b_sigma, n, p, term, S, m, a_tau, b_tau, a_nu, b_nu)

        sol = root(eq_a_sigma, self.a_sigma_, method='lm')
        a_sigma = sol.x[0]

        return np.maximum(1.1, a_sigma), b_sigma


    def _derivative_a_sigma(self, a_sigma, b_sigma, n, p, term, S, m, a_tau, b_tau, a_nu, b_nu):
        """Derivative of ELBO w.r.t. a_sigma"""
        term1 = (n / 2) * polygamma(1, a_sigma)
        term2 = -(1 / (2 * b_sigma)) * term
        term3 = -(self.gamma_beta / 2) * (
                (a_nu / (b_sigma * b_nu)) * np.sum((np.diag(S) + m ** 2) * (a_tau / b_tau)) -
                p * polygamma(1, a_sigma)
        )
        term4 = -self.gamma_sigma * (
                -polygamma(1, a_sigma) * (self.a_pi_sigma - a_sigma) +
                (self.b_pi_sigma / b_sigma) - 1
        )
        return term1 + term2 + term3 + term4

    def fit(self, X, y):
        """Fit the model to data"""
        X, y = check_X_y(X, y)
        self.n_samples_, self.n_features_ = X.shape
        self.sigma_alpha = 10.0  # Default value for sigma_alpha

        # Precomputations
        XtX = X.T @ X
        Xty = X.T @ y
        Xt1 = X.T @ np.ones(self.n_samples_) if self.fit_intercept else None

        # Parameter initialization
        self.a_sigma_ = self.a_pi_sigma
        self.b_sigma_ = self.b_pi_sigma
        self.a_tau_ = np.ones(self.n_features_) * 3.1
        self.b_tau_ = np.ones(self.n_features_) * 2
        self.a_nu_ = 3.1
        self.b_nu_ = 2.0
        self.coef_ = np.zeros(self.n_features_)
        self.m_alpha_ = np.mean(y) if self.fit_intercept else 0.0
        self.s_alpha_ = 1.0 if self.fit_intercept else 0.0

        for _ in tqdm(range(self.max_iter), disable=not self.verbose):
            # Parameter updates
            D = np.diag(self.a_tau_ / self.b_tau_)

            # 1. Update intercepto
            self.m_alpha_, self.s_alpha_ = self._update_intercept(
                X, y, self.coef_, self.a_sigma_, self.b_sigma_)

            # 2. Update coeficientes
            self.coef_, self.S_, S_diag = self._update_coefficients(
                XtX, Xty, Xt1, D, self.a_sigma_, self.b_sigma_, self.a_nu_, self.b_nu_, self.m_alpha_)

            # 3. Update parameters de tau
            a_tau_new, b_tau_new = self._update_tau_parameters(
                S_diag, self.coef_, self.a_sigma_, self.b_sigma_, self.a_nu_, self.b_nu_)

            # 4. Update parameters de nu
            a_nu_new, b_nu_new = self._update_nu_parameters(
                S_diag, self.coef_, self.a_sigma_, self.b_sigma_, a_tau_new, b_tau_new)

            # 5. Update parameters de sigma
            a_sigma_new, b_sigma_new = self._update_sigma_parameters(
                X, y, self.S_, self.coef_, self.a_sigma_, self.b_sigma_, a_tau_new, b_tau_new,
                a_nu_new, b_nu_new, self.m_alpha_, self.s_alpha_)

            # Convergence check
            tau_converged = np.allclose(
                b_tau_new / (a_tau_new - 1), self.b_tau_ / (self.a_tau_ - 1), rtol=self.tol)
            nu_converged = np.isclose(
                b_nu_new / (a_nu_new - 1), self.b_nu_ / (self.a_nu_ - 1), rtol=self.tol)
            sigma_converged = np.isclose(
                b_sigma_new / (a_sigma_new - 1), self.b_sigma_ / (self.a_sigma_ - 1), rtol=self.tol)

            if tau_converged and nu_converged and sigma_converged:
                if self.verbose:
                    print("Converged!")
                break

            # Update parameters
            self.a_tau_, self.b_tau_ = a_tau_new, b_tau_new
            self.a_nu_, self.b_nu_ = a_nu_new, b_nu_new
            self.a_sigma_, self.b_sigma_ = a_sigma_new, b_sigma_new

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using the fitted model"""
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if self.fit_intercept:
            return X @ self.coef_ + self.m_alpha_
        return X @ self.coef_

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'fit_intercept': self.fit_intercept,
            'gamma_beta': self.gamma_beta,
            'gamma_tau': self.gamma_tau,
            'gamma_nu': self.gamma_nu,
            'gamma_sigma': self.gamma_sigma,
            'gamma_alpha': self.gamma_alpha,
            'a_pi_sigma': self.a_pi_sigma,
            'b_pi_sigma': self.b_pi_sigma,
            'lambda_tau': self.lambda_tau,
            'lambda_nu': self.lambda_nu,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'verbose': self.verbose
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




class SoUGlobalLocalClassifier(BaseEstimator, ClassifierMixin):
    """Logistic SoU classifier with global-local shrinkage."""
    def __init__(self, gamma_alpha=1.0, gamma_beta=1.0, gamma_tau=1.0, gamma_nu=1.0,
                 sigma_alpha=1.0, lambda_tau=1.0, lambda_nu=1.0,
                 tol=1e-6, max_iter=1000, verbose=False):
        """Initialize classifier hyperparameters and optimization settings."""
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_tau = gamma_tau
        self.gamma_nu = gamma_nu
        self.sigma_alpha = sigma_alpha
        self.lambda_tau = lambda_tau
        self.lambda_nu = lambda_nu
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _compute_lambda(self, xi):
        """Compute lambda(xi) = (sigmoid(xi) - 0.5)/(2*xi)"""
        return (sigmoid(xi) - 0.5) / (2 * xi + 1e-16)

    def _update_intercept(self, X, y, m, S, Lambda_diag):
        """Update intercept parameters m_alpha and s_alpha"""
        n = len(y)
        Xm = X @ m

        lambda_sum = np.sum(Lambda_diag)
        numerator = np.sum(y - 0.5 - 2 * Lambda_diag * Xm)
        denominator = 2 * lambda_sum + self.gamma_alpha / self.sigma_alpha ** 2
        m_alpha = numerator / denominator
        s_alpha_sq = self.gamma_alpha / denominator

        return m_alpha, s_alpha_sq

    def _update_coefficients(self, X, y, m_alpha, Lambda_diag, D_diag):
        """Update coefficients m and S"""
        n, p = X.shape

        # Compute XtLambdaX efficiently using broadcasting
        XtLambda = X.T * Lambda_diag
        XtLambdaX = XtLambda @ X

        # Compute XtLambda1 efficiently
        XtLambda1 = XtLambda.sum(axis=1)

        # Update S using Cholesky decomposition for stability
        S_inv = (2 / self.gamma_beta) * XtLambdaX + np.diag(D_diag)
        term0_inv = 2 * XtLambdaX + self.gamma_beta * np.diag(D_diag)
        try:
            L = np.linalg.cholesky(S_inv)
            S = np.linalg.inv(L)
            S = S.T @ S

            L0=np.linalg.cholesky(term0_inv)
            term0=np.linalg.inv(L0)
            term0 = term0.T @ term0

        except np.linalg.LinAlgError:
            S = np.linalg.pinv(S_inv)
            term0 = np.linalg.pinv(term0_inv)

        # Update m
        term1 = X.T @ (y - 0.5)
        term2 = 2 * m_alpha * XtLambda1
        # m = S @ (term1 - term2)
        m = term0 @ (term1 - term2)

        return m, S

    def _update_tau_params(self, S_diag, m, a_nu, b_nu):
        """Update tau parameters (local shrinkage) - vectorized"""
        p = len(m)
        m_sq = m ** 2
        ratio = a_nu / b_nu

        # Vectorized computation of b_tau - much faster!
        terms = (self.gamma_tau - self.gamma_beta / 2) * (self.a_tau_ - 1)

        sqrt_terms = np.sqrt(
            (self.gamma_tau - self.gamma_beta / 2) ** 2 +
            (2 * self.lambda_tau * self.gamma_tau * self.gamma_beta *
             self.a_tau_ * ratio * (S_diag + m_sq)) / (self.a_tau_ - 1)#denominator_aux
        )

        b_tau = (terms + (self.a_tau_ - 1) * sqrt_terms) / (2 * self.lambda_tau * self.gamma_tau)
        b_tau = np.maximum(1e-16, b_tau)

        # Numerical update of a_tau (ainda precisa ser loop)
        a_tau = np.empty(p)
        for j in range(p):
            def eq_a_tau(a):
                return self._derivative_a_tau(a, b_tau[j], S_diag[j], m[j], a_nu, b_nu)

            sol = root(eq_a_tau, self.a_tau_[j], method='lm')
            a_tau[j] = sol.x[0]

        return np.maximum(1.1, a_tau), b_tau
    def _derivative_a_tau(self, a, b, S_jj, m_j, a_nu, b_nu):
        """Derivative of ELBO w.r.t. a_tau_j"""
        term1 = -self.gamma_beta / 2 * (
                (a_nu / (b * b_nu)) * (S_jj + m_j ** 2) - polygamma(1, a))
        term2 = -self.gamma_tau * (
                polygamma(1, a) * (a + 1) - 1 - (self.lambda_tau * b) / (a - 1) ** 2)
        return term1 + term2

    def _update_nu_params(self, S_diag, m, a_tau, b_tau):
        """Update nu parameters (global shrinkage)"""
        sum_term = np.sum((S_diag + m ** 2) * (a_tau / b_tau))
        p = len(m)

        # Update b_nu
        numerator = (self.gamma_nu - p * self.gamma_beta / 2) * (self.a_nu_ - 1)
        sqrt_term = np.sqrt(
            (self.gamma_nu - p * self.gamma_beta / 2) ** 2 +
            (2 * self.lambda_nu * self.gamma_nu * self.gamma_beta * self.a_nu_ * sum_term) / (self.a_nu_ - 1))
        denominator = 2 * self.lambda_nu * self.gamma_nu
        b_nu = (numerator + (self.a_nu_ - 1) * sqrt_term) / denominator
        b_nu = np.maximum(1e-16, b_nu) 

        # Update a_nu numerically
        def eq_a_nu(a):
            return self._derivative_a_nu(a, b_nu, S_diag, m, a_tau, b_tau)

        sol = root(eq_a_nu, self.a_nu_, method='lm')
        a_nu = sol.x[0]

        # return np.maximum(1.1, a_nu), np.maximum(1e-16, b_nu)
        return np.maximum(1.1, a_nu), b_nu 


    def _derivative_a_nu(self, a, b, S_diag, m, a_tau, b_tau):
        """Derivative of ELBO w.r.t. a_nu"""
        p = len(m)
        sum_term = np.sum((S_diag + m ** 2) * (a_tau / b_tau))

        term1 = -self.gamma_beta / 2 * (
                sum_term / b - p * polygamma(1, a)
        )
        term2 = -self.gamma_nu * (
                polygamma(1, a) * (a + 1) - 1 - (self.lambda_nu * b) / (a - 1) ** 2
        )
        return term1 + term2

    def fit(self, X, y):
        """Fit the model to data"""
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.m_alpha_ = 0.0
        self.s_alpha_sq_ = 1.0
        self.m_ = np.zeros(n_features)
        self.S_ = np.eye(n_features)
        self.a_tau_ = np.ones(n_features) * 3.1
        self.b_tau_ = np.ones(n_features) * 2
        self.a_nu_ = 3.1
        self.b_nu_ = 2.0
        self.xi_ = np.ones(n_samples)

        for iteration in tqdm(range(self.max_iter), disable=not self.verbose):
            # Update Jaakkola-Jordan parameters
            eta_mean = self.m_alpha_ + X @ self.m_
            eta_var = self.s_alpha_sq_ + np.sum(X @ self.S_ * X, axis=1)
            self.xi_ = np.sqrt(eta_mean ** 2 + eta_var)
            Lambda_diag = self._compute_lambda(self.xi_)

            # Update D matrix (local shrinkage) - store only diagonal
            D_diag = self.a_tau_ / self.b_tau_ * self.a_nu_ / self.b_nu_

            # Update intercept
            self.m_alpha_, self.s_alpha_sq_ = self._update_intercept(
                X, y, self.m_, self.S_, Lambda_diag)

            # Update coefficients
            self.m_, self.S_ = self._update_coefficients(
                X, y, self.m_alpha_, Lambda_diag, D_diag)

            # Get diagonal of S for efficiency
            S_diag = np.diag(self.S_)

            # Update tau parameters
            a_tau_new, b_tau_new = self._update_tau_params(
                S_diag, self.m_, self.a_nu_, self.b_nu_)

            # Update nu parameters
            a_nu_new, b_nu_new = self._update_nu_params(
                S_diag, self.m_, a_tau_new, b_tau_new)

            # Check convergence
            tau_converged = np.allclose(
                b_tau_new / (a_tau_new - 1), self.b_tau_ / (self.a_tau_ - 1),
                rtol=self.tol)
            nu_converged = np.isclose(
                b_nu_new / (a_nu_new - 1), self.b_nu_ / (self.a_nu_ - 1),
                rtol=self.tol)

            if tau_converged and nu_converged:
                break

            # Update parameters
            self.a_tau_, self.b_tau_ = a_tau_new, b_tau_new
            self.a_nu_, self.b_nu_ = a_nu_new, b_nu_new

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """Return predicted positive-class probabilities."""
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        linear_pred = self.m_alpha_ + X @ self.m_
        return sigmoid(linear_pred)

    def predict(self, X):
        """Return binary predictions using a 0.5 probability threshold."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_params(self, deep=True):
        """Return estimator parameters for scikit-learn compatibility."""
        return {
            'gamma_alpha': self.gamma_alpha,
            'gamma_beta': self.gamma_beta,
            'gamma_tau': self.gamma_tau,
            'gamma_nu': self.gamma_nu,
            'sigma_alpha': self.sigma_alpha,
            'lambda_tau': self.lambda_tau,
            'lambda_nu': self.lambda_nu,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set estimator parameters for scikit-learn compatibility."""
        for param, value in params.items():
            setattr(self, param, value)
        return self


# Backward-compatible aliases (to be removed after migration).
SoU_Regression_GlobalLocal_eff = SoUGlobalLocalRegressor
SoU_LogisticRegression_GlobalLocal_eff = SoUGlobalLocalClassifier




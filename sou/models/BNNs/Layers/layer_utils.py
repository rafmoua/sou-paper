from __future__ import annotations

import numpy as np
import torch
from torch.nn import Parameter


gammaln = torch.lgamma


def diag_gaussian_entropy(log_std, D):
    """Return entropy of a diagonal Gaussian."""
    return 0.5 * D * (1.0 + torch.log(2 * np.pi)) + torch.sum(log_std)


def inv_gamma_entropy(a, b):
    """Return entropy of an inverse-gamma distribution."""
    return torch.sum(a + torch.log(b) + torch.lgamma(a) - (1 + a) * torch.digamma(a))


def log_normal_entropy(log_std, mu, D):
    """Return entropy of a log-normal distribution."""
    return torch.sum(log_std + mu + 0.5) + (D / 2) * np.log(2 * np.pi)


class InvGammaHalfCauchyLayer(torch.nn.Module):
    """Variational half-Cauchy layer using inverse-gamma auxiliary variables."""

    def __init__(self, out_features, b):
        """Initialize variational parameters for node or layer scales."""
        super(InvGammaHalfCauchyLayer, self).__init__()
        self.b = b
        self.out_features = out_features
        self.mu = Parameter(torch.FloatTensor(out_features))
        self.log_sigma = Parameter(torch.FloatTensor(out_features))
        self.register_buffer("ahat", torch.tensor([1.], dtype=torch.float32))
        self.register_buffer("bhat", torch.ones(out_features, dtype=torch.float32) * (1.0 / self.b ** 2))
        self.register_buffer("const", torch.tensor([0.5], dtype=torch.float32))
        self.initialize_from_prior()

    def initialize_from_prior(self):
        """Initialize variational parameters from prior samples."""
        sample = np.abs(self.b * (np.random.randn(self.out_features) / np.random.randn(self.out_features)))
        self.mu.data = torch.FloatTensor(np.log(sample))
        self.log_sigma.data = torch.FloatTensor(np.random.randn(self.out_features) - 10.)

    def expectation_wrt_prior(self):
        """Compute the expected prior log-density terms."""
        expected_a_given_lambda = -gammaln(self.const) - 0.5 * (torch.log(self.bhat) - torch.digamma(self.ahat)) + (
            -1.5 * self.mu - torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2) * (self.ahat / self.bhat)
        )
        expected_lambda = -gammaln(self.const) - np.log(self.b) + (-self.const - 1.) * (
            torch.log(self.bhat) - torch.digamma(self.ahat)
        ) - (1. / self.b ** 2) * (self.ahat / self.bhat)
        return torch.sum(expected_a_given_lambda) + torch.sum(expected_lambda)

    def entropy(self):
        """Compute entropy of q(log a^2) and q(lambda)."""
        return self.entropy_lambda() + self.entropy_a2()

    def entropy_lambda(self):
        """Compute entropy of q(lambda)."""
        return inv_gamma_entropy(self.ahat, self.bhat)

    def entropy_a2(self):
        """Compute entropy of q(log a^2)."""
        return log_normal_entropy(self.log_sigma, self.mu, self.out_features)

    def kl(self):
        """Compute KL divergence to the half-Cauchy prior."""
        return -self.expectation_wrt_prior() - self.entropy()

    def fixed_point_updates(self):
        """Update inverse-gamma moments using fixed-point equations."""
        self.bhat = torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2) + (1. / self.b ** 2)


class InvGammaLayer(torch.nn.Module):
    """Variational inverse-gamma layer with log-normal approximation."""

    def __init__(self, a, b, out_features=1):
        """Initialize inverse-gamma prior and variational parameters."""
        super(InvGammaLayer, self).__init__()
        self.register_buffer("a", torch.tensor([a], dtype=torch.float32))
        self.register_buffer("b", torch.tensor([b], dtype=torch.float32))
        self.mu = Parameter(torch.FloatTensor(out_features))
        self.log_sigma = Parameter(torch.FloatTensor(out_features))
        self.out_features = out_features
        self.initialize_from_prior()

    def initialize_from_prior(self):
        """Initialize variational parameters at the prior mode."""
        self.mu.data = torch.log(self.b / (self.a + 1) * torch.ones(self.out_features))
        self.log_sigma.data = torch.FloatTensor(np.random.randn(self.out_features) - 10.)

    def expectation_wrt_prior(self):
        """Compute the expected prior log-density term."""
        return self.a * torch.log(self.b) - gammaln(self.a) + (- self.a - 1) * self.mu - \
               self.b * torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2)

    def entropy(self):
        """Compute entropy of q(log c^2)."""
        return log_normal_entropy(self.log_sigma, self.mu, 1)

    def kl(self):
        """Compute KL divergence to the inverse-gamma prior."""
        return -self.expectation_wrt_prior().sum() - self.entropy()

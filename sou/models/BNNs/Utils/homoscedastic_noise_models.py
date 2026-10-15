from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import gammaln
from torch.nn import Parameter

from sou.models.BNNs.Utils.noisemodel import AbstractNoiseModel


def transform(a):
    """Apply a softplus-like transform to keep parameters positive."""
    return torch.log(1 + torch.exp(a))


class GaussianNoiseGammaPrecision(torch.nn.Module, AbstractNoiseModel):
    """Gaussian likelihood with Gamma precision variational posterior."""

    def __init__(self, a0=6, b0=6, cuda=False):
        """Initialize prior hyperparameters and variational parameters."""
        super(GaussianNoiseGammaPrecision, self).__init__()
        self.cuda = cuda
        self.a0 = a0
        self.b0 = b0
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))
        self.ahat = Parameter(torch.FloatTensor([10.]))
        self.bhat = Parameter(torch.FloatTensor([3.]))

    def loss(self, y_pred=None, y_true=None):
        """Return expected negative log-likelihood under q(lambda)."""
        n = y_pred.shape[0]
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return -1 * (-0.5 * n * self.const + 0.5 * n * (torch.digamma(ahat) - torch.log(bhat))
                     - 0.5 * (ahat / bhat) * ((y_pred - y_true) ** 2).sum())

    def kl(self):
        """Return KL divergence between q(lambda) and Gamma prior."""
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return (ahat - self.a0) * torch.digamma(ahat) - torch.lgamma(ahat) + gammaln(self.a0) + \
            self.a0 * (torch.log(bhat) - np.log(self.b0)) + ahat * (self.b0 - bhat) / bhat

    def get_noise_var(self):
        """Return the current noise variance estimate."""
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return (bhat / ahat).data.numpy()[0]


class GaussianNoiseFixedPrecision(torch.nn.Module, AbstractNoiseModel):
    """Gaussian likelihood with fixed, known observation variance."""

    def __init__(self, std_y=1., cuda=False):
        """Initialize fixed-noise parameters."""
        super(GaussianNoiseFixedPrecision, self).__init__()
        self.cuda = cuda
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))
        self.sigma_y = std_y

    def loss(self, y_pred=None, y_true=None):
        """Return negative log-likelihood for fixed Gaussian noise."""
        ll = -0.5 * self.const - np.log(self.sigma_y) - 0.5 * (1. / self.sigma_y ** 2) * ((y_pred - y_true) ** 2)
        return -ll.sum(dim=0)

    def get_noise_var(self):
        """Return the fixed noise variance."""
        return self.sigma_y ** 2

    def kl(self):
        """Return zero KL because the precision is fixed."""
        return 0

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sou.models.BNNs.Layers.layer_utils import InvGammaHalfCauchyLayer


def reparam(mu, logvar, do_sample=True, mc_samples=1):
    """Sample from a diagonal Gaussian via reparameterization."""
    if do_sample:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        for _ in np.arange(1, mc_samples):
            eps = torch.randn_like(std)
            sample += mu + eps * std
        return sample / mc_samples
    return mu


class BayesianLinearLayer(torch.nn.Module):
    """Affine layer with Gaussian variational posterior on weights and bias."""

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, prior_stdv=None):
        """Initialize layer parameters and prior hyperparameters."""
        super(BayesianLinearLayer, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features

        self.weights = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.weights_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))

        self.fudge_factor = 1e-8
        if not prior_stdv:
            prior_stdv_tensor = torch.tensor([1. / np.sqrt(self.weights.size(1))], dtype=torch.float32)
        else:
            prior_stdv_tensor = torch.tensor([prior_stdv], dtype=torch.float32)
        self.register_buffer("prior_stdv", prior_stdv_tensor)
        self.register_buffer("prior_mean", torch.tensor([0.], dtype=torch.float32))
        self.register_buffer("prior_bias_stdv", torch.tensor([1.], dtype=torch.float32))
        self.register_buffer("prior_bias_mean", torch.tensor([0.], dtype=torch.float32))

        self.init_parameters(init_weight, init_bias)

    def init_parameters(self, init_weight, init_bias):
        """Initialize variational means and log-variances."""
        if init_weight is not None:
            self.weights.data = torch.Tensor(init_weight)
        else:
            self.weights.data.normal_(0, float(self.prior_stdv.numpy()[0]))

        if init_bias is not None:
            self.bias.data = torch.Tensor(init_bias)
        else:
            self.bias.data.normal_(0, 1)

        self.weights_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def forward(self, x, do_sample=True, scale_variances=False):
        """Compute stochastic affine transformation."""
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        if scale_variances:
            activ = reparam(mu_activations, var_activations.log() - np.log(self.in_features), do_sample=do_sample)
        else:
            activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl(self):
        """Return KL divergence between posterior and prior."""
        weights_logvar = self.weights_logvar
        kld_weights = self.prior_stdv.log() - weights_logvar.mul(0.5) + \
                      (weights_logvar.exp() + (self.weights.pow(2) - self.prior_mean)) / \
                      (2 * self.prior_stdv.pow(2)) - 0.5
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
                   (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / \
                   (2 * self.prior_bias_stdv.pow(2)) - 0.5
        return kld_weights.sum() + kld_bias.sum()


class SoUHorseshoeLayer(BayesianLinearLayer):
    """Horseshoe layer with non-centered parameterization."""

    def __init__(self, in_features, out_features, cuda=False, scale=1., gamma_local=1.0):
        """Initialize local and global shrinkage components."""
        super(SoUHorseshoeLayer, self).__init__(in_features, out_features)
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.nodescales = InvGammaHalfCauchyLayer(out_features=out_features, b=1.)
        self.layerscale = InvGammaHalfCauchyLayer(out_features=1, b=scale)
        self.prior_stdv = torch.tensor([1.], dtype=torch.float32)
        self.prior_mean = torch.tensor([0.], dtype=torch.float32)
        self.gamma_local = gamma_local
        self.register_buffer('gamma_local_tensor', torch.tensor(gamma_local))

    def forward(self, x, do_sample=True, debug=False, eps_scale=None, eps_w=None):
        """Compute stochastic affine transformation with horseshoe scaling."""
        scale_mean = 0.5 * (self.nodescales.mu + self.layerscale.mu)
        scale_var = 0.25 * (self.nodescales.log_sigma.exp() ** 2 + self.layerscale.log_sigma.exp() ** 2)
        scale_sample = reparam(scale_mean, scale_var.log(), do_sample=do_sample).exp()
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ_sample = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return scale_sample * activ_sample

    def kl(self):
        """Return KL divergence including horseshoe scale terms."""
        return (super(SoUHorseshoeLayer, self).kl() +
                self.gamma_local_tensor * self.nodescales.kl() +
                self.layerscale.kl())

    def fixed_point_updates(self):
        """Update auxiliary inverse-gamma variational moments."""
        self.nodescales.fixed_point_updates()
        self.layerscale.fixed_point_updates()

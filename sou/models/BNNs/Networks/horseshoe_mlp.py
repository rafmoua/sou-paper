from __future__ import annotations
from abc import ABC

import numpy as np
import torch
from torch import nn

from sou.models.BNNs.Layers.layers import BayesianLinearLayer, SoUHorseshoeLayer
from sou.models.BNNs.Utils.homoscedastic_noise_models import GaussianNoiseGammaPrecision


class SOUHshoeBNN(nn.Module, ABC):
    """Bayesian neural network with horseshoe shrinkage layers."""

    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu', num_layers=1,
                 hshoe_scale=1e-1, gamma_local=1.0):
        """Initialize a horseshoe Bayesian network backbone."""
        super(SOUHshoeBNN, self).__init__()
        self.num_layers = num_layers
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            print("Activation Type not supported")

        self.fc_hidden = nn.ModuleList()
        self.fc1 = SoUHorseshoeLayer(ip_dim, num_nodes[0], scale=hshoe_scale[0], gamma_local=gamma_local[0])
        for k in np.arange(len(num_nodes) - 1):
            self.fc_hidden.append(
                SoUHorseshoeLayer(num_nodes[k], num_nodes[k + 1], scale=hshoe_scale[k + 1], gamma_local=gamma_local[k + 1])
            )

        self.fc_out = BayesianLinearLayer(num_nodes[-1], op_dim)
        self.noise_layer = None

    def forward(self, x, do_sample=True):
        """Run a forward pass through the network."""
        x = self.fc1(x, do_sample=do_sample)
        x = self.activation(x)
        for layer in self.fc_hidden:
            x = layer(x, do_sample=do_sample)
            x = self.activation(x)
        return self.fc_out(x, do_sample=do_sample, scale_variances=True)

    def kl_divergence_w(self):
        """Return the KL term for all stochastic layers."""
        kld = self.fc1.kl() + self.fc_out.kl()
        for layer in self.fc_hidden:
            kld += layer.kl()
        return kld

    def fixed_point_updates(self):
        """Run fixed-point updates for horseshoe auxiliary variables."""
        if hasattr(self.fc1, 'fixed_point_updates'):
            self.fc1.fixed_point_updates()
        if hasattr(self.fc_out, 'fixed_point_updates'):
            self.fc_out.fixed_point_updates()
        for layer in self.fc_hidden:
            if hasattr(layer, 'fixed_point_updates'):
                layer.fixed_point_updates()

    def prior_predictive_samples(self, n_sample=100):
        """Draw samples from the prior predictive process."""
        n_eval = 1000
        x = torch.linspace(-2, 2, n_eval)[:, np.newaxis]
        y = np.zeros([n_sample, n_eval])
        for i in np.arange(n_sample):
            y[i] = self.forward(x).data.numpy().ravel()
        return x.data.numpy(), y


class SoUHshoeRegressionNet(SOUHshoeBNN):
    """Horseshoe Bayesian neural network for regression."""

    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                 num_layers=1, hshoe_scale=1e-5, gamma_local=1):
        """Initialize the regression likelihood head."""
        super(SoUHshoeRegressionNet, self).__init__(ip_dim=ip_dim, op_dim=op_dim,
                                                    num_nodes=num_nodes, activation_type=activation_type,
                                                    num_layers=num_layers,
                                                    hshoe_scale=hshoe_scale,
                                                    gamma_local=gamma_local)
        self.noise_layer = GaussianNoiseGammaPrecision(a0=6., b0=6.)

    def likelihood(self, x=None, y=None):
        """Return expected log-likelihood term for regression."""
        out = self.forward(x)
        return -self.noise_layer.loss(y_pred=out, y_true=y)

    def neg_elbo(self, num_batches, x=None, y=None):
        """Compute minibatch negative ELBO."""
        Elik = self.likelihood(x, y)
        neg_elbo = (self.kl_divergence_w() + self.noise_layer.kl()) / num_batches - Elik
        return neg_elbo

    def mse(self, x, y):
        """Return the scaled squared-error objective used for diagnostics."""
        E_noise_precision = 1. / self.noise_layer.get_noise_var()
        return (0.5 * E_noise_precision * (self.forward(x, do_sample=False) - y) ** 2).sum()

    def get_noise_var(self):
        """Return the current noise variance estimate."""
        return self.noise_layer.get_noise_var()


class SoUHshoeClassificationNet(SOUHshoeBNN):
    """Horseshoe Bayesian neural network for classification."""

    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                 num_layers=1, hshoe_scale=1e-5, gamma_local=1):
        """Initialize the classification likelihood head."""
        super(SoUHshoeClassificationNet, self).__init__(ip_dim=ip_dim, op_dim=op_dim,
                                                        num_nodes=num_nodes, activation_type=activation_type,
                                                        num_layers=num_layers,
                                                        hshoe_scale=hshoe_scale,
                                                        gamma_local=gamma_local)
        self.noise_layer = torch.nn.CrossEntropyLoss(reduction='sum')

    def likelihood(self, x=None, y=None):
        """Return expected log-likelihood term for classification."""
        out = self.forward(x)
        return -self.noise_layer(out, y)

    def neg_elbo(self, num_batches, x=None, y=None):
        """Compute minibatch negative ELBO."""
        Elik = self.likelihood(x, y)
        neg_elbo = self.kl_divergence_w() / num_batches - Elik
        return neg_elbo

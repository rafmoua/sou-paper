from __future__ import annotations
from abc import ABC

import torch
from torch import nn

from sou.models.BNNs.Layers.layers import BayesianLinearLayer
from sou.models.BNNs.Utils.homoscedastic_noise_models import GaussianNoiseGammaPrecision
import numpy as np


class BayesianNN(nn.Module, ABC):
    """Bayesian neural network with Gaussian priors on weights."""

    def __init__(self, layer=BayesianLinearLayer, ip_dim=1, op_dim=1, num_nodes=50,
                 activation_type='relu', num_layers=1):
        """Initialize hidden and output layers for the Bayesian network."""
        super(BayesianNN, self).__init__()
        self.num_layers = num_layers
        if activation_type == 'relu':
            # activation
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            print("Activation Type not supported")

        self.fc_hidden = nn.ModuleList()
        self.fc1 = layer(ip_dim, num_nodes[0],)

        for k in np.arange(len(num_nodes) - 1):
            self.fc_hidden.append(layer(num_nodes[k], num_nodes[k+1], ))
        self.fc_out = layer(num_nodes[-1], op_dim, )
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

    def prior_predictive_samples(self, n_sample=100):
        """Draw samples from the prior predictive process."""
        n_eval = 1000
        x = torch.linspace(-2, 2, n_eval)[:, np.newaxis]
        y = np.zeros([n_sample, n_eval])
        for i in np.arange(n_sample):
            y[i] = self.forward(x).data.numpy().ravel()
        return x.data.numpy(), y

    def get_weights(self):
        """Return variational parameters for one-hidden-layer-free models."""
        assert len(self.fc_hidden) == 0  # only works for one layer networks.
        weight_dict = {}
        weight_dict['layerip_means'] = torch.cat([self.fc1.weights, self.fc1.bias.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerip_logvar'] = torch.cat([self.fc1.weights_logvar, self.fc1.bias_logvar.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerop_means'] = torch.cat([self.fc_out.weights, self.fc_out.bias.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerop_logvar'] = torch.cat([self.fc_out.weights_logvar, self.fc_out.bias_logvar.unsqueeze(1)], dim=1).data.numpy()
        return weight_dict

    def set_weights(self, weight_dict):
        """Set variational parameters for one-hidden-layer-free models."""
        assert len(self.fc_hidden) == 0  # only works for one layer networks.
        to_param = lambda x: nn.Parameter(torch.Tensor(x))
        self.fc1.weights = to_param(weight_dict['layerip_means'][:, :-1])
        self.fc1.weights = to_param(weight_dict['layerip_logvar'][:, :-1])
        self.fc1.bias = to_param(weight_dict['layerip_means'][:, -1])
        self.fc1.bias_logvar = to_param(weight_dict['layerip_logvar'][:, -1])

        self.fc_out.weights = to_param(weight_dict['layerop_means'][:, :-1])
        self.fc_out.weights = to_param(weight_dict['layerop_logvar'][:, :-1])
        self.fc_out.bias = to_param(weight_dict['layerop_means'][:, -1])
        self.fc_out.bias_logvar = to_param(weight_dict['layerop_logvar'][:, -1])


class BayesianRegressionNet(BayesianNN, ABC):
    """Bayesian neural net for regression with Gamma precision noise."""

    def __init__(self, layer=BayesianLinearLayer, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                     num_layers=1):
        """Initialize the regression likelihood head."""
        super(BayesianRegressionNet, self).__init__(layer=layer, ip_dim=ip_dim, op_dim=op_dim,
                                                    num_nodes=num_nodes, activation_type=activation_type,
                                                    num_layers=num_layers,
                                                    )
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
        return (0.5 * E_noise_precision * (self.forward(x, do_sample=False) - y)**2).sum()

    def get_noise_var(self):
        """Return the current noise variance estimate."""
        return self.noise_layer.get_noise_var()


class BayesianClassificationNet(BayesianNN, ABC):
    """Bayesian neural net for categorical classification."""

    def __init__(self, layer=BayesianLinearLayer, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                 num_layers=1):
        """Initialize the classification likelihood head."""
        super(BayesianClassificationNet, self).__init__(layer=layer, ip_dim=ip_dim, op_dim=op_dim,
                                                    num_nodes=num_nodes, activation_type=activation_type,
                                                    num_layers=num_layers)
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

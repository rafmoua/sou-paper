from __future__ import annotations
import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from sou.models.BNNs.Networks import bayesian_mlp
from sou.models.BNNs.Networks.horseshoe_mlp import SoUHshoeRegressionNet, SoUHshoeClassificationNet
from sou.models.BNNs.Utils.builtinuq import BuiltinUQ


def _count_below_threshold(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    """Return counts of absolute values below each threshold."""
    abs_values = np.abs(values)
    return np.array([(abs_values < thr).sum() for thr in thresholds], dtype=np.float64)


def _is_horseshoe_like_layer(module: torch.nn.Module) -> bool:
    """Return True when the layer has SoU-HS scaling components."""
    return (
        hasattr(module, "nodescales")
        and hasattr(module, "layerscale")
        and hasattr(module.nodescales, "mu")
        and hasattr(module.nodescales, "log_sigma")
        and hasattr(module.layerscale, "mu")
        and hasattr(module.layerscale, "log_sigma")
    )


def _deterministic_scale(module: torch.nn.Module, use_lognormal_mean: bool) -> torch.Tensor:
    """
    Build deterministic per-neuron scale s_k for SoU-HS layers.

    `use_lognormal_mean=False` matches the deterministic forward convention (median in log-normal space).
    """
    scale_mean = 0.5 * (module.nodescales.mu + module.layerscale.mu)
    scale_var = 0.25 * (module.nodescales.log_sigma.exp() ** 2 + module.layerscale.log_sigma.exp() ** 2)
    if use_lognormal_mean:
        return torch.exp(scale_mean + 0.5 * scale_var)
    return torch.exp(scale_mean)


def _compute_node_scale_values(
    net: torch.nn.Module,
    use_lognormal_mean: bool = True,
) -> dict[str, list[float]]:
    """
    Compute deterministic per-node/filter scales for HS/SoU layers.

    Returns a dict mapping module name -> list of scales for that layer.
    Layers without HS/SoU scales are omitted.
    """
    scales: dict[str, list[float]] = {}
    with torch.no_grad():
        for layer_name, module in net.named_modules():
            if not layer_name:
                continue
            if not _is_horseshoe_like_layer(module):
                continue

            layer_scale = _deterministic_scale(module, use_lognormal_mean=use_lognormal_mean)
            scales[layer_name] = layer_scale.detach().cpu().numpy().ravel().tolist()
    return scales


def _compute_layer_sparsity(
    net: torch.nn.Module,
    thresholds: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
) -> dict[str, dict[str, list[float]]]:
    """
    Compute per-layer sparsity metrics.

    Returns:
      - W_effective: sparsity over effective parameters (W_eff/b_eff).
      - neuron: sparsity over HS neuron scales s_k (only available for HS/SoU layers).
    """
    w_effective_sparsity: dict[str, list[float]] = {}
    neuron_sparsity: dict[str, list[float]] = {}

    with torch.no_grad():
        for layer_name, module in net.named_modules():
            if not (hasattr(module, "weights") and hasattr(module, "bias")):
                continue

            weights = module.weights.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()

            base_values = np.concatenate([np.abs(weights).ravel(), np.abs(bias).ravel()])

            if _is_horseshoe_like_layer(module):
                scale = _deterministic_scale(module, use_lognormal_mean=True).detach().cpu().numpy().ravel()
                w_eff = scale[:, None] * weights
                b_eff = scale * bias
                w_effective_values = np.concatenate([np.abs(w_eff).ravel(), np.abs(b_eff).ravel()])
                neuron_counts = _count_below_threshold(scale, thresholds)
                neuron_total = max(scale.size, 1)
                neuron_sparsity[layer_name] = (neuron_counts / float(neuron_total)).tolist()
            else:
                # For standard BNN layers, W_effective is just the base (W, b).
                w_effective_values = base_values

            w_effective_counts = _count_below_threshold(w_effective_values, thresholds)
            w_effective_total = max(w_effective_values.size, 1)
            w_effective_sparsity[layer_name] = (w_effective_counts / float(w_effective_total)).tolist()

    return {
        "W_effective": w_effective_sparsity,
        "neuron": neuron_sparsity,
    }


class SoUBnnRegression(BuiltinUQ):
    """
    Variationally trained BNNs for regression.

    Supported models:
      - model="bnn": standard Bayesian NN (Gaussian variational weights)
      - model="sou": SoU-BNN with local-global shrinkage controlled by config["gamma_local"]
                    (e.g. gamma_local=1 -> HS-like, gamma_local=1e-9 -> SoU-SGL)
    """

    def __init__(self, config, model: str = "bnn", device=None):
        """
        Args:
            config: dict specifying architecture and learning hyperparameters.
            model: "bnn" or "sou".
            device: torch device or None.
        """
        super().__init__()
        self.config = dict(config)
        self.device = device
        self.model = model

        # Resolve device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Build network ----
        if self.model == "bnn":
            self.net = bayesian_mlp.BayesianRegressionNet(
                ip_dim=self.config["ip_dim"],
                op_dim=self.config["op_dim"],
                num_nodes=self.config["num_nodes"],
                num_layers=self.config["num_layers"],
            )

        elif self.model == "sou":
            if "gamma_local" not in self.config:
                raise ValueError("For model='sou', config['gamma_local'] must be specified.")
            if "hshoe_scale" not in self.config:
                raise ValueError("For model='sou', config['hshoe_scale'] must be specified.")

            self.net = SoUHshoeRegressionNet(
                ip_dim=self.config["ip_dim"],
                op_dim=self.config["op_dim"],
                num_nodes=self.config["num_nodes"],
                num_layers=self.config["num_layers"],
                hshoe_scale=self.config["hshoe_scale"],
                gamma_local=self.config["gamma_local"],
            )

        else:
            raise NotImplementedError("`model` must be one of {'bnn','sou'}.")

        self.net = self.net.to(self.device)

    def get_params(self, deep=True):
        """Return estimator parameters for sklearn-style compatibility."""
        return {"model": self.model, "config": self.config, "device": self.device}

    def fit(self, X, y, checkpoint_epochs=None):
        """
        Fit the BNN regression model and store history, checkpoints, and best state.
        """
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        torch.manual_seed(1234)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['step_size'])
        params_store = {}
        best_elbo = float("inf")

        self.history = {"neg_elbo": [], "noise_var": [], "sparsity": [], "nodes_scales": []}

        for epoch in range(self.config['num_epochs']):

            loss = self.net.neg_elbo(num_batches=1, x=X, y=y.float().unsqueeze(dim=1)) / X.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if hasattr(self.net, 'fixed_point_updates'):
                self.net.fixed_point_updates()

            train_loss = loss.item()

            if train_loss < best_elbo:
                best_elbo = train_loss
                best_state_elbo = copy.deepcopy(self.net.state_dict())
                self.best_model_elbo_epoch = epoch + 1


            sparsity_metrics = _compute_layer_sparsity(self.net, thresholds=(1e-3, 1e-4, 1e-5))
            nodes_scales = _compute_node_scale_values(self.net, use_lognormal_mean=True)

            self.history["neg_elbo"].append(train_loss)
            self.history["noise_var"].append(self.net.get_noise_var())
            self.history["sparsity"].append(sparsity_metrics)
            self.history["nodes_scales"].append(nodes_scales)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch[{epoch + 1}/{self.config['num_epochs']}], "
                      f"neg elbo: {self.history['neg_elbo'][-1]:.6f}, "
                      f"noise var: {self.history['noise_var'][-1]:.6f}")

            if checkpoint_epochs and (epoch + 1) in checkpoint_epochs:
                params_store[epoch + 1] = copy.deepcopy(self.net.state_dict())

        self.best_model_state_elbo = best_state_elbo

        self.checkpoints = params_store

        self.best_model_state = self.best_model_state_elbo

        return self

    def predict(self, X, checkpoint_epoch=None, mc_samples=100):
        """
        Generate predictions from a selected checkpoint or the best model state.
        """
        if checkpoint_epoch:
            if checkpoint_epoch not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_epoch} not found. "
                                 f"Available checkpoints: {list(self.checkpoints.keys())}")
            state_dict = self.checkpoints[checkpoint_epoch]
        else:
            state_dict = self.best_model_state

        self.net.load_state_dict(state_dict)

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        for _ in range(mc_samples):
            preds.append(self.net(X).detach().cpu().numpy())
        preds = np.stack(preds, axis=0)

        return preds

class SoUBnnClassification(BuiltinUQ):
    """
    Variationally trained BNNs for classification.

    Supported models:
      - model="bnn": standard Bayesian NN (Gaussian variational weights)
      - model="sou": SoU-BNN with local-global shrinkage controlled by config["gamma_local"]
    """
    def __init__(self, config, model: str = "bnn", device=None):
        """
        Args:
            config: dict specifying architecture and learning hyperparameters.
            model: "bnn" or "sou".
            device: torch device or None.
        """
        super().__init__()
        self.config = dict(config)
        self.model = model
        self.device = device

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model == "bnn":
            self.net = bayesian_mlp.BayesianClassificationNet(
                ip_dim=self.config['ip_dim'],
                op_dim=self.config['op_dim'],
                num_nodes=self.config['num_nodes'],
                num_layers=self.config['num_layers'],
            )
        elif self.model == "sou":
            if 'gamma_local' not in self.config:
                raise ValueError("For model='sou', config['gamma_local'] must be specified.")
            if "hshoe_scale" not in self.config:
                raise ValueError("For model='sou', config['hshoe_scale'] must be specified.")

            self.net = SoUHshoeClassificationNet(
                ip_dim=self.config['ip_dim'],
                op_dim=self.config['op_dim'],
                num_nodes=self.config['num_nodes'],
                num_layers=self.config['num_layers'],
                hshoe_scale=self.config['hshoe_scale'],
                gamma_local=self.config['gamma_local'],
            )
        else:
            raise NotImplementedError("`model` must be one of {'bnn','sou'}.")

        if "batch_size" not in self.config:
            self.config["batch_size"] = 50
        self.net = self.net.to(self.device)

    def get_params(self, deep=True):
        """Return estimator parameters for sklearn-style compatibility."""
        return {"model": self.model, "config": self.config, "device": self.device}

    def fit(self, X=None, y=None, train_loader=None, checkpoint_epochs=None):
        """
        Fit the BNN classification model and store history, checkpoints, and best state.
        """
        if train_loader is None:
            train = data_utils.TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
            train_loader = data_utils.DataLoader(train, batch_size=self.config['batch_size'], shuffle=True)

        torch.manual_seed(1234)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['step_size'])
        neg_elbo = torch.zeros([self.config['num_epochs'], 1])
        params_store = {}
        best_elbo = float("inf")


        self.history = {"neg_elbo": [], "sparsity": [], "nodes_scales": []}

        for epoch in range(self.config['num_epochs']):
            avg_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                loss = self.net.neg_elbo(num_batches=len(train_loader), x=batch_x, y=batch_y) / batch_x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if hasattr(self.net, 'fixed_point_updates'):
                    self.net.fixed_point_updates()

                avg_loss += loss.item()

            epoch_loss = avg_loss / len(train_loader)
            neg_elbo[epoch] = epoch_loss

            if epoch_loss < best_elbo:
                best_elbo = epoch_loss
                best_state_elbo = copy.deepcopy(self.net.state_dict())
                self.best_model_elbo_epoch = epoch

            sparsity_metrics = _compute_layer_sparsity(self.net, thresholds=(1e-3, 1e-4, 1e-5))
            nodes_scales = _compute_node_scale_values(self.net, use_lognormal_mean=True)

            self.history["neg_elbo"].append(epoch_loss)
            self.history["sparsity"].append(sparsity_metrics)
            self.history["nodes_scales"].append(nodes_scales)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch[{epoch + 1}/{self.config['num_epochs']}], neg elbo: {epoch_loss:.6f}")

            if checkpoint_epochs and (epoch + 1) in checkpoint_epochs:
                params_store[epoch + 1] = copy.deepcopy(self.net.state_dict())

        self.best_model_state_elbo = best_state_elbo
        self.checkpoints = params_store

        self.best_model_state = self.best_model_state_elbo

        return self

    def predict(self, X, checkpoint_epoch=None, mc_samples=100):
        """
        Return the predictive class probability distribution for `X`.
        """
        if checkpoint_epoch:
            if checkpoint_epoch not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_epoch} not found. "
                                 f"Available: {list(self.checkpoints.keys())}")
            state_dict = self.checkpoints[checkpoint_epoch]
        else:
            state_dict = self.best_model_state

        self.net.load_state_dict(state_dict)

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_prob_samples = [F.softmax(self.net(X), dim=1).detach().cpu().numpy()
                          for _ in np.arange(mc_samples)]

        y_prob_samples_stacked = np.stack(y_prob_samples)  # (mc_samples, n_samples, n_classes)
        prob_mean = np.mean(y_prob_samples_stacked, 0)
        prob_var = np.var(y_prob_samples_stacked, 0)

        if len(np.shape(prob_mean)) == 1:
            y_pred_labels = prob_mean > 0.5

        else:
            y_pred_labels = np.argmax(prob_mean, axis=1)

        Result = namedtuple('res', ['y_pred', 'y_prob', 'y_prob_var', 'y_prob_samples'])
        res = Result(y_pred_labels, prob_mean, prob_var, y_prob_samples)

        return res

class SoUBnnClassificationHistory(BuiltinUQ):
    """
    Variationally trained BNNs for classification.

    Supported models:
      - model="bnn": standard Bayesian NN (Gaussian variational weights)
      - model="sou": SoU-BNN with local-global shrinkage controlled by config["gamma_local"]
    """

    def __init__(self, config, model: str = "bnn", device=None):
        """
        Args:
            config: dict specifying architecture and learning hyperparameters.
            model: "bnn" or "sou".
            device: torch device or None.
        """
        super().__init__()
        self.config = dict(config)
        self.device = device
        self.model = model

        # Resolve device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default batch size
        if "batch_size" not in self.config:
            self.config["batch_size"] = 50

        # ---- Build network ----
        if self.model == "bnn":
            self.net = bayesian_mlp.BayesianClassificationNet(
                ip_dim=self.config["ip_dim"],
                op_dim=self.config["op_dim"],
                num_nodes=self.config["num_nodes"],
                num_layers=self.config["num_layers"],
            )

        elif self.model == "sou":
            if "gamma_local" not in self.config:
                raise ValueError("For model='sou', config['gamma_local'] must be specified.")

            self.net = SoUHshoeClassificationNet(
                ip_dim=self.config["ip_dim"],
                op_dim=self.config["op_dim"],
                num_nodes=self.config["num_nodes"],
                num_layers=self.config["num_layers"],
                hshoe_scale=self.config["hshoe_scale"],
                gamma_local=self.config["gamma_local"],
            )

        else:
            raise NotImplementedError("`model` must be one of {'bnn','sou'}.")

        self.net = self.net.to(self.device)

    def get_params(self, deep=True):
        """Return estimator parameters for sklearn-style compatibility."""
        return {"model": self.model, "config": self.config, "device": self.device}

    def fit(self, X=None, y=None, train_loader=None, X_test=None, y_test=None, test_loader=None, checkpoint_epochs=None):
        """
        Fit the BNN classification model and track training history.
        """
        if train_loader is None:
            train = data_utils.TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
            train_loader = data_utils.DataLoader(train, batch_size=self.config['batch_size'], shuffle=True)


        X_test_t = None
        y_test_t = None
        if (X_test is not None) and (y_test is not None):
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)

        torch.manual_seed(1234)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['step_size'])
        neg_elbo = torch.zeros([self.config['num_epochs'], 1])
        params_store = {}
        best_elbo = float("inf")


        self.history = {"neg_elbo": [], "sparsity": [], "nodes_scales": [], "accuracy": []}

        for epoch in range(self.config['num_epochs']):
            avg_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                loss = self.net.neg_elbo(num_batches=len(train_loader), x=batch_x, y=batch_y) / batch_x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if hasattr(self.net, 'fixed_point_updates'):
                    self.net.fixed_point_updates()

                avg_loss += loss.item()

            epoch_loss = avg_loss / len(train_loader)
            neg_elbo[epoch] = epoch_loss

            if epoch_loss < best_elbo:
                best_elbo = epoch_loss
                best_state_elbo = copy.deepcopy(self.net.state_dict())
                self.best_model_elbo_epoch = epoch

            sparsity_metrics = _compute_layer_sparsity(self.net, thresholds=(1e-3, 1e-4, 1e-5))
            nodes_scales = _compute_node_scale_values(self.net, use_lognormal_mean=True)


            test_mc_samples=100
            acc = None
            if X_test_t is not None:
                self.net.eval()
                with torch.no_grad():
                    probs = 0.0
                    for _ in range(test_mc_samples):
                        probs = probs + F.softmax(self.net(X_test_t), dim=1)
                    probs = probs / float(test_mc_samples)

                    pred = probs.argmax(dim=1)
                    acc = (pred == y_test_t).float().mean().item()

            self.history["neg_elbo"].append(epoch_loss)
            self.history["sparsity"].append(sparsity_metrics)
            self.history["nodes_scales"].append(nodes_scales)
            self.history["accuracy"].append(acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch[{epoch + 1}/{self.config['num_epochs']}], neg elbo: {epoch_loss:.6f}")

            if checkpoint_epochs and (epoch + 1) in checkpoint_epochs:
                params_store[epoch + 1] = copy.deepcopy(self.net.state_dict())


        self.best_model_state_elbo = best_state_elbo
        self.checkpoints = params_store

        self.best_model_state = self.best_model_state_elbo

        return self

    def predict(self, X, checkpoint_epoch=None, mc_samples=100):
        """
        Return the predictive class probability distribution for `X`.
        """
        if checkpoint_epoch:
            if checkpoint_epoch not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_epoch} not found. "
                                 f"Available: {list(self.checkpoints.keys())}")
            state_dict = self.checkpoints[checkpoint_epoch]
        else:
            state_dict = self.best_model_state

        self.net.load_state_dict(state_dict)

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_prob_samples = [F.softmax(self.net(X), dim=1).detach().cpu().numpy()
                          for _ in np.arange(mc_samples)]

        y_prob_samples_stacked = np.stack(y_prob_samples)  # (mc_samples, n_samples, n_classes)
        prob_mean = np.mean(y_prob_samples_stacked, 0)
        prob_var = np.var(y_prob_samples_stacked, 0)

        if len(np.shape(prob_mean)) == 1:
            y_pred_labels = prob_mean > 0.5

        else:
            y_pred_labels = np.argmax(prob_mean, axis=1)

        Result = namedtuple('res', ['y_pred', 'y_prob', 'y_prob_var', 'y_prob_samples'])
        res = Result(y_pred_labels, prob_mean, prob_var, y_prob_samples)

        return res




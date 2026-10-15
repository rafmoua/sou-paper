from __future__ import annotations

import abc


class BuiltinUQ(abc.ABC):
    """Base class for models with built-in uncertainty quantification."""

    def __init__(self, *argv, **kwargs):
        """Initialize the base UQ model."""

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """Fit model parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *argv, **kwargs):
        """Return predictions and uncertainty quantities."""
        raise NotImplementedError

    def set_params(self, **parameters):
        """Set estimator parameters in sklearn style."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

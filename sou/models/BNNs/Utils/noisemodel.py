from __future__ import annotations

import abc


class AbstractNoiseModel(abc.ABC):
    """Base interface for observation noise models."""

    def __init__(self, *argv, **kwargs):
        """Initialize the noise model."""

    @abc.abstractmethod
    def loss(self, *argv, **kwargs):
        """Compute loss from predictions and targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_noise_var(self, *argv, **kwargs):
        """Return the current estimate of observation noise variance."""
        raise NotImplementedError

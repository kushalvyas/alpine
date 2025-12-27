import torch
import numpy as np

import torch.nn as nn

import torch
import torch.nn as nn


class Nonlinear(nn.Module):
    """Base class for activation functions that can be tracked using the FeatureExtractor context manager."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name.lower()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")


class Sine(Nonlinear):
    """Sine nonlinearity proposed by :cite:`siren2020sitzmann`"""

    def __init__(self, omega=30.0, name="sine"):
        super().__init__(name=name)
        self.omega = omega
        self.name = name.lower()

    def forward(self, x):
        return torch.sin(self.omega * x)


class ReLU(Nonlinear):
    def __init__(self, name="relu"):
        super().__init__(name=name)
        self.relu = nn.ReLU()
        self.name = name.lower()

    def forward(self, x):
        return self.relu(x)


class Gauss(Nonlinear):
    """Gaussian nonlinearity proposed by :cite:`ramasinghe2022beyond`."""

    def __init__(self, scale=1.0, name="gauss"):
        super().__init__(name=name)
        self.scale = scale
        self.name = name.lower()

    def forward(self, x):
        return torch.exp(-((self.scale * x) ** 2))


class Wavelet(Nonlinear):
    """Wavelet nonlinearty proposed by :cite:`saragadam2022wire`"""

    def __init__(self, sigma=1.0, omega=30.0, trainable=False, name="wavelet"):
        super(Wavelet, self).__init__(name=name)
        self.name = name.lower()
        self.sigma = nn.Parameter(sigma * torch.ones(1), requires_grad=trainable)
        self.omega = nn.Parameter(omega * torch.ones(1), requires_grad=trainable)

    def forward(self, x):
        return torch.exp(1j * (self.omega * x) - (self.sigma * x).abs().square())


class HOSC(Nonlinear):
    """HOSC nonlinearity proposed by :cite:`serrano2024hoscperiodicactivationfunction`"""

    def __init__(self, beta, name="hosc"):
        super().__init__(name=name)
        self.beta = beta
        self.name = name.lower()

    def forward(self, x):
        return torch.tanh(self.beta * torch.sin(x))


class Sinc(Nonlinear):
    """Sinc nonlinearity proposed by :cite:``."""

    def __init__(self, omega=30.0, name="sinc"):
        super().__init__(name=name)
        self.omega = omega
        self.name = name.lower()

    def forward(self, x):
        return torch.sinc(self.omega * x)


class FinerSine(Nonlinear):
    """Finer nonlinearity proposed by :cite: 'liu2024finer"""

    def __init__(self, omega=30.0, name="finer"):
        super().__init__(name=name)
        self.omega = omega
        self.name = name.lower()

    def forward(self, x):
        scale = torch.abs(x) + 1
        return torch.sin(self.omega * scale * x)

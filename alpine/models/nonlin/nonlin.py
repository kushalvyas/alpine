import torch
import numpy as np

import torch.nn as nn

from alpine.models.nonlin.tracked_activation import TrackedActivation

class Sine(TrackedActivation):
    """Sine nonlinearity proposed by :cite:`siren2020sitzmann`"""
    def __init__(self, omega=30.0, name='sine'):
        super().__init__()
        self.omega = omega
        self.name = name.lower()
    
    def forward(self, x):
        return torch.sin(self.omega * x)

class ReLU(TrackedActivation):
    def __init__(self, name='relu'):
        super().__init__()
        self.relu = nn.ReLU()
        self.name = name.lower()

    def forward(self, x):
        return self.relu(x)

class Gauss(TrackedActivation):
    '''Gaussian nonlinearity proposed by :cite:`ramasinghe2022beyond`. '''
    def __init__(self, scale=1.0, name='gauss'):
        super().__init__()
        self.scale = scale
        self.name = name.lower()
    
    def forward(self, x):
        return torch.exp(-(self.scale * x)**2)


class Wavelet(TrackedActivation):
    ''' Wavelet nonlinearty proposed by :cite:`saragadam2022wire` '''
    def __init__(self, sigma=1.0, omega=30.0, trainable=False, name="wavelet"):
        super(Wavelet, self).__init__()
        self.name = name.lower()
        self.sigma = nn.Parameter(sigma * torch.ones(1), requires_grad=trainable)
        self.omega = nn.Parameter(omega * torch.ones(1), requires_grad=trainable)
    
    def forward(self, x):
        return torch.exp(1j * (self.omega * x)  - (self.sigma * x).abs().square())


class HOSC(TrackedActivation):
    '''HOSC nonlinearity proposed by :cite:`serrano2024hoscperiodicactivationfunction` '''
    def __init__(self, beta, name='hosc'):
        super().__init__()
        self.beta = beta
        self.name = name.lower()
    
    def forward(self, x):
        return torch.tanh(self.beta * torch.sin(x))

class Sinc(TrackedActivation):
    '''Sinc nonlinearity proposed by :cite:``. '''
    def __init__(self, omega=30.0, name='sinc'):
        super().__init__()
        self.omega = omega
        self.name = name.lower()
    
    def forward(self, x):
        return torch.sinc(self.omega * x)
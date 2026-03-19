from collections import OrderedDict
from enum import Enum
import numpy as np

import torch
from torch import nn, Tensor

from ..trainers import AlpineBaseModule
from .nonlin import ReLU, Sine
from .utils import fourier_bases


class InitType(str, Enum):
    """ Enumeration class for the different initialization types."""
    relu = 'relu'
    sine = 'sine'
    fourier = 'fourier'


class NonlinClass:
    """ Data class for the different nonlinearities."""
    relu = ReLU
    sine = Sine


def init_layer(layer: nn.Linear, init_type: InitType, is_first: bool, omega: float | None = None):
    """Initializes the weights of the given linear layer based on the specified initialization type.

    Args:
        layer (nn.Linear): The linear layer to be initialized.
        init_type (str): Initialization type. Must be included in InitType.
        is_first (bool): Boolean indicating whether the layer is the first in the model.
        omega (float, optional): Controls the bandwidth of each layer of the siren.
    """
    if init_type == InitType.relu:
        # Use default pytorch init
        pass
    elif init_type == InitType.sine:
        if is_first:
            bound = 1 / layer.in_features
        else:
            bound = np.sqrt(6 / layer.in_features) / omega
        layer.weight.data.uniform_(-bound, bound)
    elif init_type == InitType.fourier:
        # Init is handled by FourierLinear
        pass
    else:
        raise ValueError(f"Unknown init type: {init_type}")


def get_linear_layer(
        in_features: int,
        out_features: int,
        num_phases: int,
        num_frequencies: int,
        scaling_factor: float,
        init_type: str,
        omega: float | None = None,
        is_first: bool = False,
        is_last: bool = False,
        bias: bool = True) -> nn.Module:
    """
    Determines and returns an appropriate linear layer instance based on initialization type
    and layer position in the model. Supports both FourierLinear and standard linear layers.

    Args:
        in_features (int): Input feature dimensionality of the linear layer.
        out_features (int): Output feature dimensionality of the linear layer.
        num_phases (int): Number of phases for FourierLinear initialization.
        num_frequencies (int): Number of frequencies for FourierLinear initialization.
        scaling_factor (float): Scaling factor for frequency initialization.
        init_type (str): Specifies the initialization type (e.g., standard or Fourier).
        omega (float, optional): Optional omega coefficient for initialization.
        is_first (bool): Indicates if this is the first layer in the model. Defaults to False.
        is_last (bool): Indicates if this is the last layer in the model. Defaults to False.
        bias (bool): Determines whether the linear layer includes a bias term. Defaults to True.

    Returns:
        nn.Module: An instance of FourierLinear or a standard linear layer.

    """

    if init_type == InitType.fourier and not is_first and not is_last:
        return FourierLinear(
            in_features,
            out_features,
            num_phases,
            num_frequencies,
            omega,
            scaling_factor,
        )

    # fallback to standard linear
    layer = nn.Linear(in_features, out_features, bias=bias)
    init_layer(layer, init_type, is_first, omega)

    return layer


class FourierLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_phases: int,
                 num_frequencies: int,
                 omega: float | None = None,
                 scaling_factor: float = 1.0,
                 ):
        """Fourier Reparametrization layer.
        
        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            num_phases (int): number of phases (P) for the Fourier basis.
            num_frequencies (int): number of frequencies (F) for the Fourier basis. 2F phases are used.
            omega (float, optional): Controls the bandwidth of each layer of the siren. Defaults to None for linear layers.
            scaling_factor (float, optional): Controls the scaling factor for the Fourier basis. Defaults to 1.0.

        """
        super(FourierLinear, self).__init__()
        # Create the Fourier basis
        bases_ = fourier_bases(num_phases, num_frequencies, in_features, scaling_factor)
        # Adjust omega for relu inits
        self.omega = omega if omega is not None else 1.0
        # Initialize the learnable weights, bases, and bias
        self.bases = nn.Parameter(bases_, requires_grad=False)
        self.lambda_weights = self._init_weights(num_phases, num_frequencies, out_features)
        self.bias = self._init_bias(out_features)

    def _init_weights(self, num_phases: int, num_frequencies: int, out_features: int) -> nn.Parameter:
        """
        Initializes weights for a layer following the principles outlined in a specific paper.

        The method computes the weights based on the number of frequencies, phases, and
        target output features. It uses bounds derived from normalization of basis vectors
        and scaling to initialize the weights.

        Args:
            num_phases (int): Number of phases, used to calculate the total number of basis
                vectors.
            num_frequencies (int): Number of frequencies to consider when determining the
                bounds for weight initialization. Uses *num_frequencies* low frequencies and
                *num_frequencies* high frequencies.
            out_features (int): Number of output features.

        Returns:
            nn.Parameter: Initialized weight tensor.
        """
        # Following the paper M = 2FP
        m = 2 * num_frequencies * num_phases
        # Compute norms of each basis vector (shape: m)
        normalization = torch.norm(self.bases, p=2, dim=1)  # (m,)
        scale = torch.sqrt(torch.tensor(6.0 / m, device=self.bases.device, dtype=self.bases.dtype))
        # Compute correct bounds for linear layer (omega=1) or sine layer (omega!=1)
        bounds = (scale / normalization) / self.omega
        weights = torch.empty(out_features, m).uniform_(-1.0, 1.0)
        weights = weights * bounds
        return nn.Parameter(weights, requires_grad=True)

    def _init_bias(self, out_features: int) -> nn.Parameter:
        """Initializes the bias term for the layer.
        Args:
            out_features (int): Number of output features.
        """
        bias = torch.zeros(out_features, dtype=self.bases.dtype, device=self.bases.device)
        return nn.Parameter(bias, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass of the Fourier Reparametrization layer."""
        x = x @ (self.lambda_weights @ self.bases).mT + self.bias
        return x


class LinearFR(AlpineBaseModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_phases: int,
        num_frequencies: int,
        nonlinearity: str = 'relu',
        scaling_factor: float = 1.0,
        omegas: list[float] | None = None,
        outermost_linear: bool = True,
        bias: bool = True,
    ):
        """Implements the Fourier Reparametrization INR using relu or SIREN activations :cite:`shi2024improved`.

        Args:
            in_features (int): number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D)
            hidden_features (int): width of each layer in the INR. Number of neurons per layer.
            hidden_layers (int): Total number of hidden layers in the INR.
            out_features (int): number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this field is 3.
            num_phases (int): number of phases (P) for the Fourier basis.
            num_frequencies (int): number of frequencies (F) for the Fourier basis. 2F frequencies are used.
            nonlinearity (str, optional): Specifies the nonlinearity to be used (sine or relu). Defaults to 'relu'.
            scaling_factor (float, optional): Controls the scaling factor for the Fourier basis. Defaults to 1.0.
            omegas (list[float], optional): Controls the bandwidth of each layer of the siren. Defaults to [30.0].
            outermost_linear (bool, optional): Ensures that the last layer is a linear layer with no activation. Defaults to True.
            bias (bool, optional): Sets bias for each layer in the INR. Defaults to True.
        """
        super(LinearFR, self).__init__()
        self.nonlinearity = NonlinClass.__dict__[nonlinearity]
        self.model = nn.ModuleList()

        # Check if omegas are provided, if not, use default values
        if omegas is None:
            # Empty omegas are only valid with relu nonlinearities
            assert nonlinearity == 'relu', 'Omegas must be specified for non-ReLU nonlinearities'
            self.omegas = [omegas] * hidden_layers
        else:
            self.omegas = (
                omegas if len(omegas) == hidden_layers else [omegas[0]] * (hidden_layers)
            )

        self.model.append(
            get_linear_layer(
                in_features=in_features,
                out_features=hidden_features,
                num_phases=num_phases,
                num_frequencies=num_frequencies,
                scaling_factor=scaling_factor,
                omega=self.omegas[0],
                init_type=nonlinearity,
                is_first=True,
                bias=bias,
            )
        )
        self.model.append(self.nonlinearity())

        for i in range(hidden_layers - 2):
            self.model.append(
                get_linear_layer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    num_phases=num_phases,
                    num_frequencies=num_frequencies,
                    scaling_factor=scaling_factor,
                    omega=self.omegas[i + 1],
                    init_type='fourier',
                    is_first=False,
                    bias=bias,
                )
            )
            self.model.append(self.nonlinearity())

        self.model.append(
            get_linear_layer(
                in_features=hidden_features,
                out_features=out_features,
                num_phases=num_phases,
                num_frequencies=num_frequencies,
                scaling_factor=scaling_factor,
                omega=self.omegas[-1],
                init_type=nonlinearity,
                is_first=False,
                is_last=outermost_linear,
                bias=bias,
            )
        )

    def forward(self, coords: Tensor, return_features: bool = False) -> dict:
        """Compute the forward pass of the Fourier Reparametrization model.

        Args:
            coords (torch.Tensor): Input coordinates or features to the model of shape :math:`b \\times \cdots \\times d` where b is batch and d is the number of input features.
            return_features (bool, optional): Set flag to True to return intermediate layer features along with computed output. Defaults to False.

        Returns:
            dict: Returns a dict with keys: output, features. The output key contains the output of the model. The features key contains the intermediate features of the model.
        """

        if return_features:
            return self.forward_w_features(coords)

        output = coords.clone()
        for i, layer in enumerate(self.model):
            output = layer(output)
        return {"output": output}

    def load_weights(self, weights: OrderedDict[str, Tensor]):
        """ Load weights from a state dict. """
        self.load_state_dict(weights)





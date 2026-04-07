from collections import OrderedDict
from enum import Enum
import numpy as np

import torch
from torch import nn, Tensor

from .utils import fourier_bases
from ..trainers import AlpineBaseModule
from .nonlin import ReLU, Sine


class NonlinClass(str, Enum):
    """ Data class for the different nonlinearities."""
    relu = 'relu'
    sine = 'sine'


def init_layer(layer: nn.Linear, nonlinearity: NonlinClass, is_first: bool, omega: float | None = None):
    """Initializes the weights of the given linear layer based on the specified initialization type.

    Args:
        layer (nn.Linear): The linear layer to be initialized.
        nonlinearity (NonlinClass): Nonlinearity class type (affects initialization).
        is_first (bool): Boolean indicating whether the layer is the first in the model.
        omega (float, optional): Controls the bandwidth of each layer of the siren.
    """

    if nonlinearity == NonlinClass.relu:
        # Use default pytorch init
        pass
    elif nonlinearity == NonlinClass.sine:
        if is_first:
            bound = 1 / layer.in_features
        else:
            bound = np.sqrt(6 / layer.in_features) / omega
        layer.weight.data.uniform_(-bound, bound)
    else:
        raise ValueError(f"Unknown init type: {nonlinearity}")


def get_linear_layer(
        in_features: int,
        out_features: int,
        nonlinearity: NonlinClass,
        omega: float | None = None,
        is_first: bool = False,
        bias: bool = True) -> nn.Module:
    """
    Determines and returns an appropriate linear layer instance based on initialization type
    and layer position in the model. Supports both FourierLinear and standard linear layers.

    Args:
        in_features (int): Input feature dimensionality of the linear layer.
        out_features (int): Output feature dimensionality of the linear layer.
        nonlinearity (NonlinClass): Specifies the nonlinearity to be used (sine or relu).
        omega (float, optional): Optional omega coefficient for initialization.
        is_first (bool): Indicates if this is the first layer in the model. Defaults to False.
        bias (bool): Determines whether the linear layer includes a bias term. Defaults to True.

    Returns:
        nn.Module: An instance of FourierLinear or a standard linear layer.

    """
    # Instance standard linear layer with ReLU or Sine activation
    layer = nn.Linear(in_features, out_features, bias=bias)
    init_layer(layer, nonlinearity, is_first, omega)

    return layer


def get_nonlinearity(nonlinearity: str, omega: float) -> nn.Module:
    """ Factory function to get the appropriate nonlinearity based on the specified type/args."""
    if nonlinearity == "sine":
        return Sine(omega=omega)
    elif nonlinearity == "relu":
        return ReLU()
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")


class FourierReparameterization(AlpineBaseModule):
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
        super(FourierReparameterization, self).__init__()
        try:
            self.nonlinearity = NonlinClass(nonlinearity)
        except ValueError:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        self.model = nn.ModuleList()

        assert hidden_layers > 1, 'Need at least 2 hidden layers'

        # Check if omegas are provided, if not, use default values
        if omegas is None:
            # Empty omegas are only valid with relu nonlinearities
            assert nonlinearity == 'relu', 'Omegas must be specified for non-ReLU nonlinearities'
            self.omegas = [omegas] * hidden_layers
        elif len(omegas) == 1:
            self.omegas = omegas * hidden_layers
        elif len(omegas) == hidden_layers:
            self.omegas = omegas
        else:
            raise ValueError(f"omegas must have length 1 or hidden_layers. Got {len(omegas)}")

        self.model.append(
            get_linear_layer(
                in_features=in_features,
                out_features=hidden_features,
                omega=self.omegas[0],
                nonlinearity=self.nonlinearity,
                is_first=True,
                bias=bias,
            )
        )
        self.model.append(get_nonlinearity(self.nonlinearity, self.omegas[0]))

        for i in range(hidden_layers - 2):
            self.model.append(
                FourierLinear(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    num_phases=num_phases,
                    num_frequencies=num_frequencies,
                    scaling_factor=scaling_factor,
                    omega=self.omegas[i + 1]
                )
            )
            self.model.append(get_nonlinearity(self.nonlinearity, self.omegas[i+1]))


        self.model.append(
            get_linear_layer(
                in_features=hidden_features,
                out_features=out_features,
                omega=self.omegas[-1],
                nonlinearity=self.nonlinearity,
                bias=bias,
            )
        )
        if not outermost_linear:
            self.model.append(get_nonlinearity(self.nonlinearity, self.omegas[-1]))

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


class FourierLinear(nn.Linear):
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
        super(FourierLinear, self).__init__(in_features, out_features)
        # Create the Fourier basis
        bases_ = fourier_bases(num_phases, num_frequencies, in_features, scaling_factor)
        # Adjust omega for relu inits
        self.omega = omega if omega is not None else 1.0
        # Initialize the learnable weights, bases, and bias
        self.bases = nn.Parameter(bases_, requires_grad=False)
        self.weight = self._init_weights(num_phases, num_frequencies, out_features)
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
        x = x @ (self.weight @ self.bases).mT + self.bias
        return x

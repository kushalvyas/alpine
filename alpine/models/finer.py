import numpy as np

from torch import nn
import torch

from ..trainers import AlpineBaseModule
from .nonlin import FinerSine


def get_linear_layer(
    in_features,
    out_features,
    omega=30.0,
    first_bias_scale=None,
    bias=True,
    is_first=False,
):
    layer = nn.Linear(in_features, out_features, bias=bias)
    # Initialize weights
    if is_first:
        layer.weight.data.uniform_(-1 / in_features, 1 / in_features)
    else:
        layer.weight.data.uniform_(
            -np.sqrt(6 / in_features) / omega, np.sqrt(6 / in_features) / omega
        )

    # Initialize bias
    if is_first and first_bias_scale is not None and bias:
        with torch.no_grad():
            layer.bias.data.uniform_(-first_bias_scale, first_bias_scale)

    return layer


class Finer(AlpineBaseModule):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        omegas=[30.0],
        first_bias_scale=None,
        bias=True,
    ):
        """Implements the FINER INR :cite:`liu2024finer`.

        FINER (Flexible Spectral-Bias Tuning in Implicit Neural Representation) extends SIREN by using
        a variable-periodic activation function and a tunable bias initialization scheme to control
        the frequency spectrum of the learned representation.

        Args:
            in_features (int): Number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D).
            hidden_features (int): Width of each layer in the INR. Number of neurons per layer.
            hidden_layers (int): Total number of hidden layers in the INR.
            out_features (int): Number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this is 3.
            omegas (list[float], optional): Controls the bandwidth of each layer. If a single value is provided, it is applied to all layers. Defaults to [30.0].
            first_bias_scale (float, optional): The scale 'k' for initializing the bias of the first layer.
                Larger values (e.g., 10-20) shift the activation to higher frequencies. If None, uses standard initialization. Defaults to None.
            bias (bool, optional): Sets bias for each layer in the INR. Defaults to True.
        """
        super(Finer, self).__init__()

        self.finer_sine = FinerSine
        self.model = nn.ModuleList()
        self.omegas = (
            omegas if len(omegas) == hidden_layers else [omegas[0]] * (hidden_layers)
        )

        self.model.append(
            get_linear_layer(
                in_features,
                hidden_features,
                is_first=True,
                omega=omegas[0],
                first_bias_scale=first_bias_scale,
                bias=bias,
            )
        )
        self.model.append(FinerSine(omega=self.omegas[0]))

        for i in range(hidden_layers - 2):
            self.model.append(
                get_linear_layer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega=self.omegas[i + 1],
                    bias=bias,
                )
            )
            self.model.append(FinerSine(omega=self.omegas[i + 1]))

        # Final layer
        self.model.append(
            get_linear_layer(
                hidden_features,
                out_features,
                is_first=False,
                omega=self.omegas[-1],
                bias=False,
            )
        )

    def forward(self, coords, return_features=False):
        """Compute the forward pass of the FINER model.


        Args:
            coords (torch.Tensor): Input coordinates of shape (batch_size, in_features).
            return_features (bool, optional): If True, returns intermediate features along with output. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally intermediate features.
        """
        if return_features:
            return self.forward_w_features(coords)

        output = coords.clone()
        for i, layer in enumerate(self.model):
            output = layer(output)
        return {"output": output}

    def load_weights(self, weights):
        """Load weights from a state dict.

        Args:
            weight_dict (dict): state dict containing the weights to load.
        """
        self.load_state_dict(weights)

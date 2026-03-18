import numpy as np

import torch
from torch import nn

from ..trainers import AlpineBaseModule
from .nonlin import ReLU, Sine
from .utils import fourier_bases


def get_linear_layer(
        in_features,
        out_features,
        num_phases: int,
        num_frequencies: int,
        scaling_factor: float,
        is_first=False,
        is_last=False,
        bias=True
):

    if is_first:
        layer = nn.Linear(in_features, out_features, bias=bias)

    elif is_last:
        layer = nn.Linear(in_features, out_features, bias=bias)
        layer.weight.data.uniform_(-np.sqrt(6 / in_features), np.sqrt(6 / in_features))
    else:
        layer = FourierLinear(in_features=in_features,
                              out_features=out_features,
                              num_phases=num_phases,
                              num_frequencies=num_frequencies,
                              scaling_factor=scaling_factor)

    return layer


class FourierLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_phases: int,
                 num_frequencies: int,
                 scaling_factor: float = 1.0
                 ):
        super(FourierLinear, self).__init__()
        bases_ = fourier_bases(num_phases, num_frequencies, in_features, scaling_factor)
        self.bases = nn.Parameter(bases_, requires_grad=False)
        self.lambda_weights = self._init_weights(num_phases, num_frequencies, out_features)
        self.bias = self._init_bias(out_features)

    def _init_weights(self, num_phases: int, num_frequencies: int, out_features: int) -> nn.Parameter:
        # Following the paper M = 2FP
        m = 2 * num_frequencies * num_phases
        # Compute norms of each basis vector (shape: m)
        normalization = torch.norm(self.bases, p=2, dim=1)  # (m,)
        scale = torch.sqrt(torch.tensor(6.0 / m, device=self.bases.device, dtype=self.bases.dtype))
        bounds = (scale / normalization)
        weights = torch.empty(out_features, m).uniform_(-1.0, 1.0)
        weights = weights * bounds
        return nn.Parameter(weights, requires_grad=True)

    def _init_bias(self, out_features: int) -> nn.Parameter:
        bias = torch.zeros(out_features, dtype=self.bases.dtype, device=self.bases.device)
        return nn.Parameter(bias, requires_grad=True)

    def forward(self, x):
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
        scaling_factor: float = 1.0,
        omegas: list[float] | None = None,
        outermost_linear: bool = True,
        bias: bool = True,
    ):
        super(LinearFR, self).__init__()
        self.nonlinearity = ReLU
        self.model = nn.ModuleList()

        if omegas is None:
            self.omegas = [1.0] * hidden_layers
        else:
            self.omegas = (
                omegas if len(omegas) == hidden_layers else [omegas[0]] * (hidden_layers)
            )

        self.model.append(
            get_linear_layer(
                in_features,
                hidden_features,
                num_phases,
                num_frequencies,
                scaling_factor,
                is_first=True,
                bias=bias,
            )
        )
        self.model.append(self.nonlinearity())

        for i in range(hidden_layers - 2):
            self.model.append(
                get_linear_layer(
                    hidden_features,
                    hidden_features,
                    num_phases,
                    num_frequencies,
                    scaling_factor,
                    is_first=False,
                    bias=bias,
                )
            )
            self.model.append(self.nonlinearity())

        self.model.append(
            get_linear_layer(
                hidden_features,
                out_features,
                num_phases,
                num_frequencies,
                scaling_factor,
                is_first=False,
                is_last=outermost_linear,
                bias=bias,
            )
        )

    def forward(self, coords, return_features=False):
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

    def load_weights(self, weights):
        self.load_state_dict(weights)





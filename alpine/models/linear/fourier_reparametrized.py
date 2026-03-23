import torch
from torch import nn, Tensor

from alpine.models.utils import fourier_bases


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

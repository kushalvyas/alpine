from torch import nn

from ..trainers import AlpineBaseModule
from .nonlin.nonlin import Gauss as GaussActivation

class Gauss(AlpineBaseModule):
    """Implements the Gauss INR :cite:`ramasinghe2022beyond`.
    
    Gauss utilizes non-periodic activations to improve performance of random initializations within coordinate 
    based MLPs
    
    Args:
        in_features (int): Number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D).
        hidden_features (int): Width of each layer in the INR. Number of neurons per layer.
        hidden_layers (int): Total number of hidden layers in the INR.
        out_features (int): Number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this is 3.
        scale (float, optional): Controls the bandwidth of each layer.
        bias (bool, optional): Sets bias for each layer in the INR. Defaults to True.
    """
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        scale=30.0,
        bias=True,
    ):
        super(Gauss, self).__init__()
        
        self.model = nn.ModuleList()
        
        # First layer
        self.model.append(nn.Linear(in_features, hidden_features, bias=bias))
        self.model.append(GaussActivation(scale))
        
        # Hideen layers
        for _ in range(hidden_layers - 2):
            self.model.append(nn.Linear(hidden_features, hidden_features, bias=bias))
            self.model.append(GaussActivation(scale))
            
        # Final layer
        self.model.append(nn.Linear(hidden_features, out_features))
        
    def forward(self, coords, return_features=False):
        """Compute the forward pass of the Gauss model
        
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
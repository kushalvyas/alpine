
import numpy as np

from torch import nn

from ..trainers import AlpineBaseModule
from .nonlin import Sine

def get_linear_layer(in_features, out_features, omega=30.0, bias=True, is_first=False, is_last=False):
    layer =  nn.Linear(in_features, out_features, bias=bias)    
    if is_first:
            layer.weight.data.uniform_(-1 / in_features, 1 / in_features)
    else:
        layer.weight.data.uniform_(-np.sqrt(6 / in_features) / omega, 
                                              np.sqrt(6 / in_features) / omega)
        
    if is_last:
        last_init = np.sqrt(6/in_features)/max(omega, 1e-12)
        layer.weight.data.uniform_(-last_init, last_init)
    
    return layer
    
class Siren(AlpineBaseModule):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 omegas = [30.0], bias=True):
        """Implements the Siren model by Sitzmann et.al.

        Args:
            in_features (int): number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D)
            hidden_features (int): width of each layer in the INR. Number of neurons per layer.
            hidden_layers (int): Total number of hidden layers in the INR.
            out_features (int): number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this field is 3.
            outermost_linear (bool, optional): Ensures that the last layer is a linear layer with no activation. Defaults to True.
            omegas (list[float], optional): Controls the bandwidth of each layer of the siren. Defaults to [30.0].
            bias (bool, optional): Sets bias for each layer in the INR. Defaults to True.
        """
        super(Siren, self).__init__()

        self.sine_layer = Sine
        self.model = nn.ModuleList()
        self.omegas = omegas if len(omegas) == hidden_layers else [omegas[0]]*(hidden_layers)

        self.model.append(get_linear_layer(in_features, hidden_features,
                                  is_first=True, omega=self.omegas[0], bias=bias))
        self.model.append(Sine(omega=self.omegas[0]))

        for i in range(hidden_layers-2):
            self.model.append(get_linear_layer(hidden_features, hidden_features, 
                                      is_first=False, omega=self.omegas[i+1], bias=bias))
            self.model.append(Sine(omega=self.omegas[i+1]))
        
        self.model.append(get_linear_layer(hidden_features, out_features, is_first=False, is_last=outermost_linear,bias=bias))
        if not outermost_linear:
            self.model.append(Sine(omega=self.omegas[-1]))

    
    def forward(self, coords, return_features=False):
        """Compute the forward pass of the Siren model.

        Args:
            coords (torch.Tensor): Input coordinates or features to the model of shape :math:`b \\times \cdots \\times d` where b is batch and d is the number of input features.
            return_features (bool, optional): Set flag to True to return intermediate layer features along with computed output. Defaults to False.

        Returns:
            dict: Returns a dict with keys: output, features. The output key contains the output of the model. The features key contains the intermediate features of the model.
        """
        if return_features:
            return self.forward_w_features(coords)
        
        output = coords.clone()
        for i,  layer in enumerate(self.model):
            output = layer(output)
        return {'output':output}

        
    
    def forward_w_features(self, coords):
        """Compute the forward pass of the Siren model and return intermediate features.

        Args:
            coords (torch.Tensor): Input coordinates or features to the model of shape $b \times * \times d$

        Returns:
            dict: Returns a dict with keys: output, features. The output key contains the output of the model. The features key contains the intermediate features of the model.
        """
        features = []
        output = coords.clone()
        for i,  layer in enumerate(self.model):
            output = layer(output)
            if hasattr(layer, 'name') and layer.name == 'Sine':
                features.append(output.detach().clone())
        return {'output':output, 'features':features}
        
    def load_weights(self, weights):
        self.load_state_dict(weights)


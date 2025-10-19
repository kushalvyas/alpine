import torch
import torch.nn as nn
from ..trainers import AlpineBaseModule
from .nonlin import ReLU

class PosEncoding(nn.Module):
    def __init__(self, in_features, mapping_size, scale ):
        """Positional encoding module to encode the input coordinates. Adapted from FFN by :cite:`tancik2020` .

        Args:
            in_features (_type_): Input features to the network such as coordinate dimensions
            mapping_size (_type_): dimensions to map input features.
            scale (_type_): deviation of distribution for sampling random frequencies.
        """
        super(PosEncoding, self).__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.scale = scale

        self.B = nn.Parameter(torch.randn(self.mapping_size, self.in_features) * scale, requires_grad=False)
        self.out_dim = 2 * self.mapping_size

    def forward(self, x):
        sin_x = torch.sin(2.0 * torch.pi * x @ self.B.T)
        cos_x = torch.cos(2.0 * torch.pi * x @ self.B.T)
        posenc_coords = torch.cat([sin_x, cos_x], dim=-1)
        
        return posenc_coords


class FFN(AlpineBaseModule):
    def __init__(self,in_features, hidden_features, 
                 hidden_layers, out_features,  outermost_linear=True, positional_encoding='fourier', positional_encoding_kwargs={'mapping_size':256, 'scale': 10.0} ):
        """Fourier Feature Networks by :cite:`tancik2020` . 
    
        Args:
            in_features (int): number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D)
            hidden_features (int): width of each layer in the INR. Number of neurons per layer.
            hidden_layers (int): Total number of hidden layers in the INR.
            out_features (int): number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this field is 3.
            outermost_linear (bool, optional): Ensures that the last layer is a linear layer with no activation. Defaults to True.
            positional_encoding (str, optional): Encoding scheeme. Currently only fourier encoding is supported. Defaults to 'fourier'.
            positional_encoding_kwargs (dict, optional): Parameters for position encoding layer. Defaults to {'mapping_size':256, 'scale': 10.0}.

        Raises:
            NotImplementedError: Rasies an exception for custom / non-fourier position encoding. 
        """
        
        super(FFN, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        if positional_encoding != 'fourier' and positional_encoding is not None:
            raise NotImplementedError("Only fourier encoding is supported.")

        self.positional_encoding = positional_encoding
        posencoding_outdim = in_features
    
        self.model = nn.ModuleList()
        if self.positional_encoding is not None:
            self.pos_encode = PosEncoding(in_features=in_features, **positional_encoding_kwargs)
            self.model.append(self.pos_encode)
            posencoding_outdim = self.pos_encode.out_dim

        self.model.append(nn.Linear(posencoding_outdim, hidden_features))
        self.model.append(ReLU())

        for i in range(self.hidden_layers-2):
            self.model.append(nn.Linear(hidden_features, hidden_features))
            self.model.append(ReLU())
        
        self.model.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.model.append(ReLU())
        
    def forward(self, coords, return_features=False):
        """Compute the forward pass of the FFN model.

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
        
        output = output
        return {'output':output}
    
    def load_weights(self, weights):
        self.load_state_dict(weights)


            
        


        
        
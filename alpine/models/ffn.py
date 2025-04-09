import torch
import torch.nn as nn
from ..trainers import AlpineBaseModule
from .nonlin import ReLU

class PosEncoding(nn.Module):
    '''Positional encoding module to encode the input coordinates. Adapted from FFN.'''
    def __init__(self, in_features, mapping_size, scale ):
        """_summary_

        Args:
            in_features (_type_): _description_
            max_freq (_type_): _description_
            num_levels (_type_): _description_
            log_sampling (bool, optional): _description_. Defaults to True.
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
        
        super(FFN, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        if positional_encoding != 'fourier' and positional_encoding is not None:
            raise NotImplementedError(f"Positional encoding {positional_encoding} not supported. Only fourier and no encoding (None) is supported.")

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
            if hasattr(layer, 'name') and layer.name == 'relu':
                features.append(output.detach().clone())
        
        output = output
        return {'output':output, 'features':features}
        
    def load_weights(self, weights):
        self.load_state_dict(weights)


            
        


        
        
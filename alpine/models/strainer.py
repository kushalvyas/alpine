import torch
import torch.nn as nn
from ..trainers import BaseINR

import numpy as np

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

class Strainer(BaseINR):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, num_shared_layers, num_decoders, outermost_linear=True,
                 omegas = [30.0], bias=True):
        """Implements Strainer model by Vyas et.al.

        Args:
            in_features (int): number of input features. For a coordinate based model, input the number of coordinate dims (2 for 2D, 3 for 3D)
            hidden_features (int): width of each layer in the INR. Number of neurons per layer.
            hidden_layers (int): Total number of hidden layers in the INR.
            out_features (int): number of output features. For a scalar or grayscale field, this is 1. For an RGB image, this field is 3.
            num_shared_layers (int): Number of layers shared for the encoder layer.
            num_decoders (int): Number of decoder heads for strainer. Use 1 for single decoder.
            outermost_linear (bool, optional): Ensures that the last layer is a linear layer with no activation. Defaults to True.
            omegas (list[float], optional): Controls the bandwidth of each layer of the siren. Defaults to [30.0].
            bias (bool, optional): Sets bias for each layer in the INR. Defaults to True.
        """
        super().__init__()
        assert hidden_layers > num_shared_layers, "Number of shared layers should be less than total number of layers"

        self.sine_layer = Sine
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.num_decoders = num_decoders
        self.omegas = omegas if len(omegas) == hidden_layers else [omegas[0]]*(hidden_layers)
        
        self.num_decoder_layers = hidden_layers - num_shared_layers
        assert self.num_decoder_layers >= 1, "Number of decoders should be atleast 1"

        for i in range(num_shared_layers):
            self.encoder.append(get_linear_layer(in_features if i == 0 else hidden_features, hidden_features,
                                  is_first=(i==0), omega=self.omegas[i], bias=bias))
            self.encoder.append(Sine(omega=self.omegas[i]))

        for i in range(self.num_decoders):
            _decoder = []
            for j in range(num_shared_layers, hidden_layers):
                _decoder.append(get_linear_layer(hidden_features, hidden_features if j != hidden_layers-1 else out_features, 
                                      is_first=False, omega=self.omegas[j], bias=bias, is_last=(j==hidden_layers-1)))
                if j != hidden_layers-1:
                    _decoder.append(Sine(omega=self.omegas[j]))
            
            self.decoder.append(nn.ModuleList(_decoder))
    

    def forward(self, coords, return_features=False):
        """Compute the forward pass of the Strainer model. 

        Args:
            coords (torch.Tensor): Input coordinates or features to the model of shape :math:`b \\times \cdots \\times d` where b is batch and d is the number of input features.
            return_features (bool, optional): Set flag to True to return intermediate layer features along with computed output. Defaults to False.

        Returns:
            dict: Returns a dict with keys: output, features. The output key contains the output of the model. The features key contains the intermediate features of the model.
        """
        if return_features:
            return self.forward_w_features(coords)
    
        output = coords
        for layer in self.encoder:
            output = layer(output)
        enc_output = output.clone()
        
        dec_outputs = []
        for i, decoder in enumerate(self.decoder):
            dec_output = enc_output.clone()
            for dec_layer in decoder:
                dec_output = dec_layer(dec_output)
            dec_outputs.append(dec_output)
        
        dec_outputs = torch.stack(dec_outputs, dim=1)
        return {'output': dec_outputs}
    
    def forward_w_features(self, coords):
        """Compute the forward pass of the Strainer model and return intermediate features.

        Args:
            coords (torch.Tensor): Input coordinates or features to the model of shape $b \\times num\_decoder \cdots * \\times d$

        Returns:
            dict: Returns a dict with keys: output, enc_features and dec_features. The output key contains the output of the model while enc_features and dec_features contain the intermediate features of the encoder and decoder respectively.
        """
        output = coords
        enc_features = []
        for layer in self.encoder:
            output = layer(output)
            if hasattr(layer, 'name') and layer.name == 'Sine':
                enc_features.append(output.detach().clone())
        enc_output = output.clone()
        
        dec_outputs = []
        dec_features = [[] for _ in range(self.num_decoders)]
        for i, decoder in enumerate(self.decoder):
            dec_output = enc_output.clone()
            for dec_layer in decoder:
                dec_output = dec_layer(dec_output)
                dec_features[i].append(dec_output.detach().clone())
            dec_outputs.append(dec_output)
        
        dec_outputs = torch.stack(dec_outputs, dim=1)
        return {'features': {'enc_features': enc_features, 'dec_features': dec_features}, 'output': dec_outputs}
    
    def load_weights(self, weights):
        self.load_state_dict(weights)


    def load_encoder_weights(self, weights):
        self.encoder.load_state_dict(weights)
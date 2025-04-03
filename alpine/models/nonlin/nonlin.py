import torch
import numpy as np

import torch.nn as nn

class Sine(nn.Module):
    def __init__(self, omega=30.0, name='Sine'):
        super().__init__()
        self.omega = omega
        self.name = name
    
    def forward(self, x):
        return torch.sin(self.omega * x)

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class Gauss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return torch.exp(-(self.scale * x)**2)


class Wire(nn.Module):
    def __init__(self, scale=1.0, omega=30.0):
        super().__init__()
        self.scale = scale
        self.omega = omega
    
    def forward(self, x):
        return torch.exp( -1 * (self.scale * x)**2 + 1j * (self.omega * x) )


class HOSC(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return torch.tanh(self.beta * torch.sin(x))

class Sinc(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x):
        return torch.sinc(self.omega * x)
    

# class Finer(nn.Module):
#     """Finer activation function, 
#     https://github.com/liuzhen0212/FINERplusplus/blob/main/models.py#L129

#     Args:
#         nn (_type_): _description_
#     """
#     def __init__(self, in_features, out_features, bias=True, omega=30, 
#                  is_first=False, is_last=False, 
#                  init_method='sine', init_gain=1, fbs=None, hbs=None, 
#                  alphaType=None, alphaReqGrad=False):
#         super().__init__()
#         self.omega = omega
#         self.is_last = is_last ## no activation
#         self.alphaType = alphaType
#         self.alphaReqGrad = alphaReqGrad
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
        
#         # init weights
#         init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
#         # init bias
#         init_bias_cond(self.linear, fbs, is_first)
    
#     def forward(self, input):
#         wx_b = self.linear(input) 
#         if not self.is_last:
#             return finer_activation(wx_b, self.omega)
#         return wx_b # is_last==True
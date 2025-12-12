import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import logging


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Union    
from ..losses import MSELoss
from collections import OrderedDict, defaultdict

from tqdm.autonotebook import tqdm
from ..utils.checkers import (check_opt_types, check_sch_types, check_lossfn_types)
from ..utils.checkers import wrap_signal_instance




class MAML(nn.Module):
    ''' Metalearning INR weights module. This module is used to train the INR weights using MAML algorithm.
    Reference: https://github.com/vsitzmann/metasdf and https://github.com/YannickStruempler/inr_based_compression
    '''
    
    def __init__(self, 
                 model, 
                 inner_steps, 
                 loss_fn=None,
                 init_lr = 1e-2,
                 opt_lr = 1e-4, 
                 lr_type = 'static',
                 first_order=False):
        """
        Wrapper module for MAML algorithm.
        
        Args:
            model (nn.Module): `alpine.model` to be trained.
            inner_steps (int): Number of inner steps for MAML algorithm.
            loss_fn (callable, optional): Loss function. Defaults to Mse loss.
            init_lr (float, optional): Inner loop learning rate. Defaults to 1e-2.
            opt_lr (float, optional): Optimizer learning rate. Defaults to 1e-4.
            lr_type (str, optional): Learning rate type for inner loop. Defaults to 'static'.
            first_order (bool, optional): Whether to use first order approximation. Defaults to False.
            
        
        """
        super(MAML, self).__init__()

        self.model = model
        self.inner_steps = inner_steps
        self.loss_fn = loss_fn if loss_fn is not None else self.loss_fn_mse
        self.lr_type = lr_type
        self.setup_init_lr(init_lr)
        self.opt_lr = opt_lr
        
        self.first_order = first_order
        self.mse_loss = nn.MSELoss().to(next(model.parameters()).device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt_lr)
        
    def setup_init_lr(self, init_lr):
        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(self.inner_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ModuleList([])
            _model_params = self.mdoel.parameters()
            for param in _model_params:
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(self.inner_steps)]))
        elif self.lr_type == 'simple_per_parameter':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr])) for _ in self.mdoel.parameters()])
        

    def loss_fn_mse(self, output, signal):
        mse_loss = self.mse_loss(output, signal)
        loss_val = mse_loss
        return loss_val, {'loss':float(mse_loss)}
    
    def loss_fn_inner(self, output, signal):
        loss = ((output- signal)**2).sum(0).mean()
        return loss, {'loss':float(loss)}

    def forward_w_params(self, coords, params=None):
        if params is None:
            raise ValueError("Params must be provided")
        
        batch_size = coords.shape[0]
        if batch_size > 1:
            def forward_single(params_dict, coords_single):
                return torch.func.functional_call(self.model, params_dict, (coords_single,))

            batched_out = torch.func.vmap(forward_single, in_dims=(0, 0))(params, coords)
            
            return batched_out
        else:
            return torch.func.functional_call(self.model, params, (coords,))
    
    
    def generate_params(
        self, 
        coords, 
        signals,
        num_meta_steps=None,
        **kwargs,   
    ) -> tuple[OrderedDict, list]:
        """Generates parameters for the model.
        
        Args:
            coords (torch.Tensor): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            signals (torch.Tensor): Signal of shape ( B x * x D) where B is batch size, and D is the dimensionality of the signal.
            num_meta_steps (int): Number of meta steps.
            **kwargs: Additional keyword arguments.
            
        Returns:
            tuple[OrderedDict, list]: A tuple containing the adapted parameters to the training batch.
        """
        
        meta_batch_size = coords.shape[0]
        with torch.enable_grad():
            adapted_parameters = OrderedDict()
            for name, param in self.model.named_parameters():#meta_named_parameters() #TODO: will replace alpine.base with two version, one for normal and one for meta.s
                    adapted_parameters[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))
                    
            for j in range(num_meta_steps):
                coords.requires_grad_()
                predictions = self.forward_w_params(coords, params=adapted_parameters)
                loss, loss_info = self.loss_fn_inner(predictions['output'], signals)
                
                grads = torch.autograd.grad(loss, 
                                            adapted_parameters.values(), 
                                            allow_unused=False, 
                                            create_graph=(True if (not self.first_order or j == num_meta_steps-1) else False))
                
                for i, ((name, param), grad) in enumerate(zip(adapted_parameters.items(), grads)):                    
                        if self.lr_type in ['static', 'global']:
                            lr = self.lr
                        elif self.lr_type in ['per_step']:
                            lr = self.lr[j]
                        elif self.lr_type in ['per_parameter']:
                            lr = self.lr[i][j] if num_meta_steps <= self.num_meta_steps else 1e-2
                        elif self.lr_type in ['simple_per_parameter']:
                            lr = self.lr[i]
                        else:
                            raise NotImplementedError
                        # print(f"Grad: {grad.shape}, Param: {param.shape}, Name: {name}")
                        adapted_parameters[name] = param - lr * grad
    
        return adapted_parameters
    

    
    def fit_signal(self, 
                   *,
                   input : torch.Tensor = None,
                   signal : Union[torch.Tensor, dict] = None,
                #    dataloader: torch.utils.data.DataLoader = None,
                #    n_iters : int = None,
                #    closure: callable = None,
                #    enable_tqdm : bool = False,
                #    return_features : bool = False,
                #    track_loss_history : bool = False,
                #    metric_trackers : dict = None,
                #    save_best_weights : bool = False,
                #    kwargs : dict = {},
                   ) -> dict:
        """ For a given batch of input coordinates and signal, this function performs MAML update to the model parameters.

        Args:
            input (torch.Tensor, optional): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            signal (Union[torch.Tensor, dict], optional): PyTorch tensor of shape ( B x * x D) where B is batch size, and D is the dimensionality of the signal.
            

        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True, loss, and other metrics if provided.
        """
        assert input is not None and signal is not None, "Input and signal must be provided"
        # assert dataloader is None, "Dataloader must be None when input and signal are provided"
        
        
        return self._fit_signal_tensors(
            input,
            signal,
        )
        
    def _fit_signal_tensors(self,
                    input,
                    signal,
                    ):
        """ For a given batch of input coordinates and signal, this function performs MAML update to the model parameters.

        Args:
            input (torch.Tensor, optional): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            signal (Union[torch.Tensor, dict], optional): PyTorch tensor of shape ( B x * x D) where B is batch size, and D is the dimensionality of the signal.
            

        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True, loss, and other metrics if provided.
        """
                
        _device = next(self.parameters()).device
        loss_history = []
        

        params = self.generate_params(input, 
                                        signal, 
                                        num_meta_steps=self.inner_steps)
        
        output  = self.forward_w_params(input, params=params)
        
        self.optimizer.zero_grad()
        loss, loss_info = self.loss_fn(output['output'], signal)
        loss.backward()
        self.optimizer.step()
                
                
        params = OrderedDict({k:v.clone().detach() for k,v in self.model.state_dict().items()})
        return params, loss_info

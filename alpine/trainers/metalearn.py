import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import logging

class MAMLMetaLearner():
    """Metalearning is an early experimental feature. We are in the process of integrating `torchmeta` with Alpine"""
    def __init__(self, model, inner_steps, config={}, custom_loss_fn=None, outer_optimizer='adam', inner_loop_loss_fn=None):
        super(MAMLMetaLearner, self).__init__()
        logging.warning("Metalearning is an early experimental feature. We are in the process of integrating `torchmeta` with Alpine.")
        self.model = model
        self.inner_steps = inner_steps
        self.config = config

        self.model_params = OrderedDict({
            k: nn.Parameter(v.detach().clone(), requires_grad=True) for k,v in self.model.state_dict().items()
        })

        self.outer_optimizer = outer_optimizer

        self.configure_optimizers()

        self.outer_loss_fn = None
        self.inner_loss_fn = None

        self.loss_fn = custom_loss_fn if custom_loss_fn is not None else self.loss_fn_mse
        self.inner_loop_loss_fn = inner_loop_loss_fn

    def configure_optimizers(self ):
        if self.outer_optimizer == 'adam':
            self.opt_thetas = torch.optim.Adam(self.model_params.values(), lr=self.config.get("outer_lr", 1e-4))
        else:
            self.opt_thetas = torch.optim.SGD(self.model_params.values(), lr=self.config.get("outer_lr", 1e-2))


    def get_parameters(self,copy=True):
        return OrderedDict({
            k: v.clone().detach() if copy else v for k, v in self.model_params.items()
        })
    
    def get_inr_parameters(self, copy=True):
        return OrderedDict({
            k: v.clone().detach() if copy else v for k, v in self.model_params.items()
        })
    
    def set_parameters(self, params):
        self.model_params.update(params)

    def mse_loss(self, x, y):
        return nn.functional.mse_loss(x, y)
    
    def loss_fn_mse(self, data_packet):
        mse_loss = self.mse_loss(data_packet['output'], data_packet['gt'])
        loss_val = mse_loss
        return loss_val, {'mse_loss':float(mse_loss)}
    
    def inner_loop(self, coords, data_packet):
        """
        learns the inr for inner_steps.
        gradient is taken w.r.t inr parameters.
        so regular back prop will do.
        """
        coords_shape = coords.shape
        logging.info(f"Coords shape: {coords_shape}")
        opt_inner = torch.optim.Adam(self.model_params.values(), lr=self.config.get("inner_lr",1e-4))
        
        gt = data_packet['gt']
        # Forward pass
        for i in range(self.inner_steps):
            opt_inner.zero_grad()
            output = torch.func.functional_call(self.model, self.model_params, coords)
            output = self.squeeze_output(output, gt)
            data_packet['output'] = output
            if self.inner_loop_loss_fn is not None:
                loss, _ = self.inner_loop_loss_fn(data_packet)
            else:
                loss, loss_info = self.loss_fn(data_packet)
            loss.backward()
            opt_inner.step()
            logging.info(f"Inner step: {i}/{self.inner_steps}. Loss: {loss}.")
        
        if self.inner_loop_loss_fn is not None:
            opt_inner.zero_grad()
            output = torch.func.functional_call(self.model, self.model_params, coords)
            output = self.squeeze_output(output, gt)
            data_packet['output'] = output
            loss, loss_info = self.loss_fn(data_packet)
            loss.backward()
            opt_inner.step()
        return self.model_params
    
    def squeeze_output(self, output, gt):
        if isinstance(output, dict):
            for k,v in output.items():
                if v.dim != gt.dim:
                    output[k] = v.squeeze(1)
        elif isinstance(output, torch.Tensor):
            if output.dim != gt.dim:
                output = output.squeeze(1)
        return output
            

    def outer_loop(self, coords, data_packet):
        self.opt_thetas.zero_grad()       
        gt = data_packet['gt']
        model_params_updated = self.inner_loop(coords, data_packet)
        # use updated parameters to do a forward pass
        output = torch.func.functional_call(self.model, model_params_updated, coords)
        output = self.squeeze_output(output, gt)
        data_packet['output'] = output
        loss, loss_info = self.loss_fn(data_packet)
        loss.backward()
        self.opt_thetas.step()

        return loss, loss_info

    def forward(self, coords, data_packet):
        loss, loss_info = self.outer_loop(coords, data_packet)
        return loss, loss_info

    def render_inner_loop(self, coords, gt, inner_loop_steps=1):
        model_params_copy = OrderedDict({
            k: v.clone().detach().requires_grad_(True) for k, v in self.model_params.items()
        })
        opt_render = torch.optim.Adam(model_params_copy.values(), lr=self.config.get("inner_lr",1e-4))
        for _ in range(inner_loop_steps):
            opt_render.zero_grad()
            output = torch.func.functional_call(self.model, model_params_copy, coords)
            loss = nn.functional.mse_loss(output['output'], gt)
            loss.backward()
            opt_render.step()
        output = self.squeeze_output(output, gt)
        return {'output':output}
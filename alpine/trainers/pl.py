import torch
import numpy as np
import torch.nn as nn

import lightning  as pl
from ..utils import check_opt_types, check_sch_types


class LightningTrainer(pl.LightningModule):
    def __init__(self, model, closure = None, return_features = False, log_results = False):
        """
        Lightning Trainer class for Alpine.
        """

        super(LightningTrainer, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_function = self.model.loss_function
        self.closure = closure # set closure to model forward
        self.return_features = return_features
        self.log_results = log_results

        print(self.return_features, self.log_results)
        

    # def configure_optimizers(self, optimizer_name="adam", learning_rate=1e-4, scheduler=None):
    #     """Setup optimizers.

    #     Args:
    #         optimizer_name (str, optional): Optimizer name. Defaults to "adam".
    #         learning_rate (float, optional): Learning rate. Defaults to 1e-4.
    #         scheduler (_type_, optional): Scheduler. Defaults to None.
    #     """
    #     check_opt_types(optimizer_name)
    #     check_sch_types(scheduler)


    #     if optimizer_name == "adam":
    #         self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    #     elif optimizer_name == "sgd":
    #         self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    #     else:
    #         raise ValueError("Optimizer not supported")

    #     if scheduler is not None:
    #         self.scheduler = scheduler(optimizer=self.optimizer)
    #     else:   
    #         self.scheduler = None

    #     self.is_model_compiled = True
        
    #     return {
    #         'optimizer': self.optimizer,
    #         'lr_scheduler': {
    #             'scheduler': self.scheduler,
    #         },
    #     }

    def configure_optimizers(self):
        return {
            'optimizer': self.model.optimizer,
            'lr_scheduler': {
                'scheduler': self.model.scheduler,
            },
        }
    
    def set_closure(self, closure: callable):
        """Set closure function.

        Args:
            closure (function): Closure function.
        """
        assert callable(closure), "Closure must be a callable function"
        self.closure = closure
    
    
    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (tuple): Coordinate input data of shape ( B x * .... * x D)
            

        Returns:
            torch.Tensor: Loss value.
        """
        input = batch['input']
        signal = batch['signal']


        # Forward pass
        if self.closure is None:
            # if closure is not set, use the model forward method
            output_packet = self.model(input, return_features=self.return_features)
        else:
            output_packet = self.closure(self.model, input, signal, return_features=self.return_features)
        
        # Loss calculation
        loss = self.loss_function(output_packet, signal)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=self.log_results)
        return loss

    


    
    


        
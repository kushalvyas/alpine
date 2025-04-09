import torch
import numpy as np
import torch.nn as nn

import lightning  as pl
from ..utils import check_opt_types, check_sch_types


class LightningTrainer(pl.LightningModule):
    def __init__(self, model, dataloader = None, closure = None, return_features = False, log_results = False):
        """
        Lightning Trainer class for Alpine.
        """

        super(LightningTrainer, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        if dataloader is not None:
            self.dataloader = dataloader
        # self.train_dataloader = dataloader
        self.loss_function = self.model.loss_function
        self.closure = closure # set closure to model forward
        self.return_features = return_features
        self.log_results = log_results
        self.test_outputs = []

        print(self.return_features, self.log_results)

    def train_dataloader(self):
        assert self.train_dataloader is not None, "Dataloader is not set. Please set the dataloader using the set_dataloader method."
        return self.train_dataloader
        

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

    
    def test_step(self, batch, batch_idx):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
    
        input = batch['input']
        signal = batch['signal']

        # Forward pass
        if self.closure is None:
            # if closure is not set, use the model forward method
            output_packet = self.model(input, return_features=self.return_features)
        else:
            output_packet = self.closure(self.model, input, signal, return_features=self.return_features)
        
        output_packet.update({'loss':  torch.tensor(0.0, device=self.device)})
        self.test_outputs.append(output_packet['output'].detach().cpu())
        return output_packet

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            all_outputs = self.all_gather(torch.cat(self.test_outputs, dim=0))
            print("on_test_end: test outputs gathered.")
            print(type(all_outputs))
            self.stacked_test_output = all_outputs.cpu().numpy()
    
    def on_test_end(self):
        print(f"[on_test_end] Local rank: {self.local_rank}, Global rank: {self.global_rank}")
        if self.trainer.is_global_zero:
            self.stacked_test_output = self.stacked_test_output
            print("on_test_end: test outputs gathered.")
            print(type(self.stacked_test_output), self.stacked_test_output.shape)


        
import torch
import torch.nn as nn
from ..utils.checkers import (check_opt_types, check_sch_types, check_lossfn_types)
from ..utils.checkers import wrap_signal_instance
from tqdm.autonotebook import tqdm
from ..losses import MSELoss
from collections import OrderedDict
from .feature_extractor import FeatureExtractor
from typing import Union
import math


class AlpineBaseModule(nn.Module):
    def __init__(self):
        """Base class for all Alpine INR models. Each INR model defined in `alpine.models` inherits `AlpineBaseModule`. 
        """
        super(AlpineBaseModule, self).__init__()
        self.optimizer = None
        self.scheduler = None
        self.loss_function = MSELoss() # default loss function.
        self.is_model_compiled = False
        self.best_weights = None
        self.best_loss = math.inf


    # avoid decorators for abstractmethods to allow for simpler debugging traces
    def forward(self, *args, **kwargs):
        """Forward pass.

        Raises:
            NotImplementedError: Please implement the forward method in your subclass.
        """
        raise NotImplementedError("Please implement the forward method in your subclass.")
    
    def forward_w_features(self, *args, **kwargs):
        """Forward pass with features.

        Raises:
            NotImplementedError: Please implement the forward method in your subclass.
        """
        raise NotImplementedError("Please implement the forward_w_features method in your subclass.")
    
    def compile(self, optimizer_name="adam", learning_rate=1e-4, scheduler=None):
        """Setup optimizers.

        Args:
            optimizer_name (str, optional): Optimizer name. Defaults to "adam".
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            scheduler (torch.optim.lr.schedular, optional): PyTorch scheduler object. Defaults to None.
        """
        check_opt_types(optimizer_name)
        check_sch_types(scheduler)


        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer not supported")

        if scheduler is not None:
            self.scheduler = scheduler(optimizer=self.optimizer)
        else:   
            self.scheduler = None

        self.is_model_compiled = True

    
    def register_loss_function(self, loss_function : callable):
        """Registers a loss function to the model. Default loss function for fitting the signal is mean square error.

        Args:
            loss_function (callable): A PyTorch `nn.Module` class object or a callable function that takes in two arguments: model's output dictionary and the ground truth data signal or dictionary.
        """
        check_lossfn_types(loss_function)
        
        self.loss_function = loss_function
        
    def _forward_w_features(self, input):
        """
        Runs a forward pass and extracts features using FeatureExtractor context manager.
        Args:
            input (torch.Tensor): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
        Returns:
            dict: Returns a dictionary containing the output from the INR, with features.
        """
        with FeatureExtractor(self) as extractor:
            output = self(input) # forward pass
        # add features to output dict
        if isinstance(output, dict):
            output.update({'features': extractor.features})
        else:
            output = {'output': output, 'features': extractor.features}
        return output

    def fit_signal(self, 
                   *,
                   input : torch.Tensor = None,
                   signal : Union[torch.Tensor, dict] = None,
                   dataloader: torch.utils.data.DataLoader = None,
                   n_iters : int = 1000,
                   closure: callable = None,
                   enable_tqdm : bool = True,
                   return_features : bool = False,
                   track_loss_history : bool = False,
                   metric_trackers : dict = None,
                   save_best_weights : bool = False,
                   kwargs : dict = {},
                   ) -> dict:
        """Final 

        Args:
            input (torch.Tensor, optional): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            signal (Union[torch.Tensor, dict], optional): PyTorch tensor or dictionary containing the signal and auxiliary data. Pleaee use `key=signal` for signal or ground truth measurement. Defaults to None.
            dataloader (torch.utils.data.DataLoader, optional): Input coordinate-signal pytorch dataloader object. Defaults to None.
            n_iters (int, optional): Number of iterations for fitting signal. Defaults to 1000.
            closure (callable, optional): Callable for custom forward propagation. Defaults to None and uses `AlpineBaseModules's` forward propagation with mse losss
            enable_tqdm (bool, optional): Enables tqdm progress bar. Defaults to True.
            return_features (bool, optional): Return intermediate INR features. Defaults to False.
            track_loss_history (bool, optional): Track loss while fitting the signal. Defaults to False.
            metric_trackers (dict, optional): Dictionary of torchmetrics Metrictracker objects. Defaults to None.
            save_best_weights (bool, optional): Use best weights saved during training. Defaults to False.
            kwargs (dict, optional): Other keyword arguments that is a dict of dicts. Defaults to {}.

        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True, loss, and other metrics if provided.
        """
        if dataloader is not None and (input is not None or signal is not None):
            raise ValueError("Either dataloader or input and signal pair must be provided, not all.")
        if dataloader is None:
            if input is None or signal is None:
                raise ValueError("Either dataloader or input and signal pair must be provided.")
            return self._fit_signal_tensor(
                input = input,
                signal = signal,
                n_iters = n_iters,
                closure = closure,
                enable_tqdm = enable_tqdm,
                return_features = return_features,
                track_loss_history = track_loss_history,
                metric_trackers = metric_trackers,
                save_best_weights = save_best_weights,
                kwargs = kwargs,
            )
        else:
            return self._fit_signal_dataloader(
                dataloader = dataloader,
                n_iters = n_iters,
                closure = closure,
                enable_tqdm = enable_tqdm,
                return_features = return_features,
                track_loss_history = track_loss_history,
                metric_trackers = metric_trackers,
                save_best_weights = save_best_weights,
                kwargs = kwargs,
            )
            

    def _fit_signal_tensor(self, 
                   input : torch.Tensor,
                   signal : Union[torch.Tensor, dict],
                   n_iters : int = 1000,
                   closure: callable = None,
                   enable_tqdm : bool = True,
                   return_features : bool = False,
                   track_loss_history : bool = False,
                   metric_trackers : dict = None,
                   save_best_weights : bool = False,
                   kwargs : dict = {},
                   
                   ) -> dict:
        """Fitting function for INR. This function is subclassed by the INR class, and can be overridden by the user.

        Args:
            input (torch.Tensor): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            signal (torch.Tensor): PyTorch tensor or dictionary containing the signal and auxiliary data. Pleaee use `key=signal` for signal or ground truth measurement. Defaults to None.
            n_iters (int, optional): Number of iterations for fitting signal. Defaults to 1000.
            closure (callable, optional): Callable for custom forward propagation. Defaults to None and uses `AlpineBaseModules's` forward propagation with mse losss
            enable_tqdm (bool, optional): Enables tqdm progress bar. Defaults to True.
            return_features (bool, optional): Return intermediate INR features. Defaults to False.
            track_loss_history (bool, optional): Track loss while fitting the signal. Defaults to False.
            metric_trackers (dict, optional): Dictionary of torchmetrics Metrictracker objects. Defaults to None.
            save_best_weights (bool, optional): Use best weights saved during training. Defaults to False.
            kwargs (dict, optional): Other keyword arguments that is a dict of dicts. Defaults to {}.
        
        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True, loss, and other metrics if provided.
        """
        if not self.is_model_compiled:
            self.compile()
        
        signal = wrap_signal_instance(signal) # triggers if signal is a torch.Tensor. Alpine's workflow is with dict-based and not tensor based.
        loss_history = []

        iter_pbar = range(n_iters) if not enable_tqdm else tqdm(range(n_iters), **kwargs.get("tqdm_kwargs", {}))
        for iteration in iter_pbar:
            self.optimizer.zero_grad()
            if closure is None:
                if return_features:
                    output_packet = self._forward_w_features(input)
                else:
                    output_packet = self(input) # forward pass returns a dict.
            else:

                output_packet = closure(self, input, signal=signal, iteration=iteration, return_features=return_features, **kwargs)
            
            loss = self.loss_function(output_packet, signal) # loss function takes in the output packet and the signal.

            loss.backward() # backward pass
            if track_loss_history:
                loss_history.append(float(loss.item())) 
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            if enable_tqdm:
                best_loss_str = f"Best Loss: {self.best_loss:.6f}" if self.best_loss < math.inf else ""
                iter_pbar.set_description(f"Iteration {iteration}/{n_iters}.  Loss: {loss.item():.6f}. {best_loss_str}")
                iter_pbar.refresh()

            if save_best_weights and (float(loss.item()) < self.best_loss):
                self.best_loss = float(loss.item())
                self.best_weights = OrderedDict({k:v.clone().detach() for k,v in self.state_dict().items()})

            if metric_trackers is not None and len(metric_trackers) > 0:
                for _, metric_tracker in metric_trackers.items():
                    metric_tracker.increment()
                    metric_tracker.update(output_packet['output'], signal['signal'])
            
        
        # retvals = dict(loss=float(loss.item()), output = output_packet)
        retvals = output_packet
        retvals.update(dict(loss=float(loss.item())))
        if track_loss_history:
            retvals.update(dict(loss_history = loss_history))
        
        if metric_trackers is  not None and len(metric_trackers) > 0:
            metrics_dict = {}
            for metric_name, metric_tracker in metric_trackers.items():
                computed_metrics = metric_tracker.compute_all().detach().cpu()
                metrics_dict.update({metric_name: computed_metrics})
            
            retvals.update(dict(metrics = metrics_dict, metric_trackers = metric_trackers))
        
        return retvals

    def _fit_signal_dataloader(self,
                    dataloader : torch.utils.data.DataLoader,
                    n_iters : int = 1000,
                    closure: callable = None,
                    enable_tqdm : bool = True,
                    return_features : bool = False,
                    track_loss_history : bool = False,
                    metric_trackers : dict = None,
                    save_best_weights : bool = False,
                    kwargs : dict = {}, ):
        """_summary_

        Args:
            dataloader (torch.utils.data.DataLoader): _description_
            n_iters (int, optional): Number of iterations for fitting signal. Defaults to 1000.
            closure (callable, optional): Callable for custom forward propagation. Defaults to None and uses `AlpineBaseModules's` forward propagation with mse losss
            enable_tqdm (bool, optional): Enables tqdm progress bar. Defaults to True.
            return_features (bool, optional): Return intermediate INR features. Defaults to False.
            track_loss_history (bool, optional): Track loss while fitting the signal. Defaults to False.
            metric_trackers (dict, optional): Dictionary of torchmetrics Metrictracker objects. Defaults to None.
            save_best_weights (bool, optional): Use best weights saved during training. Defaults to False.
            kwargs (dict, optional): Other keyword arguments that is a dict of dicts. Defaults to {}.

        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True, loss, and other metrics if provided.
        """
    
        if not self.is_model_compiled:
            self.compile()
        
        _device = next(self.parameters()).device
        loss_history = []

        iter_pbar = range(n_iters) if not enable_tqdm else tqdm(range(n_iters), **kwargs.get("tqdm_kwargs", {}))
        for iteration in iter_pbar:

            loss_iteration = 0.0
            for batch_idx, batch in enumerate(dataloader):
                input = batch['input'].to(_device)
                signal = wrap_signal_instance( batch['signal'].to(_device)) 

                self.optimizer.zero_grad()
                if closure is None:
                    output_packet = self(input, return_features=return_features) # forward pass returns a dict. 
                else:
                    output_packet = closure(self, input, signal=signal, iteration=iteration, return_features=return_features, **kwargs)
                
                loss = self.loss_function(output_packet, signal) # loss function takes in the output packet and the signal.

                loss.backward() # backward pass
                loss_iteration += float(loss.item())
            
            loss_over_dl = loss_iteration/len(dataloader)
            if track_loss_history:
                loss_history.append(loss_over_dl) 
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            if enable_tqdm:
                iter_pbar.set_description(f"Iteration {iteration}/{n_iters}.  Loss (over dataloader): {loss_over_dl:.6f}")

            if metric_trackers is not None and len(metric_trackers) > 0:
                for _, metric_tracker in metric_trackers.items():
                    metric_tracker.increment()
                    metric_tracker.update(output_packet['output'], signal['signal'])
            
        
        # retvals = dict(loss=float(loss.item()), output = output_packet)
        retvals = output_packet
        retvals.update(dict(loss=float(loss.item())))
        if track_loss_history:
            retvals.update(dict(loss_history = loss_history))
        
        if metric_trackers is  not None and len(metric_trackers) > 0:
            metrics_dict = {}
            for metric_name, metric_tracker in metric_trackers.items():
                computed_metrics = metric_tracker.compute_all().detach().cpu()
                metrics_dict.update({metric_name: computed_metrics})
            
            retvals.update(dict(metrics = metrics_dict, metric_trackers = metric_trackers))
        
        return retvals
    
    def render(self, input, closure=None, return_features=False, use_best_weights=False):
        """Renders the model output for the given input. This method is used for inference or evaluation.

        Args:
            input (torch.Tensor): Input coordinates of shape ( B x * x D) where B is batch size, and D is the dimensionality of the input grid.
            closure (callable, optional): Callable for custom forward propagation. Defaults to None and uses `AlpineBaseModules's` forward propagation with mse losss
            return_features (bool, optional): Return intermediate INR features. Defaults to False.
            use_best_weights (bool, optional): Use best weights saved during training. Defaults to False.

        Returns:
            dict: Returns a dictionary containing the output from the INR, with features if return_features=True.
        """
        # _weights = OrderedDict({k:v.clone().detach() for k,v in self.state_dict().items()})
        # self.load_weights(_weights)
        if use_best_weights and self.best_weights is not None:
            self.load_state_dict(self.best_weights)

        with torch.no_grad():
            if closure is not None:
                output_quantities = closure(self, input, return_features=return_features)
            else:
                if return_features:
                    output_quantities = self._forward_w_features(input)
                else:
                    output_quantities = self(input) # forward pass returns a dict.
            return output_quantities
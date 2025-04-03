import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from collections import OrderedDict
import torchmetrics


class BaseINR(nn.Module):
    def __init__(self, *args, **kwargs):
        """Base class for INR models.
        """
        super().__init__()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = torch.nn.functional.mse_loss

        self.is_model_compiled=False
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def set_loss_function(self, fn=torch.nn.functional.mse_loss):
        self.loss_function = fn

    def compile(self, optimizer_name="adam", learning_rate=1e-4, scheduler=None):
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
        
    
    def fit_signal(self, input, signal, n_iters=1000, enable_tqdm=False, return_features=False):
        """Fit the model to the given signal

        Args:
            input (torch.Tensor): (N x 2) input coordinates
            signal (_type_): _description_
            n_iters (int, optional): _description_. Defaults to 1000.
            enable_tqdm (bool, optional): _description_. Defaults to False.
            return_features (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert self.is_model_compiled, "Model must be compiled before fitting"
        iter_range = range(n_iters) if not enable_tqdm else tqdm(range(n_iters))
        for _iter in iter_range:

            self.optimizer.zero_grad()
            output_quantities = self(input, return_features=return_features)
            if return_features:
                output = output_quantities['output']
                features = output_quantities['features']
            else:
                output = output_quantities['output']

            loss = self.loss_function(output_quantities, signal) # changed this line from output to output_quantities
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            
            loss_val = float(loss.item())
            if enable_tqdm:
                iter_range.set_description(f"Epoch: {_iter}/{n_iters}. Loss: {loss_val:.6f}")
        ret_dict = {'output': output,  'metrics': self.metrics}
        if return_features:
            ret_dict.update({'features': features})

        return ret_dict
    
    def fit_signal(self, input, signal, n_iters=1000, enable_tqdm=False, return_features=False, metrics: list[torchmetrics.Metric]=None, out_shape=None, permute_output_after_reshape=None):
        """Fit the model to the signal.
        """
        assert self.is_model_compiled, "Model must be compiled before fitting"
        iter_range = range(n_iters) if not enable_tqdm else tqdm(range(n_iters))
        self.metrics = metrics
        for _iter in iter_range:

            self.optimizer.zero_grad()
            output_quantities = self(input, return_features=return_features)
            if return_features:
                output = output_quantities['output']
                features = output_quantities['features']
            else:
                output = output_quantities['output']

            # loss = self.loss_function(output, signal)
            loss = self.loss_function(output_quantities, signal) # changed this line from output to output_quantities
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            
            loss_val = float(loss.item())
            if enable_tqdm:
                iter_range.set_description(f"Epoch: {_iter}/{n_iters}. Loss: {loss_val:.6f}")

            if self.metrics is not None:
                for metric in self.metrics:
                    if isinstance(metric, torchmetrics.MetricTracker):
                        metric.increment()
                    if out_shape is not None:   
                        b = signal.shape[0]
                        output = output.reshape((b,) + (out_shape))
                        signal = signal.reshape((b,) + (out_shape))
                    if permute_output_after_reshape is not None:
                        output = output.permute(permute_output_after_reshape)
                        signal = signal.permute(permute_output_after_reshape)
                    metric.update(output, signal)

        ret_dict = {'output': output,  'metrics': self.metrics}
        if return_features:
            ret_dict.update({'features': features})
        return ret_dict
            
    def render(self, input, return_features=False):
        """Predict the signal.
        """
        
        _weights = OrderedDict({k:v.clone().detach() for k,v in self.state_dict().items()})
        self.load_weights(_weights)

        output_quantities = self(input, return_features=return_features)
        return output_quantities

    def fetch_metrics(self, return_as_np=True):
        _metrics = []
        for metric in self.metrics:
            computed_metrics = metric.compute_all().detach().cpu().numpy() if return_as_np else metric.compute_all().detach()
            _metrics.append(computed_metrics)
        return _metrics


        

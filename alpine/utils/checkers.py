import torch
import torch.nn as nn
from functools import partial

def check_opt_types(x):
    """
    Check if the optimizer name is a string or a callable. If it is a string, check if it is in the list of supported optimizers.
    Args:
        x (str or callable): The optimizer name or callable. Must be a string or a torch.optim object.
    """
    supported_optimizers = ["adam", "sgd",]
    if isinstance(x, str):
        if x not in supported_optimizers:
            raise ValueError(f"Optimizer {x} not supported (please raise a issue on GitHub). Supported optimizers are {supported_optimizers}.")
    elif not (callable(x) and isinstance(x, torch.optim)):
        raise ValueError("Optimizer must be a string or a callable.")
    

def check_sch_types(x):
    """
    Check if the scheduler is of the right type.
    Args:
        x (callable): learning rate scheduler, must be a torch.optim.lr_scheduler or `partial` object.
    """
    if x is not None:
        if not isinstance(x, (partial, torch.optim.lr_scheduler._LRScheduler)):
            raise ValueError("Scheduler must be a partial object or a torch.optim.lr_scheduler object.")
        

def check_lossfn_types(loss_function):
    if not callable(loss_function):
        raise ValueError("Loss function must be callable.")
    if not isinstance(loss_function, nn.Module):
        raise ValueError("Loss function must be a PyTorch nn.Module class object.")
    if not hasattr(loss_function, "forward") or  not hasattr(loss_function, "__call__"):
        raise ValueError("Loss function must have a forward method or must have a __call__ method.")
    
def wrap_signal_instance(x):
    if isinstance(x, torch.Tensor):
        return {'signal' : x}
    elif isinstance(x, dict):
        if 'signal' in x:
            return x
        else:
            raise ValueError("Dictionary must contain a key 'signal'.")
    else:
        raise ValueError("Input must be a torch.Tensor or a dictionary containing a key 'signal'.")

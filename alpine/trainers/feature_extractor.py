from collections import defaultdict
import torch.nn as nn
from ..models.nonlin.nonlin import Nonlinear

class FeatureExtractor:
    def __init__(self, model, track_linear=True, track_nonlinear=True):
        """Context manager to extract features from specified layers of a model during the forward pass."""
        self.model = model
        self.track_linear = track_linear
        self.track_nonlinear = track_nonlinear
        self.hooks = [] # List to store hook handles
        self.features = defaultdict(dict) # {layer_name: {'pre': tensor, 'post': tensor}}
    
        
    def __enter__(self):
        linear_idx = 0
        nonlin_idx = 0
        for name, module in self.model.named_modules():
            if self.track_linear and isinstance(module, nn.Linear):
                
                handle = module.register_forward_hook(
                    self._save_pre_activations(f"linear_{linear_idx}_pre")
                )
                self.hooks.append(handle)
                linear_idx += 1
            if self.track_nonlinear and isinstance(module, Nonlinear):
                
                handle = module.register_forward_hook(
                    self._save_post_activations(f"{module.name}_{nonlin_idx}_post")
                )
                self.hooks.append(handle)
                nonlin_idx += 1
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Remove all registered hooks to clean up.
        Args:
            exc_type: Exception type, if any.
            exc_value: Exception value, if any.
            traceback: Traceback object, if any.
        Returns:
            None
        """
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
                
    def _save_post_activations(self, name):
        """Hook function to save the output of a layer during the forward pass.
        Args:
            module (nn.Module): The layer/module being hooked.
            input (tuple): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        def hook(module, input, output):
            self.features[name] = output.detach().clone()
        return hook
        
    def _save_pre_activations(self, name):
        """Hook function to save the input of a layer during the forward pass.
        Args:
            module (nn.Module): The layer/module being hooked.
            input (tuple): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        def hook(module, input, output):
            self.features[name] = input[0].detach().clone()
        return hook
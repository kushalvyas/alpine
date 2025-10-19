from collections import OrderedDict
import torch
import torch.nn as nn
from alpine.models.nonlin import Nonlinear

class FeatureExtractor:
    def __init__(self, model, pre_layers = [nn.Linear], post_layers = [Nonlinear]):
        """Context manager to extract features from specified layers of a model during the forward pass."""
        self.model = model
        # Convert layer lists to tuples for instance  checking
        self.pre_layers = tuple(pre_layers)
        self.post_layers = tuple(post_layers)
        self.hooks = [] # List to store hook handles
        self.features = OrderedDict()
    
    def __enter__(self):
        """Starts the recursive hook registration process."""
        # The top-level features dictionary is the result of the recursive traversal.
        # It will be None if no hooks were attached, so we default to an empty dict.
        self.features = self._register_hooks_recursively(self.model) or OrderedDict()
        return self

    def _register_hooks_recursively(self, module: nn.Module):
        """
        Recursively traverses the model. Returns an OrderedDict of relevant features 
        (modules with hooks or that contain hooked children), or None if no hooks were attached.
        """
        # This dictionary will store the feature dicts of relevant children
        children_features = OrderedDict()
        
        for name, child in module.named_children():
            # We first recurse to the deepest level
            child_feature_dict = self._register_hooks_recursively(child)

            # Check if the child is a target layer for pre or post activation hooks
            is_pre_layer = isinstance(child, self.pre_layers)
            is_post_layer = isinstance(child, self.post_layers)
            
            # A child is added to feature dict if it's a target layer itself OR it contains hooked children.
            if child_feature_dict or is_pre_layer or is_post_layer:
                # If the child had no hooked descendants, initialize its feature dict
                if not child_feature_dict:
                    child_feature_dict = OrderedDict()

                # Attach hooks directly to the child if it's a target layer
                if is_pre_layer:
                    self._attach_hook(child_feature_dict, 'pre_activation', child, 'pre')
                if is_post_layer:
                    self._attach_hook(child_feature_dict, 'post_activation', child, 'post')
                
                # Add the child's feature dictionary to the current module's feature dict
                children_features[name] = child_feature_dict

        # If this module contained any relevant children, return their feature dicts. Otherwise, return None.
        return children_features if children_features else None
        
    def _attach_hook(self, feature_dict: OrderedDict, name: str, child: nn.Module, hook_type: str):
        handle = child.register_forward_hook(self._make_hook(feature_dict, name, hook_type))
        self.hooks.append(handle)

    def _make_hook(self, feature_dict: OrderedDict, name: str, hook_type: str):
        """Creates the hook function that saves the activation tensor into the correct dictionary."""
        def hook(module, input, output):
            if hook_type == 'pre':
                feature_dict[name] = input[0].detach().clone()
            elif hook_type == 'post':
                feature_dict[name] = output.detach().clone()
            else: 
                raise ValueError("hook_type must be 'pre' or 'post'")
        return hook
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Remove all registered hooks to clean up."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

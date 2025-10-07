from ..models.nonlin.tracked_activation import TrackedActivation

class FeatureExtractor:
    def __init__(self, model, layers=None):
        """Context manager to extract features from specified layers of a model during the forward pass."""
        self.model = model,
        self.layers = layers
        self.hooks = [] # List to store hook handles
        self.features = [] # List to store extracted features
    
        
    def __enter__(self):
        """
        Register forward hooks on the specified layers to capture their outputs.
        Returns:
            self: The FeatureExtractor instance with hooks registered.
        """
        # Register hooks on specified layers or all TrackedActivation layers if none specified
        for name, module in self.model.named_modules():
            if isinstance(module, TrackedActivation):
                handle = module.register_forward_hook(self._save_forward_features)
                self.hooks.append(handle)
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
                
    def _save_forward_features(self, module, input, output):
        """Hook function to save the output of a layer during the forward pass.
        Args:
            module (nn.Module): The layer/module being hooked.
            input (tuple): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        self.features.append(output.detach())
        
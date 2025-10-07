import torch
import torch.nn as nn

class TrackedActivation(nn.Module):
    """Base class for activation functions that can be tracked using the FeatureExtractor context manager."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name.lower()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
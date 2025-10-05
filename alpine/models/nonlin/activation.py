import torch
import torch.nn as nn

class TrackedActivation(nn.Module):
    """Base class for activation functions that can store their outputs."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name.lower()
        self._last_output = None  # store most recent forward output

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def get_features(self):
        """Return the most recent stored output (detached)."""
        if self._last_output is None:
            raise RuntimeError(f"No stored features yet in {self.name}")
        return self._last_output.detach()
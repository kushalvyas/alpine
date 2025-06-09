import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """Mean squared error loss

        Args:
            reduction (str, optional): Reduction applied to the computed mse tensor. Defaults to 'mean'.
        """
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, x, y):
        # x, and y are packets

        x_output = x.get('output', None)    
        assert x_output is not None, "Output not found in the packet.Please return the output as a dict object with keys 'output' and others"

        if isinstance(y, dict):
            y_signal = y.get('signal', None)
        else:
            y_signal = y
        assert y_signal is not None, "Output not found in the packet.Please return the output as a dict object with keys 'output' and others"

        # Compute the MSE loss
        loss = self.mse_loss(x_output, y_signal)
        return loss
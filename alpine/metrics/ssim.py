import torchmetrics
import torch.nn as nn
import torch


class SSIM(torchmetrics.Metric):
    """
    Structural Similarity Index (SSIM) metric.
    Example shows how to use a subclass torchmetrics.Metric to create a custom metric.
    """

    def __init__(
        self,
        signal_shape,
        data_range=1.0,
        kernel_size=11,
        sigma=1.5,
        reduction="elementwise_mean",
    ):
        super().__init__()
        self.signal_shape = signal_shape
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=kernel_size,
            sigma=sigma,
            reduction=reduction,
        )
        self.add_state("ssim_val", default=torch.tensor(0), dist_reduce_fx=None)

    @property
    def higher_is_better(self):
        return True

    def update(self, x, y):
        b = x.shape[0]
        x_output = x.reshape((b,) + self.signal_shape + (-1,))
        y_output = y.reshape((b,) + self.signal_shape + (-1,))

        x = x_output.permute(0, 3, 1, 2)
        y = y_output.permute(0, 3, 1, 2)

        self.ssim_val = self.ssim(x, y)

    def compute(self):
        return self.ssim_val.float()

__author__ = "Kushal Vyas"
import torch
import numpy as np
from ..utils import coords


def unflat_index(index, shape_of_tensor):
    return torch.unravel_index(index, shape_of_tensor)


class BatchedCoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, grid_dims: tuple, bounds: tuple = (-1, 1), vectorized=True):
        """PyTorch dataloader for generating coordinate datasets in batches.

        Args:
            grid_dims (tuple): Input d-dimensional grid dimensions.
            bounds (tuple) : Bounds of the grid. Defaults to (-1, 1).
            vectorized (bool): If True, returns a vectorized grid of shape N x d. Defaults to True.
        """
        super(BatchedCoordinateDataset).__init__()
        assert grid_dims is not None, "Grid dimensions must be provided."
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.vectorized = vectorized
        self.n_points = np.prod(grid_dims)

        self.build_coordinate_tensors()

    def __len__(self):
        return self.n_points

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset."""

        grid_axis = (
            torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i])
            for i in range(len(self.grid_dims))
        )

        grid_meshgrids = torch.meshgrid(*grid_axis, indexing="ij")
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        self.grid_tensor = grid_tensor.float()

    def __getitem__(self, idx):
        """Returns a batch of coordinates based on the specified index.

        Args:
            idx (int): Index of the batch to be returned.

        Returns:
            torch.Tensor: Batch of coordinates.
        """
        idx = unflat_index(
            idx, self.grid_dims
        )  # always gets called because idx will be linear index.
        return {"input": self.grid_tensor[idx]}


class CoordinateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for generating a grid of coordinates as input to the network."""

    def __init__(self, grid_dims: tuple, bounds: tuple = (-1, 1), vectorized=True):
        super(CoordinateDataset).__init__()
        """PyTorch Dataset for generating a grid of coordinates as input to the network. Please use `batchsize=1` for this dataset. 
        If you prefer using a Batched coordinate dataset please use `alpine.dataloaders.BatchedCoordinateDataset` instead.
        
        Args:
            grid_dims (tuple): Dimensions of the d-dimensional grid to be generated.
            bounds (tuple, optional): Bounds of the grid. Defaults to (-1, 1).
            vectorized (bool): If True, returns a vectorized grid of shape N x d. Defaults to True.
        """
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.n_points = np.prod(grid_dims)

    def __len__(self):
        return 1  # return 1 as we return all coordinates at once.

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset."""
        grid_axis = [
            torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i])
            for i in range(len(self.grid_dims)).reshape(-1, 1)
        ]

        grid_meshgrids = torch.meshgrid(*grid_axis, indexing="ij")
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        self.grid_tensor = grid_tensor.float()

    def __getitem__(self, idx):
        return {"input": self.grid_tensor}

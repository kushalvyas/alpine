__author__ = "Kushal Vyas"
import torch
import numpy as np

class BatchedCoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, grid_dims: tuple, randomize_coordinates=False):
        super(BatchedCoordinateDataset).__init__()
        pass


class CoordinateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for generating a grid of coordinates as input to the network.
    """
    def __init__(self, grid_dims : tuple, bounds : tuple = (-1, 1), indexing='ij'):
        super(CoordinateDataset).__init__()
        """PyTorch Dataset for generating a grid of coordinates as input to the network.
        
        Args:
            grid_dims (tuple): Dimensions of the grid to be generated.
            bounds (tuple, optional): Bounds of the grid. Defaults to (-1, 1).
        """
        self.grid_dims = grid_dims
        self.bounds = bounds

        self.build_grid_dataset(self.grid_dims, self.bounds)
    
    def build_grid_dataset(self, grid_dims, bounds = (-1, 1)):
        """Builds a grid dataset of coordinates based on the specified dimensions and bounds. Used internally by CoordinateDataset.

        Args:
            grid_dims (tuple): Dimensions of the grid to be generated.
            bounds (tuple, optional): Bounds of the grid. Defaults to (-1, 1).
        """
        pass


        



import torch
import numpy as np
from typing import Union

class BatchedNDSignalLoader(torch.utils.data.Dataset):
    def __init__(self, signal : Union[np.ndarray, torch.Tensor], grid_dims:tuple,  bounds: tuple= (-1, 1),  vectorized : bool=True, normalize_signal : bool=True, normalize_fn:callable=None):
        """_summary_

        Args:
            signal (Any): Indexible object containing the signal data. Can be numpy array, list, torch.Tensor, etc. Must be of shape grid_dims[0] x grid_dims[1] x ... x grid_dims[n] x (optional channels).
            grid_dims (tuple): _description_
            bounds (tuple, optional): _description_. Defaults to (-1, 1).
            vectorized (bool, optional): _description_. Defaults to True.
            normalize_signal (bool, optional): Min max normalization of the signal. Defaults to True.
            normalize_fn (callable, optional): custom callable function to normalize signal. Function will accept the signal as an argument. Defaults to None. 
        """

        super(BatchedNDSignalLoader).__init__()
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.vectorized = vectorized
        self.normalize_signal = normalize_signal
        self.normalize_fn = normalize_fn
        self.signal = self.setup_signal(signal)
        self.grid_tensor = self.build_coordinate_tensors()
    
    def setup_signal(self, signal):
        """
        Sets up the signal for the dataset. This includes reshaping, normalizing, and converting to a tensor if necessary. Used internally by BatchedNDSignalLoader.
        """
        assert np.prod(signal.shape[:len(self.grid_dims)]) == np.prod(self.grid_dims), f"Signal shape {signal.shape} does not match grid dimensions {self.grid_dims}."
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        elif isinstance(signal, list):
            signal = torch.tensor(signal)
        elif not isinstance(signal, torch.Tensor):
            raise TypeError("Signal must be a numpy array, list, or torch.Tensor.")
        
        if self.normalize_signal:
            if self.normalize_fn is None:
                signal = (signal - signal.min()) / (signal.max() - signal.min())
            else:
                signal = self.normalize_fn(signal)
        
        if self.vectorized:
            signal = signal.reshape(np.prod(self.grid_dims), -1)
        
        return signal

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset.
        """
        
        grid_axis = (torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i]) for i in range(len(self.grid_dims)))
        grid_meshgrids = torch.meshgrid(*grid_axis, indexing='ij')
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        return grid_tensor.float()

    def __len__(self):
        return np.prod(self.grid_dims)

    def __getitem__(self, idx):
        """Returns a batch of signal data based on the specified index.

        Args:
            idx (int): Index of the batch to be returned.

        Returns:
            torch.Tensor: Batch of signal data.
        """
        idx = torch.unravel_index(torch.tensor(idx), self.grid_dims) if not self.vectorized else idx
        coords = self.grid_tensor[idx]
        signal = self.signal[idx]

        return {'input' : coords.float(), 'signal' : signal.float()}
    



class NDSignalLoader(torch.utils.data.Dataset):
    def __init__(self, signal : Union[np.ndarray, torch.Tensor], grid_dims:tuple,  bounds: tuple= (-1, 1),  vectorized : bool=True, normalize_signal : bool=True, normalize_fn:callable=None):
        """_summary_

        Args:
            signal (Any): Indexible object containing the signal data. Can be numpy array, list, torch.Tensor, etc.
            grid_dims (tuple): _description_
            bounds (tuple, optional): _description_. Defaults to (-1, 1).
            vectorized (bool, optional): _description_. Defaults to True.
            normalize_signal (bool, optional): Min max normalization of the signal. Defaults to True.
            normalize_fn (callable, optional): custom callable function to normalize signal. Function will accept the signal as an argument. Defaults to None. 
        """

        super(NDSignalLoader).__init__()
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.vectorized = vectorized
        self.normalize_signal = normalize_signal
        self.normalize_fn = normalize_fn
        self.signal = self.setup_signal(signal)
        self.grid_tensor = self.build_coordinate_tensors()
    
    def setup_signal(self, signal):
        """
        Sets up the signal for the dataset. This includes reshaping, normalizing, and converting to a tensor if necessary. Used internally by NDSignalLoader.
        """
        assert np.prod(signal.shape[:len(self.grid_dims)]) == np.prod(self.grid_dims), f"Signal shape {signal.shape} does not match grid dimensions {self.grid_dims}."
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        elif isinstance(signal, list):
            signal = torch.tensor(signal)
        elif not isinstance(signal, torch.Tensor):
            raise TypeError("Signal must be a numpy array, list, or torch.Tensor.")
        
        if self.normalize_signal:
            if self.normalize_fn is None:
                signal = (signal - signal.min()) / (signal.max() - signal.min())
            else:
                signal = self.normalize_fn(signal)
        
        if self.vectorized:
            signal = signal.reshape(np.prod(self.grid_dims), -1)
        
        return signal

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset.
        """
        
        grid_axis = [torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i]) for i in range(len(self.grid_dims))]
        
        grid_meshgrids = torch.meshgrid(*grid_axis, indexing='ij')
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        return grid_tensor.float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Returns a batch of signal data based on the specified index.

        Args:
            idx (int): Index of the batch to be returned.

        Returns:
            torch.Tensor: Batch of signal data.
        """
        coords = self.grid_tensor
        signal = self.signal

        return {'input' : coords.float(), 'signal' : signal.float()}
    

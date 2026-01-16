"""Bio data loaders for NIfTI and PDB files.

Requires: nibabel, biopandas, pandas
Install with: pip install alpine[bio]
"""

import nibabel as nib
import numpy as np
import torch
import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional


def normalize_fn(x, normalize_method="minmax"):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_nii_gz(
    filename,
    data_key="f_data",
    squeeze_dims=True,
    normalize=False,
    normalize_method="minmax",
    return_as_torch=False,
):
    """Load NIfTI (.nii.gz) files.

    Args:
        filename: Path to the .nii.gz file
        data_key: Data key to extract. Defaults to 'f_data'.
        squeeze_dims: Whether to squeeze singleton dimensions. Defaults to True.
        normalize: Whether to normalize the data. Defaults to False.
        normalize_method: Normalization method. Defaults to 'minmax'.
        return_as_torch: Whether to return as torch tensor. Defaults to False.

    Returns:
        Loaded data as numpy array or torch tensor.
    """
    data = nib.load(filename)
    if data_key == "f_data":
        data = data.get_fdata()

    if squeeze_dims:
        find_ones = list(data.shape).index(1)
        data = np.squeeze(data, axis=find_ones)

    if normalize:
        data = normalize_fn(data, normalize_method)

    if return_as_torch:
        return torch.from_numpy(data).float()

    return data


def load_pdb(filename, model_index=1):
    """Load PDB file using PandasPdb.

    Code adapted from Biopandas and @jgbrasier
    https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4

    Args:
        filename: Path to the PDB file
        model_index: Model index to load. Defaults to 1.

    Returns:
        DataFrame with concatenated ATOM and HETATM records.
    """
    ppdb = PandasPdb()
    protein_data = ppdb.read_pdb(filename)
    protein_model = protein_data.get_model(model_index=model_index)

    # get atoms
    atoms = protein_model.df["ATOM"]

    # get hetatm
    hetatm = protein_model.df["HETATM"]

    return pd.concat([atoms, hetatm])

import nibabel as nib
import numpy as np
import torch
import pandas as pd
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader
from typing import Optional

def normalize_fn(x, normalize_method = 'minmax'):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_nii_gz(filename, data_key = 'f_data', squeeze_dims=True, normalize=False, normalize_method='minmax', return_as_torch=False):
    """_summary_

    Args:
        filename (_type_): _description_
        data_key (str, optional): _description_. Defaults to 'f_data'.
        squeeze_dims (bool, optional): _description_. Defaults to True.
        normalize (bool, optional): _description_. Defaults to False.
        normalize_method (str, optional): _description_. Defaults to 'minmax'.
        return_as_torch (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    data = nib.load(filename)
    if data_key == 'f_data':
        data = data.get_fdata()
    
    if squeeze_dims:
        find_ones = list(data.shape).index(1)
        data = np.squeeze(data,axis = find_ones)

    if normalize:
        data = normalize_fn(data, normalize_method)

    if return_as_torch:
        return torch.from_numpy(data).float()
    
    
    return data



def load_pdb(filename, model_index=1):
    """Loading PDF file using PandasPdb

    Code adapted from Biopandas and @jgbrasier
    https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4

    Args:
        filename (_type_): _description_
        model_index (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    ppdb = PandasPdb()
    protein_data = ppdb.read_pdb(filename)
    protein_model = protein_data.get_model(model_index=model_index)

    # get atoms
    atoms = protein_model.df['ATOM']

    # get 
    hetatm = protein_model.df['HETATM']

    return pd.concat([atoms, hetatm])


if __name__ == '__main__':

    nii_file = "/shared/kv30/oasis_mri/OASIS_OAS1_0060_MR1/slice_norm.nii.gz"
    data = load_nii_gz(nii_file, data_key='f_data', squeeze_dims=True, normalize=True, normalize_method='minmax', return_as_torch=False)

    print(data.shape)
    print('done')

    protein_data = "/home/kv30/KV/ALL_INR_PROJECTS/ALPINE_LIBRARY/alpine/examples/data/proteins/3nir.pdb"
    protein_data = load_pdb(protein_data, model_index=1)
    print(protein_data)
    print('done')

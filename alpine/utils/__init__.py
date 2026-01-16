from .coords import *
from .functional import *
from .checkers import *
# grid_search requires pandas: pip install alpine[bio] then import from alpine.utils.grid_search


def normalize(x, mode="minmax"):
    if mode == "minmax":
        x = (x - x.min()) / (x.max() - x.min())
    elif mode == "max":
        x = x / x.max()
    elif mode == "meanstd":
        x = (x - x.mean()) / x.std()
    return x


def normalize_nd(tensor, dim=None):
    if dim == None:
        return normalize(tensor)
    return (tensor - torch.amin(tensor, dim=dim, keepdim=True)) / (
        torch.amax(tensor, dim=dim, keepdim=True)
        - torch.amin(tensor, dim=dim, keepdim=True)
    )

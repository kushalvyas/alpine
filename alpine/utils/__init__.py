from .coords import *
from .functional import *
from .checkers import *
from .grid_search import *
from . import volutils


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

import torch


def get_coords2d(H, W):
    """Get 2D coordinates for a grid size H x W.

    Args:
        H (int): Height of field / image.
        W (int): Width of field / image.

    Returns:
        torch.Tensor: returns a tensor of shape (H*W, 2) with coordinates ranging from (-1, 1).
    """
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    return coords

def get_coords3d(H, W, D):
    """Get 3D coordinates for a grid size H x W x D.

    Args:
        H (int): Height of field / image.
        W (int): Width of field / image.
        D (int): Depth of field / image.

    Returns:
        torch.Tensor: returns a tensor of shape (H*W*D, 3) with coordinates ranging from (-1, 1).
    """
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    z = torch.linspace(-1, 1, D)
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
    return coords

def get_coords3d_INT(H,W,D):
    x = torch.arange(W).long()
    y = torch.arange(H).long()
    z = torch.arange(D).long()
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)])
    return coords


def get_coords_nd(*args, bounds=(-1,1), indexing='ij'):
    """Get flattened coordinates for ND grid.

    Args:
        bounds (tuple, optional): Bounds over the coordinates. Defaults to (-1,1).
        indexing (str, optional): Indexing scheme for meshgrid. Defaults to 'ij'.

    Returns:
        _type_: Returns a tensor of shape (n, len(args)) with coordinates ranging from (-1, 1).
    """
    _coords = [torch.linspace(bounds[0], bounds[1], n) for n in args]
    meshgrid = torch.meshgrid(*_coords, indexing=indexing)
    coords = torch.stack(meshgrid, dim=-1)
    coords = coords.reshape(-1, len(args))
    return coords



def get_coords_spatial(*args, bounds=(-1,1), indexing='ij'):
    """Get spatial coordinates for ND grid.

    Args:
        bounds (tuple, optional): Bounds over the coordinates. Defaults to (-1,1).
        indexing (str, optional): Indexing scheme for meshgrid. Defaults to 'ij'.

    Returns:
        torch.Tensor: Returns a tensor of spatial coordinates across each dimension with coordinates ranging from (-1, 1).
    """
    _coords = [torch.linspace(bounds[0], bounds[1], n) for n in args]
    meshgrid = torch.meshgrid(*_coords, indexing=indexing)
    coords = torch.stack(meshgrid, dim=-1)
    return coords


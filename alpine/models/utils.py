import torch

def get_coords_nd(*args, bounds=(-1,1)):
    """Get coordinates for ND grid.
    """
    _coords = [torch.linspace(-1, 1, n) for n in args]
    meshgrid = torch.meshgrid(*_coords)
    coords = torch.stack(meshgrid, dim=-1)
    coords = coords.reshape(-1, len(args))
    return coords



def get_coords_spatial(*args, bounds=(-1,1)):
    """Get coordinates for ND grid.
    """
    _coords = [torch.linspace(-1, 1, n) for n in args]
    meshgrid = torch.meshgrid(*_coords)
    coords = torch.stack(meshgrid, dim=-1)
    return coords


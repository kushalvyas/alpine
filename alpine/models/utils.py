import torch


def get_coords_nd(*args, bounds=(-1, 1)):
    """Get coordinates for ND grid."""
    _coords = [torch.linspace(-1, 1, n) for n in args]
    meshgrid = torch.meshgrid(*_coords)
    coords = torch.stack(meshgrid, dim=-1)
    coords = coords.reshape(-1, len(args))
    return coords


def get_coords_spatial(*args, bounds=(-1, 1)):
    """Get coordinates for ND grid."""
    _coords = [torch.linspace(-1, 1, n) for n in args]
    meshgrid = torch.meshgrid(*_coords)
    coords = torch.stack(meshgrid, dim=-1)
    return coords


def fourier_bases(num_phases: int,
                  num_frequencies: int,
                  input_features: int,
                  scaling_factor: float = 1.0):
    """ Create Fourier bases given phases and frequencies."""
    step_size = (num_phases - 1) / num_phases
    phi = torch.linspace(0, 2 * torch.pi * step_size, num_phases)  # (P,)
    low_freq = torch.linspace(1/num_frequencies, 1, num_frequencies)
    high_freq = torch.linspace(1, num_frequencies, num_frequencies)
    freq = torch.cat([low_freq, high_freq])  # (2F,)
    max_period = 2*torch.pi/freq[0]
    points = torch.linspace(-max_period/2, max_period/2, input_features)  #(N,)

    # Reshape for broadcasting
    freq = freq[:, None, None]        # (2F, 1, 1)
    phi = phi[None, :, None]          # (1, P, 1)
    points = points[None, None, :]    # (1, 1, N)

    # Broadcasted computation
    bases = torch.cos(freq * points + phi)  # (2F, P, N)

    # Flatten first two dims and scale
    bases = bases.reshape(-1, input_features) * scaling_factor
    return bases





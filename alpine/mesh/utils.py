"""Mesh utilities for 3D volume to mesh conversion.

Requires: open3d, mcubes
Install with: pip install alpine[mesh]
"""

import open3d as o3d
import mcubes


def march_and_save(occupancy, mcubes_thres, savename, smoothen=False):
    """Convert volumetric occupancy cube to a 3D mesh.

    Args:
        occupancy: (H, W, T) occupancy volume with values going from 0 to 1
        mcubes_thres: Threshold for marching cubes algorithm
        savename: DAE file name to save
        smoothen: If True, the mesh is binarized, smoothened, and then the
            marching cubes is applied

    Returns:
        None
    """
    if smoothen:
        occupancy = occupancy.copy()
        occupancy[occupancy < mcubes_thres] = 0.0
        occupancy[occupancy >= mcubes_thres] = 1.0

        occupancy = mcubes.smooth(occupancy, method="gaussian", sigma=1)
        mcubes_thres = 0

    vertices, faces = mcubes.marching_cubes(occupancy, mcubes_thres)

    mcubes.export_mesh(vertices, faces, savename)

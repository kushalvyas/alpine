import sys
import importlib
import time
import itertools
import os
import pdb
import copy

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
from scipy.interpolate import RegularGridInterpolator as rgi

import cv2
import torch
import open3d as o3d
import mcubes


def march_and_save(occupancy, mcubes_thres, savename, smoothen=False):
    '''
        Convert volumetric occupancy cube to a 3D mesh
        
        Inputs:
            occupancy: (H, W, T) occupancy volume with values going from 0 to 1
            mcubes_thres: Threshold for marching cubes algorithm
            savename: DAE file name to save
            smoothen: If True, the mesh is binarized, smoothened, and then the
                marching cubes is applied
        Outputs:
            None
    '''
    if smoothen:
        occupancy = occupancy.copy()
        occupancy[occupancy < mcubes_thres] = 0.0
        occupancy[occupancy >= mcubes_thres] = 1.0
        
        occupancy = mcubes.smooth(occupancy, method='gaussian', sigma=1)
        mcubes_thres = 0
        
    vertices, faces = mcubes.marching_cubes(occupancy, mcubes_thres)
    
    #vertices /= occupancy.shape[0]
        
    mcubes.export_mesh(vertices, faces, savename)
    
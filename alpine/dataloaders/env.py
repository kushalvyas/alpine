import imageio
import numpy as np


def nlcd_to_integer_index(nlcd_arr):
    nlcd_temp_contrib = {
            11: 0,   # Open Water
            12: 1,  # Perennial Ice/Snow

            21: 2,   # Developed, Open Space
            22: 3,  # Developed, Low Intensity
            23: 4,   # Developed, Medium Intensity
            24: 5,   # Developed, High Intensity

            31: 6,  # Barren Land

            41: 7,   # Deciduous Forest
            42: 8,  # Evergreen Forest
            43: 9,  # Mixed Forest

            51: 10,   # Dwarf Scrub
            52: 11,   # Shrub/Scrub

            71: 12,  # Grassland/Herbaceous
            72: 13,  # Sedge/Herbaceous
            73: 14,   # Lichens
            74: 15,  # Moss

            81: 16,   # Pasture/Hay
            82: 17,   # Cultivated Crops

            90: 18,   # Woody Wetlands
            95: 19   # Emergent Herbaceous Wetlands
        }
    nlcd_keys = list(nlcd_temp_contrib.keys())
    nlcd_temp_contrib_arr = np.zeros((100), dtype=np.float32)
    for k, v in nlcd_temp_contrib.items():
        nlcd_temp_contrib_arr[k] = v

    nlcd_updated = nlcd_temp_contrib_arr[nlcd_arr]
    return nlcd_updated


def remap_nlcd(x):
    mapping = {0:[80,107,158],
               1: [234,241,252],
               2 : [222,206,206],
               3: [210,155,133],
               4 : [227,49,34],
               5: [157, 31, 21],
               6: [178, 175, 168],
               7 : [123, 169, 110],
               8 :[51, 100, 56],
               9: [192, 203, 153],
               10: [171, 150, 75],
               11: [205, 188, 137],
               12: [237, 236, 207],
               13 : [208,209,139],
               14 : [172, 203, 98],
               15 : [141,185,159],
               16 : [220, 216, 92],
               17 : [165, 117, 55],
               18: [191, 213, 235],
               19 : [123, 163, 189],}

    xmap = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    for _x in np.unique(x):
        rows, cols = np.where(x == _x)
        xmap[rows, cols, :] = np.array(mapping[_x])
    
    return xmap


def load_nlcd(path, return_onehot = False, use_colormap = False):
    data = imageio.v3.imread(path)
    onehot = None
    cmap_data = None
    if return_onehot:
        onehot = nlcd_to_integer_index(data)
    
    if use_colormap:
        cmap_data = remap_nlcd(nlcd_to_integer_index(data) if return_onehot is None else onehot)
    
    if not return_onehot and not use_colormap:
        return data

    return data, onehot, cmap_data

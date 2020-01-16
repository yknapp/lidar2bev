"""
    Util scripts for building features, fetching ground truths, computing IoU, etc. 
"""


import numpy as np


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where(
            (PointCloud[:, 0] >= minX) &
            (PointCloud[:, 0] <= maxX) &
            (PointCloud[:, 1] >= minY) &
            (PointCloud[:, 1] <= maxY) &
            (PointCloud[:, 2] >= minZ) &
            (PointCloud[:, 2] <= maxZ)
            )
    PointCloud = PointCloud[mask]
    PointCloud[:, 2] = PointCloud[:, 2] - minZ
    return PointCloud


def makeBVFeature(PointCloud_, BoundaryCond, img_height, img_width, Discretization):
    Height = img_height + 1
    Width = img_width + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    # because of the rounding of points, there are many identical points
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    # Some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(BoundaryCond['maxZ'] - BoundaryCond['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height
    #heightMap_normalized = (heightMap - BoundaryCond['minZ'])/abs(BoundaryCond['maxZ']-BoundaryCond['minZ'])  # Normalize to [0, 1]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # 'counts': The number of times each of the unique values comes up in the original array
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height-1, Width-1, 3))
    RGB_Map[:, :, 0] = densityMap[0:img_height, 0:img_width]  # r_map
    #RGB_Map[:, :, 1] = heightMap_normalized[0:img_height, 0:img_width]  # g_map
    RGB_Map[:, :, 1] = heightMap[0:img_height, 0:img_width]  # g_map
    RGB_Map[:, :, 2] = intensityMap[0:img_height, 0:img_width]  # / 255  # b_map

    # normalize RGB_Map from [0, 1] to [0, 255]
    #RGB_Map = (255 * RGB_Map).astype(np.uint8)

    return RGB_Map


def makeBVFeature_old(PointCloud_, BoundaryCond, img_height, img_width, Discretization):  # method origined of YOLO3D instead of ComplexYOLO
    Height = img_width + 1
    Width = img_width + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    # because of the rounding of points, there are many identical points
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    # Some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]
    #heightMap_normalized = (heightMap - BoundaryCond['minZ'])/abs(BoundaryCond['maxZ']-BoundaryCond['minZ'])  # Normalize to [0, 1]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # 'counts': The number of times each of the unique values comes up in the original array
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    RGB_Map = np.zeros((Height, Width, 3))

    # RGB channels respectively
    RGB_Map[:, :, 0] = densityMap
    #RGB_Map[:, :, 1] = heightMap_normalized
    RGB_Map[:, :, 1] = heightMap
    RGB_Map[:, :, 2] = intensityMap / 255

    # normalize RGB_Map from [0, 1] to [0, 255]
    #RGB_Map = (255 * RGB_Map).astype(np.uint8)

    save = RGB_Map[0:img_height, 0:img_width, :]
    return save

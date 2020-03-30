import os
import sys
import cv2
import utils
import imageio
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R

# Paths
KITTI_LIDAR_DIR = "/home/user/work/master_thesis/datasets/kitti/kitti/object/training/velodyne"
KITTI_OUTPUT_DIR = "/home/user/work/master_thesis/datasets/bev_images/kitti/training"
LYFT_LIDAR_DIR = "/home/user/work/master_thesis/datasets/lyft_level_5/v1.02-train/lidar"
LYFT_OUTPUT_DIR = "/home/user/work/master_thesis/datasets/bev_images/lyft"
LYFT_KITTI_LIDAR_DIR = "/home/user/work/master_thesis/datasets/lyft_kitti/object/training/velodyne"
LYFT_KITTI_OUTPUT_DIR = "/home/user/work/master_thesis/datasets/bev_images/lyft_kitti"
NUSCENES_LIDAR_DIR = "/home/user/work/master_thesis/datasets/nuscenes/samples/LIDAR_TOP"
NUSCENES_OUTPUT_DIR = "/home/user/work/master_thesis/datasets/bev_images/nuscenes/samples"
AUDI_LIDAR_DIR = "/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/lidar/cam_front_center"
AUDI_OUTPUT_DIR = "/home/user/work/master_thesis/datasets/bev_images/audi/"

# lidar boundarys for Bird's Eye View
BOUNDARY = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}


def load_lidar_file_kitti(file_path):
    n_vec = 4
    dtype = np.float32
    lidar_pc_raw = np.fromfile(file_path, dtype)
    return lidar_pc_raw.reshape((-1, n_vec))


def load_lidar_file_lyft(file_path):
    n_vec = 5
    dtype = np.float32
    lidar_pc_raw = np.fromfile(file_path, dtype)
    lidar_pc = lidar_pc_raw.reshape((-1, n_vec))
    intensity_normalized = lidar_pc[:, 3] / 100
    lidar_pc[:, 3] = intensity_normalized
    print(lidar_pc.shape)
    return lidar_pc


def load_lidar_file_nuscenes(file_path):
    n_vec = 5
    dtype = np.float32
    lidar_pc_raw = np.fromfile(file_path, dtype)
    lidar_pc = lidar_pc_raw.reshape((-1, n_vec))
    #intensity_normalized = lidar_pc[:, 3] / 255
    #lidar_pc[:, 3] = intensity_normalized
    pointcloud = LidarPointCloud.from_file(file_path)
    r = R.from_quat([0, 0, np.sin((np.pi / 4)*3), np.cos((np.pi / 4)*3)])
    pointcloud.rotate(r.as_matrix())
    lidar_pc[:, 0] = pointcloud.points[0, :]
    lidar_pc[:, 1] = pointcloud.points[1, :]
    lidar_pc[:, 2] = pointcloud.points[2, :]
    lidar_pc[:, 3] = pointcloud.points[3, :]

    return lidar_pc


def load_lidar_file_audi(file_path):
    lidar_pc_raw = np.load(file_path)
    print(list(lidar_pc_raw.keys()))
    lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
    lidar_pc[:, :3] = lidar_pc_raw['points']
    lidar_pc[:, 3] = lidar_pc_raw['reflectance']
    return lidar_pc


def check_identity_of_img(filename, lidar_bev):
    # compare stored matrix with actual matrix
    print("Check stored file for identity")
    lidar_bev_file = imageio.imread(filename)

    if np.array_equal(lidar_bev, lidar_bev_file) is False:
        print("Error: BEV image '%s' is corrupted" % filename)
        sys.exit()
    else:
        print("Identity test successful!")


def check_identity_of_img_float(filename, lidar_bev):
    # compare stored matrix with actual matrix
    # this method is for comparing matrix with file, which was automatically rounded from float64 to uint8 by imageio
    print("Check stored file for identity")
    lidar_bev_file = imageio.imread(filename)

    lidar_bev_round = np.rint((lidar_bev * 255)).astype(np.uint8)

    np.set_printoptions(threshold=sys.maxsize)
    indices = np.argwhere(lidar_bev_file != lidar_bev_round)
    for index in indices:
        print(lidar_bev_file[index[0], index[1], index[2]])
        print(lidar_bev_round[index[0], index[1], index[2]])
        print(lidar_bev[index[0], index[1], index[2]] * 255)

    if np.array_equal(lidar_bev_round, lidar_bev_file) is False:
        print("Error: BEV image '%s' is corrupted" % filename)
        sys.exit()
    else:
        print("Identity test successful!")


def main(chosen_dataset, img_height, img_width):
    load_lidar_file = None
    lidar_path = None
    output_dir = None

    if chosen_dataset == "kitti":
        load_lidar_file = load_lidar_file_kitti
        lidar_path = KITTI_LIDAR_DIR
        output_dir = KITTI_OUTPUT_DIR
    elif chosen_dataset == "lyft":
        load_lidar_file = load_lidar_file_lyft
        lidar_path = LYFT_LIDAR_DIR
        output_dir = LYFT_OUTPUT_DIR
    elif chosen_dataset == "nuscenes":
        load_lidar_file = load_lidar_file_nuscenes
        lidar_path = NUSCENES_LIDAR_DIR
        output_dir = NUSCENES_OUTPUT_DIR
    elif chosen_dataset == "lyftkitti":
        load_lidar_file = load_lidar_file_kitti
        lidar_path = LYFT_KITTI_LIDAR_DIR
        output_dir = LYFT_KITTI_OUTPUT_DIR
    elif chosen_dataset == "audi":
        load_lidar_file = load_lidar_file_audi
        lidar_path = AUDI_LIDAR_DIR
        output_dir = AUDI_OUTPUT_DIR
    else:
        print("Error: Unknown dataset '%s'" % chosen_dataset)
        sys.exit()

    # setup
    lidar_pc_filenames = os.listdir(lidar_path)
    num_lidar_pc = len(lidar_pc_filenames)

    for idx in range(num_lidar_pc):
        print("Processing file '%s' (%s|%s)" % (lidar_pc_filenames[idx], idx+1, num_lidar_pc))
        file_path = os.path.join(lidar_path, lidar_pc_filenames[idx])
        lidar_pc = load_lidar_file(file_path=file_path)

        # filter point cloud points inside fov
        lidar_pc_filtered = utils.removePoints(lidar_pc, BOUNDARY)

        # create Bird's Eye View
        discretization = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / img_height
        lidar_bev = utils.makeBVFeature(lidar_pc_filtered, BOUNDARY, img_height, img_width, discretization)
        #print("min: ", np.amin(lidar_bev[:, :, 0]))
        #print("max: ", np.amax(lidar_bev[:, :, 0]))
        #print("min: ", np.amin(lidar_bev[:, :, 1]))
        #print("max: ", np.amax(lidar_bev[:, :, 1]))
        #print("min: ", np.amin(lidar_bev[:, :, 2]))
        #print("max: ", np.amax(lidar_bev[:, :, 2]))
        ########################
        # set intensity to zero since lyft doesn't provide intensity values
        lidar_bev[:, :, 2] = 0.0
        ########################

        ########################
        # binary bev
        #lidar_bev_mask = lidar_bev[:, :, 1] > 0.0
        #lidar_bev = lidar_bev_mask.astype(int)
        ########################

        # save as PNG image with the same image name like lidar file
        output_filename = lidar_pc_filenames[idx].replace(".bin", ".png").replace(".npz", ".png")
        output_filepath = os.path.join(output_dir, output_filename)
        imageio.imwrite(output_filepath, lidar_bev)

        # check whether stored file is the same as the matrix to preserve identical values when file is the source
        #check_identity_of_img(output_filepath, lidar_bev)
        #check_identity_of_img_float(output_filepath, lidar_bev)

    print("\nSuccessfully transformed all '%s' lidar pointclouds" % chosen_dataset)


if __name__ == "__main__":
    img_height = 480
    img_width = 480
    dataset = "kitti"  # 'kitti', 'lyft', 'nuscenes', 'lyftkitti' or 'audi'
    main(dataset, img_height, img_width)

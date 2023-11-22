"""
Yi Du's point cloud Toolbox
"""

import os
import numpy as np
try:
    import pptk
except:
    print("pptk is not installed!")
import open3d as o3d
import cv2



def show_point_cloud(pc_data, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=False):
    """
    Show point cloud. If the point cloud is colored (dimension is 6), the color will be shown.
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        point_size: float, size of the point
        bg_color: list, background color
        show_grid: bool, whether to show the grid
    Return:
        window: pptk.viewer object for further operations like view manipulation and image capture
    """
    if pc_data.shape[1] == 3:
        window = pptk.viewer(pc_data)
    elif pc_data.shape[1] == 6:
        if np.max(pc_data[:, 3:]) > 1 + 0.9:
            pc_data[:, 3:] = pc_data[:, 3:]/255 # convert rgb from 0-255 to 0-1
        window = pptk.viewer(pc_data[:, :3], pc_data[:, 3:])

     # set point size
    window.set(point_size=point_size)
    # set background color
    window.set(bg_color=bg_color)
    # remove grid
    window.set(show_grid=show_grid)

    return window


def windows_close_ctrl(windows):
    """
    Close all the pptk viewer windows
    Parameter:
        windows: list, pptk.viewer object
    """
    cv2.namedWindow('Key Listener')
    while True:
        key = cv2.waitKey(0)  # Listen for a key event
        if key == 27:  # ASCII value of 'esc' is 27
            cv2.destroyAllWindows()   
            for window in windows:
                window.close()  # Close each pptk viewer window
            break


def noise_Gaussian(points, mean):
    """
    Add Gaussian noise to the point cloud
    Parameter:
        points: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        mean: float, mean of the Gaussian distribution
    Return:
        pc_out: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
    """
    noise = np.random.normal(0, mean, points.shape)
    pc_out = points + noise
    return pc_out

def read_SuperMap_hdf5(path):
    """
    Read hdf5 file in SuperMap format
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    import h5py
    f = h5py.File(path, 'r')
    data_train = f['02801938']['train']
    data_test = f['02801938']['test']
    data = {'train':np.array(data_train), 'test':np.array(data_test)}
    
    return data


def read_off(path):
    """
    Read off file format point cloud
    Parameter:
        file: path of off file
    Return:
        verts: numpy array of pointcloud [[x, y, z], ...]
        faces: numpy array of faces [[v1, v2, v3], ...]
    """
    with open(path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, __ = map(int, file.readline().strip().split(' '))
        verts = np.array([[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)])
        faces = [[int(s) for s in file.readline().strip().split(' ')[1:]] for _ in range(n_faces)]
    

    return verts, faces


def read_xyz(path):
    """
    Read xyz file 
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    data = np.genfromtxt(path, delimiter=' ', skip_header=0)
    

    return np.array(data)


def read_pcd(path):
    """
    Read pcd file generated by ./func.py/save_pc
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    xyzrgb = []
    with open(path, 'r') as f:
        content = f.readlines()
        for i in content[10:]:
            i_content = i.split(" ")
            x, y, z = float(i_content[0]), float(i_content[1]), float(i_content[2])
            r, g, b = float(i_content[3]), float(i_content[4]), float(i_content[5][:-1])

            xyzrgb.append([x,y,z,r,g,b])

    return np.array(xyzrgb)


def read_bin_pc(path, pc_show=False):
    """
    Read bin file format point cloud
    Parameter:
        path: path of pcd file
        pc_show: bool, whether to show the point cloud.
    Return:
        xyzr: numpy array of pointcloud [[x, y, z, r], ...]
    """
    point_cloud_data = np.fromfile(path, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r
    point_cloud_data = point_cloud_data[:, :3]
    if pc_show:
        pptk.viewer(point_cloud_data)

    return point_cloud_data


def save_pc_file(data, path):
    """
    Save point cloud file
    Parameter:
        data: numpy array of pointcloud [[x, y, z], ...]
        path: path of pcd file
    """
    np.savetxt(path, data, fmt='%.6f')
    # np.save(path, data)
    # data.tofile(path)





def uniform_downsample_pc(pc_data, every_k_points=10, pc_show=False):
    """
    Downsample the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_down: numpy array of downsampled pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
    """
    if pc_data.shape[1] == 6:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_data[:, 3:])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        # downpcd = pcd.voxel_down_sample(voxel_size=0.5)
        downpcd = rm.uniform_down_sample(every_k_points=every_k_points)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_rgb_down = downpcd.colors
        pcd_down = np.hstack((pcd_xyz_down, pcd_rgb_down))
        if pc_show:
            show_point_cloud(pcd_down)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        # downpcd = pcd.voxel_down_sample(voxel_size=0.5)
        downpcd = rm.uniform_down_sample(every_k_points=every_k_points)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_down = pcd_xyz_down
        if pc_show:
            pptk.viewer(pcd_down)
    else:
        assert False, "Error: point cloud data shape is not correct!"


    return pcd_down


def divide_pc_xyz_rgb(pc_data, pc_show=False):
    """
    1. Divide the point cloud into xyz and rgb.  2. Convert rgb from 0-255 to 0-1.
    Parameter:
        The input point cloud should be in the format of [[x, y, z, r, g, b], ...]
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_xyz: numpy array of pointcloud [[x, y, z], ...]
        pcd_rgb: numpy array of pointcloud [[r, g, b], ...]
        pcd_data: numpy array of pointcloud [[x, y, z, r, g, b], ...]
    """
    assert pc_data.shape[1] == 6, "Error: point cloud data shape is not correct!"

    # devide the pointcloud into xyz and rgb
    if np.max(pc_data[:, 3:]) > 1 + 0.9:
        pc_data[:, 3:] = pc_data[:, 3:]/255 # convert rgb from 0-255 to 0-1
    pcd_xyz = pc_data[:, :3]
    pcd_rgb = pc_data[:, 3:]
    pcd_data = np.hstack((pcd_xyz, pcd_rgb))

    if pc_show:
        show_point_cloud(pcd_data)

    return pcd_xyz, pcd_rgb, pcd_data


def rm_statistical_outlier(pc_data, nb_neighbors=680,std_ratio=3.8 ,pc_show=False):
    """
    Remove statistical outliers from the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
        nb_neighbors: int, The number of neighbors to use for radius/outlier removal.
        std_ratio: float, The standard deviation multiplier for the distance of points.
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_rm: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
    """
    if pc_data.shape[1] == 6:
        pcd_xyz = pc_data[:, :3]
        pcd_rgb = pc_data[:, 3:]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
        pcd_xyz_rm = rm.points
        pcd_rgb_rm = rm.colors
        pcd_rm = np.hstack((pcd_xyz_rm, pcd_rgb_rm))
        if pc_show:
            show_point_cloud(pcd_rm)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data)
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
        pcd_xyz_rm = rm.points
        pcd_rm = np.array(pcd_xyz_rm)
        if pc_show:
            pptk.viewer(pcd_rm)
    else:
        assert False, "Error: point cloud data shape is not correct!"

    return pcd_rm


if __name__ == '__main__':
    num_pts = 800
    mean = 0.88
    noise = np.random.normal(0, mean, (num_pts, 6))
    win8 = show_point_cloud(noise, point_size=0.01, bg_color=[1, 1, 1, 1], show_grid=False)
    win8.capture('./img/screenshot.png')
    # # import point cloud from the .pcd file
    # # path_to_point_cloud = './result/pcd/%06d.pcd' % int(2283)
    # # path_to_point_cloud = './result/pcd/%06d.pcd' % int(31)
    # # path_to_point_cloud = './result/pcd/single/%06d.pcd' % int(2283)
    # path_to_point_cloud = '/home/jared/SAIR_Lab/Super-Map/Grad-PU/data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/pcd_down.xyz'
    # # point_cloud_data = read_pcd(path_to_point_cloud)  # little-endian float32
    # point_cloud_data = read_xyz(path_to_point_cloud)  # little-endian float32
    # window = show_point_cloud(point_cloud_data, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=True)

    # path_to_point_cloud2 = '/home/jared/SAIR_Lab/Super-Map/Grad-PU/pretrained_model/pugan/test/4X/pcd_down.xyz'
    # point_cloud_data2 = read_xyz(path_to_point_cloud2)  # little-endian float32
    # window2 = show_point_cloud(point_cloud_data2, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=True)

    # windows = [window, window2]
    # windows_close_ctrl(windows)

    # # downsample the point cloud to get the input
    # pcd_down = uniform_downsample_pc(point_cloud_data, pc_show=False)


    # # 1. Divide the point cloud into xyz and rgb.  2. Convert rgb from 0-255 to 0-1.
    # pcd_xyz, pcd_rgb, pcd_data = divide_pc_xyz_rgb(point_cloud_data, pc_show=True)


    # # add gaussian noise
    # pcd_xyz_out = noise_Gaussian(pcd_xyz, 0.3)
    # pcd_rgb_out = noise_Gaussian(pcd_rgb, 0.1)
    # pcd_out = np.hstack((pcd_xyz_out, pcd_rgb_out))

    # # Remove statistical outliers from the point cloud
    # pcd_rm = rm_statistical_outlier(pcd_out, nb_neighbors=680,std_ratio=3.8 ,pc_show=False)
     


# ---------------------------------- For pcd file (global map) ----------------------------------------
    # PATH = "./result/pcd/000031.pcd"
    # PATH = "/home/jared/Large_datasets/data/KITTI/KITTI_Tools/kitti-map-python/map-08_0.1_0-18.pcd"

    # # Simply view
    # pcd = o3d.io.read_point_cloud(PATH)
    # o3d.visualization.draw_geometries([pcd])

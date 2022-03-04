'''
Question 5. Triangulation
In this question we move to 3D.
You are given keypoint matching between two images, together with the camera intrinsic and extrinsic matrix.
Your task is to perform triangulation to restore the 3D coordinates of the key points.
In your PDF, please visualize the 3d points and camera poses in 3D from three different viewing perspectives.
'''
import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
all_good_matches = np.load('assets/all_good_matches.npy')

K1 = np.load('assets/fountain/Ks/0000.npy')
K2 = np.load('assets/fountain/Ks/0005.npy')

R1 = np.load('assets/fountain/Rs/0000.npy')
R2 = np.load('assets/fountain/Rs/0005.npy')

t1 = np.load('assets/fountain/ts/0000.npy')
t2 = np.load('assets/fountain/ts/0005.npy')

def triangulate(K1, K2, R1, R2, t1, t2, all_good_matches):
    """
    Arguments:
        K1: intrinsic matrix for image 1, dim: (3, 3)
        K2: intrinsic matrix for image 2, dim: (3, 3)
        R1: rotation matrix for image 1, dim: (3, 3)
        R2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        all_good_matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    points_3d = None
    # --------------------------- Begin your code here ---------------------------------------------
    # credit: http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf

    # cosntruct camera projection matrix
    T1 = np.eye(4); T1[:3, :3] = R1; T1[:3, -1] = t1.flatten()
    T2 = np.eye(4); T2[:3, :3] = R2; T2[:3, -1] = t2.flatten()
    eye_3by4 = np.zeros((3,4)); eye_3by4[:, :3] = np.eye(3)
    M1 = K1 @ eye_3by4 @ T1
    M2 = K2 @ eye_3by4 @ T2

    # container for the estimated 3d points reconstructed by triangulation
    points_3d = np.empty((len(all_good_matches), 3))
    
    for i, match in enumerate(all_good_matches):
        # conduct triangulation of the given 2d matches
        x1, y1, x2, y2 = match
        A = np.vstack((y1 * M1[2,:] -      M1[1,:],
                            M1[0,:] - x1 * M1[2,:],
                       y2 * M2[2,:] -      M2[1,:],
                            M2[0,:] - x2 * M2[2,:]))
        U, S, Vh = np.linalg.svd(A)
        pt_3d = np.reshape(Vh[-1, :], (4,))
        pt_3d /= pt_3d[-1]

        # save to the array
        points_3d[i] = pt_3d[:-1]

    # --------------------------- End your code here   ---------------------------------------------
    return points_3d


points_3d = triangulate(K1, K2, R1, R2, t1, t2, all_good_matches)
if points_3d is not None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Visualize both point and camera
    # Check this link for Open3D visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Check this function for adding a virtual camera in the visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Open3D is not the only option. You could use matplotlib, vtk or other visualization tools as well.
    
    # --------------------------- Begin your code here ---------------------------------------------

    # obtain camera center in 3D world frame
    cam1 = - R1.T @ t1
    cam2 = - R2.T @ t2

    cam1 = cam1.flatten()
    cam2 = cam2.flatten()

    camera_centers = np.vstack((cam1, cam2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', label='Points')
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c='g', s=50, marker='^', label='Camera Centers')
    ax.legend(loc='best')

    plt.show()


    '''
    W = 640
    H = 320
    T1 = np.eye(4); T1[:3,:3] = R1; T1[:3,-1] = t1.flatten()
    T2 = np.eye(4); T2[:3,:3] = R2; T2[:3,-1] = t2.flatten()

    cam1 = o3d.geometry.LineSet.create_camera_visualization(W, H, K1, T1)
    cam2 = o3d.geometry.LineSet.create_camera_visualization(W, H, K2, T2)
    
    
    o3d.visualization.draw_geometries([cam1, cam2])
    o3d.visualization.draw_geometries([pcd])
    '''
    
    # --------------------------- End your code here   ---------------------------------------------
from os import fwalk
from re import I
import time
import cv2
import numpy as np
import open3d as o3d
import time
from pyparsing import col
from sklearn.neighbors import KDTree

# Question 4: deal with point_to_plane = True
def fit_rigid(src, tgt, point_to_plane=False, src_normal=None):
  # Question 2: Rigid Transform Fitting
  # Implement this function
  # -------------------------

  assert len(src) == len(tgt)

  T = np.identity(4)
  
  # point-to-point algorithm
  # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
  if not point_to_plane:
    # assume that weights are equal
    # calculate centroid
    src_avg = np.mean(src, axis=0)
    tgt_avg = np.mean(tgt, axis=0)
    
    # construct 3x3 covariance matrix
    C = (src - src_avg).T @ (tgt - tgt_avg)
    
    # do svd 
    U, S, Vh = np.linalg.svd(C)
    
    # reconstruct rotation matrix
    M = np.eye(3)
    M[2,2] = np.linalg.det(Vh.T @ U.T)
    
    R = Vh.T @ M @ U.T

    # reconstruct translation matrix
    t = tgt_avg - R @ src_avg
    
    # write on SE(3) matrix
    T[:3,:3] = R
    T[:3,-1] = t

  # point-to-plane algorithm
  # https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
  else:
    A = np.empty((6*len(src), 6))
    b = np.empty(6*len(src))

    # construct overdetermined linear system
    for i, (p,q,n) in enumerate(zip(src,tgt,src_normal)):
      c = np.cross(p,n)
      v = np.concatenate([c,n])
      A[6*i:6*(i+1)] = np.outer(v,v)
      b[6*i:6*(i+1)] = - v * np.inner(p-q, n)
    
    # get solution to the overdetermined system
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # recover translation
    t = x[3:]

    # recover rotation
    a, b, c = x[:3]
    R = np.array([[1,-c,b],
                  [c,1,-a],
                  [-b,a,1]])
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    R = Rotation.as_matrix(rot)

    # write on SE(3) matrix
    T[:3,:3] = R
    T[:3,-1] = t
  
  # -------------------------
  return T

# Question 4: deal with point_to_plane = True
def icp(source, target, max_iter=20, point_to_plane=False, inlier_thres=0.1):
  src = np.asarray(source.points)
  tgt = np.asarray(target.points)

  src_normal = np.asarray(source.normals)
  tgt_normal = np.asarray(target.normals)

  # Question 3: ICP
  # Hint 1: using KDTree for fast nearest neighbour
  # Hint 3: you should be calling fit_rigid inside the loop
  # You implementation between the lines
  # ---------------------------------------------------
  T = np.eye(4)     # initial guess of transformation matrix 
  transforms = []   # total transformation after every iteration
  delta_Ts = []     # incremental transformation for every iteration

  inlier_ratio = 0
  print("iter %d: inlier ratio: %.2f" % (0, inlier_ratio))

  for i in range(max_iter):
    # construct kd tree
    tree = KDTree(src)
    # find corresponding pairs (the nearest)
    dist, idx = tree.query(tgt, k=1)
    dist = dist.flatten(); idx = idx.flatten()
    # count inlier ratio
    inlier_ratio = len(np.where(dist < inlier_thres)[0]) / len(tgt)

    # find incremental transformation matrix
    T_delta = fit_rigid(src[idx], tgt, point_to_plane=point_to_plane, \
                        src_normal=src_normal[idx])
    # update total transformation matrix
    T = T_delta @ T
    # update src points
    src = (T_delta @ np.c_[src, np.ones(len(src))].T)[:3, :].T

    # ---------------------------------------------------

    if inlier_ratio > 0.999:
      break
    
    print("iter %d: inlier ratio: %.2f" % (i+1, inlier_ratio))
    # relative update from each iteration
    delta_Ts.append(T_delta.copy())
    # pose estimation after each iteration
    transforms.append(T.copy())
  return transforms, delta_Ts

def rgbd2pts(color_im, depth_im, K):
  # Question 1: unproject rgbd to color point cloud, provide visualiation in your document
  # Your implementation between the lines
  # ---------------------------

  # read image size 
  h, w = depth_im.shape 
  # read camera intrinsics
  f = np.linalg.norm(np.diag(K)[:-1])
  px, py = K[0:2,2]

  # convert depth data into 3d coordinate
  U, V = np.meshgrid(np.arange(w)-int(px), np.arange(h)-int(py))
  X = U * depth_im / f
  Y = V * depth_im / f
  
  # create 3d pointcloud and color data
  xyz = np.stack([X,Y,depth_im], axis=2)
  xyz = xyz.reshape((-1,3))
  color = color_im.reshape((-1,3))
  
  # ---------------------------
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd

# TODO (Shenlong): please check that I set this question up correctly, it is called on line 136
def pose_error(estimated_pose, gt_pose):
  # Question 5: Translation and Rotation Error 
  # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
  # Your implementation between the lines
  # ---------------------------
  error = 0

  error_trn = np.linalg.norm(estimated_pose[:3, -1] - gt_pose[:3, -1])
  error_rot = np.arccos((np.trace(estimated_pose[:3,:3] @ gt_pose[:3,:3].T) - 1) / 2)
  
  print(f" -- translation error: {error_trn:.4f} [m]")
  print(f" -- rotation error   : {np.rad2deg(error_rot):.4f} [deg]")
  
  error = (error_trn, error_rot)
  # ---------------------------
  return error

def read_data(ind = 0):
  K = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  depth_im = cv2.imread("data/frame-%06d.depth.png"%(ind),-1).astype(float)
  depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
  depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
  T = np.loadtxt("data/frame-%06d.pose.txt"%(ind))  # 4x4 rigid transformation matrix
  color_im = cv2.imread("data/frame-%06d.color.jpg"%(ind),-1)
  color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)  / 255.0
  return color_im, depth_im, K, T

if __name__ == "__main__":

  # pairwise ICP

  # read color, image data and the ground-truth, converting to point cloud
  color_im, depth_im, K, T_tgt = read_data(0)
  target = rgbd2pts(color_im, depth_im, K)
  color_im, depth_im, K, T_src = read_data(40)
  source = rgbd2pts(color_im, depth_im, K)

  # downsampling and normal estimatoin
  source = source.voxel_down_sample(voxel_size=0.02)
  target = target.voxel_down_sample(voxel_size=0.02)
  source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

  # conduct ICP (your code)
  # TODO (JONGWON): CHANGE PARAMETER point_to_plane
  final_Ts, delta_Ts = icp(source, target, point_to_plane=True)

  # visualization
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  ctr = vis.get_view_control()
  ctr.set_front([ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ])
  ctr.set_lookat([ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ])
  ctr.set_up([ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])
  ctr.set_zoom(0.42199999999999971)
  vis.add_geometry(source)
  vis.add_geometry(target)

  save_image = False

  # update source images
  for i in range(len(delta_Ts)):
      source.transform(delta_Ts[i])
      vis.update_geometry(source)
      vis.poll_events()
      vis.update_renderer()
      time.sleep(0.2)
      if save_image:
          vis.capture_screen_image("temp_%04d.jpg" % i)

  # visualize camera
  h, w, c = color_im.shape
  tgt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.2)
  src_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(T_src) @ T_tgt, scale = 0.2)
  pred_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(final_Ts[-1]), scale = 0.2)

  gt_pose = np.linalg.inv(T_src) @ T_tgt
  pred_pose = np.linalg.inv(final_Ts[-1])
  p_error = pose_error(pred_pose, gt_pose)
  np.set_printoptions(precision=4)
  print("Ground truth pose:\n", gt_pose)
  print("Estimated pose:\n", pred_pose)
  print("Rotation/Translation Error", p_error)

  # TODO (JONGWON): SANITY CHECK AGAIN
  # p_raw = pose_error(np.eye(4), gt_pose)
  # print("Rotation/Translation Error (before)", p_raw)

  tgt_cam.paint_uniform_color((1, 0, 0))
  src_cam.paint_uniform_color((0, 1, 0))
  pred_cam.paint_uniform_color((0, 0.5, 0.5))
  vis.add_geometry(src_cam)
  vis.add_geometry(tgt_cam)
  vis.add_geometry(pred_cam)

  vis.run()
  vis.destroy_window()

  # Provide visualization of alignment with camera poses in write-up.
  # Print pred pose vs gt pose in write-up.
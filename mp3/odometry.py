import enum
from re import I
from telnetlib import PRAGMA_HEARTBEAT
from turtle import st
import numpy as np
import open3d as o3d
from sympy import im
from urllib3 import Retry
from icp import read_data, icp, rgbd2pts, pose_error


def odometry_error(T_W2Cs_pred, T_W2Cs_gt):
  rot_err = 0
  trans_err = 0
  return rot_err, trans_err

color_im, depth_im, K, T_init = read_data(0)
pcd_init = rgbd2pts(color_im, depth_im, K)
pcd_down = pcd_init.voxel_down_sample(voxel_size=0.02)
pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
h, w, c = color_im.shape

step = 10 # try different step
end = 100
# odometry pose: blue
cam_init = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.1)
cam_init.paint_uniform_color((0, 0, 1))
# gt pose: red
gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.1)
gt_cam.paint_uniform_color((1, 0, 0))

pred_poses = {}
gt_poses = {}
pcds = {}
pred_poses[0] = np.eye(4) # assuming first frame is global coordinate center
gt_poses[0] = np.eye(4) # assuming first frame is global coordinate center
pcds[0] = pcd_down # save point cloud for init frame
vis_list = [pcd_init, cam_init, gt_cam]

T_W2C = np.eye(4)  # world to current camera frame

for frame in range(step, end, step):
  color_im, depth_im, K, T_tgt = read_data(frame - step)
  target = rgbd2pts(color_im, depth_im, K)
  color_im, depth_im, K, T_src = read_data(frame)
  source = rgbd2pts(color_im, depth_im, K)

  # some pre-processing, including computing normals and downsampling
  source_down = source.voxel_down_sample(voxel_size=0.02)
  target_down = target.voxel_down_sample(voxel_size=0.02)
  source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

  # Question 6 --- Could you update the camera poses using ICP?
  # Hint1: you could call your ICP to estimate relative pose betwen frame t and t-step
  # Hint2: based on the estimated transform between frame t and t-step, are you able to compute world to camera transform for t?
  
  # Your code
  # ------------------------

  # incremental pose difference
  final_Ts, _ = icp(source_down, target_down, inlier_thres=0.01)
  T_inc =  final_Ts[-1]
  # update latest camera pose wrt world frame
  T_W2C = np.linalg.inv(T_inc) @ T_W2C
  
  pred_poses[frame] = T_W2C
  # ------------------------
  
  # get ground-truth pose
  T_W2C_gt =  np.linalg.inv(T_src) @ T_init # world to init to current
  gt_poses[frame] = T_W2C_gt # ground truth
  pcds[frame] = source_down

  # odometry pose: blue
  current_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, T_W2C, scale = 0.1)
  current_cam.paint_uniform_color((0, 0, 1))
  # gt pose: red
  gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, T_W2C_gt, scale = 0.1)
  gt_cam.paint_uniform_color((1, 0, 0))
  source.transform(np.linalg.inv(T_W2C))
  vis_list.append(source)
  vis_list.append(current_cam)
  vis_list.append(gt_cam)
  print("Frame %d is done" % frame)

o3d.visualization.draw_geometries(vis_list,
                                  zoom=0.422,
                                  front = [ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ],
                                  lookat = [ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ],
                                  up = [ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])

# Question 7: Relative Trajectory Error
# implement the relative rotation error and relative translation error
# reference, Eq.[2] and Eq.[3] in:
# https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf
# please write down your derivations

# Your code
# ------------------------

## function calculating relative trajectory error framewise
def get_rte(pred_poses, gt_poses):
  from scipy.spatial.transform import Rotation
  assert len(pred_poses) == len(gt_poses)

  rte_rot = 0
  rte_trn = 0

  for frame in range(step, end, step):
    delT_pred = pred_poses[frame] @ np.linalg.inv(pred_poses[frame-step])
    delT_gt = gt_poses[frame] @ np.linalg.inv(gt_poses[frame-step])
    delT_pred_gt = delT_pred @ np.linalg.inv(delT_gt)
    
    t = delT_pred_gt[:3,-1]
    R = delT_pred_gt[:3,:3]
    rot = Rotation.from_matrix(R)
    
    rte_rot += rot.magnitude()
    rte_trn += np.linalg.norm(t)

  rte_rot /= (len(gt_poses)-1)
  rte_trn /= (len(gt_poses)-1)

  print(f" -- RTE (translation): {rte_trn:.2f} [m]  ")
  print(f" -- RTE (rotation)   : {np.rad2deg(rte_rot):.2f} [deg]")

  return rte_trn, rte_rot

## report RTE of the aforementioned odometry result
get_rte(pred_poses, gt_poses)

# ------------------------

# Question 8: Pose graph optimization
# Now we have an simple odometry solution, where each frame's pose in world coordinate is decided by its previous frame's pose and a relative transformation provided by ICP.
# Given the pose T_0 and T_40, please validate whether the relative transformation
# caculated from the two odometry poses will perfectly agree with the transformation we estimated from ICP?
# If not, explain why. If yes, explain why (in your pdf).

color_im, depth_im, K, T_tgt = read_data(0)
target = rgbd2pts(color_im, depth_im, K)
color_im, depth_im, K, T_src = read_data(40)
source = rgbd2pts(color_im, depth_im, K)

# some pre-processing, including computing normals and downsampling
source_down = source.voxel_down_sample(voxel_size=0.02)
target_down = target.voxel_down_sample(voxel_size=0.02)
source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
final_Ts, delta_Ts = icp(source_down, target_down, max_iter=50)

T_0 = pred_poses[0]
T_40 = pred_poses[40]
print("Relative transfrom from ICP:\n", np.linalg.inv(final_Ts[-1]))
print("Relative transfrom from odometry:\n", T_40)

p_error = pose_error(np.linalg.inv(final_Ts[-1]), T_40)
print("Rotation/Translation Error: \n", p_error)

# Question 8: to ensure the consistency, we could build a pose graph to further improve the performance.
# Each node is the a camera pose
# Each edge will describe the relative transformation between the node, provided by ICP.
# The general idea of pose graph optimization is to jointly optimize the pose such that maximum consensus has been reached:
# argmin_{T_i} \sum_i^N \sum_{j>i}^{i+K} (T_ij - inv(T_j) @ T_i)^2
# where T_ij is transformation from i to j, T_i is transformation from i to world (global coordinate)

# In this question, you are going to leverage pose graph optimization to build dense pose graph.
# A node will be connected if their difference in frame number is smaller or equal to 30
# Open3D provided us some helpful function
# 0. Building pose graph pose_graph = o3d.pipelines.registration.PoseGraph()
# 1. Add one graph node: o3d.pipelines.registration.PoseGraphNode(init)
# 2. Add one graph edge: o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, tranform_from_icp)
# 3. Optimize pose graph: o3d.pipelines.registration.global_optimization()

# Hints:
# Be careful about the transformation, before we are outputing extrinsic matrix which is world to camera.
# Now each node records a transformation that goes from camera to world.

# Your code
# ------------------------
pose_graph = o3d.pipelines.registration.PoseGraph()
frame_list = list(pred_poses.keys())

voxel_size=0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

def pairwise_registration(source, target):
  print("Apply point-to-plane ICP")
  icp_coarse = o3d.pipelines.registration.registration_icp(
      source, target, max_correspondence_distance_coarse, np.identity(4),
      o3d.pipelines.registration.TransformationEstimationPointToPlane())
  icp_fine = o3d.pipelines.registration.registration_icp(
      source, target, max_correspondence_distance_fine,
      icp_coarse.transformation,
      o3d.pipelines.registration.TransformationEstimationPointToPlane())
  transformation_icp = icp_fine.transformation
  information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
      source, target, max_correspondence_distance_fine,
      icp_fine.transformation)
  return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
  step = 10

  pose_graph = o3d.pipelines.registration.PoseGraph()
  odometry = np.identity(4)
  pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
  n_pcds = len(pcds)
  for source_id in range(0, step*(n_pcds-1), step):
    for target_id in range(source_id+step, step*n_pcds, step):
      transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id])
      print("Build o3d.pipelines.registration.PoseGraph")
      if target_id == source_id + step:  # odometry case
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                      target_id,
                                                      transformation_icp,
                                                      information_icp,
                                                      uncertain=False))
      else:  # loop closure case
        pass
        # pose_graph.edges.append(
        #     o3d.pipelines.registration.PoseGraphEdge(source_id,
        #                                               target_id,
        #                                               transformation_icp,
        #                                               information_icp,
        #                                               uncertain=True))
  return pose_graph

'''
### option 1: off-the-shelf code
# http://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html

print("Full registration ...")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
  pose_graph = full_registration(pcds,
                                 max_correspondence_distance_coarse,
                                 max_correspondence_distance_fine)

'''

### option 2: my own code

# add all nodes
for frame_id in frame_list:
  T_i2W = np.linalg.inv(pred_poses[frame_id])
  pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_i2W))

# add all edges
for i, src_id in enumerate(frame_list):
  for k in range(1,4):
    if i+k < len(frame_list):
      tgt_id = frame_list[i+k]
      print(f' -- add edge from {src_id} to {tgt_id} ... ')
      # run icp between frame i and j (i+k)
      # final_Ts, _ = icp(pcds[src_id], pcds[tgt_id], inlier_thres=0.01)
      # T_i2j =  final_Ts[-1]
      # transformation_icp =  final_Ts[-1]
      transformation_icp, information_icp = pairwise_registration(pcds[src_id], pcds[tgt_id])

      # add an edge
      pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i+k, transformation_icp))

### end

# run optimization
print("Optimizing PoseGraph ...")

option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

# return optimized nodes
opt_poses = {}
for i, frame_id in enumerate(frame_list):
  T_i2W = pose_graph.nodes[i].pose
  opt_poses[frame_id] = np.linalg.inv(T_i2W)

### report RTEs
print(" -- RTE of odometry:")
get_rte(pred_poses, gt_poses)
print(" -- RTE of graph optimization:")
get_rte(opt_poses, gt_poses)


print("Transform points and display")
vis_list = []
for point_id in range(len(pcds)):
    point_frame = frame_list[point_id]
    pcds[point_frame].transform(pose_graph.nodes[point_id].pose)
    # optimized pose: green
    # T_C2W = pose_graph.nodes[point_id].pose
    pgo_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, opt_poses[point_frame], scale = 0.1)
    pgo_cam.paint_uniform_color((0, 1, 0))
    # gt pose: red
    gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, gt_poses[point_frame], scale = 0.1)
    gt_cam.paint_uniform_color((1, 0, 0))
    # odometry pose: blue
    odometry_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, pred_poses[point_frame], scale = 0.1)
    odometry_cam.paint_uniform_color((0, 0, 1))
    vis_list.append(pgo_cam)
    vis_list.append(odometry_cam)
    vis_list.append(gt_cam)
    vis_list.append(pcds[point_frame])

o3d.visualization.draw_geometries(vis_list,
                                  zoom=0.422,
                                  front = [ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ],
                                  lookat = [ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ],
                                  up = [ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])

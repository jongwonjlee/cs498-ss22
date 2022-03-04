from email.mime import base
import enum
import random
from turtle import left
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from sympy import randMatrix

# read intrinsics, extrinsincs and camera images
K1 = np.load('assets/fountain/Ks/0005.npy')
K2 = np.load('assets/fountain/Ks/0004.npy')
R1 = np.load('assets/fountain/Rs/0005.npy')
R2 = np.load('assets/fountain/Rs/0004.npy')
t1 = np.load('assets/fountain/ts/0005.npy')
t2 = np.load('assets/fountain/ts/0004.npy')
img1 = cv2.imread('assets/fountain/images/0005.png')
img2 = cv2.imread('assets/fountain/images/0004.png')
h, w, _ = img1.shape

# resize the image to reduce computation
scale = 8 # you could try different scale parameters, e.g. 4 for better quality & slower speed.
img1 = cv2.resize(img1, (w//scale, h//scale))
img2 = cv2.resize(img2, (w//scale, h//scale))
h, w, _ = img1.shape

# visualize the left and right image
plt.figure()
# opencv default color order is BGR instead of RGB so we need to take care of it when visualization
plt.imshow(cv2.cvtColor(np.concatenate((img1, img2), axis=1), cv2.COLOR_BGR2RGB))
plt.title("Before rectification")
plt.show()

# Q6.a: How does intrinsic change before and after the scaling?
# You only need to modify K1 and K2 here, if necessary. If you think they remain the same, leave here as blank and explain why.
# --------------------------- Begin your code here ---------------------------------------------

np.set_printoptions(suppress=True)

print("K1 (unscaled): \n", K1)
print("K2 (unscaled): \n", K2)

K1[:2, :] //= scale
K2[:2, :] //= scale

print("K1 (scaled): \n", K1)
print("K2 (scaled): \n", K2)

# --------------------------- End your code here   ---------------------------------------------

# Compute the relative pose between two cameras
T1 = np.eye(4)
T1[:3, :3] = R1
T1[:3, 3:] = t1
T2 = np.eye(4)
T2[:3, :3] = R2
T2[:3, 3:] = t2
T = T2.dot(np.linalg.inv(T1)) # c1 to world and world to c2
R = T[:3, :3]
t = T[:3, 3:]

# Rectify stereo image pair such that they are frontal parallel. Here we call cv2 to help us
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, None, K2, None,(w // 4, h // 4), R, t, 1, newImageSize=(0,0))
left_map  = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w, h), cv2.CV_16SC2)
right_map = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w, h), cv2.CV_16SC2)
left_img = cv2.remap(img1, left_map[0],left_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
right_img = cv2.remap(img2, right_map[0],right_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
plt.figure()
plt.imshow(cv2.cvtColor(np.concatenate((left_img, right_img), axis = 1), cv2.COLOR_BGR2RGB))
plt.title("After stereo rectification")
plt.show()

# Visualize images after rectification and report K1, K2 in your PDF report.


def stereo_matching_ssd(left_im, right_im, max_disp=128, block_size=21):
  """
  Using sum of square difference to compute stereo matching.
  Arguments:
      left_im: left image (h x w x 3 numpy array)
      right_im: right image (h x w x 3 numpy array)
      mask_disp: maximum possible disparity
      block_size: size of the block for computing matching cost
  Returns:
      disp_im: disparity image (h x w numpy array), storing the disparity values
  """
  # --------------------------- Begin your code here ---------------------------------------------
  assert left_im.shape == right_im.shape

  # convert to grayscale
  left_gray = cv2.cvtColor(left_im, cv2.COLOR_BGR2GRAY)
  right_gray = cv2.cvtColor(right_im, cv2.COLOR_BGR2GRAY)
  
  h, w = left_gray.shape
  
  # create padded arrays
  left_gray_padded = np.zeros((h+block_size-1,w+block_size-1), dtype=left_gray.dtype)
  right_gray_padded = np.zeros((h+block_size-1,w+block_size-1,max_disp), dtype=right_gray.dtype)
  
  left_gray_padded[block_size//2:h+block_size//2, block_size//2:w+block_size//2] = left_gray
  for d in range(max_disp):
    right_gray_padded[block_size//2:h+block_size//2, block_size//2+d:w+min(block_size-1, block_size//2+d), d] =\
       right_gray[:, :w+min(block_size-1, block_size//2+d)-block_size//2-d]

  # create strided arrays
  # left_gray_strided[j,i] indicates (block_size, block_size) np.array around left_gray[i,j]
  # right_gray_strided[j,i,d] indicates (block_size, block_size) np.array around right_gray[i,j]
  from numpy.lib.stride_tricks import as_strided

  left_gray_strided = np.empty(shape=left_gray.shape+(block_size,block_size), dtype=left_gray.dtype)
  right_gray_strided = np.empty(shape=right_gray.shape+(max_disp,)+(block_size,block_size), dtype=right_gray.dtype)

  left_gray_strided = as_strided(left_gray_padded, shape=left_gray.shape+(block_size,block_size), strides=left_gray_padded.strides*2)
  for d in range(max_disp):
    right_gray_strided[:,:,d,:,:] = as_strided(right_gray_padded[:,:,d], shape=right_gray.shape+(block_size,block_size), strides=right_gray_padded.strides[:-1]*2)

  # compute squared sum of distance
  # disp_map_candidate[j,i,d] indicates the ssd at (j,i) with a disparity offset d
  # disp_map_candidate = np.einsum('ijmn,ijdmn->ijd', left_gray_strided, right_gray_strided)
  # disp_map_candidate = np.einsum('ijdmn,ijdmn->ijd', right_gray_strided-left_gray_strided[:,:,np.newaxis], right_gray_strided-left_gray_strided[:,:,np.newaxis])
  disp_map_candidate = np.empty(shape=left_gray.shape+(max_disp,), dtype=np.int64)
  for d in range(max_disp):
    for j in range(h):
      for i in range(w):
        disp_map_candidate[j,i,d] = ((left_gray_strided[j,i] - right_gray_strided[j,i,d])**2).sum()
  

  # extract disparity
  # disp_map[j,i] indicates the disparity of two images at (j,i)
  disp_map = np.argmin(disp_map_candidate, axis=2)
  
  # to avoid zero-division, covert zero elements to be -1
  disp_map[np.where(disp_map==0)] = -1
  disp_map = disp_map.astype(np.float64)

  # --------------------------- End your code here   ---------------------------------------------
  return disp_map

disparity = stereo_matching_ssd(left_img, right_img, max_disp=128, block_size=21)
# Depending on your implementation, runtime could be a few minutes.
# Feel free to try different hyper-parameters, e.g. using a higher-resolution image, or a bigger block size. Do you see any difference?

plt.figure()
plt.imshow(disparity)
plt.title("Disparity map")
plt.show()

# Compare your method and an off the shelf CV2's stereo matching results.
# Please list a few directions which you think could improve your own results
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
plt.imshow(np.concatenate((left_gray, right_gray), axis = 1), 'gray')
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=21)
disparity_cv2 = stereo.compute(left_gray, right_gray) / 16.0
plt.imshow(np.concatenate((disparity, disparity_cv2), axis = 1))
plt.show()

# Visualize disparity map and comparison against disparity_cv2 in your report.


# Q6 Bonus:


# --------------------------- Begin your code here ---------------------------------------------

# import RGB info pixelwise 
color = (cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)).astype(np.float64).reshape(-1, 3)

# cast non-positive disparities to nan
disparity[np.where(disparity==-1)] == np.nan
disparity_cv2[np.where(disparity_cv2==-1)] == np.nan

# read baseline and focal length
baseline = np.linalg.norm(t)
f = np.linalg.norm(np.diag(K1)[:-1])

# generate meshgrid in a normalized image coordinate
U, V = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)

fig = plt.figure()

for i, disp in enumerate([disparity, disparity_cv2]):
  # project pointclouds in 3d space upon estimated disparity
  Z = baseline / disp * f
  X = Z / f * U
  Y = -Z / f * V
  
  # create position and corresponding color for projected 3d points
  xyz = np.stack([X,Y,Z], axis=2)
  xyz = np.reshape(xyz, (-1, 3))

  # draw 3d points with corresponding color
  ax = fig.add_subplot(f'12{i+1}', projection='3d')
  ax.scatter(xyz[:, 2], xyz[:, 0], xyz[:, 1], c=color/255.0, label='Points')
  ax.legend(loc='best')
  ax.view_init(azim=-145, elev=30)
  ax.set_xlim((6, 15))
  ax.set_ylim((-5, 5))
  ax.set_zlim((-4, 4))

plt.show()


# --------------------------- End your code here   ---------------------------------------------

# Hints:
# What is the focal length? How large is the stereo baseline?
# Convert disparity to depth
# Unproject image color and depth map to 3D point cloud
# You can use Open3D to visualize the colored point cloud

if xyz is not None:
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.colors = o3d.utility.Vector3dVector(color)
  o3d.visualization.draw_geometries([pcd])
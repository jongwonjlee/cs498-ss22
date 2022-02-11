import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import open3d as o3d

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0

# Read eight keypoints, including four court corners and four backboard corners
keypoints_im = np.array([
 [ 642.89378381,  589.79713627],
 [1715.31354164,  773.80704813],
 [1087.51501881, 1049.40560391],
 [  74.2115675 ,  637.2567063 ],
 [ 375.62146838,  464.07090689],
 [ 439.73351912,  462.40565882],
 [ 441.39876719,  496.54324428],
 [ 376.45409242,  499.87374042]
 ])

plt.figure()
plt.imshow(im)
plt.scatter(keypoints_im[:, 0], keypoints_im[:, 1])
plt.plot(keypoints_im[[0, 1, 2, 3, 0], 0], keypoints_im[[0, 1, 2, 3, 0], 1], 'g')
plt.plot(keypoints_im[[0+4, 1+4, 2+4, 3+4, 0+4], 0], keypoints_im[[0+4, 1+4, 2+4, 3+4, 0+4], 1], 'g')
for ind, corner in enumerate(keypoints_im):
		plt.text(corner[0] + 30.0, corner[1] + 30.0, '#'+str(ind),
						 c='b', family='sans-serif', size='x-large')
plt.title("Keypoints")
plt.show()



'''
Question 4: specify the keypoints' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis,
upward direction perpendicular to the ground as z axis
Output:
		 - corners_3D: a numpy array (8x3 matrix)
'''

# Predefined constants on basketball court
lower_rim = 3.05 - 0.305 # height of backboard's lower rim
backboard_width = 1.83
backboard_height = 1.22
court_length = 28.65
court_width = 15.24
board_to_baseline = 1.22 # board to baseline distance

# --------------------------- Begin your code here ---------------------------------------------
corners_3d = np.array([(0, court_width, 0), (court_length, court_width, 0), (court_length, 0, 0), (0, 0, 0),
                       (board_to_baseline, (court_width-backboard_width)/2, lower_rim+backboard_height),
                       (board_to_baseline, (court_width+backboard_width)/2, lower_rim+backboard_height),
                       (board_to_baseline, (court_width+backboard_width)/2, lower_rim),
                       (board_to_baseline, (court_width-backboard_width)/2, lower_rim)])
# --------------------------- End your code here   ---------------------------------------------

print("corners_3d:\n", corners_3d)

'''
Question 5: complete the findProjection function
Arguments:
     xyz - Each row corresponds to an actual point in 3D with homogeneous coordinate (Nx4 matrix)
     uv - Each row is the pixel location in the homogeneous image coordinate (Nx3 matrix)
Returns:
     P - The projection matrix (4x3 matrix) such that uv = P @ xyz

Hints:
    - you might find the function vstack, hstack to be handy for getting homogenous coordinate;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''
def findProjection(xyz, uv):
# --------------------------- Begin your code here ---------------------------------------------
    N, _ = xyz.shape
    
    A = np.zeros((2*N, 12))
    for i in range(N):
        A[2*i, 4:8] = xyz[i]
        A[2*i+1, 0:4] = xyz[i]
        
        A[2*i, 8:12] = - uv[i][1] * xyz[i]
        A[2*i+1, 8:12] = - uv[i][0] * xyz[i]
    
    U, S, Vh = np.linalg.svd(A)
    P = Vh[-1]
    
    return P.reshape(3, 4) / P[-1]

# --------------------------- End your code here   ---------------------------------------------

# Get homogeneous coordinate (using np concatenate)
uv = np.concatenate([keypoints_im, np.ones((len(keypoints_im), 1))], axis = 1)
xyz = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis = 1)

# Find the projection matrix from correspondences
P = findProjection(xyz, uv)
print("P:\n", P)

# Recalculate the projected point location
uv_project = P.dot(xyz.T).T
uv_project = uv_project / np.expand_dims(uv_project[:, 2], axis = 1)

# Plot reprojection.
plt.clf()
plt.imshow(im)
plt.scatter(uv[:, 0], uv[:, 1], c='r', label = 'original keypoints')
plt.scatter(uv_project[:, 0], uv_project[:, 1], c='b', label = 'reprojected keypoints')
plt.title('Reprojection')
plt.legend()
plt.show()



# Load the stanford bunny 3D mesh
bunny = o3d.io.read_triangle_mesh('./bunny.ply')
bunny.compute_vertex_normals()
# Today we will only consider using its vertices
verts = np.array(bunny.vertices)

'''
Question 6: project the stanford bunny onto the center of the basketball court

Output:
		- bunny_uv: all the vertices on image coordinate (35947x2 matrix)

Hints:
    - Transform the bunny from its object-centric 3D coordinate to basketball court 3D coordinate;
    - Make sure the bunny is above the ground
    - Do not forget to use homomgeneous coordinate for projection
'''

# --------------------------- Begin your code here ---------------------------------------------

T_bunny_xyz = np.eye(4); T_bunny_xyz[:3, -1] = np.array([court_length/2, court_width/2, -verts[:, -1].min()])
verts = T_bunny_xyz.dot(np.concatenate([verts, np.ones((len(verts), 1))], axis = 1).T).T
bunny_uv = P.dot(verts.T).T
bunny_uv = bunny_uv / np.expand_dims(bunny_uv[:, 2], axis = 1)

# --------------------------- End your code here   ---------------------------------------------


# Visualize the Projection
plt.clf()
plt.imshow(im)
plt.scatter(bunny_uv[:, 0], bunny_uv[:, 1], c='b', s = 0.01, label = 'bunny')
plt.title('Stanford Bunny on State Farm Center')
plt.legend()
plt.show()

# Dump the results for autograde
outfile = 'solution_perspective.npz'
np.savez(outfile, corners_3d, P, bunny_uv)
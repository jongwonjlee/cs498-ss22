'''
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation
For this question, your task is to implement normalized and unnormalized eight-point algorithms to find out the fundamental matrix between two cameras.
We've provided a method to compute the average geometric distance, which is the distance between each projected keypoint from one image to its corresponding epipolar line in the other image.
You might consider reading that code below as a reminder for how we can use the fundamental matrix.
For more information on the normalized eight-point algorithm, please see this link: https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm

Question 3. RANSAC
Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100, and 10000 iterations of RANSAC.
Please also visualize the inliers with your best estimated fundamental matrix in your solution for both images (we provide a visualization function).
In your PDF, please also explain why we do not perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epipolar Lines
Please visualize the epipolar line for both images for your estimated F in Q2 and Q3.

To draw it on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
'''

from locale import normalize
import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random
from pathlib import Path

basedir= Path('assets/fountain')
img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
img2 = cv2.imread(str(basedir /'images/0005.png'), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap='gray')
axarr[1].imshow(img2, cmap='gray')
plt.show()

# --------------------- Question 2

def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geomtric distance from each projected keypoint from one image to its corresponding epipolar line in another image.
    Note that you should take the average of the geometric distance in two direction (image 1 to 2, and image 2 to 1)
    Arguments:
        all_matches: all matched keypoint pairs that loaded from disk (#all_matches, 4).
        F: estimated fundamental matrix, (3, 3)
    Returns:
        average geomtric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2)/2
    return dist

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
eight_good_matches = np.load('assets/eight_good_matches.npy')
all_good_matches = np.load('assets/all_good_matches.npy')

def estimate_fundamental_matrix(matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    F = np.eye(3)
    # --------------------------- Begin your code here ---------------------------------------------
    assert len(matches) >= 8
    matches = matches.copy()

    A = np.empty((len(matches), 9))

    if normalize:
        all_p1_mean = matches[:,:2].mean(axis=0)
        all_p1_stdv = matches[:,:2].std(axis=0)
        all_p2_mean = matches[:,2:].mean(axis=0)
        all_p2_stdv = matches[:,2:].std(axis=0)

        # normalize points
        matches[:, :2] = (matches[:, :2] - all_p1_mean) / all_p1_stdv
        matches[:, 2:] = (matches[:, 2:] - all_p2_mean) / all_p2_stdv

        # construct transformation matrices
        T1 = np.eye(3)
        T1[:2, -1] -= all_p1_mean
        T1[:2, :] /= all_p1_stdv[:, np.newaxis]
        T2 = np.eye(3)
        T2[:2, -1] -= all_p2_mean
        T2[:2, :] /= all_p2_stdv[:, np.newaxis]
        
    # formulate a homogeneous linear equation
    for i, match in enumerate(matches):
        x1, y1, x2, y2 = match
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    
    # get solution to the constructed homogeneous linear equation (in the form of Af = 0)
    U, S, Vh = np.linalg.svd(A)
    F = np.reshape(Vh[-1, :], (3, 3))   # reshape the min singular value into a 3 by 3 matrix (get f minimizing |Af|^2 subject to |f|^2 = 1)

    # enforce rank-2 constraint to the reconstructed fundamental matrix F
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh
    
    # return to the original units with transformation matrices T1 and T2
    if normalize:
        F = T2.T @ F @ T1

    # --------------------------- End your code here   ---------------------------------------------
    return F

F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
F_without_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=False)

# Evaluation (these numbers should be quite small)
print(f"F_with_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_with_normalization)}")
print(f"F_without_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_without_normalization)}")


# --------------------- Question 3

def ransac(all_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        num_iteration: total number of RANSAC iteration
        estimate_fundamental_matrix: your eight-point algorithm function but use normalized version
        inlier_threshold: threshold to decide if one point is inlier
    Returns:
        best_F: best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_F: (#inliers, 4)
        avg_geo_dis_with_best_F: float
    """

    best_F = np.eye(3)
    inlier_matches_with_best_F = None
    avg_geo_dis_with_best_F = 0.0

    ite = 0
    # --------------------------- Begin your code here ---------------------------------------------

    while ite < num_iteration:
        ## step 1. obtain a fundamental matrix candidate
        # sample candidate matches randomly (at least eight-points are needed)
        num_samples = 8
        candidate_matches = all_good_matches[np.random.choice(len(all_good_matches), num_samples)]
        
        # estimate fundamental matrix of the candidate matches
        F = estimate_fundamental_matrix(candidate_matches, normalize=True)
        
        ## step 2. check the inliers by applying the estimated fundamental matrix to the whole matches
        # def apply_fundamental_matrix(F, matches):
        # transform the set of image points into homogeneous coordinates
        ones = np.ones((len(all_good_matches), 1))
        all_p1 = np.concatenate((all_good_matches[:, 0:2], ones), axis=1)
        all_p2 = np.concatenate((all_good_matches[:, 2:4], ones), axis=1)
        # get coefficients of epipolar lines
        F_p2 = np.dot(F.T, all_p2.T).T  # coefficients of epipolar lines appearing in img1 (corresponding to epipoles all_p1)
        F_p1 = np.dot(F, all_p1.T).T    # coefficients of epipolar lines appearing in img2 (corresponding to epipoles all_p2)
        # get geometric distances between reconstructed epipolar lines and corresponding epipoles
        p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
        p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
        dist_all_p1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]    # set of distances between the estimated epipolar lines (projected from p2) and epipoles (p1)
        dist_all_p2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]    # set of distances between the estimated epipolar lines (projected from p1) and epipoles (p2)
        dist_all = (dist_all_p1 + dist_all_p2) / 2

        # get indices of inliers
        inlier_idx = np.where(dist_all < inlier_threshold)[0]

        # step 3. update the best solution upon the number of inliers 
        if (inlier_matches_with_best_F is None) or (len(inlier_idx) > len(inlier_matches_with_best_F)):
            best_F = F
            inlier_matches_with_best_F = all_good_matches[inlier_idx]
            avg_geo_dis_with_best_F = dist_all.sum() / len(all_matches)

        ite += 1

    # --------------------------- End your code here   ---------------------------------------------
    return best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F

def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i+1)
        plt.imshow(im, cmap='gray')
        plt.scatter(inlier_coords[:, 2*i], inlier_coords[:, 2*i+1], marker="x", color="red", s=10)
    plt.show()

num_iterations = [1, 100, 10000]
inlier_threshold = 1 # TODO: change the inlier threshold by yourself
for num_iteration in num_iterations:
    best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold)
    if inlier_matches_with_best_F is not None:
        print(f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};")
        visualize_inliers(img1, img2, inlier_matches_with_best_F)

# --------------------- Question 4

def visualize(estimated_F, img1, img2, kp1, kp2):
    # --------------------------- Begin your code here ---------------------------------------------
    assert len(kp1) == len(kp2)
    
    all_p1 = np.concatenate((kp1, np.ones((len(kp1),1))), axis=1)
    all_p2 = np.concatenate((kp2, np.ones((len(kp2),1))), axis=1)
    
    F_p2 = np.dot(estimated_F.T, all_p2.T).T  # coefficients of epipolar lines appearing in img1 (corresponding to epipoles all_p1)
    F_p1 = np.dot(estimated_F, all_p1.T).T    # coefficients of epipolar lines appearing in img2 (corresponding to epipoles all_p2)

    # get geometric distances between reconstructed epipolar lines and corresponding epipoles
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:,np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:,np.newaxis]
    dist_all_p1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:,np.newaxis]    # set of distances between the estimated epipolar lines (projected from p2) and epipoles (p1)
    dist_all_p2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:,np.newaxis]    # set of distances between the estimated epipolar lines (projected from p1) and epipoles (p2)

    # get points on every reconstructed line closest to the corresponding epipole
    all_p1_closest = kp1 - F_p2[:,:2] / np.linalg.norm(F_p2[:,:2], axis=1)[:,np.newaxis] * np.einsum('ij,ij->i', all_p1, F_p2)[:,np.newaxis]
    all_p2_closest = kp2 - F_p1[:,:2] / np.linalg.norm(F_p1[:,:2], axis=1)[:,np.newaxis] * np.einsum('ij,ij->i', all_p2, F_p1)[:,np.newaxis]

    # find endpoints of segment on every epipolar line
    # offset from the closest point is 10 pixels
    l = 100
    all_p1_src = all_p1_closest + np.hstack((F_p2[:,1][:,np.newaxis], -F_p2[:,0][:,np.newaxis])) / np.linalg.norm(F_p2[:,:2], axis=1)[:, np.newaxis] * l
    all_p1_dst = all_p1_closest - np.hstack((F_p2[:,1][:,np.newaxis], -F_p2[:,0][:,np.newaxis])) / np.linalg.norm(F_p2[:,:2], axis=1)[:, np.newaxis] * l
    all_p2_src = all_p2_closest + np.hstack((F_p1[:,1][:,np.newaxis], -F_p1[:,0][:,np.newaxis])) / np.linalg.norm(F_p1[:,:2], axis=1)[:, np.newaxis] * l
    all_p2_dst = all_p2_closest - np.hstack((F_p1[:,1][:,np.newaxis], -F_p1[:,0][:,np.newaxis])) / np.linalg.norm(F_p1[:,:2], axis=1)[:, np.newaxis] * l

    # Display points and segments of corresponding epipolar lines.
    # You will see points in red corsses, epipolar lines in green 
    # and a short cyan line that denotes the shortest distance between
    # the epipolar line and the corresponding point.
    
    plt.tight_layout()

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.plot(kp1[:,0], kp1[:,1],  '+r')
    plt.plot([kp1[:,0], all_p1_closest[:,0]],[kp1[:,1], all_p1_closest[:,1]], 'r')
    plt.plot([all_p1_src[:,0], all_p1_dst[:,0]],[all_p1_src[:,1], all_p1_dst[:,1]], 'g')
    plt.xlim(0, img1.shape[1])
    plt.ylim(img1.shape[0], 0)

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.plot(kp2[:,0], kp2[:,1],  '+r')
    plt.plot([kp2[:,0], all_p2_closest[:,0]],[kp2[:,1], all_p2_closest[:,1]], 'r')
    plt.plot([all_p2_src[:,0], all_p2_dst[:,0]],[all_p2_src[:,1], all_p2_dst[:,1]], 'g')
    plt.xlim(0, img2.shape[1])
    plt.ylim(img2.shape[0], 0)
    
    plt.show()
    # --------------------------- End your code here   ---------------------------------------------

all_good_matches = np.load('assets/all_good_matches.npy')
F_Q2 = F_with_normalization # link to your estimated F in Q3
F_Q3 = best_F # link to your estimated F in Q3
visualize(F_Q2, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
visualize(F_Q3, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
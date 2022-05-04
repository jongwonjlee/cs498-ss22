import numpy as np
from numpy import dot, zeros, eye, isscalar
from copy import deepcopy
from math import log
import sys

from utils import within_range

class Kalman:
    # we've set the dimensions of x, z for you
    def __init__(self, bbox3D, info, ID, dim_x=10, dim_z=7, dim_u=0):
        self.initial_pos = bbox3D       # the initial_pos also initializes self.x below
        self.time_since_update = 0      # keep track of how long since a detection was matched to this tracker
        self.hits = 1                   # number of total hits including the first detection
        self.info = info                # some additional info, mainly used for evaluation code
        self.ID = ID                    # each tracker has a unique ID, so that we can see consistency across frames

        # -------------------- above are some bookkeeping params, below is the Kalman Filter

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))          # state
        self.Sigma = eye(dim_x)             # uncertainty covariance
        self.Q = eye(dim_x)                 # process uncertainty

        self.A = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # measurement function
        self.R = eye(dim_z)               # measurement uncertainty

        self.z = np.array([[None]*self.dim_z]).T

        # Kalman gain and residual are computed during the update step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # initialize data
        self.x[:7] = self.initial_pos.reshape((7, 1))

        # IMPORTANT, call define_model
        self.define_model()

    # Q3.
    # Here is where you define the model, namely self.A and self.H
    # Though you may likely want to make modifications to uncertainty as well,
    #   ie. self.R, self.Q, self.Sigma
    # TODO: Your code
    def define_model(self):
        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        # self.A = ...

        # Measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        # self.H = ...

        # Hint: the initial velocity is very uncertain for new detections

        # --------------------------- Begin your code here ---------------------------------------------
        # Uncertainty matrix: We are not confident in initial values to every state element
        self.Sigma = eye(self.dim_x)*100
        # State transition matrix: Populate elements corresponding to linear velocity dynamics
        self.A[:3,-3:] = eye(3)
        # Measurement matrix: Initialize elements mapping from x (with linear velocity) to z (without linear velocity)
        self.H[:7,:7] = eye(7)
        # Process noise matrix: It describes how we uncertain about propagations to every state element.
        # For position (0~3), they are set to zero under the assumption that we 'trust' linear velocity.
        # Others (orientation, bounding box dimension, velocity) are set to 10
        self.Q[:3,:3] = eye(3)*0
        self.Q[3:,3:] = eye(7)*10
        # Measurement noise matrix: It describes how we uncertain about measurements.
        # For measurements (position, orientation, bounding box dimension), they are set to 1
        self.R = eye(self.dim_z)*1
        # --------------------------- End your code here   ---------------------------------------------
        return

    # Q5.
    # TODO: Your code
    def predict(self, kalman_prediction=True):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        Note that often we might give predict a control vector u,
        but of course we don't know what controls each tracked object is applying
        """
        # Hint: you should be modifying self.x and self.Sigma
        # --------------------------- Begin your code here ---------------------------------------------
        if kalman_prediction:
            self.x = self.A @ self.x
            self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q
        else:
            pass
        # --------------------------- End your code here   ---------------------------------------------

        # Leave this at the end, within_range ensures that the angle is between -pi and pi
        self.x[3] = within_range(self.x[3])
        return

    # Q2 and Q4.
    # TODO: Your code
    def update(self, z, kalman_update=True):
        """
        Add a new measurement (z) to the Kalman filter.
        ----------
        z : (dim_z, 1): array_like measurement for this update.
        """
        z = z.reshape(-1,1)

        # --------------------------- Begin your code here ---------------------------------------------

        if kalman_update:  # For Q4
            # Apply bayesian filtering to relate the measurement (z) to the state (self.x)
            # Compute required matrices
            self.S = self.H @ self.Sigma @ self.H.T + self.R
            self.SI = np.linalg.inv(self.S)
            self.K = self.Sigma @ self.H.T @ self.SI
            self.y = self.H @ self.x

            # Update state estimate
            self.x = self.x + self.K @ (z - self.y)
            # Update covariance estimate
            self.Sigma = self.Sigma - self.K @ self.H @ self.Sigma

        else:      # For Q2
            # Trust measurement (detection) as it is
            self.x[:7] = z
        
        # --------------------------- End your code here   ---------------------------------------------

        # Leave this at the end, within_range ensures that the angle is between -pi and pi
        self.x[3] = within_range(self.x[3])
        return

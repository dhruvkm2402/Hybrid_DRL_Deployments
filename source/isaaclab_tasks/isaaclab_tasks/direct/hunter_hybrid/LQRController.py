"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# === Parameters =====

# LQR parameter
lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.0083  # time tick[s]
L =  0.608 # Wheel base of the vehicle [m]
max_steer = np.deg2rad(30.0)  # maximum steering angle[rad]

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v




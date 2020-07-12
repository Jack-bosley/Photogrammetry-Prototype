# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:25:32 2020

@author: Jack
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D


from Utilities.virtual_realtime import Virtual_Realtime_Accelerometer, Virtual_Realtime_Camera


# Class to compute the orientation and position of the camera approximately
class Orienter:
    
    def __init__(self, x0, v0, r0):
        # Current position
        self.pos = x0
        self.vel = v0
        self.rot = r0
        
        # Histories for plotting
        self.X = []
        self.R = []
        
        
    def plot(self):
        
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Positions
        z, x, y = np.multiply(np.array(self.X).T, 1)
        ax.plot(x, y, z)
        
        n = 50
        
        # Camera direction
        rz, rx, ry = np.array(self.R).T
        ax.quiver(x[::n], y[::n], z[::n],
                  rx[::n], ry[::n], rz[::n], 
                  colors='k', length=0.2, normalize=True)
        
        # Format the plot
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        ax.set_xlabel('z position')
        ax.set_ylabel('x position')
        ax.set_zlabel('y position')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
        
        
    def update(self, dt, a, r):
        self.rot = r
        
        # Use orientation to update velocity
        R = Orienter.rotation_matrix(*self.rot)
        dv = np.matmul(R, np.multiply(a, dt).T)
        self.vel = np.add(self.vel, dv)
        
        # Use velocity to update position
        dx = np.multiply(self.vel, dt)
        self.pos = np.add(self.pos, dx)
        
        # Store for plotting
        self.X.append(self.pos)
        self.R.append(np.matmul(R, np.array([0, -1, 0]).T).T)
    
    def get_camera_pose_guess(self):
        T_guess = [[] for i in range(len(self.X))]
        
        for i, data in enumerate(zip(self.X, self.R)):
            x, r = data
            T_guess[i] = [r[0], r[1], r[2], x[0], x[1], x[2]]
            
        return T_guess
    
    # Get camera rotation matrix from Euler angles
    @staticmethod
    def rotation_matrix(a, b, c):   
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
    
        # Compute and return the rotation matrix
        return np.array([[ca*cb, (ca*sb*sc) - (sa*cc), (ca*sb*cc) + (sa*sc)], 
                         [sa*cb, (sa*sb*sc) + (ca*cc), (sa*sb*cc) - (ca*sc)], 
                         [  -sb,                cb*sc,                cb*cc]])  
        



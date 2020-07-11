# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:25:32 2020

@author: Jack
"""

import os
import numpy as np
import time, threading
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

# Class to compute the orientation and position of the camera approximately
class Orientor:
    
    def __init__(self, x0, v0, r0):
        # Current position
        self.pos = x0
        self.vel = v0
        self.rot = r0
        
        # Histories for plotting
        self.T = [0]
        self.X = [self.pos]
        self.R = [self.rot]
       
        
    def plot(self):
        
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Positions
        x, z, y = np.multiply(np.array(self.X).T, 1)
        ax.plot(x, y, z)
        
        # Camera direction
        rx, rz, ry = np.array(self.R).T
        ax.quiver(x, y, z, rx, ry, rz, colors='k', length=0.2, normalize=True)
        
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
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.set_zlabel('z position (m)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
        
        
    def update(self, dt, a, w):
        # Update the orientation of the camera
        dr = np.multiply(w, dt)
        self.rot = np.add(self.rot, dr)
        
        
        # Use orientation to update velocity
        R = Orientor.rotation_matrix(self.rot[1], self.rot[0], self.rot[2])
        dv = np.matmul(R, np.multiply(a, dt).T)

        self.vel = np.add(self.vel, dv)
        
        # Use velocity to update position
        dx = np.multiply(self.vel, dt)
        self.pos = np.add(self.pos, dx)
        
        # Store for plotting
        self.T.append(self.T[-1] + dt)
        self.X.append(self.pos)
        self.R.append(np.matmul(R, np.array([0, 0, -1]).T).T)
    
    
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
        

# Immitate a real-time gyroscope / accelerometer data stream
class Virtual_Realtime_Streamer:
    def __init__(self, times, accel_data, gyro_data):
        self.times = times
        self.a_x, self.a_y, self.a_z = accel_data
        self.w_x, self.w_y, self.w_z = gyro_data
    
    
    def plot(self):
        fig = plt.figure()
        axs = fig.subplots(2)
        
        axs[0].plot(self.times, self.a_x)
        axs[0].plot(self.times, self.a_y)
        axs[0].plot(self.times, self.a_z)
        axs[0].legend(["x", "y", "z"])
        axs[0].set_ylabel("acceleration (m/s^2)")
                
        axs[1].plot(self.times, self.w_x)
        axs[1].plot(self.times, self.w_y)
        axs[1].plot(self.times, self.w_z)
        axs[1].legend(["ω_x", "ω_y", "ω_z"])
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("rotational velocity (ω)")
        
        
    
    def bundle(self, t, dt):
        t_i = np.where((self.times >= t) & (self.times < t + dt))
        
        n = len(t_i[0])
        if n == 0:
            return dt, [0, 0, 0], [0, 0, 0]
        
        a_x, a_y, a_z = self.a_x[t_i], self.a_y[t_i], self.a_z[t_i]
        w_x, w_y, w_z = self.w_x[t_i], self.w_y[t_i], self.w_z[t_i]
        
        ax = np.mean(a_x)
        ay = np.mean(a_y)
        az = np.mean(a_z)
        
        wx = np.mean(w_x)
        wy = np.mean(w_y)
        wz = np.mean(w_z)
        
        return dt, [ax, ay, az], [wx, wy, wz]
    
    
    def run(self, callback, delta_t=-1, speed_multiplier=1.0, t_start=0, t_end=-1):
        def caller():
            if delta_t != -1:
                t = t_start
                end_time = self.times[-1] if t_end == -1 else t_end
                
                while t < end_time:
                    dt, a, w = self.bundle(t, delta_t)
                    
                    callback(dt, a, w)
                    
                    t += dt
                    time.sleep(delta_t / speed_multiplier)
            else:
                for i in range(len(self.times)-1):
                    dt = self.times[i+1] - self.times[i]
                    a = [self.a_x[i], self.a_y[i], self.a_z[i]]
                    w = [self.w_x[i], self.w_y[i], self.w_z[i]]
                    callback(dt, a, w)
                
                
        thread = threading.Thread(target=caller)
        thread.start()
        thread.join()
    
    
def test_projector(directory, data_file):
    data = open(directory + "/" + data_file)

    T = []
    A = []
    W = []
    
    for i, d in enumerate(data):        
        if i <= 1:
            continue
            
        try:
            t, ax, ay, az, wx, wy, wz = d.split(',')[:7]
            t = float(t)
            ax, ay, az = float(ax), float(ay), float(az)
            wx, wy, wz = float(wx), float(wy), float(wz)
            
            T.append(t)
            A.append([ax, ay, az])
            W.append([wz, wy, wz])
        except:
            break

    
    orientor = Orientor([0, 0, 0], [0, 0, 0], [0, 0, 0])
    virtual_streamer = Virtual_Realtime_Streamer(np.array(T).T, np.array(A).T, np.array(W).T)
    virtual_streamer.plot()
    virtual_streamer.run(orientor.update, delta_t=0.1, speed_multiplier=100, t_start=8, t_end=16)
    orientor.plot()
    
    
if __name__ == '__main__':
    test_projector('../../Data/Video3', 'accel_gyro_data.csv')



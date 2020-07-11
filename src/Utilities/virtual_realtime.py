# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:32:01 2020

@author: Jack
"""


import numpy as np
import time, threading
import matplotlib.pyplot as plt

from PIL import Image

# Immitate a real-time gyroscope / accelerometer data stream
# Mimics the raspberry pi sense hat
class Virtual_Realtime_Accelerometer:
    def __init__(self, times, accel_data, gyro_data):
        self.is_running = False
        self.t_start = 0
        self.t_length = times[-1]
        
        self.times = times
        self.a_x, self.a_y, self.a_z = accel_data
        self.w_x, self.w_y, self.w_z = gyro_data
        
        dt = [0]
        for i in range(1, len(self.times)):
            dt.append(self.times[i] - self.times[i - 1])
        
        self.o_x, self.o_y, self.o_z = [0], [0], [0]
        for i in range(len(dt)):
            self.o_x.append(self.o_x[-1] + (self.w_x[i] * dt[i]))
            self.o_y.append(self.o_y[-1] + (self.w_y[i] * dt[i]))
            self.o_z.append(self.o_z[-1] + (self.w_z[i] * dt[i]))
        
        del self.o_x[0]
        del self.o_y[0]
        del self.o_z[0]


    def plot(self):
        fig = plt.figure()
        axs = fig.subplots(2)
        
        axs[0].plot(self.times, self.a_x)
        axs[0].plot(self.times, self.a_y)
        axs[0].plot(self.times, self.a_z)
        axs[0].legend(["x", "y", "z"])
        axs[0].set_ylabel("acceleration (m/s^2)")
                
        axs[1].plot(self.times, self.o_x)
        axs[1].plot(self.times, self.o_y)
        axs[1].plot(self.times, self.o_z)
        axs[1].legend(["θ_x", "θ_y", "θ_z"])
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("Orientation (rads)")
        
           
    def get_accelerometer_raw(self):
        t = time.time() - self.t_start
        if t > self.t_length:
            self.is_running = False
        
        if self.is_running:
            ax = np.interp(t, self.times, self.a_x)
            ay = np.interp(t, self.times, self.a_y)
            az = np.interp(t, self.times, self.a_z)
            
            return ax, ay, az
        else:
            return 0, 0, 0
    
    def get_orientation_radians(self):
        t = time.time() - self.t_start
        if t > self.t_length:
            self.is_running = False
        
        if self.is_running:
            t = time.time() - self.t_start
            ox = np.interp(t, self.times, self.o_x)
            oy = np.interp(t, self.times, self.o_y)
            oz = np.interp(t, self.times, self.o_z)
            
            return ox, oy, oz
        else:
            return 0, 0, 0        
    
    
    def start(self):
        self.is_running = True
        self.t_start = time.time()
        

        
        
class Virtual_Realtime_Camera:
    def __init__(self, times, files):
        self.is_running = False
        self.t_start = time.time()
        self.t_length = times[-1]
        
        self.times = times
        self.files = files
        
        
    def capture(self):
        t = time.time() - self.t_start
        if t > self.t_length:
            self.is_running = False
        
        if self.is_running:
            array = np.asarray(self.times)
            i = (np.abs(array - t)).argmin()
            
            return Image.open(self.files[i])
        else:
            return None
        
        
    def start(self):
        self.is_running = True
        self.t_start = time.time()
        
        
        
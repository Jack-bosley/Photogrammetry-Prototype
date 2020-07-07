# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:34:35 2020

@author: Jack
"""

import numpy as np


# Class containing pre-calibrated camera information and camera specific methods
class Camera:
    
    def __init__(self, f_x, f_y, k1, k2, k3, p1, p2, w, h):
        # Focal lengths
        self.f_x, self.f_y = f_x, f_y
        
        # Radial distortion coefficients
        self.k1, self.k2, self.k3 = k1, k2, k3
        
        # Tangential distortion coefficients
        self.p1, self.p2 = p1, p2
        
        # Output image resolution
        self.w, self.h = w, h
    
    
    # Get camera rotation matrix from Euler angles
    @staticmethod
    def rotation_matrix(pitch, roll, yaw):   
        # yaw
        cy, sy = np.cos(roll), np.sin(roll)
        
        # pitch
        cp, sp = np.cos(yaw), np.sin(yaw)
        
        # roll
        cr, sr = np.cos(pitch), np.sin(pitch)
    
        # Compute and return the rotation matrix
        return np.array([[cy*cp, (cy*sp*sr) - (sy*cr), (cy*sp*cr) + (sy*sr)], 
                         [sy*cp, (sy*sp*sr) + (cy*cr), (sy*sp*cr) - (cy*sr)], 
                         [  -sp,                cp*sr,                cp*cr]])   
    
    
    # Getters for grouped variables
    def focal_lengths(self):
        return self.f_x, self.f_y
    
    def radial_distortion_coefficients(self):
        return self.k1, self.k2, self.k3
    
    def tangential_distortion_coefficients(self):
        return self.p1, self.p2
    
    def image_resolution(self):
        return self.w, self.h
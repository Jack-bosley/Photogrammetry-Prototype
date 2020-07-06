# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import numpy as np
from matplotlib import pyplot as plt


class Bundle_adjuster:
    
    
    # Camera roll from euler angles  
    def __get_R_camera(self):
        # yaw
        cy = np.cos(self.R_camera_data[0])
        sy = np.sin(self.R_camera_data[0])
        
        # pitch
        cp = np.cos(self.R_camera_data[1])
        sp = np.sin(self.R_camera_data[1])
        
        # roll
        cr = np.cos(self.R_camera_data[2])
        sr = np.sin(self.R_camera_data[2])
    
        return np.array([[cy*cp, (cy*sp*sr) - (sy*cr), (cy*sp*cr) + (sy*sr)], 
                         [sy*cp, (sy*sp*sr) + (cy*cr), (sy*sp*cr) - (cy*sr)], 
                         [  -sp,                cp*sr,                cp*cr]])   
    R_camera = property(__get_R_camera)
    
    
    def __init__(self):
        
        # Dummy coordinates for testing 
        self.X = [np.array([1,0,0]).T,
                  np.array([0,1,0]).T,
                  np.array([0,0,1]).T,
                  np.array([1,1,2]).T]
    
        # Camera extrinsic parameters       
        # Rotation (roll, yaw, pitch)
        self.R_camera_data = [0, 0, 0]
        # Translation
        self.T_camera = np.array([0, 0, -2]).T
        
        # Camera intrinsic parameters
        # (x, y) focal lengths
        self.K_camera_focal = [9/16, 1]
        # radial distortion parameters
        self.K_camera_radial = [0, 0, 0]
        # tangential distortion parameters
        self.K_camera_tangential = [0, 0]
        # pixel space parameters
        self.K_camera_pixel_space = [1280, 720]
        

      
    def reproject(self):
        
        # Translate into camera reference frame
        X_camera = [np.matmul(self.R_camera, np.subtract(X, self.T_camera)) for X in self.X]
        #  Cull points too near the camera
        X_camera = [X for X in X_camera if X[2] > 0.01]
        
        
        # Project onto camera plane
        u_norm_image = [np.array([self.K_camera_focal[0] * X_c[0] / X_c[2], 
                                  self.K_camera_focal[1] * X_c[1] / X_c[2], 
                                  1]) for X_c in X_camera]
        
            
        # Distortions
        u_norm_image_r_sq = [np.square(np.add(np.square(u[0]), 
                                              np.square(u[1]))) for u in u_norm_image]         
        k1, k2, k3 = self.K_camera_radial
        p1, p2 = self.K_camera_tangential
        #  Radial
        u_radial_corrections = [(k1 * u_r +
                                 k2 * np.power(u_r, 2) +
                                 k3 * np.power(u_r, 3)) for u_r in u_norm_image_r_sq]
        u_c_r = [np.array([u[0] * u_rad_corr, 
                           u[1] * u_rad_corr]) for u, u_rad_corr in zip(u_norm_image, u_radial_corrections)]        
        #  Tangential
        p1, p2 = self.K_camera_tangential
        u_c_t = [np.array([(2*p1*u[0]*u[1]) + (p2*(r + (2*np.square(u[0])))),
                           (p1*(r + (2*np.square(u[1])))) + (2*p2*u[0]*u[1])]) for u,r in zip(u_norm_image,u_norm_image_r_sq)]
        #  Apply distortions
        u_norm_image = [np.array([u[0] + cr[0] + ct[0],
                                  u[1] + cr[1] + ct[1]]) for (u, cr, ct) in zip(u_norm_image, u_c_r, u_c_t)]
            
            
        # Project into pixel plane
        w, h = self.K_camera_pixel_space
        w_half, h_half = w/2, h/2
        p_image = [np.array([w*u_x + w_half,h*u_y + h_half]) for (u_x, u_y) in u_norm_image]
        
        
        # Plot points for debugging purposes
        p_x, p_y = zip(*p_image)
        plt.scatter(p_x, p_y)
        plt.axis('equal')
        plt.xlim(0, w)
        plt.ylim(0, h)


def main():
    ba = Bundle_adjuster()
    
    ba.reproject()
    
    
    
    
if __name__ == '__main__':
    main()
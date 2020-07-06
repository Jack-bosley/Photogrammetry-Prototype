# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import numpy as np
from matplotlib import pyplot as plt


class Bundle_adjuster:
    
    # Camera roll from euler angles
    @staticmethod
    def __get_R_camera(roll, yaw, pitch):
        # yaw
        cy = np.cos(roll)
        sy = np.sin(roll)
        
        # pitch
        cp = np.cos(yaw)
        sp = np.sin(yaw)
        
        # roll
        cr = np.cos(pitch)
        sr = np.sin(pitch)
    
        return np.array([[cy*cp, (cy*sp*sr) - (sy*cr), (cy*sp*cr) + (sy*sr)], 
                         [sy*cp, (sy*sp*sr) + (cy*cr), (sy*sp*cr) - (cy*sr)], 
                         [  -sp,                cp*sr,                cp*cr]])   
    
    def __init__(self):
        print("init bundle adjuster")

    
    # Reproject coordinates from world space to pixel space subject to
    #  X world coordinates
    #  (T, K) camera properties
    #  (w, h) image size
    def reproject(self, X, T, K, w, h, plot_reprojection = False):
        
        # Get camera properties
        R_camera = Bundle_adjuster.__get_R_camera(T[0], T[1], T[2])
        T_camera = np.array([T[3], T[4], T[5]])       
        f_x, f_y, k1, k2, k3, p1, p2 = K
        
        # Translate into camera reference frame
        X_camera = [np.matmul(R_camera, np.subtract(x, T_camera)) for x in X]
        #  Cull points too near the camera
        X_camera = [x for x in X_camera if x[2] > 0.01]
        
        
        # Project onto camera plane
        u_norm_camera = [np.array([f_x * X_c[0] / X_c[2], 
                                   f_y * X_c[1] / X_c[2]]) for X_c in X_camera]
        
            
        # Distortions
        u_norm_camera_r_sq = [np.square(np.add(np.square(u[0]), 
                                               np.square(u[1]))) for u in u_norm_camera]         

        #  Radial
        u_radial_corrections = [(k1 * u_r +
                                 k2 * np.power(u_r, 2) +
                                 k3 * np.power(u_r, 3)) for u_r in u_norm_camera_r_sq]
        u_c_r = [np.array([u[0] * u_rad_corr, 
                           u[1] * u_rad_corr]) for u, u_rad_corr in zip(u_norm_camera, u_radial_corrections)]        
        #  Tangential
        u_c_t = [np.array([(2*p1*u[0]*u[1]) + (p2*(r + (2*np.square(u[0])))),
                           (p1*(r + (2*np.square(u[1])))) + (2*p2*u[0]*u[1])]) for u,r in zip(u_norm_camera,u_norm_camera_r_sq)]
        #  Apply distortions
        u_norm_camera = [np.array([u[0] + cr[0] + ct[0],
                                   u[1] + cr[1] + ct[1]]) for (u, cr, ct) in zip(u_norm_camera, u_c_r, u_c_t)]
            
            
        # Project into pixel plane
        w_half, h_half = w/2, h/2
        p_image = [np.array([w*u_x + w_half,h*u_y + h_half]) for (u_x, u_y) in u_norm_camera]
        
        if plot_reprojection:
            # Plot points for debugging purposes
            p_x, p_y = zip(*p_image)
            plt.scatter(p_x, p_y)
            plt.axis('equal')
            plt.xlim(0, w)
            plt.ylim(0, h)

        
        return p_image


    # Error in reprojection subject to 
    #  P pixel coords
    #  X world coordinates
    #  (T, K) camera properties
    #  (w, h) image size
    def reprojection_error(self, P, X, T, K, w, h):
        residules = np.subtract(P, self.reproject(X, T, K, w, h))
        residules_sq = np.sum(np.square(residules), axis=1)
        print(residules_sq)
    
    
    def corrections(self, P, X, T, K, w, h):
        # Compute derivation matrices
        print("hello")
        
    

def main():
    ba = Bundle_adjuster()
    
    X = [np.array([1,0,2]).T,
         np.array([0,1,2]).T,
         np.array([0,0,3]).T,
         np.array([1,1,4]).T]
    
    
#    P = [np.array([990, 370]).T,
#         np.array([650, 715]).T,
#         np.array([635, 370]).T,
#         np.array([825, 530]).T]
#    
    
    # Camera extrinsic parameters       
    #  (roll, yaw, pitch, x, y, z)
    T_camera = np.array([[0, 0, 0, 0.0, 0, 0],
                         [0, 0, 0, 0.2, 0, 0],
                         [0, 0, 0, 0.4, 0, 0],
                         [0, 0, 0, 0.6, 0, 0],
                         [0, 0, 0, 0.8, 0, 0]])
    T_camera_1 = T_camera[0]
    
    # Camera intrinsic parameters
    #  (focal (x, y), radial distortion (2, 4, 6), tangential distortion (x, y))
    K_camera = np.array([9/16, 1, 0, 0, 0, 0, 0])

    #ba.reprojection_error(P, X, T_camera, K_camera, 1280, 720)
    
    print(ba.reproject(X, T_camera_1, K_camera, 1280, 720, True))
    
    
if __name__ == '__main__':
    main()
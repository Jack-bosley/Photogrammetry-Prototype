# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

from camera_model import Camera

class Bundle_adjuster:
    
    def __init__(self):
        print("init bundle adjuster")

    
    # Reproject coordinates from world space to pixel space subject to
    #  X point world coordinates
    #  T extrinsic camera properties
    #  C camera model
    def reproject(self, X, T, C : Camera, plot_reprojection = False):
        
        # Get camera properties for each frame
        R_camera = Camera.rotation_matrix(T[0], T[1], T[2])
        T_camera = np.array([T[3], T[4], T[5]])
        f_x, f_y = C.focal_lengths()
        k1, k2, k3 = C.radial_distortion_coefficients()
        p1, p2 = C.tangential_distortion_coefficients()
        w, h = C.image_resolution()
        
        
        # Translate into camera reference frames
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
        p_image = [np.array([w * (u_x + 0.5),h * (u_y + 0.5)]) for (u_x, u_y) in u_norm_camera]
        
        if plot_reprojection:
            # Plot points for debugging purposes
            p_x, p_y = zip(*p_image)
            plt.scatter(p_x, p_y)
            plt.axis('equal')
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.show()

        
        return p_image
    
    


    # Error in reprojection subject to 
    #  P pixel coords
    #  X world coordinates
    #  (T, K) camera properties
    #  (w, h) image size
    def reprojection_errors(self, P, X, T, C):
        P_reprojected = [self.reproject(X, T_j, C) for T_j in T]
        residules = np.subtract(P, P_reprojected)
        
        x_res, y_res = np.dsplit(residules,2)
  
        # Returns MxN array for M cameras and N points
        new_shape = (len(T), len(X))
        return np.reshape(x_res, new_shape), np.reshape(y_res, new_shape)
    
    
    def corrections(self, P, X, T, C):
        # Compute derivatives
        x_err, y_err = self.reprojection_errors(P, X, T, C)
        
        optimize.minimize()
        
        print(x_err)
        
        
    

def main():
    ba = Bundle_adjuster()
    C = Camera(9/16, 1, 0, 0, 0, 0, 0, 1280, 720)
    
    number_of_points = 4
    number_of_cameras = 5
    
    # Use solutions and reprojection to set a dummy problem
    _X = [np.array([1,0,2]).T,
          np.array([0,1,2]).T,
          np.array([0,0,3]).T,
          np.array([1,1,4]).T]
    
    _T_camera = [[0,     0, 0, 0.0, 0, 0],
                 [0,  -0.1, 0, 0.2, 0, 0],
                 [0, -0.15, 0, 0.4, 0, 0],
                 [0, -0.25, 0, 0.6, 0, 0],
                 [0,  -0.3, 0, 0.8, 0, 0]]
    
    ba.reproject(_X, _T_camera[0], C, True)
    
    
#    
#    # Attempt to recover _X, _T, _K without knowing them from P
#    P = [ba.reproject(_X, _T_camera, C) for _T_camera in _T_camera]
#
#    X_guess = [np.array([0, 0, 1]).T for i in range(number_of_points)]
#    T_camera_guess = [[0, 0, 0, 0, 0, 0] for i in range(number_of_cameras)]
#    
#    ba.corrections(P, X_guess, T_camera_guess, C)
#    
    
    
if __name__ == '__main__':
    main()
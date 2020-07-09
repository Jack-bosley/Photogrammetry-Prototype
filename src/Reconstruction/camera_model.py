# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:34:35 2020

@author: Jack
"""

import autograd.numpy as np
import matplotlib.pyplot as plt

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
    
    
    # Returns pixel coordinates of projection of X with camera at T
    def project(self, X, T):
                
        # Get camera properties
        R_camera = Camera.rotation_matrix(T[0], T[1], T[2])
        T_camera = np.array([T[3], T[4], T[5]])
        
        
        # Translate into camera reference frames
        X_camera = np.matmul(R_camera, np.subtract(X, T_camera))
        

        # Project onto camera plane
        u_norm_camera = np.array([self.f_x * X_camera[0] / X_camera[2], 
                                  self.f_y * X_camera[1] / X_camera[2]])
        
            
        # Distortions
        u_norm_camera_r_sq = np.square(np.add(np.square(u_norm_camera[0]), 
                                              np.square(u_norm_camera[1])))         

        #  Radial
        u_radial_correction = (self.k1 * u_norm_camera_r_sq +
                               self.k2 * np.power(u_norm_camera_r_sq, 2) +
                               self.k3 * np.power(u_norm_camera_r_sq, 3))
        u_c_r = np.array([u_norm_camera[0] * u_radial_correction, 
                          u_norm_camera[1] * u_radial_correction])
        #  Tangential
        u_c_t = np.array([(2*self.p1*u_norm_camera[0]*u_norm_camera[1]) + 
                          (self.p2*(u_norm_camera_r_sq + (2*np.square(u_norm_camera[0])))),
                          (self.p1*(u_norm_camera_r_sq + (2*np.square(u_norm_camera[1])))) + 
                          (2*self.p2*u_norm_camera[0]*u_norm_camera[1])])
    
        #  Apply distortions
        u_norm_camera = np.array([u_norm_camera[0] + u_c_r[0] + u_c_t[0],
                                  u_norm_camera[1] + u_c_r[1] + u_c_t[1]])
            
            
        # Project into pixel plane
        p_image = np.array([self.w * (u_norm_camera[0] + 0.5),
                            self.h * (u_norm_camera[1] + 0.5)])
    
        return p_image


    def reproject_all(self, X, T, plot_reprojection=False):
        reprojection = np.array([[self.project(X, T) for X in X] for T in T])
             
        # Plot points for debugging purposes
        if plot_reprojection:
            for p in reprojection:
                p_x, p_y = zip(*p)
                plt.scatter(p_x, p_y)
                plt.axis('equal')
                plt.xlim(0, self.w)
                plt.ylim(0, self.h)
                plt.show()
            
        return reprojection
    
    # Getters for grouped variables
    def focal_lengths(self):
        return self.f_x, self.f_y
    
    def radial_distortion_coefficients(self):
        return self.k1, self.k2, self.k3
    
    def tangential_distortion_coefficients(self):
        return self.p1, self.p2
    
    def image_resolution(self):
        return self.w, self.h
        
    def jacobian(self, X, T):
        
        def dU_dX(X_c_prime):
            X_x, X_y, X_z = X_c_prime
            f_x, f_y = self.f_x, self.f_y
            return np.array([f_x * ((R_camera[0] * X_z) - (X_x * R_camera[2])), 
                             f_y * ((R_camera[1] * X_z) - (X_y * R_camera[2]))]) / (X_z ** 2)
        
        def dUr_dX(U, U_X):
            k1, k2, k3 = self.k1, self.k2, self.k3          
            u_r_corr_X = ((k1 * r_sq_X) +
                          (k2 * 2 * r_sq * r_sq_X) + 
                          (k3 * 3 * np.square(r_sq) * r_sq_X))
            return np.multiply(u_r_corr, U_X) + np.outer(U.T, u_r_corr_X)
        
        def dUt_dX(U, U_X):
            p1, p2 = self.p1, self.p2
            a = ((U[0]*U_X[1])+(U[1]*U_X[0]))
            return np.array([(2*p1*a) + (p2*(r_sq_X + (4*U[0]*U_X[0]))),
                             (p1*(r_sq_X + (4*U[1]*U_X[1]))) + (2*p2*a)])
    
    
    
        def dU_dR(X_c, X_c_prime):
            X_x, X_y, X_z = X_c_prime
            f_x, f_y = self.f_x, self.f_y
            
            return np.array([f_x * np.concatenate((X_z * X_c, [0, 0, 0], -X_x * X_c)),
                             f_y * np.concatenate(([0, 0, 0], X_z * X_c, -X_y * X_c))]) / (X_z ** 2)

        def dUr_dR(U, U_R):
            k1, k2, k3 = self.k1, self.k2, self.k3          
            u_r_corr_R = ((k1 * r_sq_R) +
                          (k2 * 2 * r_sq * r_sq_R) + 
                          (k3 * 3 * np.square(r_sq) * r_sq_R))
            return np.multiply(u_r_corr, U_R) + np.outer(U.T, u_r_corr_R)
        
        def dUt_dR(U, U_R):
            p1, p2 = self.p1, self.p2
            a = ((U[0]*U_R[1])+(U[1]*U_R[0]))
            return np.array([(2*p1*a) + (p2*(r_sq_R + (4*U[0]*U_R[0]))),
                             (p1*(r_sq_R + (4*U[1]*U_R[1]))) + (2*p2*a)])
    
        def dR_dE():
            sa, ca = np.sin(T[0]), np.cos(T[0])
            sb, cb = np.sin(T[1]), np.cos(T[1])
            sc, cc = np.sin(T[2]), np.cos(T[2])

            return np.array([np.array([-sa*cb, -ca*sb, 0]),
                             np.array([-(sa*sb*sc)-(ca*cc), ca*cb*sc, (ca*sb*cc)+(sa*sc)]),
                             np.array([-(sa*sb*cc)+(ca*sc), ca*cb*cc,-(ca*sb*sc)+(sa*cc)]),
                             np.array([ ca*cb, -sa*sb, 0]),
                             np.array([ (ca*sb*sc)-(sa*cc), sa*cb*sc, (sa*sb*cc)-(ca*sc)]),
                             np.array([ (ca*sb*cc)+(sa*sc), sa*cb*cc,-(sa*sb*sc)-(ca*cc)]),
                             np.array([ 0, -cb, 0]),
                             np.array([ 0, -sb*sc, cb*cc]),
                             np.array([ 0, -sb*cc,-cb*sc])])
    
        # Compute values used in Jacobian
        R_camera = Camera.rotation_matrix(T[0], T[1], T[2])
        T_camera = np.array([T[3], T[4], T[5]])
        
        X_c = np.subtract(X, T_camera)
        X_c_prime = np.matmul(R_camera, X_c)
        
        U = np.array([self.f_x * X_c_prime[0] / X_c_prime[2], 
                      self.f_y * X_c_prime[1] / X_c_prime[2]])
    
        r_sq = np.square(np.add(np.square(U[0]), np.square(U[1])))
        u_r_corr = ((self.k1 * r_sq) + (self.k2 * np.power(r_sq, 2)) + (self.k3 * np.power(r_sq, 3)))
        
        

        
        # Compute the derivatives with respect to the point position
        U_X = dU_dX(X_c_prime)
        r_sq_X = 4 * np.multiply(np.dot(U, U), np.dot(U, U_X))
        Ur_X = dUr_dX(U, U_X)
        Ut_X = dUt_dX(U, U_X)
        f_X = U_X + Ur_X + Ut_X; f_X[0] *= self.w; f_X[1] *= self.h
        
        # Derivatives with respect to camera position is negative of f_X
        f_C = -f_X
        
        # Derivatives with respect to camera orientation
        U_R = dU_dR(X_c, X_c_prime)
        r_sq_R = 4 * np.multiply(np.dot(U, U), np.dot(U, U_R))
        Ur_R = dUr_dR(U, U_R)
        Ut_R = dUt_dR(U, U_R)
        f_R = U_R + Ur_R + Ut_R; f_R[0] *= self.w; f_R[1] *= self.h
        f_E = np.matmul(f_R, dR_dE())
        
        # Concatenate into 1 long jacobian matrix and return
        Jacobian = np.concatenate([f_X, f_E, f_C], axis=1)
        return Jacobian
        
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
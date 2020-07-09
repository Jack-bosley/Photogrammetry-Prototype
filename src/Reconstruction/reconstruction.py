# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import autograd as ag


from camera_model import Camera

class Bundle_adjuster:
    
    def __init__(self):
        print("init bundle adjuster")


    def corrections(self, P, X, T, C:Camera):
        
        # Lambda for reprojection taking concatenated coordinates
        reprojection = lambda _X: C.project(_X[:3], _X[3:])
        # Jacobian of reprojection
        proj_jac = ag.jacobian(reprojection)

        
#        # Store errors
#        r = [[np.zeros((1, 2)) for i in range(len(X))] for j in range(len(T))]
#        
#        
#        A = [[np.zeros((2, 6)) for i in range(len(X))] for j in range(len(T))]
#        B = [[np.zeros((2, 3)) for i in range(len(X))] for j in range(len(T))]
#
#        U =  [np.zeros((3, 3)) for i in range(len(T))]
#        V =  [np.zeros((6, 6)) for j in range(len(X))]
#        W = [[np.zeros((3, 6)) for i in range(len(X))] for j in range(len(T))]

        # For every point
        for i, _X in enumerate(X):
            # For every camera
            for j, _T in enumerate(T):

                # Get the error vector
                #r[j][i] = P[j, i] - C.project(_X, _T)
                
                # Get the jacobian for position and camera position dependency
                concatenated_params = np.concatenate((_X, _T))
                
                t0 = time.time()
                J_all = np.array(proj_jac(concatenated_params))
                
                t1 = time.time()
                
                f_R_ja = C.jacobian(_X, _T)
                
                t2 = time.time()
                
                print(t1 - t0)
                print(np.round(J_all, 1))
                
                print(t2 - t1)
                print(np.round(f_R_ja, 1))
                
                break
            break
        
#                # Store the jacobian in the 
#                A[j][i] = J_all.T[:3].T
#                B[j][i] = J_all.T[3:].T
#                
#                
#                U[j] += np.matmul(A[j][i].T, A[j][i])
#                V[i] += np.matmul(B[j][i].T, B[j][i])
#                W[j][i] = np.matmul(A[j][i].T, B[j][i])
#                
##                H_pp = np.matmul(J_p.T, J_p)
##                H_pc = np.matmul(J_p.T, J_c)
##                H_cc = np.matmul(J_c.T, J_c)
#        
        
                
        
    

def main():
    ba = Bundle_adjuster()
    C = Camera(9/16, 1, 2, 3, 4, 1, 1, 1280, 720)
    
    number_of_points = 4
    number_of_cameras = 3
    
    # Use solutions and reprojection to set a dummy problem
    _X = [np.array([ 0,0,5]).T,
          np.array([ 0,1,5]).T,
          np.array([ 1,0,4]).T,
          np.array([ 1,1,4]).T]
    
    _T_camera = [[0, 0, -0.1, 0.0, 0, 0],
                 [0, 0, -0.2, 0.2, 0, 0],
                 [0, 0, -0.4, 0.4, 0, 0]]
    
    # Attempt to recover _X, _T, _K without knowing them, from P
    P = C.reproject_all(_X, _T_camera, False)
    
    X_guess = [np.array([0., 0., 1.]).T for i in range(number_of_points)]
    T_camera_guess = [np.array([0.8, 0.5, 0.2, 1, 0, 0.]) for i in range(number_of_cameras)]
    ba.corrections(P, X_guess, T_camera_guess, C)
    
    
    
if __name__ == '__main__':
    main()
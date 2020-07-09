# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


from camera_model import Camera

class Bundle_adjuster:
    
    def __init__(self):
        print("init bundle adjuster")


    def corrections(self, P, X, T, C:Camera, damping = 10):

        # Store errors
        r = [[np.zeros((1, 2)) for i in range(len(X))] for j in range(len(T))]
        
        # Jacobians
        A = [[np.zeros((2, 6)) for i in range(len(X))] for j in range(len(T))]
        B = [[np.zeros((2, 3)) for i in range(len(X))] for j in range(len(T))]

        # Temporal variables
        U =  [np.zeros((6, 6)) for i in range(len(T))]
        V =  [np.zeros((3, 3)) for j in range(len(X))]
        W = [[np.zeros((6, 3)) for i in range(len(X))] for j in range(len(T))]      
        r_c = [np.zeros((1, 6)) for j in range(len(T))]
        r_p = [np.zeros((1, 3)) for i in range(len(X))]
        
        
        
        # For every point
        for i, _X in enumerate(X):
            # For every camera
            for j, _T in enumerate(T):

                # Get the error vector
                r[j][i] = P[j, i] - C.project(_X, _T)
                
                # Compute and store the jacobian in A and B
                jacobi = C.jacobian(_X, _T).T
                A[j][i] = jacobi[3:].T
                B[j][i] = jacobi[:3].T
                
                # Compute temporal variables
                U[j] += np.matmul(A[j][i].T, A[j][i])
                V[i] += np.matmul(B[j][i].T, B[j][i])
                W[j][i] = np.matmul(A[j][i].T, B[j][i])
                
                r_c[j] += np.matmul(A[j][i].T, r[j][i])
                r_p[i] += np.matmul(B[j][i].T, r[j][i])
        
        
        # Augment U and V by the LM damping coefficient
        for u in U:
            u += damping * np.identity(6)
        for v in V:
            v += damping * np.identity(3)
        
        
        # More temporal variables
        T = [[np.matmul(W[j][i], np.linalg.inv(V[i])) for i in range(len(X))] for j in range(len(T))]
        

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
    T_camera_guess = [np.array([0, 0, 0, 1, 0, 0.]) for i in range(number_of_cameras)]
    ba.corrections(P, X_guess, T_camera_guess, C)
    
    
    
if __name__ == '__main__':
    main()
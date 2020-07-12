# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


from Reconstruction.camera_model import Camera

class Bundle_adjuster:
    
    def __init__(self, presence, pixel_coords, C:Camera, X_guess=None, T_guess=None):
        
        self.presence = presence
        self.pixel_coords = pixel_coords
        self.C = C
        
        self.n_cameras = np.shape(pixel_coords)[0]
        self.n_points = np.shape(pixel_coords)[1]
        
        
        self.X = [np.array([0, 0, 5.]).T for i in range(self.n_points)] if X_guess == None else X_guess
        self.T = [np.array([0, 0, 0, 0, 0, 0.]) for i in range(self.n_cameras)] if T_guess == None else T_guess
        

    def optimise(self, num_steps=5):
        damping = 1.0
        
        c0 = self.cost()
        
        for i in range(num_steps):
            print(i)
            dp, dc = self.corrections(damping)
        
            self.update_guess(dp, dc)
            
            if c0 > self.cost():
                damping /= 3
            else:
                damping *= 2
                self.revert_guess(dp, dc)
              
        
    def update_guess(self, dp, dc):       
        for i in range(self.n_points):
            self.X[i] = np.squeeze(np.subtract(self.X[i], dp[i]))
        for j in range(self.n_cameras):
            self.T[j] = np.squeeze(np.subtract(self.T[j], -dc[j]))
    def revert_guess(self, dp, dc):
        for i in range(self.n_points):
            self.X[i] = np.squeeze(np.add(self.X[i], dp[i]))
        for j in range(self.n_cameras):
            self.T[j] = np.squeeze(np.add(self.T[j], -dc[j]))


    def cost(self):
        P_guess = self.C.reproject_all(self.X, self.T)
        
        error = np.subtract(self.pixel_coords, P_guess)
        absents = np.where(self.presence == False)
        error[absents] = np.array([0, 0])
        return np.sum(np.square(self.robustifier(error)))


    def robustifier(self, x, sigma=0.3):
        return np.log(np.add(1, np.divide(np.square(np.divide(x, sigma)), 2)))

    def corrections(self, damping):

        # Store errors
        r = [[np.zeros((1, 2)) for i in range(self.n_points)] for j in range(self.n_cameras)]
        r_robust = [[0 for i in range(self.n_points)] for j in range(self.n_cameras)]
        
        # Jacobians
        A = [[np.zeros((2, 6)) for i in range(self.n_points)] for j in range(self.n_cameras)]
        B = [[np.zeros((2, 3)) for i in range(self.n_points)] for j in range(self.n_cameras)]

        # Temporal variables
        U =  [np.zeros((6, 6)) for j in range(self.n_cameras)]
        V =  [np.zeros((3, 3)) for i in range(self.n_points)]
        W = [[np.zeros((6, 3)) for i in range(self.n_points)] for j in range(self.n_cameras)]      
        r_c = [np.zeros((1, 6)) for j in range(self.n_cameras)]
        r_p = [np.zeros((1, 3)) for i in range(self.n_points)]
        
        
        
        # For every point
        for i, _X in enumerate(self.X):
            # For every camera
            for j, _T in enumerate(self.T):

                # Get the error vector
                r[j][i] = self.pixel_coords[j][i] - self.C.project(_X, _T) if self.presence[j][i] else np.array([0, 0])
                dx, dy = r[j][i]
                r_robust[j][i] = self.robustifier(np.sqrt(dx**2 + dy**2))
                
                # Compute and store the jacobian in A and B
                jacobi = self.C.jacobian(_X, _T).T
                A[j][i] = -jacobi[3:].T
                B[j][i] = -jacobi[:3].T
                
                # Compute temporal variables
                U[j] += r_robust[j][i] * np.matmul(A[j][i].T, A[j][i])
                V[i] += r_robust[j][i] * np.matmul(B[j][i].T, B[j][i])
                W[j][i] = r_robust[j][i] * np.matmul(A[j][i].T, B[j][i])
                
                r_c[j] += r_robust[j][i] * np.matmul(A[j][i].T, r[j][i])
                r_p[i] += r_robust[j][i] * np.matmul(B[j][i].T, r[j][i])
        
        
        # Augment U and V by the LM damping coefficient
        for u in U:
            u += damping * np.identity(6)
        for v in V:
            v += damping * np.identity(3)
        
        
        # More temporal variables
        V_inv = [np.linalg.inv(V[i]) for i in range(self.n_points)]
        T = [[np.matmul(W[j][i], V_inv[i]) for i in range(self.n_points)] for j in range(self.n_cameras)]
        
        S = [[np.zeros((6, 6)) for i in range(self.n_cameras)] for k in range(self.n_cameras)]
        for j in range(self.n_cameras):
            for k in range(self.n_cameras):
                for i in range(self.n_points):
                    S[k][j] -= np.matmul(T[k][i], W[j][i].T)      

            S[j][j] -= U[j]

        r_j = [r_c[j] for j in range(self.n_cameras)]
        for i in range(self.n_points):
            for j in range(self.n_cameras):
                r_j[j] -= np.matmul(T[j][i], r_p[i].T).T
        
        # Compute camera updates
        A = np.block(S)
        dc = np.reshape(np.matmul(np.linalg.inv(A), np.block(r_j).T), (self.n_cameras, 6))
        
        dc[0] = [0,0,0,0,0,0]
        
        # Compute point updates
        dp = [np.zeros((1, 3)) for i in range(self.n_points)]
        for i in range(self.n_points):
            Wdc = np.zeros((1, 3))
            for j in range(self.n_cameras):
                Wdc += np.matmul(W[j][i].T, dc[j])
            
            dp[i] = np.matmul(V_inv[i], (r_p[i] - Wdc).T).T
        
        dp[0] = [0,0,0]
        
        return dp, dc
                    
    def __str__(self):
        string = "Bundle Adjuster \n"
        for x in self.X:
            _x = np.round(x, 1)
            string += ("Point position (%.1f, %.1f, %.1f)\n" % (_x[0], _x[1], _x[2]))
        for t in self.T:
            _t = np.round(t, 1)
            string += ("Rotation (%.1f, %.1f, %.1f),\tPosition(%.1f, %.1f, %.1f)\n" % (_t[0], _t[1], _t[2], _t[3], _t[4], _t[5]))
        return string

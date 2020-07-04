# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:40:45 2020

@author: Jack
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage.filters as filters
from scipy.optimize import minimize
from scipy.spatial import KDTree
import time

# Sobel kernels
#------------------------------------------------------------------------------
d_dx = np.array([[ -1,  0,  1],
                 [ -2,  0,  2],
                 [ -1,  0,  1]])

d_dy = np.array([[ -1, -2, -1],
                 [  0,  0,  0],
                 [  1,  2,  1]])

d_dxx = np.array([[  1,  0, -2,  0,  1],
                  [  4,  0, -8,  0,  4],
                  [  6,  0,-12,  0,  6],
                  [  1,  0, -2,  0,  1],
                  [  4,  0, -8,  0,  4]])

d_dyy = np.array([[  1,  4,  6,  4,  1],
                  [  0,  0,  0,  0,  0],
                  [ -2, -8,-12, -8, -2],
                  [  0,  0,  0,  0,  0],
                  [  1,  4,  6,  4,  1]])

d_dxy = np.array([[ -1, -2,  0,  2,  1],
                  [ -2, -4,  0,  4,  2],
                  [  0,  0,  0,  0,  0],
                  [  2,  4,  0, -4, -2],
                  [  1,  2,  0, -2,  1]])

# laplace
l_xy = np.array([[ -1, -1, -1],
                 [ -1,  8, -1],
                 [ -1, -1, -1]])

# Finding common features
#------------------------------------------------------------------------------
def get_corners_fast(image, plot_Harris_response = False):
    d = 8
    k = 0.04
        
    # Contruct the aperture to convolve with the gradients for the structure tensor
    aperture = np.outer(signal.gaussian(d, d/2), signal.gaussian(d, d/2))


    # Get image gradients
    I_dx = signal.fftconvolve(image, d_dx, mode='same')
    I_dy = signal.fftconvolve(image, d_dy, mode='same')
    
    
    # Compute squares of gradients 
    I_xx = np.square(I_dx)
    I_yy = np.square(I_dy)
    I_xy = np.multiply(I_dx, I_dy)
    

    # Structure tensor elements can be found by convolving
    #  squared gradients with aperture function
    A_11 = signal.fftconvolve(I_xx, aperture, mode='same')
    A_12 = signal.fftconvolve(I_xy, aperture, mode='same')
    A_22 = signal.fftconvolve(I_yy, aperture, mode='same')
    

    # Compute harris response
    detM = np.multiply(A_11, A_22) - np.square(A_12)
    trM = np.add(A_11, A_12)
    M = detM - k * np.square(trM)
    M = np.subtract(M, np.min(M))
    M = np.divide(M, np.max(M))

    # Find the local maxima
    max_values = filters.maximum_filter(M, d)
    rows, cols = np.where(M == max_values)
    
    
    # Find intensity of peaks
    #  and ilter low intensity points
    M_intensity = signal.fftconvolve(M, l_xy, mode='same')
    avg = np.average(M_intensity)
    high_contrast_points = np.where(M_intensity[rows, cols] > 2*avg)
    rows = [rows[i] for i in high_contrast_points][0]
    cols = [cols[i] for i in high_contrast_points][0]


    # Plot the harris response graph if desired
    if plot_Harris_response:
        plt.contourf(M_intensity, 50)
        plt.colorbar()
        plt.axis('equal')
        plt.show()
    
    # Return best points
    return list(zip(rows, cols))



# Guess which features correspond to the same point in 3d space
#------------------------------------------------------------------------------
def match_features(features_1, features_2):
    
    threshold_distance = 5
    threshold_ratio = 0.7
    
    # Construct a KD tree with the first image's feature points
    T1 = KDTree(features_1)
    
    # Find the closest points in 1 to the points in 2
    pair_distances_k2, paired_2_k2 = T1.query(features_2, 2)
    paired_2 = list(zip(*paired_2_k2))[0]
    neighbour_distance_ratio = np.array([p_d[0] / p_d[1] for p_d in pair_distances_k2])
    neighbour_distance = np.array([p_d[0] for p_d in pair_distances_k2])
    
    # Cull all pairs with neighbour distance ratios greater than threshold_ratio
    #  and greater separation than threshold_distance
    pairs = [(j, i) for i, j in enumerate(paired_2)]
    valid_pairs = list(np.where((neighbour_distance_ratio < threshold_ratio) & 
                                (neighbour_distance < threshold_distance))[0])
    pairs = [pairs[i] for i in valid_pairs]
    confidences = [1/(neighbour_distance_ratio[i] + 0.01) for i in valid_pairs] 

    return zip(*reversed(sorted(zip(confidences, pairs))))


# Maintain the dictionary of features
#------------------------------------------------------------------------------
def initialise_feature_dictionary(feature_histories, feature_weights, 
                                  features_1, features_2):
    
    # Attempt to match features between first and second set
    confidences, pairs = match_features(features_1, features_2)

    for i,data in enumerate(zip(confidences, pairs)):
        c, p = data
        p1, p2 = p
        
        feature_weights[i] = c
        feature_histories[i] = [features_1[p1], features_2[p2]]
        
    print(feature_histories)


def update_feature_dictionary(feature_histories, feature_weights,
                              features):
    print("update")
















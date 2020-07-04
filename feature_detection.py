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
def match_features(features_prev, features_curr):
    
    threshold_distance = 10
    threshold_ratio = 0.4
    
    # Construct a KD tree with the current image's feature points
    T1 = KDTree(features_curr)
    
    # Find the closest 2 points in current to the points in prev
    paired_distances, paired_current = T1.query(features_prev, 2)
    paired_curr_closest = list(zip(*paired_current))[0]
    neighbour_distance_ratio = np.array([p_d[0] / p_d[1] for p_d in paired_distances])
    neighbour_distance = np.array([p_d[0] for p_d in paired_distances])
    
    # Cull all pairs with neighbour distance ratios greater than threshold_ratio
    #  and greater separation than threshold_distance
    pairs = [(i, j) for i, j in enumerate(paired_curr_closest)]
    valid_pairs = list(np.where((neighbour_distance_ratio < threshold_ratio) & 
                                (neighbour_distance < threshold_distance))[0])
    pairs = [pairs[i] for i in valid_pairs]
    confidences = [1/((neighbour_distance_ratio[i] * neighbour_distance[i]) + 0.01) \
                   for i in valid_pairs] 
    
    # Sort by confidence
    confidences, pairs = zip(*reversed(sorted(zip(confidences, pairs))))
    confidences = list(confidences)
    pairs = list(pairs)
    
    # Delete duplicates with lowest confidence
    pairs_prev, pairs_curr = zip(*pairs)
    dupes = [i for i,x in enumerate(pairs_curr) if pairs_curr.count(x) > 1]
    best_options = []
    deleted = 0
    for d in dupes:
        if pairs_curr[d] not in best_options:
            best_options.append(pairs_curr[d])
        else:
            del confidences[d - deleted]
            del pairs[d - deleted]
            deleted += 1

    return confidences, pairs


# Maintain the dictionary of features
#------------------------------------------------------------------------------
def initialise_feature_dictionary(feature_histories, feature_weights, 
                                  features_prev, features_curr):
    
    # Attempt to match features between first and second set
    confidences, pairs = match_features(features_prev, features_curr)

    # Add all pairs to the dictionary
    for i, data in enumerate(zip(confidences, pairs)):
        c, p = data
        p1, p2 = p
        
        feature_weights[i] = c
        feature_histories[i] = [features_prev[p1], features_curr[p2]]
        


def update_feature_dictionary(feature_histories, feature_weights,
                              features_prev, features_curr):
    
    # Get the coordinates of features in the dictionary from prev frame
    prev_matched_features = \
        list(list(zip(*list(feature_histories.values())))[-1])
    
    # Attempt to match features between current and dictionary
    confidences, pairs = match_features(features_curr, prev_matched_features)
    
    print(pairs)
    

    














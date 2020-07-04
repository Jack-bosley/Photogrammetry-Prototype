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


# Finding common features
#------------------------------------------------------------------------------
def get_dx(image):
    im_left  = np.array(image.transform(image.size, Image.AFFINE, 
                                        (1, 0, -1, 0, 1, 0)), dtype=np.int8)
    im_right = np.array(image.transform(image.size, Image.AFFINE, 
                                        (1, 0,  1, 0, 1, 0)), dtype=np.int8)

    return np.subtract(im_right, im_left).astype(np.int32)

def get_dy(image):
    im_up    = np.array(image.transform(image.size, Image.AFFINE, 
                                        (1, 0, 0, 0, 1,  1)), dtype=np.int8)
    im_down  = np.array(image.transform(image.size, Image.AFFINE, 
                                        (1, 0, 0, 0, 1, -1)), dtype=np.int8)

    return np.subtract(im_up, im_down).astype(np.int32)

def get_corners_fast(image, plot_Harris_response = False, contrast_grad_threshold=5):
    d = 8
    k = 0.05
        
    # Contruct the aperture to convolve with the gradients for the structure tensor
    aperture = np.outer(signal.gaussian(d, d/2), signal.gaussian(d, d/2))


    # Get image gradients
    I_x = get_dx(image)
    I_y = get_dy(image)
    
    
    # Compute squares of gradients 
    I_xx = np.square(I_x)
    I_yy = np.square(I_y)
    I_xy = np.multiply(I_x, I_y)
    

    # Structure tensor elements can be found by convolving
    #  squared gradients with aperture function
    A_11 = signal.fftconvolve(I_xx, aperture, mode='same')
    A_12 = signal.fftconvolve(I_xy, aperture, mode='same')
    A_22 = signal.fftconvolve(I_yy, aperture, mode='same')
    

    # Compute harris response
    detM = np.multiply(A_11, A_22) - np.square(A_12)
    trM = np.add(A_11, A_12)
    M = detM - k * np.square(trM)
    

    # Find the local maxima
    max_values = filters.maximum_filter(M, 4)
    rows, cols = np.where(M == max_values)
    
    # Filter low contrast points
    high_contrast_points = np.where(I_xy[rows, cols] > contrast_grad_threshold)
    rows_high_contrast = [rows[i] for i in high_contrast_points][0]
    cols_high_contrast = [cols[i] for i in high_contrast_points][0]
    
    points = list(zip(rows_high_contrast, cols_high_contrast))
    
    if plot_Harris_response:
        plt.contourf(M, 50)
        plt.show()
    
    # Return best points
    return points



# Guess which features correspond to the same point in 3d space
#------------------------------------------------------------------------------
def match_features(features_1, features_2):
    
    # Construct a KD tree with the first image's feature points
    T1 = KDTree(features_1)
    
    # Find the closest points in 1 to the points in 2
    pair_distances_k2, paired_2_k2 = T1.query(features_2, 2)
    paired_2 = list(zip(*paired_2_k2))[0]
    neighbour_distance_ratio = np.array([p_d[0] / p_d[1] for p_d in pair_distances_k2])
    
    # Cull all pairs with neighbour distance ratios greater than 0.5
    pairs = [(j, i) for i, j in enumerate(paired_2)]
    valid_pairs = list(np.where(neighbour_distance_ratio < 0.5)[0])
    pairs = [pairs[i] for i in valid_pairs]

    return pairs


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
def get_corners_fast(image, plot_Harris_response = False, dist_to_edge_threshold = -1):
    d = 8
    k = 0.04
    if dist_to_edge_threshold == -1:
        dist_to_edge_threshold = d 
        
    # Contruct the aperture to convolve with the gradients for the structure tensor
    aperture = np.outer(signal.gaussian(d, d/2), signal.gaussian(d, d/2))

#    t0 = time.time()

    # Get image gradients (small kernels prefer convolve2d)
    I_dx = signal.convolve2d(image, d_dx, mode='same')
    I_dy = signal.convolve2d(image, d_dy, mode='same')
      
    # Compute squares of gradients 
    I_xx = np.square(I_dx)
    I_yy = np.square(I_dy)
    I_xy = np.multiply(I_dx, I_dy)
    
#    t1 = time.time()

    # Structure tensor elements can be found by convolving
    #  squared gradients with aperture function
    A_11 = signal.fftconvolve(I_xx, aperture, mode='same')
    A_12 = signal.fftconvolve(I_xy, aperture, mode='same')
    A_22 = signal.fftconvolve(I_yy, aperture, mode='same')
    
#    t2 = time.time()

    # Compute harris response
    detM = np.multiply(A_11, A_22) - np.square(A_12)
    trM = np.add(A_11, A_12)
    M = detM - k * np.square(trM)
    M = np.subtract(M, np.min(M))
    M = np.divide(M, np.max(M))


    # Find the local maxima
    max_values = filters.maximum_filter(M, d)
    rows, cols = np.where(M == max_values)
    
#    t3 = time.time()
    
    # Find intensity of peaks
    #  and filter low intensity points
    M_intensity = signal.convolve2d(M, l_xy, mode='same')
    avg = np.average(M_intensity)
    high_contrast_points = np.where(M_intensity[rows, cols] > 2*avg)
    rows = [rows[i] for i in high_contrast_points][0]
    cols = [cols[i] for i in high_contrast_points][0]

#    t4 = time.time()

    # Filter out points on the edge
    r_max, c_max = np.shape(image)
    dist_to_edge = np.array([np.min([rows[i], 
                                     cols[i], 
                                     r_max - rows[i], 
                                     c_max - cols[i]]) \
                             for i in range(len(rows))])
    rows = rows[np.where(dist_to_edge - 1 > dist_to_edge_threshold)]
    cols = cols[np.where(dist_to_edge - 1 > dist_to_edge_threshold)]

#    t5 = time.time()
#
#    print("Image Gradients " + str(t1 - t0),
#          "\nConvolutions" + str(t2 - t1),
#          "\nHarris Response" + str(t3 - t2),
#          "\nFilter by laplace" + str(t4 - t3),
#          "\nFilter edge points" + str(t5 - t4) + "\n")

    # Plot the harris response graph if desired
    if plot_Harris_response:
        plt.contourf(M_intensity, 50)
        plt.colorbar()
        plt.axis('equal')
        plt.show()
    
    # Return best points
    return np.array(list(zip(rows, cols)))






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


# For finding derivatives in images
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



# Finding common features
#------------------------------------------------------------------------------                 
def get_corners_fast(image, plot_Harris_response = False, contrast_grad_threshold=10):
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


def compare(features_1, features_2, plot_feature_scatter=False): 
    # Computes the average squared distances between pairs of points
    def fit_score(delta, pairs):
        dx, dy = delta
        
        summed_sq_dist = 0
        for p in pairs:
            p1, p2 = features_1[p[0]], features_2[p[1]]
            x1, y1 = p1; x1 += dx; y1 += dy
            x2, y2 = p2
            
            summed_sq_dist += ((x1 - x2)**2) + ((y1 - y2)**2)
    
        return summed_sq_dist / len(pairs)
    
    
    def plot(delta, pairs):
        dx, dy = delta
        P1, P2 = zip(*pairs)
        
        F1 = [features_1[i] for i in P1]
        F2 = [features_2[i] for i in P2]
          
        x1, y1 = zip(*F1); x1 += dx; y1 += dy
        x2, y2 = zip(*F2)
        
        plt.scatter(y1, x1, marker='.')
        plt.scatter(y2, x2, marker='.')
        
#        d_x = np.subtract(x2, x1)
#        d_y = np.subtract(y2, y1)     
#        for data in zip(x1, y1, d_x, d_y):
#            x, y, dx, dy = data
#            dx *= 4; dy *= 4
#            plt.arrow(y, x, dy, dx)
        
        plt.axis('equal')
        plt.show()
    

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
    
    # Shift pairs around until optimal
    bnds = ((-50, 50), (-50, 50))
    res = minimize(fit_score, [0, 0], args=pairs, bounds=bnds, method='SLSQP')

    if plot_feature_scatter: 
        plot(res.x, pairs)

    

    return res.x

def compare_bad(features_1, features_2, dx_prev = 0, dy_prev = 0, plot = False):
    def fit_score(delta):
        dx, dy = delta
        
        SSR = 0
        for p1, w1 in zip(points_1, weights_1):
            DX = np.subtract(points_2[:][0], (p1[0] + dx))
            DY = np.subtract(points_2[:][1], (p1[1] + dy))
            
            R_sq = np.add(np.square(DX), np.square(DY))
            R_sq_min = np.min(R_sq)
            SSR += R_sq_min * w1  
        
        for p2, w2 in zip(points_2, weights_2):
            DX = np.subtract(points_1[:][0], (p2[0] - dx))
            DY = np.subtract(points_1[:][1], (p2[1] - dy))
            
            R_sq = np.add(np.square(DX), np.square(DY))
            R_sq_min = np.min(R_sq)
            SSR += R_sq_min * w2
        
        return SSR
            
    points_1, weights_1 = features_1  
    points_2, weights_2 = features_2
    
    bnds = ((-50, 50), (-50, 50))
    res = minimize(fit_score, [dx_prev, dy_prev], bounds=bnds, method='COBYLA')

    if plot:
        dy, dx = res.x
        x1, y1 = zip(*points_1)
        x2, y2 = zip(*points_2)
        
        plt.scatter(np.add(y1, -dy), np.add(x1, -dx))
        plt.scatter(y2, x2, marker='x')
        plt.axis('equal')
        plt.show()

    return res.x
    
    
    
## WORKS BUT VERY SLOW
def get_corners_true(image):
    d = 5
    k = 0.04
    
    def w():
        X = np.linspace(-d, d, num=(2*d)+1)
        XX, YY = np.meshgrid(X, X)
        return np.exp(-((XX**2) + (YY**2)) / (2 * ((d / 2)**2)))
    
    @np.vectorize
    def M(x, y):
        # Harris response function
        
        # Get region around current pixel
        _I_x = I_x[x-d-1:x+d, y-d-1:y+d]
        _I_y = I_y[x-d-1:x+d, y-d-1:y+d]
        
        # Structure tensor elements are averaged, weighted, local gradient values
        A_11 = np.sum(np.multiply(w, np.square(_I_x)))
        A_12 = np.sum(np.multiply(w, np.multiply(_I_x, _I_y)))
        A_22 = np.sum(np.multiply(w, np.square(_I_y)))
        
        # Harris response is smallest eigenvalue of tensor element,
        #  approximately det(A) - k*Tr(A)^2
        return ((A_11 * A_22) - (A_12 ** 2)) - (k * ((A_11 + A_22) ** 2))
    
    # Harris corner detection
    
    # Find derivatives
    I_x = get_dx(image)
    I_y = get_dy(image)  
    
    # Get aperture function
    w = w()
    
    # Meshgrid the image for parallel computation
    x, y = np.shape(image)
    X, Y = [i+d+1 for i in range(x-(2*d)-1)],[i+d+1 for i in range(y-(2*d)-1)]
    XX, YY = np.meshgrid(X, Y)
    
    # Get the Harris response
    M = M(XX, YY)
    
    # Normalise and return as an image
    M = np.divide(M, np.max(M) / 255)
    return Image.fromarray(M.T)
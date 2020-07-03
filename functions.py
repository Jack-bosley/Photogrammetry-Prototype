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
def get_corners_fast(image, use_fft = False):
    d = 16
    d2 = d*2
    k = 0.05
    
    width = image.width; height = image.height
    
    
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
    

    # Compute harris response (normalised)
    detM = np.multiply(A_11, A_22) - np.square(A_12)
    trM = np.add(A_11, A_12)
    M = detM - k * np.square(trM)
    

    # Find the local maxima
    max_values = filters.maximum_filter(M, d2)
    rows, cols = np.where(M == max_values)
    points = list(zip(rows, cols))


    # Compute normalised weights for each point
    weights = M[np.where(M == max_values)]
    distance_to_edge = 1 - ((np.abs(cols - (width / 2)) / width) +
                            (np.abs(rows - (height / 2)) / height))
    weights = np.multiply(weights, distance_to_edge)
    weights = np.divide(weights, np.max(weights))
     
    
    # Sort points by weights
    points_sorted = [p for _,p in reversed(sorted(zip(weights, points)))]
    weights_sorted = list(reversed(sorted(weights)))

    
    # Return best points
    return points_sorted[:20], weights_sorted[:20]
        


def compare(features_1, features_2, dx_prev = 0, dy_prev = 0, plot = False):
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
        plt.scatter(y2, x2)
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
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:22:53 2020

@author: Jack
"""

import ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage.filters as filters

# Brief classifier
class BRIEF_classifier: 
    
    # Initialise the classifier with datasets needed for classification
    def __init__(self, n, S): 
        self.n = n
        self.S = S
        
        # Generate the locations of the tests
        self.X1 = np.array(np.clip(np.random.normal(scale=0.04*(S**2), size=(n)), -S, S), 
                           dtype=np.int)
        self.Y1 = np.array(np.clip(np.random.normal(scale=0.04*(S**2), size=(n)), -S, S),
                           dtype=np.int)
        self.X2 = np.array(np.clip(np.random.normal(scale=0.04*(S**2), size=(n)), -S, S),
                           dtype=np.int)
        self.Y2 = np.array(np.clip(np.random.normal(scale=0.04*(S**2), size=(n)), -S, S),
                           dtype=np.int)
        
        # Create a kernel to smooth images
        self.smoothing_kernel = np.outer(signal.gaussian(7, 2), signal.gaussian(7, 2))

    # This classifier is function-like, return descriptors of all feature points in image
    def __call__(self, image, features):
        
        # Smooth the image to mitigate effects of noise
        image_smoothed = signal.fftconvolve(image, self.smoothing_kernel, mode='same')
            
        # Compare pixel values according to test locations
        #  encode comparison data in uint8's for compatibility with cv2 FLANN
        descriptor = [np.packbits(image_smoothed[fx + self.X1, fy + self.Y1] >
                                  image_smoothed[fx + self.X2, fy + self.Y2]).view(np.uint8) for (fx, fy) in features]

        # Return the description of the feature points
        return np.array(descriptor)
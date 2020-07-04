# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""

import os
import numpy as np
import time
from PIL import Image

from feature_detection import get_corners_fast, \
    initialise_feature_dictionary, update_feature_dictionary
from debugging import impose_features, impose_persistent_features

def get_images(directory, name):
    
    # Find all pictures containing name in directory
    files = os.listdir(directory)
    numbers = []
    for f in files:
        if name not in f:
            files.remove(f)
        else:
            numbers.append(int(f.split(name)[1].split('.')[0]))
    
    # Modify files to contain full path
    files = [directory + "/" + f for f in files]
    
    # Order by number in name and return 
    return zip(*sorted(zip(numbers, files)))
    

def main():  
    # Get all files in order
    numbers, files = get_images("Video2", "frame")
    
    
    feature_histories = {}
    feature_weights = {}
    
    features_prev = ""
    
    # Iterate through files
    scale_factor = 2
    for i, name in enumerate(files):
        # Open current image and scale down for speed
        image = Image.open(name)
        image_scaled = np.array(image.resize((image.width // scale_factor, 
                                              image.height // scale_factor)).convert("L"), 
                                dtype=np.int16)
                
        # Get the locations of the features
        features = get_corners_fast(image_scaled, False)  
        
        # If multiple images have been scanned for features, attempt to match them
        if i == 1:
            initialise_feature_dictionary(feature_histories, feature_weights, 
                                          features_prev, features)
        elif i > 1:
            update_feature_dictionary(feature_histories, feature_weights, 
                                      features)
            
            break

        # Store previous features
        features_prev = features
        
    impose_persistent_features(feature_histories, scale_factor, files[:2], "Video2/Imposed", "Corner")
    
if __name__ == '__main__':
    main()
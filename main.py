# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""

import os
import numpy as np
import time
from PIL import Image

from feature_detection import get_corners_fast
from feature_classifier import BRIEF_classifier
from feature_matching import Feature_dictionary
from debugging import impose_features

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
    directory = "Video1"
    stop_after = 1
    
    # Get all files in order
    numbers, files = get_images(directory, "frame")
    
    # Generate a BRIEF classifier
    brief = BRIEF_classifier(128, 25)
    
    # Iterate through files
    scale_factor = 2
    for i, name in enumerate(files):
        if i >= stop_after:
            break
        
        # Open current image and scale down for speed
        image = Image.open(name)
        image_scaled = np.array(image.resize((image.width // scale_factor, 
                                              image.height // scale_factor)).convert("L"), 
                                dtype=np.int16)
                
        # Get the locations and descriptors of the features
        feature_locations = get_corners_fast(image_scaled, False, brief.S)
        feature_descriptors = brief(image_scaled, feature_locations)

        print(feature_descriptors)

if __name__ == '__main__':
    main()
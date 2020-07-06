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
    stop_after = 1000
    
    # Get all files in order
    numbers, files = get_images(directory, "frame")
    
    
    # Generate a BRIEF classifier
    brief = BRIEF_classifier(128, 25)
    
    # Also keep track of all feature descriptors for matching against
    feature_dict = Feature_dictionary()
    
    
    locs, descs = [], []
    
    T_loc = []
    T_dec = []
    T_mat = []
    
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
    
    
        t_0 = time.time()
        # Get the locations and descriptors of the features
        feature_locations = get_corners_fast(image_scaled, False, brief.S)
        
        t_1 = time.time()
        
        feature_descriptors = brief(image_scaled, feature_locations)
        
        t_2 = time.time()
        
        feature_dict.update_dictionary(feature_descriptors)
        
        t_3 = time.time()
        
        T_loc.append(t_1 - t_0)
        T_dec.append(t_2 - t_1)
        T_mat.append(t_3 - t_2)
       
        
        locs.append(feature_locations)
        descs.append(feature_descriptors)
    
    print("\nLocate corners \t" + str(np.mean(T_loc)),
          "\nDescriptors of corners \t" + str(np.mean(T_dec)), 
          "\nMatch descriptors \t" + str(np.mean(T_mat)) + "\n") 
    
    for i in range(stop_after):
        impose_features(feature_dict, locs[i], descs[i], Image.open(files[i]),
                        scale_factor, directory+"/Imposed1", "Corner"+str(i))
    

if __name__ == '__main__':
    main()
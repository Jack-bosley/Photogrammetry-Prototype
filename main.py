# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""

import os
import numpy as np
import time
from PIL import Image

from feature_detection import get_corners_fast, match_features
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
    # Get all files in order
    numbers, files = get_images("Video2", "frame")
    
    # Iterate through files
    scale_factor = 2   
    fp1, fp2 = "", ""
    for i, name in enumerate(files):
        if i > 5:
            break
        
        # If the first image's features have been found, reassign to fp2
        if fp1 != "":
            fp2 = fp1
        
        # Open current image and scale down for speed
        image = Image.open(name)
        image_scaled = image.resize((image.width // scale_factor, 
                                     image.height // scale_factor)).convert("L")
                
        # Get the locations of the features
        fp1 = get_corners_fast(image_scaled, False)
        
        # If the second image's features have been found,
        #  compare current with previous features
        if fp2 != "":
            print(match_features(fp1, fp2))
            break
               
    
if __name__ == '__main__':
    main()
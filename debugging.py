# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:39:31 2020

@author: Jack
"""

import os
from PIL import Image, ImageDraw, ImageFont
from feature_matching import Feature_dictionary

def impose_features(feature_dictionary, feature_points, feature_descriptors, 
                    image, scale_factor = 1, directory = "", name="Corners"):

    # Create directory to save images to if does not exist
    if directory != "" and not os.path.isdir(directory):
        os.mkdir(directory)


    # Set up image for drawing
    d = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("arial.ttf", 20)
    dot_size = 2
    
    
    # Iterate over all features
    for i, data in enumerate(feature_points):
        # Scale position to image
        r_s, c_s = data
        r = r_s * scale_factor
        c = c_s * scale_factor        
        
        
        # Get feature index in dictionary
        feature_number = feature_dictionary.descriptor_index(feature_descriptors[i])
        
        # Draw feature on picture
        d.line([(c-dot_size, r-dot_size), (c+dot_size, r+dot_size)], width=1)
        d.line([(c-dot_size, r+dot_size), (c+dot_size, r-dot_size)], width=1)        
        d.text((c, r + 10), str(feature_number), font=fnt)
    
    image.save(((directory + "/") if directory != "" else "") + (name + ".bmp"))
    image.close()

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:39:31 2020

@author: Jack
"""

import os
from PIL import Image, ImageDraw, ImageFont

def impose_features(feature_points, image, scale_factor = 1, directory = "", name="Corners"):

    if directory != "" and not os.path.isdir(directory):
        os.mkdir(directory)

    rows, cols = zip(*feature_points)
    
    d = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("arial.ttf", 20)
    dot_size = 5
    for i, data in enumerate(zip(rows, cols)):
        
        r_s, c_s = data
        r = r_s * scale_factor
        c = c_s * scale_factor        
        
        d.line([(c-dot_size, r-dot_size), (c+dot_size, r+dot_size)], width=2)
        d.line([(c-dot_size, r+dot_size), (c+dot_size, r-dot_size)], width=2)        
        d.text((c, r + 10), str(i), font=fnt)
    
    image.save(((directory + "/") if directory != "" else "") + (name + ".bmp"))
    image.close()


def impose_persistent_features(feature_histories, scale_factor, files, directory, name):
    
    if directory != "" and not os.path.isdir(directory):
        os.mkdir(directory)
        
    # Drawing data
    fnt = ImageFont.truetype("arial.ttf", 20)
    dot_size = 5
    
    # Iterate over all files considered
    for i, f in enumerate(files):
        # Open the image for editing
        image = Image.open(f)
        d = ImageDraw.Draw(image)
        
        # Iterate over all feature points
        for j in feature_histories:
            r_s, c_s = feature_histories[j][i]
            r = r_s * scale_factor
            c = c_s * scale_factor  
            
            d.line([(c-dot_size, r-dot_size), (c+dot_size, r+dot_size)], width=2)
            d.line([(c-dot_size, r+dot_size), (c+dot_size, r-dot_size)], width=2)
            d.text((c - 4*dot_size, r - 4*dot_size), str(j), font=fnt)
        
        # Save the edited image to specified directory
        image.save(((directory + "/") if directory != "" else "") + (name + str(i) + ".bmp"))
        image.close()
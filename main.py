# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time


from functions import compare, get_corners_fast


def impose_features(feature_points, image, scale_factor = 1, directory = "", name="Corners"):
    points, weights = feature_points
    rows, cols = zip(*points)
    
    d = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("arial.ttf", 40)
    dot_size = 24
    for i, data in enumerate(zip(rows, cols, weights)):
        r_s, c_s, w = data
        
        r = scale_factor * r_s; c = scale_factor * c_s
        colour = (int(w * 255), 0, int((1 - w) * 255))
        dot_size = (w * 24) + 5
        
        d.line([(c-dot_size, r-dot_size), (c+dot_size, r+dot_size)], fill=colour, width=5)
        d.line([(c-dot_size, r+dot_size), (c+dot_size, r-dot_size)], fill=colour, width=5)
        
        d.text((c, r + 10), str(i), font=fnt)
    
    image.save(((directory + "/") if directory != "" else "") + (name + ".bmp"))
    image.close()
    
    
def main():
    # Iterate through pictures
    directory = "PhotosG"
    scale_factor = 12  
    
    begin = time.time()
    fp1, fp2 = "", ""
    dx_prev, dy_prev = 0, 0
    for i, name in enumerate(os.listdir(directory)):
        # If the first image's features have been found, reassign to fp2
        if fp1 != "":
            fp2 = fp1
        
        # Open current image and scale down for speed
        image = Image.open(directory + "/" + name)
        image_scaled = image.resize((image.width // scale_factor, 
                                     image.height // scale_factor)).convert("L")
        
        start = time.time()
        
        # Get the locations of the features
        fp1 = get_corners_fast(image_scaled)
        
        fp1 = get_corners_fast(image_scaled, True)
        
        
        comp_start = time.time()
        
        # If the second image's features have been found,
        #  compare current with previous features
#        if fp2 != "":
#            dx_prev, dy_prev = compare(fp1, fp2, dx_prev, dy_prev, True)
#            print(dx_prev, dy_prev)
            
            
        end = time.time()
        print("Corner Detection %f\nFeature Matching %f\nTotal %f" % (comp_start - start, end - comp_start, end - start))
    finish = time.time()
    print("total time %f" % (finish - begin))
            
if __name__ == '__main__':
    main()
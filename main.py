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
    
    
def main():    
    # Find all pictures containing name
    directory = "Video2"
    name = "frame"
    files = os.listdir(directory)
    numbers = []
    for f in files:
        if name not in f:
            files.remove(f)
        else:
            numbers.append(int(f.split(name)[1].split('.')[0]))
    numbers, files = zip(*sorted(zip(numbers, files)))
    
    
    scale_factor = 2
    
    
    fp1, fp2 = "", ""
    for i, name in enumerate(files):
        if i > 5:
            break
        
        # If the first image's features have been found, reassign to fp2
        if fp1 != "":
            fp2 = fp1
        
        # Open current image and scale down for speed
        image = Image.open(directory + "/" + name)
        image_scaled = image.resize((image.width // scale_factor, 
                                     image.height // scale_factor)).convert("L")
                
        # Get the locations of the features
        fp1 = get_corners_fast(image_scaled, False)
        
        # If the second image's features have been found,
        #  compare current with previous features
        if fp2 != "":
            compare(fp1, fp2, True)
            #print(dx * scale_factor,"\t",dy * scale_factor)
            break
               
    
if __name__ == '__main__':
    main()
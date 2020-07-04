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

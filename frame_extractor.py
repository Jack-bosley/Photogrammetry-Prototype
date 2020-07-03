# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:34:03 2020

@author: Jack
"""

import cv2
import os

def extract(directory, file):
    vidcap = cv2.VideoCapture(directory + "/" + file)
    
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(directory + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1

def rotate(directory):
    
    for f in os.listdir(directory):
        if '.' not in f:
            break
        
        rotated = cv2.rotate(cv2.imread(directory + "/" + f), cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(directory + "/" + f, rotated)


if __name__ == '__main__':
    rotate('Video2')

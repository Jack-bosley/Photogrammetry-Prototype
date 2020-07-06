# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:57:31 2020

@author: Jack
"""

import numpy as np
import cv2
import sys

img1_path  = 'test1.jpg'
img2_path  = 'test2.jpg'

img1 = cv2.imread(img1_path,0) # queryImage
img2 = cv2.imread(img2_path,0) # trainImage

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

if len(matches)>0:
    print("%d total matches found" % (len(matches)))
else:
    print("No matches were found - %d")
    sys.exit()
   
#
# store all the good matches as per Lowe's ratio test.
good = []
for _m in matches:
    m, n = _m
    if m.distance < 0.6*n.distance:
        good.append(m)
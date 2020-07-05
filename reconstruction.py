# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:24:47 2020

@author: Jack
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


#
#des1 = np.array([[3], [5], [4], [15]], dtype=np.float32)
#des2 = np.array([[5], [5], [1], [14]], dtype=np.float32)
#
#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)
#
#flann = cv2.FlannBasedMatcher(index_params, search_params)
#flann.add(des1)
#
#matches = flann.knnMatch(des2, 1)
#
#for m in matches:
#    print(m[0].queryIdx)
#    print(m[0].trainIdx)
#    print()

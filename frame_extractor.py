# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:34:03 2020

@author: Jack
"""

import cv2


file = 'Video/VID_20200703_163332_LS.mp4'

def main():
    vidcap = cv2.VideoCapture(file)
    
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:39:31 2020

@author: Jack
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation


def features_on_video(frames, features):
    fig = plt.figure(dpi=500)
    
    ims = []
    
    for f, locs in zip(frames, features):
        for i, l in enumerate(locs):
            f = cv2.circle(f, (int(l[1]), int(l[0])), radius=4, color=(255, 255, 255), thickness=2)
            f = cv2.putText(f, str(i), (int(l[1]), int(l[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        
        im = plt.imshow(f, animated=True)
        ims.append([im])
        
    anim = animation.ArtistAnimation(fig, ims, interval=200)
    
    
    if not os.path.exists("../Reconstructions"):
        os.mkdir("../Reconstructions")
        
    file_number = len(os.listdir("../Reconstructions"))
    writergif = animation.PillowWriter(fps=2)
    anim.save('../Reconstructions/debugging(%d).gif' % file_number, writer=writergif)
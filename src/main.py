# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""


import os
import numpy as np
import time, threading
import cv2

from FeatureExtraction.feature_detection import get_corners_fast
from FeatureExtraction.feature_classifier import BRIEF_classifier
from FeatureExtraction.feature_matching import Feature_dictionary

from Reconstruction.reconstruction import Bundle_adjuster
from Reconstruction.camera_model import Camera

from Utilities.debugging import features_on_video
from Utilities.virtual_realtime import Virtual_Realtime_Accelerometer, Virtual_Realtime_Camera
from Utilities.orienter import Orienter


def main():  
    
    # Generate a surf classifier
    brief = BRIEF_classifier(128, 25)
    # Also keep track of all feature descriptors for matching against
    feature_dict = Feature_dictionary()
    
    # Track the camera position (using accelerometer data)
    orienter = Orienter([0,0,0], [0,0,0], [0,0,0])
    
    # Create a model camera for bundle adjustment                                   (To Do: Calibrate)
    camera = Camera(9/16, 1, 0, 0, 0, 0, 0, 384, 512)
    
    
    images = [cv2.resize(cv2.imread(image, 0), (0, 0), fx=0.25, fy=0.25) for image in get_images()]
    print("got images")
    
    locations = [get_corners_fast(image, dist_to_edge_threshold=25) for image in images]
    print("got locations")
    
    descriptors = [brief(image, location) for image, location in zip(images, locations)]
    print("got descriptors")

    for locs, descs in zip(locations, descriptors):
        feature_dict.update_dictionary(locs, descs)
    presence, feature_locations = feature_dict.get_reproj_targets()
    
    features_on_video(images,feature_locations )

#     
#    presence, locations = feature_dict.get_reproj_targets()
#    features_on_video(frames, locations)
#        
#    presence, locations = feature_dict.get_reproj_targets(5)
#    
#    T_guess = orienter.get_camera_pose_guess()
#    
#    true_x, true_y = np.array(locations).T[0].T, np.array(locations).T[1].T
#    
#    # Create the bundle adjuster
#    bundle_adjuster = Bundle_adjuster(presence, locations, camera, T_guess = T_guess)
#    bundle_adjuster.optimise(10)
#    camera.plot_reprojection(bundle_adjuster.X, bundle_adjuster.T, true_x, true_y)
##    camera.plot_3d(bundle_adjuster.X)
#    
    

def get_images():
    directory = "../Data/Photos1"
    image_files = []
    for f in os.listdir(directory):
        image_files.append(directory + "/" + f)
    return image_files

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:55:51 2020

@author: Jack
"""


import os
import numpy as np
import time, threading
from PIL import Image

from FeatureExtraction.feature_detection import get_corners_fast
from FeatureExtraction.feature_classifier import BRIEF_classifier
from FeatureExtraction.feature_matching import Feature_dictionary

from Reconstruction.reconstruction import Bundle_adjuster
from Reconstruction.camera_model import Camera

from Utilities.debugging import impose_features
from Utilities.virtual_realtime import Virtual_Realtime_Accelerometer, Virtual_Realtime_Camera
from Utilities.orienter import Orienter


def main():  
    
    vra, vrc = get_virtual_realtime_streams("../Data/Video3", "frame", 
                                            "../Data/Video3/AccelerometerData", "accel_gyro_data.csv")
    
    # Generate a BRIEF classifier
    brief = BRIEF_classifier(128, 25)  
    # Also keep track of all feature descriptors for matching against
    feature_dict = Feature_dictionary()
    
    # Track the camera position (using accelerometer data)
    orienter = Orienter([0,0,0], [0,0,0], [0,0,0])
    
    # Create a model camera for bundle adjustment                                   (To Do: Calibrate)
    camera = Camera(9/16, 1, 0, 0, 0, 0, 0, 384, 512)
    
    
    vrc.start()
    vra.start()
    t_prev = time.time()
    
    # Iterate through files
    scale_factor = 2
    while vrc.is_running and vra.is_running:
        
        # Open current image and scale down for speed
        image = vrc.capture()
        if image == None:
            break
        image_scaled = np.array(image.resize((image.width // scale_factor, image.height // scale_factor)).convert("L"), 
                                dtype=np.int16)
        
        # Get the translation of the camera
        ax, ay, az = vra.get_accelerometer_raw()
        ox, oy, oz = vra.get_orientation_radians()
        
        orienter.update(time.time() - t_prev, [ax, ay, az], [ox, oy, oz])
        t_prev = time.time()
    
        # Get the locations and descriptors of the features
        feature_locations = get_corners_fast(image_scaled, False, brief.S)
        feature_descriptors = brief(image_scaled, feature_locations)
        
        
        # Update the dictionary with newly spotted features
        feature_dict.update_dictionary(feature_locations, feature_descriptors)
        
            
    presence, locations = feature_dict.get_reproj_targets()
    
    T_guess = orienter.get_camera_pose_guess()
    
    # Create the bundle adjuster
    bundle_adjuster = Bundle_adjuster(presence, locations, camera, T_guess = T_guess)
    bundle_adjuster.optimise(20)
    
    camera.plot_reprojection(bundle_adjuster.X, bundle_adjuster.T)
    
        
        


def get_virtual_realtime_streams(images_directory, images_file_name, data_directory, data_file):
    def get_images(directory, name):  
        # Find all pictures containing name in directory
        files = []
        numbers = []
        for f in os.listdir(directory):
            if name in f:
                numbers.append(int(f.split(name)[1].split('.')[0]))
                files.append(f)
        
        # Modify files to contain full path
        files = [directory + "/" + f for f in files]
        
        # Get times when images would be seen by camera
        times = []
        t = 0
        for f in files:
            times.append(t)
            t += (1.0 / 30)
        
        # Order by number in name and return 
        numbers, files = list(zip(*sorted(zip(numbers, files))))
        return times, files
    
    def get_data(directory, data_file):
        data = open(directory + "/" + data_file)

        T = []
        A = []
        W = []
        
        for i, d in enumerate(data):        
            if i <= 1:
                continue
                
            try:
                t, ax, ay, az, wx, wy, wz = d.split(',')[:7]
                
                t = float(t)
                ax, ay, az = float(ax), float(ay), float(az)
                wx, wy, wz = float(wx), float(wy), float(wz)
                
                if t < 8:
                    continue
                else:
                    if t > 16:
                        break
                    t -= 8
                
                T.append(t)
                A.append([ax, ay, az])
                W.append([wx, wy, wz])
            except:
                break
        
        data.close()
        
        return np.array(T).T, np.array(A).T, np.array(W).T
    
    vra = Virtual_Realtime_Accelerometer(*get_data(data_directory, data_file))
    vrc = Virtual_Realtime_Camera(*get_images(images_directory, images_file_name))
    
    return vra, vrc



if __name__ == '__main__':
    main()
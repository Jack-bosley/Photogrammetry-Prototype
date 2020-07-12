# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:51:49 2020

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

class Feature_dictionary:
    
    def __init__(self, mode='lsh'):
        
        # Set up FLANN matcher for Local Sensitivity Hashing mode (Needed for Hamming)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
        
        search_params = dict(checks = 50)
        
        # Create the flann matcher
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

            
        # Store all feature descriptors
        self.all_features = []
        
        self.current_image = 0
        self.all_feature_locations = {}
        
        
        
    
    def descriptor_indices_in_dictionary(self, descriptor):
        matches = self.get_matches(descriptor)
        
        indices = []
        # Check all matches
        for m_2 in matches:
            
            if len(m_2) == 2:
                
                # Append indices of good matches or -1 if not a good match
                m, n = m_2
                indices.append(m.trainIdx if m.distance < 0.7*n.distance else -1)
            
            # Also append -1 if no match found
            else:
                indices.append(-1)
                
        return indices
        
    def update_dictionary(self, locations, features):
        
        # If already know about some features
        if len(self.all_features) > 0:
            
            # Match new features against old ones
            matches = self.get_matches(features)
            
            # Check all matches
            for m_2 in matches:
                
                # Ignore invalid matches
                if len(m_2) == 2:
                    
                    m, n = m_2
                    dict_index = m.trainIdx
                    new_feat_index = m.queryIdx
                    
                    # If a poor match with both assume it is a new descriptor and keep track of it
                    if m.distance > 0.9*n.distance:
                        self.all_feature_locations[len(self.all_features)] = {self.current_image: locations[new_feat_index]}
                        self.all_features.append(features[new_feat_index])
                    elif m.distance < 0.7*n.distance:
                        self.all_feature_locations[dict_index][self.current_image] = locations[new_feat_index]
                    
        
        # Otherwise all features are new, so add all to array of descriptors
        else:
            for i, f in enumerate(features):
                self.all_features.append(f)
                self.all_feature_locations[i] = {self.current_image: locations[i]}
            
        self.current_image += 1
            
    def get_reproj_targets(self, min_required_frames=5):
        
        presence = [[] for i in range(len(self.all_feature_locations))]
        p_count = np.array([0 for i in range(len(self.all_feature_locations))])
        locations = [[] for i in range(len(self.all_feature_locations))]
        for feature_locations in self.all_feature_locations:
            for image in range(self.current_image):
                if image in self.all_feature_locations[feature_locations]:
                    locations[feature_locations].append(self.all_feature_locations[feature_locations][image])
                    presence[feature_locations].append(True)
                    p_count[feature_locations] += 1
                else:
                    locations[feature_locations].append(np.array([0, 0]))
                    presence[feature_locations].append(False)
                    
        

        to_keep = np.where(p_count > min_required_frames)
        presence = np.array(presence)[to_keep].T.tolist()
        locations= np.transpose(np.array(locations)[to_keep], axes=(1, 0, 2)).tolist()  
           
        print(np.shape(presence), np.shape(locations))
        
        return presence, locations
        
    def get_matches(self, features, k=2):
        # k nearest neighbours (query, train, k)
        return self.flann.knnMatch(features, np.array(self.all_features), k=k)
    

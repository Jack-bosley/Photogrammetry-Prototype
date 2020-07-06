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
    
    def __init__(self):
        
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
        self.all_features = np.array([])
        
    
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
        
    def update_dictionary(self, features):
        
        # If already know about some features
        if len(self.all_features) > 0:
            
            # Match new features against old ones
            matches = self.get_matches(features)
            
            # Check all matches
            for m_2 in matches:
                
                # Ignore invalid matches
                if len(m_2) == 2:
                    
                    # If not a good match, assume it is a new descriptor and keep track of it
                    m, n = m_2
                    if m.distance > 0.9*n.distance:
                        self.all_features = np.vstack([self.all_features, features[m.queryIdx]])
        
        # Otherwise all features are new, so add all to array of descriptors
        else:
            self.all_features = features
        
        
        
    def get_matches(self, features, k=2):
        # k nearest neighbours (query, train, k)
        return self.flann.knnMatch(features, self.all_features, k=k)
    

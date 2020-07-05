# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:51:49 2020

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt

def get_matches(feature_descriptors_prev, feature_descriptors_curr, threshold):
    
    print("find matches")
    
    # Keep track of indices of matching features and their separation
    matches_1 = []
    matches_2 = []
    match_dist = []
    
    # Iterate over all features in set 1
    for i_1, f_d_1_i in enumerate(feature_descriptors_prev):
        
        # Find the minimum nearest neighbour in set 2 and their separation
        min_dist = np.inf
        min_dist_index = -1
        for i_2, f_d_2_i in enumerate(feature_descriptors_curr):
            dist = np.sum(f_d_1_i != f_d_2_i)
            if dist < min_dist:
                min_dist = dist
                min_dist_index = i_2
         
        # If the separation is sufficiently low
        if min_dist < threshold:
            
            # Need to decide if this match should be added to list
            to_add = True
            
            # Check that the nearest neighbour in set 2 is not a duplicate
            for i_2, m_2 in enumerate(matches_2):
                # If it is a duplicate, remove the existing match
                #  if it is a worse fit than the current match
                #  otherwise just don't add current match
                if m_2 == min_dist_index:
                    if match_dist[i_2] > min_dist:
                        del matches_1[i_2]
                        del matches_2[i_2]
                        del match_dist[i_2]
                        
                        break
                    else:
                        to_add = False
                        break
            
            # Add the current match to the list if it is unique or if it is the smallest
            if to_add:
                matches_1.append(i_1)
                matches_2.append(min_dist_index)
                match_dist.append(min_dist)
        
    
    return matches_1, matches_2
    
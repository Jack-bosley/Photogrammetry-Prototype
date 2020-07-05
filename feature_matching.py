# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:51:49 2020

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt

class Feature_dictionary:
    def __init__(self, threshold):
        self.descriptor_dictionary = {}
        self.threshold = threshold
        
        print("new dictionary")


    # Gets how different the two descriptors are
    def descriptor_difference(self, d_1, d_2):
        return np.sum(d_1 != d_2)


    # Returns whether or not a match is made
    def descriptors_match(self, d_1, d_2):
        return self.descriptor_difference(d_1, d_2) < self.threshold
    

    # Returns the key of the descriptor in the dictionary (or -1 if not present)
    def descriptor_index(self, descriptor):
        
        # Scan through dictionary
        for d in self.descriptor_dictionary:
            dict_descriptor = self.descriptor_dictionary[d]
            # Return key if a match is made
            if self.descriptors_match(dict_descriptor, descriptor):
                return d
            
        return -1
    
    
    # Appends supplied feature descriptors if not already in dictionary
    def update_dictionary(self, feature_descriptors):      
        # Iterate over all features in image
        for f_d in feature_descriptors:
            
            # Get the index of the feature in the dictionary
            index_in_dictionary = self.descriptor_index(f_d)
            
            # If it isn't present, add it to the end
            if index_in_dictionary == -1:
                index_in_dictionary = len(self.descriptor_dictionary)
            
                self.descriptor_dictionary[index_in_dictionary] = f_d
                
    
    
    def __str__(self):
        return "Descriptor dictionary object -- length %d " % len(self.descriptor_dictionary)
        
    
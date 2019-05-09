import model
import decoder
import math


class Model_builder(object):

    def __init__(self):
        self.class_labels = {}
        self.all_features = {}
        
    def read_line(self, line):
    
        elements = line.split() ## a list of elements in line
            
        claimed_class_name = elements[0] ## this first element is the claimed class name
        
        if claimed_class_name not in self.class_labels:
            self.class_labels[claimed_class_name] = True
            
        list_of_elements = elements[1:]

        for pair in list_of_elements:
            feature = pair.split(':')[0]
            
            if feature not in self.all_features:
                self.all_features[feature] = 1
            else:
                self.all_features[feature] += 1
            
import sys
import math
import operator
import model


class Decoder(object):

    def __init__(self, model):
    
        self.weight_dictionary = model.feature_weight_dictionary ## get the weight_dictionary
        
        self.instance_parameter = [] ## a list of two elements of each testing instance;
                                     ## the first one is the index of this testing instance;
                                     ## the second one is the claimed_class_name of this testing instance
    
        self.all_features = set()
        self.class_prob = {} ## not normalized yet for each testing instance
        self.Z = 0 ## common base to normalize with
        self.norm_production = {} ## final output for each testing instance
        self.predicted_class_name = ''

    def read_instance(self, line, index):
        
        elements = line.strip().split() ## a list of elements in line
            
        claimed_class_name = elements[0] ## this first element is the claimed class name
            
        list_of_elements = elements[1:]
        
        self.instance_parameter.append(str(index))
        self.instance_parameter.append(claimed_class_name)
        
        for pair in list_of_elements:
            feature = pair.split(':')[0]
            self.all_features.add(feature)
       
    def fill_class_prob(self):
    
        for class_label in self.weight_dictionary:
            power = self.weight_dictionary[class_label]['<default>']

            for feature in self.all_features: ## loop all the presented feature in testing instance
                
                ## if it is presented in weight_dictionary 
                if feature in self.weight_dictionary[class_label]: 
                    power += self.weight_dictionary[class_label][feature]

            conditional_prob = math.exp(power)
            
            self.class_prob[class_label] = conditional_prob

    def compute_normal(self):
        for class_label in self.class_prob:
            self.Z += self.class_prob[class_label]

    def fill_final_prob(self):
        
        self.compute_normal()
        
        for class_label in self.class_prob:
            self.norm_production[class_label] = float(self.class_prob[class_label] / self.Z)
            
    def find_predicted_class_label(self):
        for class_label in self.norm_production:
            if self.predicted_class_name == '':
                self.predicted_class_name = class_label
            elif self.norm_production[class_label] >= self.norm_production[self.predicted_class_name]:
                self.predicted_class_name = class_label
                
        return self.predicted_class_name                
            
    def report_sys_string(self):
        
        sorted_dictionary = sorted(self.norm_production.items(), key=operator.itemgetter(1), 
                                   reverse=True)
                                   
        sequence = ''

        for element in sorted_dictionary:
            sequence = sequence + element[0] + '\t' + str(element[1]) + '\t'            
                                   
        ret = 'array:' + self.instance_parameter[0] + '\t' + self.instance_parameter[1] + '\t' + self.predicted_class_name + '\t' + sequence

        return ret                          
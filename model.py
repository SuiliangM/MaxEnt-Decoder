from tqdm import tqdm
""" This class encapsulates 
    a model class that will be shared 
    and used in multiple files
"""

class Model(object):

    ## constructor
    def __init__(self, model_file):
            
        self.feature_weight_dictionary = {}
        self.fill_feature_weight_dictionary(model_file)

    def fill_feature_weight_dictionary(self, model_file):
        with open(model_file, "r") as rfp:
        
            class_label = ''
        
            for i, line in enumerate(tqdm(rfp)): 
            
                line = line.strip()
                
                if 'FEATURES FOR CLASS' in line: ## this marks the start of a new class
                    class_label = line.split()[-1]
                    
                    if class_label not in self.feature_weight_dictionary:
                        self.feature_weight_dictionary[class_label] = {}
                        
                else:
                    pair = line.split()
                    self.feature_weight_dictionary[class_label][pair[0]] = float(pair[1])                                                                        
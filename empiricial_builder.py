import math
import sys
from tqdm import tqdm
from collections import OrderedDict

def process_training_data(training_data_file):
    
    training_dictionary = {}
    total_number = 0
    
    with open(training_data_file, "r") as rfp:
    
        for index, line in enumerate(tqdm(rfp)): 
        
            total_number = index + 1
        
            elements = line.strip().split()
            
            class_label = elements[0]
            
            if class_label not in training_dictionary:
                training_dictionary[class_label] = {}
            
            pairs = elements[1:]
            
            for pair in pairs:
                feature = pair.split(':')[0]
                
                if feature not in training_dictionary[class_label]:
                    training_dictionary[class_label][feature] = 1
                else:
                    training_dictionary[class_label][feature] += 1

    return training_dictionary, total_number                        
                    
def main():
    training_data_file = sys.argv[1]
    output_file = sys.argv[2]
    
    training_dictionary, total_number = process_training_data(training_data_file)
    
    training_dictionary = OrderedDict(sorted(training_dictionary.items()))
    
    with open(output_file, "w") as wfp:
    
        for class_label in training_dictionary:
        
            for feature in sorted(training_dictionary[class_label]):
            
                empiricial = format(float(training_dictionary[class_label][feature] / total_number), '.5f')
                
                wfp.write(class_label + '\t' + feature + '\t' + str(empiricial) + '\t' + str(training_dictionary[class_label][feature]) + '\n')
            
        
if __name__ == '__main__':
    main()        
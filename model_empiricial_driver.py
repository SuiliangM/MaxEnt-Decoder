import math
import sys
from tqdm import tqdm
from collections import OrderedDict
import model
import model_empiricial_builder as builder
import decoder

def main():

    training_data_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if len(sys.argv) == 3: ## no model passed as argument
    
        modelExp = builder.Model_builder()
        
        total = 0
        
        with open(training_data_file, "r") as rfp:
        
            with open(output_file, "w") as wfp:
            
                ## loop over each training instance x
            
                for index, line in enumerate(tqdm(rfp)): 
                
                    line = line.strip()
                    
                    total = index + 1
                    
                    modelExp.read_line(line)
                    
                p = float(1 / len(modelExp.class_labels))
                
                sorted_class_label = OrderedDict(sorted(modelExp.class_labels.items()))
                
                for class_name in sorted_class_label:
                    for feature in sorted(modelExp.all_features):
                        count = float(p * modelExp.all_features[feature])
                        model_out = format(float(count / total), '.5f')
                        wfp.write(str(class_name) + '\t' + str(feature) + '\t' + str(model_out) + '\t' + str(format((count), '.5f')) + '\n')                
                
    if len(sys.argv) == 4: ## model passed as argument
        model_file = sys.argv[3]
        
        m = model.Model(model_file)
        
        modelExp = builder.Model_builder()
        
        total = 0
        
        with open(output_file, "w") as wfp:
        
            output_dictionary = {}

            for class_label in m.feature_weight_dictionary:
                if class_label not in output_dictionary:
                    output_dictionary[class_label] = {}                
        
            with open(training_data_file, "r") as rfp:
        
                for index, line in enumerate(tqdm(rfp)): 
            
                    line = line.strip()
                    
                    modelExp.read_line(line)
                    
                    total = index + 1
                    
                ## we have total_number of training_instance and modelExp for later usage
                
            ## initialize the output_dictionary    
                
            for feature in OrderedDict(sorted(modelExp.all_features.items())):
                for class_label in output_dictionary:
                    output_dictionary[class_label][feature] = 0.0
                                            
            with open(training_data_file, "r") as rfp:                                       
                    
                for index, line in enumerate(tqdm(rfp)): 
                
                    line = line.strip()

                    d = decoder.Decoder(m)
                    d.read_instance(line, index)
                    d.fill_class_prob()                
                    d.fill_final_prob()
                    
                    ## d.norm_production is a dictionary that 
                    ## maps a class_name to its normalized probability,
                    ## for each training instance    
                    
                    pairs = line.split()[1:]
                    
                    for pair in pairs:
                        feature = pair.split(':')[0]
                        
                        for class_label in output_dictionary:
                            output_dictionary[class_label][feature] += float((1 / total) * d.norm_production[class_label])
                        
            ## write output_dictionary to output_file
            ## write output_dictionary to output_file
            sb = OrderedDict(sorted(modelExp.class_labels.items()))
            
            
            for class_label in sb:
                for feature in OrderedDict(sorted(output_dictionary[class_label].items())):
                    
                    expectation = float(output_dictionary[class_label][feature])
                    count = format(float(expectation * total), '.5f')
                    
                    wfp.write(class_label + '\t' + feature + '\t' + str(format((expectation), '.5f')) + '\t' + str(count) + '\n')


if __name__ == '__main__':
    main()        
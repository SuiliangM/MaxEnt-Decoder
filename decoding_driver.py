import model
import decoder
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys



def main():
    test_data_file = sys.argv[1]
    model_file = sys.argv[2]
    sys_file = sys.argv[3]
    
    m = model.Model(model_file)
    
    y_true = []
    y_pred = []
    
    with open(sys_file, "w") as wfp:
    
        wfp.write('%%%%% test data:\n')
    
        with open(test_data_file, "r") as rfp:
        
            for index, line in enumerate(tqdm(rfp)): 
            
                line = line.strip()
                
                d = decoder.Decoder(m)
                
                d.read_instance(line, index)
                
                d.fill_class_prob()
                
                d.fill_final_prob()
                
                y_pred.append(d.find_predicted_class_label())
                y_true.append(line.split()[0])
                
                wfp.write(d.report_sys_string() + '\n')            
                
    
    sorted_list = sorted(m.feature_weight_dictionary.items(), key=lambda x: x[0])

    label_list = []

    for element in sorted_list:
        label_list.append(element[0])

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=label_list), index=label_list, columns=label_list)

    print("Confusion matrix for the testing data:")

    print("row is the truth, column is the system output\n")

    print(cm.to_string() + "\n")

    print("accuracy=%s\n\n" % (accuracy_score(y_true, y_pred)))
    
if __name__ == '__main__':
    main()    
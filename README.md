# MaxEnt-Decoder
maximum entropy decoder

A MaxEnt classifier can be called by the following command:
./maxent_classify.sh test_data model_file sys_output > acc_file

A typical format of model_file is m1.txt. 

The following command can be used to calculate empiricial expectation:
./calc_emp_exp.sh training_data output_file

Note that output_file has the format “class_label feat_name expectation_raw_count” (an example is emp_count_ex):
raw_count is the number of training instances with that class_label and contains that feat_name;expectation is the empirical expectation.

The following command can be used to calculate model expectation: 
./calc_model_exp.sh training_data output_file {model_file}

Notice that model_file is optional. If it is given, it has the same format as model_file described above (an example is m1.txt) and this model_file is used to compute p(y|xi); if it is not given, then p(y|xi) = 1 / |C|, where |C| is the number of class labels.

Also notice that output_file has the format “class_label feat_name expectation_count” (e.g., emp_count_ex):
expectation is the model expectation; count is expectation multiplied by the number of training
instances. Note that the count is often a real number, not an integer, so output it as a real
number.

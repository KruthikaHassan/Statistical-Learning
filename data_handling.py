
import csv

def load_data(data_file_path):
    ''' Parse the csv file and get raw data '''
    csv_file = open(data_file_path)
    csv_reader = csv.reader(csv_file, delimiter='\t')
    
    all_data = []
    for row in csv_reader:
        all_data.append(row)

    return all_data

def organize(raw_data, num_vars, cat_vars):
    ''' Organize data and substitue the missing values '''
    
    req_vars =  num_vars + cat_vars + ['Order'] + ['SalePrice']
   
    # Fill the missing values and organize
    cleaned_data = {}
    for clean_var in req_vars:
        for raw_var_indx in range(len(raw_data[0])):
            if clean_var == raw_data[0][raw_var_indx]:
                cleaned_data[clean_var] = []
                for i in range(1, len(raw_data)):
                    if raw_data[i][raw_var_indx] == '':
                        if clean_var in cat_vars:
                            raw_data[i][raw_var_indx] = 'Missing'
                        else:
                            raw_data[i][raw_var_indx] = 0
                            
                    cleaned_data[clean_var].append(raw_data[i][raw_var_indx])

    
    # Transform all catagorical features as hotvectors
    organized_data = {}
    for feature in cleaned_data:
        if feature in cat_vars:
            organized_data.update(transform_to_hot_vector(feature, cleaned_data[feature]))
        else:
            organized_data[feature] = cleaned_data[feature]

    return organized_data

def split_data(data):
    ''' we will use order feature of cleaned data as index to split the data '''
    
    # sub function to split the data
    def get_data_from_index(indecies):
        req_data = {}
        for key in data:
            # Dropping the order
            if key != 'Order':
                req_data[key] = [ data[key][indx] for indx in indecies]
        return req_data
    
    # Get the indecies of split
    val_data_indecies = []
    test_data_indecies = []
    train_data_indecies = []
    
    for var in (data['Order']):
        indx = int(var)
        if indx % 5 == 3:
            val_data_indecies.append(indx-1)
        elif indx % 5 == 4:
            test_data_indecies.append(indx-1)
        else:
            train_data_indecies.append(indx-1)
    
    # Now split the data
    train_data = get_data_from_index(train_data_indecies)
    test_data  = get_data_from_index(test_data_indecies)
    val_data   = get_data_from_index(val_data_indecies)
    
    return (train_data, test_data, val_data)

def transform_to_hot_vector(feature_name, feature_data):
    # First find unique values 
    uniq_vals = []
    for val in feature_data:
        if val not in uniq_vals:
            uniq_vals.append(val)
    
    # now for real
    hot_vectors = []
    for val in feature_data:
        hot_vec = []
        for i in range(len(uniq_vals)):
            if uniq_vals[i] == val:
                hot_vec.append(1)
            else:
                hot_vec.append(0)
                        
        hot_vectors.append(hot_vec)

    transformed_data = {}
    for i in range(len(uniq_vals)):
        f_d = []
        for j in range(len(hot_vectors)):
            f_d.append(hot_vectors[j][i])
                
        transformed_data[feature_name+'-'+str(uniq_vals[i])] = f_d
    
    return transformed_data
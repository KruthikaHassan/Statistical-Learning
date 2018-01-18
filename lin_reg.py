import numpy as np

def construct_feature_matrix(features, data, add_x0=True):
    ''' Gives out the feature matrix '''
    
    # get data length (not so ideal way.. )
    data_len = len(data[features[0]])

    # Starting with empty feature matrix
    feature_matrix = []

    # First add the x0's i.e 1s in the first row
    if add_x0:
        x0 = [1 for i in range(data_len)]
        feature_matrix.append(x0)
    
    # Now add each feature data
    for feature in features:
        feature_matrix.append(data[feature])
    
    return np.matrix(feature_matrix)

def perform_lin_regression(features, data, target_feature):
    # Lets first constrict feature matrix
    fm = construct_feature_matrix(features, data).astype(float)
   
    # psudo inverse
    fm_inv = np.linalg.pinv(fm)

    # target data ( observed values )
    target_vec  = np.array([data[target_feature]]).astype(float)
    
    # weights
    weight_vector =  target_vec * fm_inv

    return weight_vector

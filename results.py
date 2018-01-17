
import math
from lin_reg import construct_feature_matrix

def predict_target(weight_vector, data, features):
    # feature matrix
    fm = construct_feature_matrix(features, data)
    fm = fm.astype(float)

    predictions = fm * weight_vector

    return predictions 
    
def cal_rmse(predictions, target):
    sqrs_sum = 0
    for indx in range(len(predictions)):
        err = predictions[indx] - float(target[indx])
        sqrs_sum = sqrs_sum + (err * err)
    
    rmse = math.sqrt(sqrs_sum / len(predictions))
    return rmse
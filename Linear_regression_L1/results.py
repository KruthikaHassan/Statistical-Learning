
import math
import numpy as np
import matplotlib.pyplot as plt
from lin_reg import construct_feature_matrix

def predict_target(weight_vector, data, features, add_x0=True):
    # feature matrix
    fm = construct_feature_matrix(features, data, add_x0).astype(float)
    predictions =  weight_vector * fm

    return np.transpose(predictions)
    
def cal_rmse(predictions, target):
    sqrs_sum = 0
    for indx in range(len(predictions)):
        err = predictions[indx] - float(target[indx])
        sqrs_sum = sqrs_sum + (err * err)
    
    rmse = math.sqrt(sqrs_sum / len(predictions))
    return rmse


def plot_warm_up(train_data, target_feature, warmup_feature, warmup_weights, warmup_preds):

    feature_data = np.array(train_data[warmup_feature]).astype(float)
    target_data = np.array(train_data[target_feature]).astype(float)
    predicted_data = np.array(warmup_preds).astype(float)
    
    plt.scatter(feature_data, target_data)
    plt.scatter(feature_data, predicted_data)
    plt.legend(['Observed Data', 'Predicted Data'], loc='upper right')
    plt.ylabel(target_feature)
    plt.xlabel(warmup_feature)
    plt.show()
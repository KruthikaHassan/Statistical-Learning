import numpy as np
from sklearn import linear_model
from lin_reg import *
from results import *
import matplotlib.pyplot as plt

def lasso_regression(features, target_feature, train_data, validation_data, test_data):


    # first get feature data of train in the form of matrix
    fm = construct_feature_matrix(features, train_data, False).astype(float)
    fm = np.transpose(fm)
    vm = construct_feature_matrix(features, validation_data, False).astype(float)
    # initialize alpha
    l = 50
    train_rmse_vals = []
    validation_rmse_vals = []
    alpha_vals = []
    while l <= 500:

        # Init lasso 
        alpha_factor = 0.00005
        clf = linear_model.Lasso(max_iter=100000, alpha = l * alpha_factor)

        # Fit the model and get weights for train data
        clf.fit(fm, train_data[target_feature])
        weights = np.array([clf.coef_])

        # predict values for train and validation
        train_pred = predict_target(weights, train_data, features, False)
        train_rmse = cal_rmse(train_pred, train_data[target_feature])

        val_pred = predict_target(weights, validation_data, features, False)
        val_rmse = cal_rmse(val_pred, validation_data[target_feature])
        
        #print("Train rmse for alpha(%d) = %f" % (l, train_rmse))
        #print("Validation rmse for alpha(%d) = %f" % (l, val_rmse))

        train_rmse_vals.append(train_rmse)
        validation_rmse_vals.append(val_rmse)
        alpha_vals.append(l)

        l = l + 50

    '''plt.plot(alpha_vals, train_rmse_vals)
    plt.plot(alpha_vals, validation_rmse_vals)
    plt.show()'''


def lasso_with_corss_validation(features, target_feature, train_data, validation_data, test_data):

    train_matrix = construct_feature_matrix(features, train_data, False).astype(float)
    train_matrix = np.transpose(train_matrix)

    validation_matrix = construct_feature_matrix(features, validation_data, False).astype(float)
    validation_matrix = np.transpose(validation_matrix)

    combined_matrix = np.concatenate((train_matrix, validation_matrix), axis=0)

    total_data_len = np.shape(combined_matrix)[0]

    

    


    ###################################### cross-validation ##########################
    vmt = np.transpose(vm)
    fm1 = np.concatenate((fm,vmt),axis=0)
    print(np.shape(fm1))
    first_fold = fm1[0:585,:]
    second_fold = fm1[586:1171,:]
    third_fold = fm1[1172:1757,:]
    fourth_fold = fm1[1758:2344,:]


    



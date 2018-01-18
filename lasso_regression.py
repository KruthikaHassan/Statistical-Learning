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
    first_fold  = combined_matrix[0:total_data_len//4, :]
    second_fold = combined_matrix[(total_data_len//4)+1 : total_data_len//2,  :]
    third_fold  = combined_matrix[(total_data_len//2)+1 : 3*total_data_len//4, :]
    fourth_fold = combined_matrix[(3*total_data_len//4)+1 : total_data_len, :]

    combined_target_vals = train_data[target_feature] + validation_data[target_feature]
    first_fold_targets  = combined_target_vals[0:total_data_len//4]
    second_fold_targets = combined_target_vals[(total_data_len//4)+1 : total_data_len//2]
    third_fold_targets  = combined_target_vals[(total_data_len//2)+1 : 3*total_data_len//4]
    fourth_fold_targets = combined_target_vals[(3*total_data_len//4)+1 : total_data_len]

    # fourth as validation
    training = np.concatenate((first_fold, second_fold, third_fold), axis=0)
    training_target = first_fold_targets + second_fold_targets + third_fold_targets
    (alpha_vals1, train_rmse_vals1, validation_rmse_vals1, non_zero_weights1) = run_lasso_reg(training, fourth_fold, training_target, fourth_fold_targets)

    # third as validation
    training = np.concatenate((first_fold, second_fold, fourth_fold), axis=0)
    training_target = first_fold_targets + second_fold_targets + fourth_fold_targets
    (alpha_vals2, train_rmse_vals2, validation_rmse_vals2, non_zero_weights2) =run_lasso_reg(training, third_fold, training_target, third_fold_targets)

    # second as validation
    training = np.concatenate((first_fold, fourth_fold, third_fold), axis=0)
    training_target = first_fold_targets + fourth_fold_targets + third_fold_targets
    (alpha_vals3, train_rmse_vals3, validation_rmse_vals3, non_zero_weights3) =run_lasso_reg(training, second_fold, training_target, second_fold_targets)

    # first as validation
    training = np.concatenate((fourth_fold, second_fold, third_fold), axis=0)
    training_target = fourth_fold_targets + second_fold_targets + third_fold_targets
    (alpha_vals4, train_rmse_vals4, validation_rmse_vals4, non_zero_weights4) =run_lasso_reg(training, first_fold, training_target, first_fold_targets)

    train_rmse_vals = []
    validation_rmse_vals = []
    non_zero_weights = []
    for indx in range(len(alpha_vals1)):
        avg_train_rmse = (train_rmse_vals1[indx] + train_rmse_vals2[indx]+ train_rmse_vals3[indx] +train_rmse_vals4[indx]) / 4
        train_rmse_vals.append(avg_train_rmse)

        avg_val_rmse = (validation_rmse_vals1[indx] + validation_rmse_vals2[indx]+ validation_rmse_vals3[indx] +validation_rmse_vals4[indx]) / 4
        validation_rmse_vals.append(avg_val_rmse)

        avg_nnz_wts = (non_zero_weights1[indx] + non_zero_weights2[indx]+ non_zero_weights3[indx] + non_zero_weights4[indx]) / 4
        non_zero_weights.append(avg_nnz_wts)

    plt.plot(alpha_vals1, train_rmse_vals)
    plt.plot(alpha_vals1, validation_rmse_vals)
    plt.legend(['Train RMSE', 'Validation RMSE'], loc='upper right')
    plt.ylabel('RMSE')
    plt.xlabel('ALPHA')
    plt.show()

    plt.plot(alpha_vals1, non_zero_weights)
    plt.legend(['Non zero weights'], loc='upper right')
    plt.ylabel('Non zero weights')
    plt.xlabel('ALPHA')
    plt.show()



def run_lasso_reg(training, validation, training_target, validation_target):

    l = 50
    train_rmse_vals = []
    validation_rmse_vals = []
    alpha_vals = []
    non_zero_weights = []
    while l <= 500:

        # Init lasso 
        alpha_factor = 1
        clf = linear_model.Lasso(max_iter=100000, normalize=True, alpha = l * alpha_factor)

        # Fit the model and get weights for train data
        clf.fit(training, training_target)
        weights = np.transpose(np.array([clf.coef_]))

        # predict values for train and validation
        train_pred = training * weights
        train_rmse = cal_rmse(train_pred, training_target)

        val_pred =  validation * weights
        val_rmse = cal_rmse(val_pred, validation_target)
    
        train_rmse_vals.append(train_rmse)
        validation_rmse_vals.append(val_rmse)
        alpha_vals.append(l * alpha_factor)

        nnz_wts = 0
        for wt in weights:
            if wt != 0:
                nnz_wts = nnz_wts + 1
        non_zero_weights.append(nnz_wts)
       
        l = l + 50

    print(validation_rmse_vals)

    return (alpha_vals, train_rmse_vals, validation_rmse_vals, non_zero_weights)
    





''' Main module  '''


from data_handling import *
from lin_reg import *
from results import *
from lasso_regression import *

def main():
    ''' Main function '''

    numerical_variables = ['Lot Area', 'Lot Frontage', 'Year Built',   \
                       'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', \
                       'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',   \
                       '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', \
                       'Garage Area', 'Wood Deck SF', 'Open Porch SF', \
                       'Enclosed Porch', '3Ssn Porch', 'Screen Porch', \
                       'Pool Area']
    
    discrete_variables = ['MS SubClass', 'MS Zoning', 'Street',        \
                      'Alley', 'Lot Shape', 'Land Contour',            \
                      'Utilities', 'Lot Config', 'Land Slope',         \
                      'Neighborhood', 'Condition 1', 'Condition 2',    \
                      'Bldg Type', 'House Style', 'Overall Qual',      \
                      'Overall Cond', 'Roof Style', 'Roof Matl',       \
                      'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',  \
                      'Exter Qual', 'Exter Cond', 'Foundation',        \
                      'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',       \
                      'BsmtFin Type 1', 'Heating', 'Heating QC',       \
                      'Central Air', 'Electrical', 'Bsmt Full Bath',   \
                      'Bsmt Half Bath', 'Full Bath', 'Half Bath',      \
                      'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual',\
                      'TotRms AbvGrd', 'Functional', 'Fireplaces',     \
                      'Fireplace Qu', 'Garage Type', 'Garage Cars',    \
                      'Garage Qual', 'Garage Cond', 'Paved Drive',     \
                      'Pool QC', 'Fence', 'Sale Type', 'Sale Condition']

    # First get parse the raw data
    raw_data = load_data('AmesHousing.txt')
    
    # Organize raw data into features
    clean_data = organize(raw_data, numerical_variables, discrete_variables)
    all_features = list(clean_data.keys())
    all_features.remove('Order')
    all_features.remove('SalePrice')

    # Split the data into train, validation and test
    train_data, validation_data, test_data = split_data(clean_data)

    # Normalize
    # train_data, validation_data, test_data = normalize_data(train_data, test_data, validation_data)

    # Our target feature:
    target_feature = 'SalePrice'

    # Test for single feature
    
    warmup_feature = ['Gr Liv Area']
    warmup_weights = perform_lin_regression(warmup_feature, train_data, target_feature)
    print(warmup_weights)
    warmup_preds   = predict_target(warmup_weights, train_data, warmup_feature)
    #plot_warm_up(train_data, target_feature, warmup_feature[0], warmup_weights, warmup_preds)
    
    warmup_val_preds = predict_target(warmup_weights, validation_data, warmup_feature)
    warmup_val_rmse = cal_rmse(warmup_val_preds, validation_data[target_feature])
    print("validation RMSE %f" % warmup_val_rmse)

    warmup_test_preds = predict_target(warmup_weights, test_data, warmup_feature)
    warmup_test_rmse = cal_rmse(warmup_test_preds, test_data[target_feature])
    print("Test RMSE %f" % warmup_test_rmse)
    
    
    # Now train using linear regression
    weights = perform_lin_regression(all_features, train_data, target_feature)

    # Predict from validation
    predicted_vals = predict_target(weights, validation_data, all_features)
    
    # Calculate error
    rmse = cal_rmse(predicted_vals, validation_data[target_feature])
    print(rmse)

    predicted_vals_test = predict_target(weights, test_data, all_features)
    rmse_test = cal_rmse(predicted_vals_test, test_data[target_feature])
    print(rmse_test)

    #lasso_regression(all_features, target_feature, train_data, validation_data, test_data)
    lasso_with_corss_validation(all_features, target_feature, train_data, validation_data, test_data)


main()

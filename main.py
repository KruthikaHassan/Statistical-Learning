''' Main module  '''


from data_handling import *
from lin_reg import *
from results import *

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

    # Our target feature:
    target_feature = 'SalePrice'

    # Now train using linear regression
    weights = perform_lin_regression(all_features, train_data, target_feature)

    # Predict from validation
    predicted_vals = predict_target(weights, validation_data, all_features)
    
    # Calculate error
    rmse = cal_rmse(predicted_vals, validation_data[target_feature])
    
    print(rmse)




main()

import csv       # csv file reading module

# Function to read raw data

def load_data(data_file_path):
    csv_file = open(data_file_path)
    csv_reader = csv.reader(csv_file, delimiter = '\t')
    raw_data = []
    for row in csv_reader:
        raw_data.append(row) 
    return raw_data

# Function to handle missing values

def handle_missing_values(raw_data):
    numerical_variables = ['Lot Area', 'Lot Frontage', 'Year Built','Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2','Bsmt Unf SF',
                           'Total Bsmt SF', '1st Flr SF','2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area','Garage Area', 
                           'Wood Deck SF', 'Open Porch SF','Enclosed Porch', '3Ssn Porch', 'Screen Porch','Pool Area'];
    discrete_variables = ['MS SubClass', 'MS Zoning', 'Street','Alley', 'Lot Shape', 'Land Contour','Utilities', 'Lot Config', 'Land Slope',
                          'Neighborhood', 'Condition 1', 'Condition 2','Bldg Type', 'House Style', 'Overall Qual','Overall Cond', 'Roof Style', 'Roof Matl',
                          'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type','Exter Qual', 'Exter Cond', 'Foundation','Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
                          'BsmtFin Type 1', 'Heating', 'Heating QC','Central Air', 'Electrical', 'Bsmt Full Bath','Bsmt Half Bath', 'Full Bath', 'Half Bath',
                          'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual','TotRms AbvGrd', 'Functional', 'Fireplaces','Fireplace Qu', 'Garage Type', 'Garage Cars',
                          'Garage Qual', 'Garage Cond', 'Paved Drive','Pool QC', 'Fence', 'Sale Type', 'Sale Condition']
    variables = numerical_variables + discrete_variables + ['order']
    raw_data_variables = raw_data[0]
    new_data_1 = []
    for k in range(0,len(variables)):
        for i,j in enumerate(raw_data_variables):   # function returns both the index and object, i = index, j = object
            if variables[k] == j:
                #print(raw_data[:][i])
                new_data_1.append(raw_data[:][i])  
                break  
    return new_data_1

# Main function
raw_data =load_data('AmesHousing.txt')
dat = handle_missing_values(raw_data)
print(len(dat))




            





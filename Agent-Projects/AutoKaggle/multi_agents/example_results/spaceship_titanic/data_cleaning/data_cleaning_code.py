import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load the data
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Make copies of the dataframes
    train_clean = train.copy()
    test_clean = test.copy()
    
    # Define columns to fill missing values
    categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Fill missing values for categorical columns using mode
    train_clean = fill_missing_values(train_clean, categorical_columns, method='mode')
    test_clean = fill_missing_values(test_clean, categorical_columns, method='mode')
    
    # Fill missing values for numerical columns using median
    train_clean = fill_missing_values(train_clean, numerical_columns, method='median')
    test_clean = fill_missing_values(test_clean, numerical_columns, method='median')
    
    print("Missing values handled.")
    
    
    # Define columns to treat outliers
    outlier_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Treat outliers by clipping them to the acceptable range
    train_clean = detect_and_handle_outliers_iqr(train_clean, outlier_columns, factor=1.5, method='clip')
    test_clean = detect_and_handle_outliers_iqr(test_clean, outlier_columns, factor=1.5, method='clip')
    
    print("Outliers treated.")
    
    
    # Set expense values to 0 for CryoSleep passengers
    expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Update the training set
    train_clean.loc[train_clean['CryoSleep'] == True, expense_features] = 0
    
    # Update the test set
    test_clean.loc[test_clean['CryoSleep'] == True, expense_features] = 0
    
    print("Data consistency ensured for CryoSleep passengers.")
    
    
    # Remove duplicates
    train_clean = remove_duplicates(train_clean)
    test_clean = remove_duplicates(test_clean)
    
    # Convert data types
    boolean_columns = ['CryoSleep', 'VIP']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_columns = ['HomePlanet', 'Cabin', 'Destination', 'Name']
    
    # Convert boolean columns
    train_clean = convert_data_types(train_clean, boolean_columns, target_type='bool')
    test_clean = convert_data_types(test_clean, boolean_columns, target_type='bool')
    
    # Convert numerical columns
    train_clean = convert_data_types(train_clean, numerical_columns, target_type='float')
    test_clean = convert_data_types(test_clean, numerical_columns, target_type='float')
    
    # Convert categorical columns
    train_clean = convert_data_types(train_clean, categorical_columns, target_type='str')
    test_clean = convert_data_types(test_clean, categorical_columns, target_type='str')
    
    print("Final checks completed and data types converted.")
    
    
    # Save cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_test.csv'
    
    train_clean.to_csv(cleaned_train_path, index=False)
    test_clean.to_csv(cleaned_test_path, index=False)
    
    print("Cleaned datasets saved.")
    


if __name__ == "__main__":
    generated_code_function()
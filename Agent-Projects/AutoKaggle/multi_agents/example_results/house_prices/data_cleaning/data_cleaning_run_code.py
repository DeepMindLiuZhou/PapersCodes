import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    import numpy as np
    
    # Load the data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/test.csv')
    
    # Define columns with missing values
    numerical_cols_with_missing = [
        'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
    ]
    
    categorical_cols_with_missing = [
        'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
        'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
    ]
    
    # Handle missing values for numerical columns using median
    train_df = fill_missing_values(train_df, numerical_cols_with_missing, method='median')
    test_df = fill_missing_values(test_df, numerical_cols_with_missing, method='median')
    
    # Handle missing values for categorical columns using 'Missing'
    train_df = fill_missing_values(train_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    test_df = fill_missing_values(test_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    
    # Remove columns with more than 50% missing values if needed
    train_df = remove_columns_with_missing_data(train_df, thresh=0.5)
    test_df = remove_columns_with_missing_data(test_df, thresh=0.5)
    
    print("Missing values handled and columns with excessive missing data removed.")
    
    # Define columns to check for outliers
    outlier_columns = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
    
    # Handle outliers by capping them to the IQR bounds
    train_df = detect_and_handle_outliers_iqr(train_df, outlier_columns, factor=1.5, method='clip')
    test_df = detect_and_handle_outliers_iqr(test_df, outlier_columns, factor=1.5, method='clip')
    
    print("Outliers detected and handled.")
    
    # Convert data types if necessary
    numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = train_df.select_dtypes(include=[object]).columns.tolist()
    
    train_df = convert_data_types(train_df, numerical_columns, target_type='float')
    # Assuming numerical_columns is a list of numerical columns including 'SalePrice'
    existing_columns = [col for col in numerical_columns if col in test_df.columns]
    if existing_columns:
        test_df = convert_data_types(test_df, existing_columns, target_type='float')
    else:
        print("None of the specified columns were found in the DataFrame.")
    
    train_df = convert_data_types(train_df, categorical_columns, target_type='str')
    
    # Ensure categorical_columns only contains columns present in both train_df and test_df
    common_categorical_columns = [col for col in categorical_columns if col in train_df.columns and col in test_df.columns]
    
    # Now call the convert_data_types function with the cleaned list
    test_df = convert_data_types(test_df, common_categorical_columns, target_type='str')
    
    # Drop 'FireplaceQu' from train_df if it is not in test_df
    if 'FireplaceQu' not in test_df.columns and 'FireplaceQu' in train_df.columns:
        train_df = train_df.drop(columns=['FireplaceQu'])
    
    # Remove duplicates if any
    train_df = remove_duplicates(train_df, columns=None)
    test_df = remove_duplicates(test_df, columns=None)
    
    print("Data consistency ensured.")
    
    # Save cleaned datasets
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test.csv', index=False)
    
    print("Cleaned datasets saved as 'cleaned_train.csv' and 'cleaned_test.csv'.")
    


if __name__ == "__main__":
    generated_code_function()
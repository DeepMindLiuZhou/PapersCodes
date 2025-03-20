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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Define numerical and categorical features
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Fill missing values in numerical features
    train_df = fill_missing_values(train_df, numerical_features, method='mean')
    test_df = fill_missing_values(test_df, numerical_features, method='mean')
    
    # Fill missing values in categorical features
    train_df = fill_missing_values(train_df, categorical_features, method='mode')
    test_df = fill_missing_values(test_df, categorical_features, method='mode')
    
    
    # Handle outliers in numerical features using IQR method
    train_df = detect_and_handle_outliers_iqr(train_df, numerical_features, factor=1.5, method='clip')
    test_df = detect_and_handle_outliers_iqr(test_df, numerical_features, factor=1.5, method='clip')
    
    
    # Convert categorical features to lowercase
    for feature in categorical_features:
        train_df[feature] = train_df[feature].str.lower()
        test_df[feature] = test_df[feature].str.lower()
    
    # Convert data types for numerical features and 'id' column
    train_df = convert_data_types(train_df, numerical_features, target_type='float')
    test_df = convert_data_types(test_df, numerical_features, target_type='float')
    
    train_df = convert_data_types(train_df, 'id', target_type='int')
    test_df = convert_data_types(test_df, 'id', target_type='int')
    
    
    # Remove duplicates from the dataset
    train_df = remove_duplicates(train_df, keep='first')
    test_df = remove_duplicates(test_df, keep='first')
    
    
    # Save the cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_test.csv'
    
    train_df.to_csv(cleaned_train_path, index=False)
    test_df.to_csv(cleaned_test_path, index=False)
    


if __name__ == "__main__":
    generated_code_function()
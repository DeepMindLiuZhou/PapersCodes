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
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/test.csv')
    
    # Make copies of the original DataFrames
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    # Handle missing values for 'Age'
    fill_missing_values(train_df_copy, columns=['Age'], method='mean')
    fill_missing_values(test_df_copy, columns=['Age'], method='mean')
    
    # Handle missing values for 'Cabin'
    fill_missing_values(train_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    fill_missing_values(test_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    
    # Handle missing values for 'Embarked'
    fill_missing_values(train_df_copy, columns=['Embarked'], method='mode')
    fill_missing_values(test_df_copy, columns=['Embarked'], method='mode')
    
    # Handle missing values for 'Fare'
    fill_missing_values(train_df_copy, columns=['Fare'], method='mean')
    fill_missing_values(test_df_copy, columns=['Fare'], method='mean')
    
    print("Missing values handled successfully.")
    
    # Treat outliers for 'Fare'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Fare'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Fare'], factor=1.5, method='clip')
    
    # Treat outliers for 'Age'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Age'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Age'], factor=1.5, method='clip')
    
    print("Outliers treated successfully.")
    
    # Convert data types
    convert_data_types(train_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(train_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    convert_data_types(test_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(test_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    # Remove duplicates
    remove_duplicates(train_df_copy, inplace=True)
    
    print("Data consistency ensured successfully.")
    
    # Save cleaned datasets
    train_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_train.csv', index=False)
    test_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_test.csv', index=False)
    
    print("Cleaned datasets saved successfully.")
    


if __name__ == "__main__":
    generated_code_function()
import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load data
    train_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/train.csv')
    test_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/test.csv')
    
    # Remove duplicates using the provided tool
    train_data = remove_duplicates(data=train_data)
    test_data = remove_duplicates(data=test_data)
    
    # Print the shape to verify the removal of duplicates
    print("Train data shape after removing duplicates:", train_data.shape)
    print("Test data shape after removing duplicates:", test_data.shape)
    
    
    # Detect and handle outliers in train and test data using the provided tool
    train_data = detect_and_handle_outliers_iqr(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    test_data = detect_and_handle_outliers_iqr(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    
    # Print a summary to verify outliers handling
    print("Outliers handled in train and test data.")
    
    
    # Ensure consistency in data types for train and test data using the provided tool
    train_data = convert_data_types(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    train_data = convert_data_types(data=train_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    test_data = convert_data_types(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    test_data = convert_data_types(data=test_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    # Print data types to verify consistency
    print("Data types after conversion in train data:\n", train_data.dtypes)
    print("Data types after conversion in test data:\n", test_data.dtypes)
    
    
    # Align categorical values for train and test data
    train_data['Geography'] = pd.Categorical(train_data['Geography'])
    train_data['Gender'] = pd.Categorical(train_data['Gender'])
    
    test_data['Geography'] = pd.Categorical(test_data['Geography'])
    test_data['Gender'] = pd.Categorical(test_data['Gender'])
    
    # Print unique values to verify alignment
    print("Unique values in Geography (train):", train_data['Geography'].unique())
    print("Unique values in Geography (test):", test_data['Geography'].unique())
    print("Unique values in Gender (train):", train_data['Gender'].unique())
    print("Unique values in Gender (test):", test_data['Gender'].unique())
    
    
    # Save cleaned data
    train_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv', index=False)
    test_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv', index=False)
    
    print("Cleaned data saved successfully.")
    


if __name__ == "__main__":
    generated_code_function()
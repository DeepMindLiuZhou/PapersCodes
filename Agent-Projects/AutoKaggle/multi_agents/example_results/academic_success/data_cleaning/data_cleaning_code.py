import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Define the data directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    CLEANED_DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    
    # Load the datasets
    train_path = DATA_DIR + 'train.csv'
    test_path = DATA_DIR + 'test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Create copies to avoid modifying original data
    cleaned_train_df = train_df.copy()
    cleaned_test_df = test_df.copy()
    
    # Define numerical and categorical columns
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Task 1: Fill Missing Values
    
    # Fill missing numerical features with median
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=numerical_cols,
        method='median'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=numerical_cols,
        method='median'
    )
    
    # Fill missing categorical features with mode
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=categorical_cols,
        method='mode'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=categorical_cols,
        method='mode'
    )
    
    # Remove columns with more than 60% missing values
    cleaned_train_df = remove_columns_with_missing_data(
        data=cleaned_train_df,
        thresh=0.6
    )
    cleaned_test_df = remove_columns_with_missing_data(
        data=cleaned_test_df,
        thresh=0.6
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step1.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step1.csv', index=False)
    
    
    # Task 2: Detect and Handle Outliers in Numerical Features
    
    # Define the numerical columns again in case some were removed in Task 1
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    # Handle possible removal of some numerical columns in Task 1
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Detect and handle outliers by clipping
    cleaned_train_df = detect_and_handle_outliers_iqr(
        data=cleaned_train_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    cleaned_test_df = detect_and_handle_outliers_iqr(
        data=cleaned_test_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step2.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step2.csv', index=False)
    
    
    # Task 3: Normalize and Standardize Categorical Features
    
    # Define categorical columns again in case some were removed in Task 1
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Handle possible removal of some categorical columns in Task 1
    categorical_cols = [col for col in categorical_cols if col in cleaned_train_df.columns]
    
    # Convert categorical columns to string type
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    # Normalize categorical strings: lowercase and strip whitespaces
    for col in categorical_cols:
        cleaned_train_df[col] = cleaned_train_df[col].str.lower().str.strip()
        cleaned_test_df[col] = cleaned_test_df[col].str.lower().str.strip()
    
    # Placeholder for correcting common typos
    # Define a dictionary for typo corrections if available
    # Example:
    # typo_mapping = {
    #     'marial status': 'marital status',
    #     'scholarship holder': 'scholarship_holder',
    #     # Add more mappings as identified
    # }
    
    # Uncomment and modify the following lines if typo mappings are available
    # for col, mapping in typo_mapping.items():
    #     cleaned_train_df[col] = cleaned_train_df[col].replace(mapping)
    #     cleaned_test_df[col] = cleaned_test_df[col].replace(mapping)
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step3.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step3.csv', index=False)
    
    
    # Task 4: Remove Duplicate Rows and Convert Data Types
    
    # Include 'id' in the columns to consider for duplicate removal
    columns_to_consider_train = cleaned_train_df.columns.tolist()  # This includes 'id'
    cleaned_train_df = remove_duplicates(
        data=cleaned_train_df,
        columns=columns_to_consider_train,
        keep='first'
    )
    
    # Remove duplicates based on all columns except 'id' for the test dataset
    columns_to_consider_test = [col for col in cleaned_test_df.columns if col != 'id']
    cleaned_test_df = remove_duplicates(
        data=cleaned_test_df,
        columns=columns_to_consider_test,
        keep='first'
    )
    
    # Define numerical columns again after Task 1 and Task 2
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP', 'Age at enrollment', 'Application order'
    ]
    
    # Ensure numerical columns exist after previous tasks
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Convert numerical columns to appropriate data types
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=numerical_cols,
        target_type='float'  # Assuming all are floats; adjust if some are integers
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=numerical_cols,
        target_type='float'
    )
    
    # Convert 'id' column to string
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=['id'],
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=['id'],
        target_type='str'
    )
    
    # Verify data types (optional)
    # print(cleaned_train_df.dtypes)
    # print(cleaned_test_df.dtypes)
    
    # Save the final cleaned datasets
    cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train.csv', index=False)
    cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test.csv', index=False)
    
    print("Data Cleaning Phase Completed: 'cleaned_train.csv' and 'cleaned_test.csv' have been saved successfully.")
    


if __name__ == "__main__":
    generated_code_function()
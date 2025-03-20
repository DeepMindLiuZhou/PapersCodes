import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    import os
    
    # Define file paths
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    cleaned_train_file = os.path.join(data_dir, 'cleaned_train.csv')
    cleaned_test_file = os.path.join(data_dir, 'cleaned_test.csv')
    
    # Load the datasets
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    # Define numerical columns
    numerical_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
    
    # TASK 1: Detect and Handle Outliers in Numerical Features Using the IQR Method
    
    # Handle outliers in training data by capping them instead of removing
    try:
        train_df_original = train_df.copy()  # Preserve original training data for comparison
        
        train_df = detect_and_handle_outliers_iqr(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=3.0,              # Increased factor to reduce sensitivity
            method='clip'            # Changed method from 'remove' to 'clip'
        )
        print(f"Training data shape after handling outliers: {train_df.shape}")
        
        # Optional: Log the number of values capped per column
        for col in numerical_cols:
            # Compare before and after to determine if capping occurred
            original_max = train_df_original[col].max()
            original_min = train_df_original[col].min()
            capped_max = train_df[col].max()
            capped_min = train_df[col].min()
            
            if capped_max < original_max or capped_min > original_min:
                print(f"Outliers in '{col}' capped to [{capped_min}, {capped_max}].")
    except Exception as e:
        print(f"Error handling outliers in training data: {e}")
        raise
    
    # Handle outliers in testing data by clipping them
    try:
        test_df = detect_and_handle_outliers_iqr(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=1.5,
            method='clip'
        )
        print(f"Testing data shape after clipping outliers: {test_df.shape}")
    except Exception as e:
        print(f"Error handling outliers in testing data: {e}")
        raise
    
    # Save the cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 1: Outliers handled and cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving cleaned datasets: {e}")
        raise
    
    
    # Reload the cleaned datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Cleaned datasets reloaded successfully.")
    except Exception as e:
        print(f"Error reloading cleaned datasets: {e}")
        raise
    
    # TASK 2: Verify and Ensure Consistency in Categorical Features Across Datasets
    
    # Identify unique categories in both datasets
    train_colors = set(train_df['color'].dropna().unique())
    test_colors = set(test_df['color'].dropna().unique())
    
    print(f"Unique colors in training data before standardization: {train_colors}")
    print(f"Unique colors in testing data before standardization: {test_colors}")
    
    # Step 1: Standardize categories by converting to lowercase
    train_df['color'] = train_df['color'].str.lower()
    test_df['color'] = test_df['color'].str.lower()
    
    # Step 2: Re-identify unique categories after standardization
    train_colors_std = set(train_df['color'].dropna().unique())
    test_colors_std = set(test_df['color'].dropna().unique())
    
    print(f"Unique colors in training data after standardization: {train_colors_std}")
    print(f"Unique colors in testing data after standardization: {test_colors_std}")
    
    # Step 3: Verify consistency
    if not train_colors_std.issubset(test_colors_std) or not test_colors_std.issubset(train_colors_std):
        # Find discrepancies
        all_colors = train_colors_std.union(test_colors_std)
        print("Discrepancies found in 'color' categories. Handling inconsistencies...")
    
        # For this example, we'll map any unseen categories in the test set to 'unknown'
        # Identify categories in test not present in train
        unseen_in_test = test_colors_std - train_colors_std
        if unseen_in_test:
            test_df['color'] = test_df['color'].replace(list(unseen_in_test), 'unknown')
            print(f"Replaced unseen colors in test data with 'unknown': {unseen_in_test}")
    
        # Similarly, handle any categories in train not present in test, if necessary
        unseen_in_train = train_colors_std - test_colors_std
        if unseen_in_train:
            train_df['color'] = train_df['color'].replace(list(unseen_in_train), 'unknown')
            print(f"Replaced unseen colors in training data with 'unknown': {unseen_in_train}")
    else:
        print("No discrepancies found in 'color' categories. No additional handling needed.")
    
    # Step 4: Re-validate unique categories after handling discrepancies
    train_colors_final = set(train_df['color'].dropna().unique())
    test_colors_final = set(test_df['color'].dropna().unique())
    
    print(f"Final unique colors in training data: {train_colors_final}")
    print(f"Final unique colors in testing data: {test_colors_final}")
    
    # Save the standardized datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 2: Categorical feature 'color' standardized and cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving standardized datasets: {e}")
        raise
    
    
    # Reload the cleaned and standardized datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Cleaned and standardized datasets reloaded successfully.")
    except Exception as e:
        print(f"Error reloading cleaned datasets: {e}")
        raise
    
    # TASK 3: Validate and Convert Data Types for All Features
    
    # Define data type conversions for training data
    type_conversions_train = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str',
        'type': 'str'
    }
    
    # Define data type conversions for testing data
    type_conversions_test = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str'
    }
    
    # Convert data types for training data
    try:
        # Convert categorical columns first to avoid issues during numeric conversions
        train_df = convert_data_types(
            data=train_df,
            columns=['color', 'type'],
            target_type='str'
        )
        
        # Convert numerical columns
        train_df = convert_data_types(
            data=train_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        train_df['id'] = pd.to_numeric(train_df['id'], errors='coerce').astype('Int64')
        
        print("Training data types after conversion:")
        print(train_df.dtypes)
    except Exception as e:
        print(f"Error converting data types for training data: {e}")
        raise
    
    # Convert data types for testing data
    try:
        # Convert categorical columns first
        test_df = convert_data_types(
            data=test_df,
            columns=['color'],
            target_type='str'
        )
        
        # Convert numerical columns
        test_df = convert_data_types(
            data=test_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        test_df['id'] = pd.to_numeric(test_df['id'], errors='coerce').astype('Int64')
        
        print("Testing data types after conversion:")
        print(test_df.dtypes)
    except Exception as e:
        print(f"Error converting data types for testing data: {e}")
        raise
    
    # Save the datasets with updated data types
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 3: Data types validated and converted. Cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving datasets after type conversion: {e}")
        raise
    
    
    # Reload the datasets with updated data types
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Datasets with updated data types reloaded successfully.")
    except Exception as e:
        print(f"Error reloading datasets for duplicate removal: {e}")
        raise
    
    # TASK 4: Confirm Absence of Duplicate Records
    
    # Remove duplicates in training data based on 'id'
    try:
        initial_train_shape = train_df.shape
        train_df = remove_duplicates(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_train_shape = train_df.shape
        print(f"Training data shape before removing duplicates: {initial_train_shape}")
        print(f"Training data shape after removing duplicates: {final_train_shape}")
    except Exception as e:
        print(f"Error removing duplicates from training data: {e}")
        raise
    
    # Remove duplicates in testing data based on 'id'
    try:
        initial_test_shape = test_df.shape
        test_df = remove_duplicates(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_test_shape = test_df.shape
        print(f"Testing data shape before removing duplicates: {initial_test_shape}")
        print(f"Testing data shape after removing duplicates: {final_test_shape}")
    except Exception as e:
        print(f"Error removing duplicates from testing data: {e}")
        raise
    
    # Verify absence of duplicates
    train_duplicates = train_df.duplicated(subset=['id']).sum()
    test_duplicates = test_df.duplicated(subset=['id']).sum()
    
    print(f"Number of duplicate 'id's in training data after removal: {train_duplicates}")
    print(f"Number of duplicate 'id's in testing data after removal: {test_duplicates}")
    
    # Save the final cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 4: Duplicate records removed. Cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving datasets after duplicate removal: {e}")
        raise
    


if __name__ == "__main__":
    generated_code_function()